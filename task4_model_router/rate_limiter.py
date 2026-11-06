"""Token-aware sliding-window rate limiter backed by on-disk SQLite.

This is a sliding log: each admitted request is one row (tenant, timestamp,
tokens), and a tenant's usage is the sum of its rows inside the last window.
Unlike a fixed window, it cannot be doubled by bursting across a minute boundary.

Token counts are only known after the model replies, so a request goes through
reserve -> call the model -> settle:

    reserve()    atomically checks the window and records an estimate
    reconcile()  swaps the estimate for the provider's real usage
    release()    removes the row if no model served the request

Reserving before the call is what stops 100 concurrent requests from all seeing
the same free capacity.

Concurrency, at two levels:
    * In-process: every transaction is one synchronous function on a dedicated
      single-thread executor. Coroutines cannot interleave inside a
      check-then-insert, and the event loop never blocks on disk.
    * Across processes: BEGIN IMMEDIATE takes SQLite's write lock before the
      read, so gateway workers sharing the file serialize correctly as well.

Timestamps use wall-clock time rather than a monotonic clock, because they are
persisted and compared across restarts and processes.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

STARTUP_LOCK_WAIT_S = 10.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id     INTEGER PRIMARY KEY,
    tenant TEXT    NOT NULL,
    ts_ms  INTEGER NOT NULL,
    tokens INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_tenant_ts ON token_usage (tenant, ts_ms);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON token_usage (ts_ms);
"""


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    reservation_id: int | None = None
    # None on a denial means waiting will never help: the request exceeds the limit.
    retry_after_ms: int | None = None


class TokenRateLimiter:
    def __init__(
        self,
        db_path: str | Path,
        limit: int = 50_000,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.limit = limit
        self._window_ms = int(window_s * 1000)
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rate-limiter")
        # The connection is only ever used on the executor's one thread.
        self._conn: sqlite3.Connection = self._executor.submit(self._open, str(db_path)).result()

    @staticmethod
    def _open(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA busy_timeout=5000")
        # SQLite can refuse the switch to WAL at once, without honouring
        # busy_timeout, when workers open a new file together. So startup waits
        # for the lock itself. Every step here is safe to repeat.
        deadline = time.monotonic() + STARTUP_LOCK_WAIT_S
        while True:
            try:
                conn.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
                conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, far fewer fsyncs
                conn.executescript(SCHEMA)
                return conn
            except sqlite3.OperationalError as error:
                busy = error.sqlite_errorcode & 0xFF == sqlite3.SQLITE_BUSY
                if not busy or time.monotonic() >= deadline:
                    conn.close()
                    raise
                time.sleep(0.05)

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    async def _run(self, fn: Callable[[], T]) -> T:
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn)

    def _transaction(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = fn(conn)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return result

    async def reserve(self, tenant: str, tokens: int) -> Decision:
        if tokens <= 0:
            raise ValueError("tokens must be positive")

        def work(conn: sqlite3.Connection) -> Decision:
            # The clock is read inside the lock, so a slow lock wait cannot make
            # the check run against a stale "now".
            now = self._now_ms()
            cutoff = now - self._window_ms
            conn.execute("DELETE FROM token_usage WHERE tenant = ? AND ts_ms <= ?", (tenant, cutoff))
            used = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ?", (tenant,)
            ).fetchone()[0]
            remaining = max(0, self.limit - used)

            if tokens > self.limit:
                return Decision(False, self.limit, remaining)
            if used + tokens > self.limit:
                retry = self._retry_after_ms(conn, tenant, used, tokens, now)
                return Decision(False, self.limit, remaining, retry_after_ms=retry)

            cursor = conn.execute(
                "INSERT INTO token_usage (tenant, ts_ms, tokens) VALUES (?, ?, ?)",
                (tenant, now, tokens),
            )
            return Decision(True, self.limit, remaining - tokens, cursor.lastrowid)

        return await self._run(lambda: self._transaction(work))

    def _retry_after_ms(
        self, conn: sqlite3.Connection, tenant: str, used: int, tokens: int, now: int
    ) -> int:
        """Exact wait until enough old rows expire to fit this request."""
        freed = 0
        rows = conn.execute(
            "SELECT ts_ms, tokens FROM token_usage WHERE tenant = ? ORDER BY ts_ms", (tenant,)
        )
        for ts_ms, row_tokens in rows:
            freed += row_tokens
            if used - freed + tokens <= self.limit:
                return max(1, ts_ms + self._window_ms - now)
        return self._window_ms

    async def reconcile(self, reservation_id: int, actual_tokens: int) -> None:
        def work(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE token_usage SET tokens = ? WHERE id = ?",
                (max(0, actual_tokens), reservation_id),
            )

        await self._run(lambda: self._transaction(work))

    async def release(self, reservation_id: int) -> None:
        def work(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM token_usage WHERE id = ?", (reservation_id,))

        await self._run(lambda: self._transaction(work))

    async def usage(self, tenant: str) -> int:
        def work() -> int:
            cutoff = self._now_ms() - self._window_ms
            return self._conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND ts_ms > ?",
                (tenant, cutoff),
            ).fetchone()[0]

        return await self._run(work)

    async def sweep(self) -> int:
        """Evict expired rows for all tenants.

        reserve() only evicts its own tenant, so without this, rows belonging to
        tenants that went quiet would stay on disk forever.
        """

        def work(conn: sqlite3.Connection) -> int:
            cutoff = self._now_ms() - self._window_ms
            return conn.execute("DELETE FROM token_usage WHERE ts_ms <= ?", (cutoff,)).rowcount

        return await self._run(lambda: self._transaction(work))

    async def row_count(self) -> int:
        return await self._run(
            lambda: self._conn.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]
        )

    async def close(self) -> None:
        await self._run(self._conn.close)
        self._executor.shutdown(wait=True)
