from __future__ import annotations

import asyncio
import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "rate_limiter", Path(__file__).with_name("rate_limiter.py")
)
module = importlib.util.module_from_spec(spec)
sys.modules["rate_limiter"] = module
spec.loader.exec_module(module)
TokenRateLimiter = module.TokenRateLimiter


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
async def limiter(tmp_path, clock):
    instance = TokenRateLimiter(tmp_path / "limits.db", limit=50_000, clock=clock)
    yield instance
    await instance.close()


async def test_allows_up_to_the_limit_then_denies(limiter):
    assert (await limiter.reserve("t", 50_000)).allowed
    denied = await limiter.reserve("t", 1)
    assert not denied.allowed
    assert denied.remaining == 0


async def test_remaining_is_reported(limiter):
    assert (await limiter.reserve("t", 12_000)).remaining == 38_000


async def test_tenants_are_isolated(limiter):
    await limiter.reserve("a", 50_000)
    assert (await limiter.reserve("b", 50_000)).allowed


async def test_request_bigger_than_the_limit_can_never_succeed(limiter):
    decision = await limiter.reserve("t", 50_001)
    assert not decision.allowed
    assert decision.retry_after_ms is None


@pytest.mark.parametrize("tokens", [0, -5])
async def test_non_positive_tokens_rejected(limiter, tokens):
    with pytest.raises(ValueError):
        await limiter.reserve("t", tokens)


async def test_window_edge_is_exact(limiter, clock):
    await limiter.reserve("t", 50_000)
    clock.now += 59.999
    assert not (await limiter.reserve("t", 1)).allowed
    clock.now += 0.001  # exactly 60s old: expired
    assert (await limiter.reserve("t", 50_000)).allowed


async def test_window_slides_rather_than_resets(limiter, clock):
    await limiter.reserve("t", 40_000)
    clock.now += 30
    await limiter.reserve("t", 10_000)
    clock.now += 30  # the 40k has expired, the 10k is 30s old and still counts
    assert not (await limiter.reserve("t", 40_001)).allowed
    assert (await limiter.reserve("t", 40_000)).allowed


async def test_no_double_spend_across_a_minute_boundary(limiter, clock):
    """A fixed window would let 100k through around the boundary."""
    clock.now = 1_000_059.0  # just before a clock-minute rollover
    await limiter.reserve("t", 50_000)
    clock.now = 1_000_061.0
    assert not (await limiter.reserve("t", 1)).allowed


async def test_retry_after_is_exact_and_honouring_it_works(limiter, clock):
    await limiter.reserve("t", 30_000)
    clock.now += 10
    await limiter.reserve("t", 20_000)
    clock.now += 5

    decision = await limiter.reserve("t", 25_000)
    assert decision.retry_after_ms == 45_000  # when the 30k row expires

    clock.now += 44.999
    assert not (await limiter.reserve("t", 25_000)).allowed
    clock.now += 0.001
    assert (await limiter.reserve("t", 25_000)).allowed


async def test_retry_after_walks_past_rows_that_free_too_little(limiter, clock):
    await limiter.reserve("t", 10_000)
    clock.now += 10
    await limiter.reserve("t", 40_000)
    decision = await limiter.reserve("t", 20_000)  # needs both rows gone
    assert decision.retry_after_ms == 60_000


async def test_reconcile_replaces_estimate_with_actual(limiter):
    decision = await limiter.reserve("t", 40_000)
    await limiter.reconcile(decision.reservation_id, 1_000)
    assert await limiter.usage("t") == 1_000
    assert (await limiter.reserve("t", 49_000)).allowed


async def test_reconcile_upward_can_overshoot_and_blocks_later_requests(limiter):
    decision = await limiter.reserve("t", 1_000)
    await limiter.reconcile(decision.reservation_id, 60_000)
    assert not (await limiter.reserve("t", 1)).allowed


async def test_release_refunds_the_reservation(limiter):
    decision = await limiter.reserve("t", 50_000)
    await limiter.release(decision.reservation_id)
    assert await limiter.usage("t") == 0
    assert (await limiter.reserve("t", 50_000)).allowed


async def test_reserve_evicts_own_expired_rows(limiter, clock):
    for _ in range(5):
        await limiter.reserve("t", 100)
    clock.now += 60
    await limiter.reserve("t", 100)
    assert await limiter.row_count() == 1


async def test_sweep_evicts_quiet_tenants_and_keeps_live_rows(limiter, clock):
    await limiter.reserve("idle", 100)
    clock.now += 30
    await limiter.reserve("live", 100)
    clock.now += 30
    assert await limiter.sweep() == 1
    assert await limiter.usage("live") == 100
    assert await limiter.row_count() == 1


async def test_concurrent_reservations_never_over_admit(limiter):
    results = await asyncio.gather(*(limiter.reserve("t", 1_000) for _ in range(200)))
    assert sum(r.allowed for r in results) == 50
    assert await limiter.usage("t") == 50_000


async def test_two_limiters_on_one_file_share_state_without_over_admitting(tmp_path, clock):
    """Stands in for two gateway workers: separate connections, one database."""
    path = tmp_path / "shared.db"
    first = TokenRateLimiter(path, limit=50_000, clock=clock)
    second = TokenRateLimiter(path, limit=50_000, clock=clock)
    try:
        results = await asyncio.gather(
            *((first if i % 2 else second).reserve("t", 1_000) for i in range(200))
        )
        assert sum(r.allowed for r in results) == 50
    finally:
        await first.close()
        await second.close()


WORKER = """
import asyncio, sys
sys.path.insert(0, sys.argv[1])
from rate_limiter import TokenRateLimiter

async def main():
    limiter = TokenRateLimiter(sys.argv[2], limit=50_000)
    results = await asyncio.gather(*(limiter.reserve("t", 1_000) for _ in range(50)))
    print(sum(r.allowed for r in results))
    await limiter.close()

asyncio.run(main())
"""


def test_separate_processes_never_over_admit(tmp_path):
    """Four real OS processes race for one tenant's budget in one database file."""
    args = [sys.executable, "-c", WORKER, str(Path(__file__).parent), str(tmp_path / "multi.db")]
    workers = [
        subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(4)
    ]
    outputs = [worker.communicate(timeout=60) for worker in workers]
    assert all(worker.returncode == 0 for worker in workers), outputs
    assert sum(int(stdout) for stdout, _ in outputs) == 50


async def test_state_survives_a_restart(tmp_path, clock):
    path = tmp_path / "persist.db"
    first = TokenRateLimiter(path, limit=50_000, clock=clock)
    await first.reserve("t", 50_000)
    await first.close()

    assert path.stat().st_size > 0  # genuinely on disk
    second = TokenRateLimiter(path, limit=50_000, clock=clock)
    try:
        assert not (await second.reserve("t", 1)).allowed
    finally:
        await second.close()


async def test_failed_transaction_rolls_back_and_limiter_recovers(limiter, monkeypatch):
    def boom(*_):
        raise sqlite3.OperationalError("disk I/O error")

    await limiter.reserve("t", 1_000)
    monkeypatch.setattr(limiter, "_retry_after_ms", boom)
    await limiter.reserve("t", 49_000)
    with pytest.raises(sqlite3.OperationalError):
        await limiter.reserve("t", 1)
    monkeypatch.undo()
    assert await limiter.usage("t") == 50_000  # usable afterwards, nothing corrupted
