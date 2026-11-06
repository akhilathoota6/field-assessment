"""Primary/secondary model routing with a hard per-attempt deadline.

Attempts run in sequence, not as a hedged race, so only one provider is ever
billed for a request, and a primary reply arriving after its deadline cannot
produce a second answer: ``asyncio.timeout`` cancels the attempt and closes its
connection before the secondary is tried.

The deadline is ``asyncio.timeout`` around the whole attempt rather than an httpx
timeout. httpx timeouts apply per network operation, so a provider that drips one
byte every 2 seconds would never trip them. The brief asks for 3000ms in total.

Failover policy:
    429, 408, 5xx, 401/403/404, timeout, connection error, unusable body -> try next
    400, 413, 422 -> stop: the request itself is bad, and the backup would only
    repeat the rejection while adding latency and cost.

401/403/404 fail over because at this layer they mean our credentials or
configuration for that one provider are wrong, not that the tenant did anything.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Final

import httpx

log = logging.getLogger("model-router")

CLIENT_FAULT_STATUSES: Final = frozenset({400, 413, 422})


@dataclass(frozen=True, slots=True)
class Provider:
    role: str  # "primary" or "secondary": the only name that may leave the gateway
    url: str
    model: str
    api_key: str = ""
    timeout_s: float = 3.0


@dataclass(frozen=True, slots=True)
class RouteResult:
    role: str
    body: dict[str, Any]
    total_tokens: int | None


class ProviderFailure(Exception):
    def __init__(self, reason: str, *, failover: bool = True) -> None:
        super().__init__(reason)
        self.reason = reason
        self.failover = failover


class RequestRejected(Exception):
    """A provider rejected the request itself."""


class AllProvidersFailed(Exception):
    pass


class ModelRouter:
    def __init__(self, providers: list[Provider], client: httpx.AsyncClient | None = None) -> None:
        self._providers = providers
        # httpx's own timeouts are off: asyncio.timeout is the single deadline.
        self._client = client or httpx.AsyncClient(timeout=None)

    async def complete(self, payload: dict[str, Any]) -> RouteResult:
        for provider in self._providers:
            try:
                return await self._attempt(provider, payload)
            except ProviderFailure as failure:
                # Details stay in the logs and never reach the client.
                log.warning("provider=%s failed reason=%s", provider.role, failure.reason)
                if not failure.failover:
                    raise RequestRejected from failure
        raise AllProvidersFailed

    async def _attempt(self, provider: Provider, payload: dict[str, Any]) -> RouteResult:
        headers = {"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else {}
        body = {**payload, "model": provider.model, "stream": False}
        try:
            async with asyncio.timeout(provider.timeout_s):
                response = await self._client.post(provider.url, json=body, headers=headers)
        except TimeoutError:
            raise ProviderFailure("timeout") from None
        except httpx.HTTPError as exc:
            raise ProviderFailure(f"transport:{type(exc).__name__}") from None

        if response.status_code in CLIENT_FAULT_STATUSES:
            raise ProviderFailure(f"http:{response.status_code}", failover=False)
        if response.status_code != 200:
            raise ProviderFailure(f"http:{response.status_code}")

        try:
            parsed = response.json()
            choices = parsed["choices"]
        except (ValueError, KeyError, TypeError):
            raise ProviderFailure("malformed_body") from None
        if not isinstance(choices, list) or not choices:
            raise ProviderFailure("empty_choices")

        usage = parsed.get("usage")
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        valid = isinstance(total, int) and not isinstance(total, bool) and total >= 0
        return RouteResult(provider.role, parsed, total if valid else None)

    async def close(self) -> None:
        await self._client.aclose()
