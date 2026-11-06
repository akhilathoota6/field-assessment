"""Task 4 - gateway HTTP surface: auth, rate limiting, routing, error sanitization.

Every failure leaves through ``error_response``, so clients only ever see this:

    {"error": {"type": "rate_limit_error", "code": "tenant_rate_limited",
               "message": "...", "request_id": "req_ab12", "retryable": true}}

The request_id ties a sanitized reply to the full detail in the gateway's logs.

Run: python task4_model_router/app.py   (settings are listed in main() below)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any, Final, Literal

import uvicorn
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

sys.path.insert(0, str(Path(__file__).parent))
from rate_limiter import TokenRateLimiter  # noqa: E402
from router import AllProvidersFailed, ModelRouter, Provider, RequestRejected  # noqa: E402

log = logging.getLogger("gateway")

MAX_BODY_BYTES: Final = 1024 * 1024
DEFAULT_MAX_TOKENS: Final = 1024
CHARS_PER_TOKEN: Final = 4  # rough heuristic; reconcile() corrects it after the call
SWEEP_INTERVAL_S: Final = 60.0
PROVIDER_RETRY_AFTER_S: Final = 5


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(min_length=1)]


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None  # accepted but not honoured: routing belongs to the gateway
    messages: Annotated[list[Message], Field(min_length=1)]
    max_tokens: Annotated[int, Field(gt=0)] = DEFAULT_MAX_TOKENS
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    stream: Literal[False] = False  # this module serves non-streaming completions

    def estimated_tokens(self) -> int:
        chars = sum(len(message.content) for message in self.messages)
        return math.ceil(chars / CHARS_PER_TOKEN) + self.max_tokens

    def upstream_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude={"model", "stream"}, exclude_none=True)


def error_response(
    request_id: str,
    status: int,
    error_type: str,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "type": error_type,
                "code": code,
                "message": message,
                "request_id": request_id,
                "retryable": retryable,
            }
        },
        status_code=status,
        headers={"X-Request-Id": request_id, **(headers or {})},
    )


def hash_key(api_key: str) -> str:
    # Raw keys never touch the disk. The key is high-entropy, so plain SHA-256
    # is enough here; slow password hashing would only add per-request latency.
    return hashlib.sha256(api_key.encode()).hexdigest()


def extract_key(request: Request) -> str | None:
    if key := request.headers.get("x-api-key"):
        return key.strip() or None
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None


async def read_limited(request: Request) -> bytes | None:
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            return None
    return bytes(body)


def create_app(
    limiter: TokenRateLimiter, router: ModelRouter, tenant_keys: set[str]
) -> Starlette:
    tenants = {hash_key(key) for key in tenant_keys}

    async def handle(request: Request, request_id: str) -> Response:
        key = extract_key(request)
        if key is None:
            return error_response(
                request_id, 401, "authentication_error", "missing_api_key", "An API key is required"
            )
        tenant = hash_key(key)
        if tenant not in tenants:
            return error_response(
                request_id, 401, "authentication_error", "invalid_api_key", "The API key is invalid"
            )

        raw = await read_limited(request)
        if raw is None:
            return error_response(
                request_id, 413, "invalid_request_error", "request_too_large", "Body is too large"
            )
        try:
            completion = CompletionRequest.model_validate_json(raw)
        except ValidationError as exc:
            # Field paths only. Pydantic's full text echoes the input back.
            fields = sorted({".".join(map(str, e["loc"])) or "body" for e in exc.errors()})
            return error_response(
                request_id,
                400,
                "invalid_request_error",
                "invalid_request",
                f"Invalid or missing fields: {', '.join(fields)}",
            )

        estimate = completion.estimated_tokens()
        decision = await limiter.reserve(tenant, estimate)
        limit_headers = {
            "X-RateLimit-Limit": str(decision.limit),
            "X-RateLimit-Remaining": str(decision.remaining),
        }
        if not decision.allowed:
            if decision.retry_after_ms is None:
                return error_response(
                    request_id,
                    413,
                    "invalid_request_error",
                    "request_exceeds_token_limit",
                    "This request is larger than the per-minute token limit",
                    headers=limit_headers,
                )
            seconds = math.ceil(decision.retry_after_ms / 1000)
            return error_response(
                request_id,
                429,
                "rate_limit_error",
                "tenant_rate_limited",
                "Token rate limit exceeded",
                retryable=True,
                headers={**limit_headers, "Retry-After": str(seconds)},
            )

        reservation = decision.reservation_id
        settled = False
        try:
            result = await router.complete(completion.upstream_payload())
            actual = result.total_tokens if result.total_tokens is not None else estimate
            await limiter.reconcile(reservation, actual)
            settled = True
        except RequestRejected:
            return error_response(
                request_id,
                400,
                "invalid_request_error",
                "request_rejected",
                "The model could not process this request",
                headers=limit_headers,
            )
        except AllProvidersFailed:
            return error_response(
                request_id,
                503,
                "upstream_error",
                "all_providers_unavailable",
                "No model provider is available right now",
                retryable=True,
                headers={**limit_headers, "Retry-After": str(PROVIDER_RETRY_AFTER_S)},
            )
        finally:
            # Runs for handled failures and for cancellation (client hung up), so
            # a tenant is never charged for a request that no model served.
            if not settled:
                await asyncio.shield(limiter.release(reservation))

        remaining = max(0, decision.remaining + estimate - actual)
        body = result.body
        return JSONResponse(
            {
                "id": body.get("id", request_id),
                "object": "chat.completion",
                "model": completion.model or "default",
                "choices": body["choices"],
                "usage": body.get("usage"),
            },
            headers={
                "X-Request-Id": request_id,
                "X-RateLimit-Limit": str(decision.limit),
                "X-RateLimit-Remaining": str(remaining),
                "X-Gateway-Route": result.role,
            },
        )

    async def completions(request: Request) -> Response:
        request_id = f"req_{uuid.uuid4().hex[:16]}"
        try:
            return await handle(request, request_id)
        except sqlite3.Error:
            # Fail closed: with no limiter there is no way to enforce quotas.
            log.exception("rate limiter failure request_id=%s", request_id)
            return error_response(
                request_id,
                503,
                "gateway_error",
                "rate_limiter_unavailable",
                "The gateway is temporarily unavailable",
                retryable=True,
                headers={"Retry-After": str(PROVIDER_RETRY_AFTER_S)},
            )
        except Exception:
            log.exception("unhandled error request_id=%s", request_id)
            return error_response(
                request_id, 500, "gateway_error", "internal_error", "An internal error occurred"
            )

    async def http_error(_: Request, exc: HTTPException) -> Response:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return error_response(
            f"req_{uuid.uuid4().hex[:16]}",
            exc.status_code,
            "invalid_request_error",
            code,
            "The requested route is not available",
        )

    async def sweeper() -> None:
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_S)
            try:
                evicted = await limiter.sweep()
                if evicted:
                    log.info("evicted %d expired usage rows", evicted)
            except sqlite3.Error:
                log.exception("sweep failed")

    @asynccontextmanager
    async def lifespan(_: Starlette):
        task = asyncio.create_task(sweeper())
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await router.close()
        await limiter.close()

    return Starlette(
        routes=[Route("/v1/chat/completions", completions, methods=["POST"])],
        exception_handlers={HTTPException: http_error},
        lifespan=lifespan,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    env = os.environ.get
    keys = {k.strip() for k in env("GATEWAY_TENANT_KEYS", "").split(",") if k.strip()}
    if not keys:
        raise SystemExit("Set GATEWAY_TENANT_KEYS to a comma-separated list of tenant API keys")
    providers = [
        Provider(
            "primary",
            env("PRIMARY_URL", "http://127.0.0.1:9201/v1/chat/completions"),
            env("PRIMARY_MODEL", "primary-model"),
            env("PRIMARY_API_KEY", ""),
            timeout_s=3.0,
        ),
        Provider(
            "secondary",
            env("SECONDARY_URL", "http://127.0.0.1:9202/v1/chat/completions"),
            env("SECONDARY_MODEL", "backup-model"),
            env("SECONDARY_API_KEY", ""),
            timeout_s=float(env("SECONDARY_TIMEOUT_S", "10")),
        ),
    ]
    limiter = TokenRateLimiter(env("GATEWAY_DB_PATH", "gateway.db"))
    uvicorn.run(create_app(limiter, ModelRouter(providers), keys), host="127.0.0.1", port=9200)


if __name__ == "__main__":
    main()
