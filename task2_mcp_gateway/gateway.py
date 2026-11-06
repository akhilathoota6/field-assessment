"""Task 2 - MCP security gateway: an authenticating reverse proxy for JSON-RPC.

    agent --Bearer JWT--> gateway --(no client token)--> downstream MCP server

Run:
    set MCP_GATEWAY_JWT_SECRET=<at least 32 characters>
    python task2_mcp_gateway/gateway.py

Rules enforced on every POST /mcp:
    * a valid, unexpired HS256 JWT with a known ``role`` claim (else HTTP 401)
    * ``tools/call`` on an ``admin_*`` tool needs role ``admin``; otherwise the
      gateway answers -32001 itself and the downstream server is never called
    * everything else, including ``tools/list``, is forwarded byte-for-byte

Bypass attempts this design closes (each has a test):
    * JSON-RPC batches: every element is authorized, not just the first
    * duplicate JSON keys: rejected, so gateway and downstream cannot disagree
      about which "method" or "name" is the real one
    * name variants (ADMIN_x, " admin_x", full-width letters): normalized first
    * alg=none and algorithm-confusion JWTs: only HS256 is accepted
    * spoofed identity headers: stripped from inbound requests
"""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from contextlib import asynccontextmanager
from typing import Any, Final

import httpx
import jwt
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

log = logging.getLogger("mcp-gateway")

# JSON-RPC 2.0 codes, plus the -32000..-32099 range reserved for implementations.
PARSE_ERROR: Final = -32700
INVALID_REQUEST: Final = -32600
INVALID_PARAMS: Final = -32602
UNAUTHENTICATED: Final = -32000
UNAUTHORIZED_TOOL: Final = -32001
DOWNSTREAM_ERROR: Final = -32002

KNOWN_ROLES: Final = frozenset({"admin", "viewer"})
ADMIN_TOOL_PREFIX: Final = "admin_"
JWT_AUDIENCE: Final = "mcp-gateway"
MAX_BODY_BYTES: Final = 1024 * 1024

# Hop-by-hop headers (RFC 9110) apply to one connection and must not be relayed.
HOP_BY_HOP: Final = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
# Authorization is dropped because the MCP spec forbids token passthrough: the
# client's token was issued for this gateway, not for the downstream server.
STRIPPED_REQUEST_HEADERS: Final = HOP_BY_HOP | {"authorization", "host", "content-length"}


class BadRequest(ValueError):
    """The body cannot be treated as JSON-RPC."""


def rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BadRequest(f"duplicate key {key!r}")
        result[key] = value
    return result


def parse_body(body: bytes) -> Any:
    try:
        return json.loads(body, object_pairs_hook=_reject_duplicate_keys)
    except (ValueError, RecursionError) as exc:  # includes JSONDecodeError, BadRequest
        raise BadRequest(str(exc)) from exc


def authenticate(header: str | None, secret: str) -> dict[str, Any] | None:
    """Return verified claims for a role-bearing token, or None."""
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    try:
        claims = jwt.decode(
            token.strip(),
            secret,
            algorithms=["HS256"],  # pinned: never trust the token's own "alg"
            audience=JWT_AUDIENCE,
            options={"require": ["exp", "sub", "aud"]},
        )
    except jwt.InvalidTokenError:
        return None
    return claims if claims.get("role") in KNOWN_ROLES else None


def authorize(message: Any, role: str) -> dict[str, Any] | None:
    """Return a JSON-RPC error if ``message`` must not be forwarded, else None."""
    if not isinstance(message, dict):
        return rpc_error(None, INVALID_REQUEST, "Invalid Request")

    request_id = message.get("id")
    if isinstance(request_id, bool) or not isinstance(request_id, (str, int, float, type(None))):
        return rpc_error(None, INVALID_REQUEST, "Invalid Request: bad id")
    if message.get("jsonrpc") != "2.0":
        return rpc_error(request_id, INVALID_REQUEST, "Invalid Request: jsonrpc must be 2.0")

    method = message.get("method")
    if method is None:
        # A client's reply to a server-initiated request carries no method.
        if "id" in message and ("result" in message or "error" in message):
            return None
        return rpc_error(request_id, INVALID_REQUEST, "Invalid Request: missing method")
    if not isinstance(method, str):
        return rpc_error(request_id, INVALID_REQUEST, "Invalid Request: method must be a string")
    if normalize(method) != "tools/call":
        return None

    # A tools/call without an id would be a notification, which could execute
    # downstream while returning nothing. MCP requires an id, so refuse it.
    if request_id is None:
        return rpc_error(None, INVALID_REQUEST, "Invalid Request: tools/call requires an id")
    params = message.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if not isinstance(name, str) or not name:
        return rpc_error(request_id, INVALID_PARAMS, "Invalid params: name must be a string")
    if normalize(name).startswith(ADMIN_TOOL_PREFIX) and role != "admin":
        return rpc_error(request_id, UNAUTHORIZED_TOOL, "Unauthorized Tool Call")
    return None


def create_app(
    downstream_url: str,
    jwt_secret: str,
    client: httpx.AsyncClient | None = None,
) -> Starlette:
    if len(jwt_secret) < 32:
        raise ValueError("JWT secret must be at least 32 characters")

    http = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    async def read_body(request: Request) -> bytes:
        # Counted while streaming: Content-Length is client-supplied and optional.
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BODY_BYTES:
                raise OverflowError
        return bytes(body)

    def forwarded_headers(request: Request, claims: dict[str, Any]) -> dict[str, str]:
        headers = {
            key: value
            for key, value in request.headers.items()
            if key not in STRIPPED_REQUEST_HEADERS and not key.startswith("x-gateway-")
        }
        headers["x-gateway-role"] = claims["role"]
        headers["x-gateway-subject"] = str(claims["sub"])
        return headers

    async def forward(
        request: Request, claims: dict[str, Any], body: bytes, request_id: Any = None
    ) -> Response:
        """Relay to the downstream server, streaming so SSE responses keep working."""
        # GET opens a long-lived SSE stream, so only POST gets a read timeout.
        timeout = httpx.USE_CLIENT_DEFAULT
        if request.method == "GET":
            timeout = httpx.Timeout(30.0, connect=5.0, read=None)
        outbound = http.build_request(
            request.method,
            downstream_url,
            headers=forwarded_headers(request, claims),
            content=body,
            timeout=timeout,
        )
        try:
            upstream = await http.send(outbound, stream=True)
        except httpx.TimeoutException:
            log.error("downstream timed out")
            return JSONResponse(
                rpc_error(request_id, DOWNSTREAM_ERROR, "Downstream timeout"), status_code=504
            )
        except httpx.TransportError as exc:
            log.error("downstream unreachable: %s", type(exc).__name__)
            return JSONResponse(
                rpc_error(request_id, DOWNSTREAM_ERROR, "Downstream unavailable"), status_code=502
            )
        return StreamingResponse(
            upstream.aiter_raw(),  # raw: Content-Encoding stays valid end to end
            status_code=upstream.status_code,
            headers={k: v for k, v in upstream.headers.items() if k not in HOP_BY_HOP},
            background=BackgroundTask(upstream.aclose),
        )

    async def forward_partial_batch(
        request: Request,
        claims: dict[str, Any],
        allowed: list[Any],
        denied: list[dict[str, Any]],
    ) -> Response:
        """Forward only the permitted batch elements, then merge in the denials."""
        response = await forward(request, claims, json.dumps(allowed).encode())
        if not isinstance(response, StreamingResponse):
            return response  # downstream unreachable: already a sanitized error
        chunks = [chunk async for chunk in response.body_iterator]
        if response.background:
            await response.background()
        raw = b"".join(chunks)
        replies: list[Any] = []
        if raw and response.status_code < 300:
            try:
                parsed = json.loads(raw)
            except ValueError:
                return JSONResponse(
                    rpc_error(None, DOWNSTREAM_ERROR, "Invalid downstream response"),
                    status_code=502,
                )
            replies = parsed if isinstance(parsed, list) else [parsed]
        elif raw:
            return Response(raw, response.status_code, media_type="application/json")
        return JSONResponse(replies + denied)

    async def mcp_endpoint(request: Request) -> Response:
        claims = authenticate(request.headers.get("authorization"), jwt_secret)
        if claims is None:
            return JSONResponse(
                rpc_error(None, UNAUTHENTICATED, "Missing or invalid bearer token"),
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            body = await read_body(request)
        except OverflowError:
            return JSONResponse(
                rpc_error(None, INVALID_REQUEST, "Request too large"), status_code=413
            )
        except ClientDisconnect:
            return Response(status_code=499)

        if request.method != "POST":  # GET (SSE) and DELETE (end session) have no payload
            return await forward(request, claims, body)

        try:
            payload = parse_body(body)
        except BadRequest:
            return JSONResponse(rpc_error(None, PARSE_ERROR, "Parse error"), status_code=400)

        role = claims["role"]
        if isinstance(payload, list):
            if not payload:
                return JSONResponse(
                    rpc_error(None, INVALID_REQUEST, "Invalid Request: empty batch"),
                    status_code=400,
                )
            verdicts = [authorize(message, role) for message in payload]
            denied = [verdict for verdict in verdicts if verdict]
            if not denied:
                return await forward(request, claims, body)
            for verdict in denied:
                log.warning(
                    "blocked batch element sub=%r role=%s code=%s",
                    claims["sub"],
                    role,
                    verdict["error"]["code"],
                )
            allowed = [m for m, verdict in zip(payload, verdicts) if verdict is None]
            if not allowed:
                return JSONResponse(denied)
            return await forward_partial_batch(request, claims, allowed, denied)

        verdict = authorize(payload, role)
        if verdict is not None:
            log.warning(
                "blocked call sub=%r role=%s code=%s", claims["sub"], role, verdict["error"]["code"]
            )
            # HTTP 200: the transport worked and the failure is in the JSON-RPC
            # layer, which is where the client's error handling looks for it.
            return JSONResponse(verdict)
        return await forward(request, claims, body, payload.get("id"))

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    @asynccontextmanager
    async def lifespan(_: Starlette):
        yield
        await http.aclose()

    return Starlette(
        routes=[
            Route("/mcp", mcp_endpoint, methods=["GET", "POST", "DELETE"]),
            Route("/healthz", health, methods=["GET"]),
        ],
        lifespan=lifespan,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    secret = os.environ.get("MCP_GATEWAY_JWT_SECRET")
    if not secret:
        raise SystemExit("Set MCP_GATEWAY_JWT_SECRET (at least 32 characters)")
    downstream = os.environ.get("MCP_DOWNSTREAM_URL", "http://127.0.0.1:9001/mcp")
    uvicorn.run(create_app(downstream, secret), host="127.0.0.1", port=9000)


if __name__ == "__main__":
    main()
