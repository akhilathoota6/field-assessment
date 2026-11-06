"""Runs mock providers and the gateway on real ports through each failure scenario.

Run: python task4_model_router/demo.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

sys.path.insert(0, str(Path(__file__).parent))
from app import create_app  # noqa: E402
from rate_limiter import TokenRateLimiter  # noqa: E402
from router import ModelRouter, Provider  # noqa: E402

GATEWAY = "http://127.0.0.1:9200/v1/chat/completions"
HEADERS = {"X-API-Key": "tenant-demo"}
BODY = {"messages": [{"role": "user", "content": "Hello there"}], "max_tokens": 5000}


def mock_provider(name: str, state: dict) -> Starlette:
    async def completions(_: Request) -> Response:
        mode = state[name]
        if mode == "hang":
            await asyncio.sleep(10)
        if mode == "429":
            return JSONResponse({"error": "rate limited"}, status_code=429)
        if mode == "500":
            return Response("Traceback: File /srv/provider/secret.py", status_code=500)
        return JSONResponse(
            {
                "id": f"cmpl-{name}",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": f"hi from {name}"}}],
                "usage": {"total_tokens": 5003},
            }
        )

    return Starlette(routes=[Route("/v1/chat/completions", completions, methods=["POST"])])


@contextmanager
def serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    try:
        yield
    finally:
        server.should_exit = True
        thread.join()


def request(label: str) -> httpx.Response:
    start = time.perf_counter()
    response = httpx.post(GATEWAY, json=BODY, headers=HEADERS, timeout=30)
    elapsed = (time.perf_counter() - start) * 1000
    route = response.headers.get("x-gateway-route", "-")
    print(f"{label:<38} HTTP {response.status_code}  route={route:<10} {elapsed:6.0f}ms")
    return response


def main() -> None:
    state = {"primary": "ok", "secondary": "ok"}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        db_path = Path(folder) / "gateway.db"
        limiter = TokenRateLimiter(db_path)
        router = ModelRouter(
            [
                Provider("primary", "http://127.0.0.1:9201/v1/chat/completions", "p", timeout_s=3.0),
                Provider("secondary", "http://127.0.0.1:9202/v1/chat/completions", "s", timeout_s=3.0),
            ]
        )
        with (
            serve(mock_provider("primary", state), 9201),
            serve(mock_provider("secondary", state), 9202),
            serve(create_app(limiter, router, {"tenant-demo"}), 9200),
        ):
            request("primary healthy")
            state["primary"] = "429"
            request("primary returns 429")
            state["primary"] = "hang"
            request("primary hangs (3000ms deadline)")
            state["primary"] = "500"
            state["secondary"] = "500"
            failed = request("both providers down")
            print(f"  sanitized body: {failed.text}\n")

            state["primary"] = state["secondary"] = "ok"
            print("Burst of 12 requests at about 5,000 tokens each (limit 50,000/min):")
            for i in range(12):
                response = request(f"  request {i + 1}")
            print(f"  remaining={response.headers['x-ratelimit-remaining']}"
                  f" retry-after={response.headers.get('retry-after')}s")
            print(f"  body: {response.text}")
            print(f"\nSQLite file on disk: {db_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
