"""OpenAI-compatible mock provider that splits PII across tokens on purpose.

Run: python task3_llm_guardrail/mock_llm.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

TOKENS = [
    "Sure", ",", " the", " customer", " email", " is", " john", ".doe", "@exa", "mple", ".com",
    ",", " their", " SSN", " is", " 123", "-4", "5-67", "89", ",", " and", " the", " card",
    " is", " 4111", " 1111", " 11", "11 1", "111", ".", " Let", " me", " know", "!",
]  # fmt: skip
TOKEN_DELAY_S = float(os.environ.get("MOCK_TOKEN_DELAY_S", "0.03"))


def chunk(delta: dict, finish_reason: str | None = None) -> bytes:
    body = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "mock-1",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(body)}\n\n".encode()


async def generate():
    yield chunk({"role": "assistant", "content": ""})
    for token in TOKENS:
        await asyncio.sleep(TOKEN_DELAY_S)
        yield chunk({"content": token})
    yield chunk({}, "stop")
    yield b"data: [DONE]\n\n"


async def completions(request: Request) -> Response:
    payload = await request.json()
    if payload.get("stream"):
        return StreamingResponse(generate(), media_type="text/event-stream")
    return JSONResponse(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "mock-1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(TOKENS)},
                    "finish_reason": "stop",
                }
            ],
        }
    )


app = Starlette(routes=[Route("/v1/chat/completions", completions, methods=["POST"])])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9101, log_level="warning")
