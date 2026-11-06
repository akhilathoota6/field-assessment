"""Task 3 - LLM gateway endpoint that redacts PII from a live SSE stream.

Run:
    python task3_llm_guardrail/mock_llm.py        # or point at a real provider
    python task3_llm_guardrail/gateway.py

    LLM_UPSTREAM_URL   OpenAI-compatible /v1/chat/completions URL
    LLM_API_KEY        provider key, held here and never taken from the client

Pipeline, one upstream event at a time, with no stage holding more than its
bounded buffer:

    provider bytes -> SSEFramer -> JSON -> StreamRedactor per choice -> client

Back-pressure comes for free. The generator only pulls the next upstream chunk
when the server has written the last one, so a slow client slows the upstream
read instead of filling memory. If the client disconnects, the generator is
cancelled and ``async with`` closes the upstream connection.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

sys.path.insert(0, str(Path(__file__).parent))
from redactor import StreamRedactor, redact_text  # noqa: E402

log = logging.getLogger("llm-gateway")

MAX_EVENT_BYTES: Final = 1024 * 1024
_EVENT_END: Final = re.compile(rb"(?:\r\n|\n|\r){2}")
_LINE_END: Final = re.compile(rb"\r\n|\n|\r")


class SSEFramer:
    """Splits a byte stream into SSE events.

    Works on bytes and only decodes complete events, so a multi-byte UTF-8
    character split across network reads cannot be corrupted.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buffer += chunk
        events: list[bytes] = []
        start = 0
        while match := _EVENT_END.search(self._buffer, start):
            events.append(bytes(self._buffer[start : match.start()]))
            start = match.end()
        del self._buffer[:start]
        if len(self._buffer) > MAX_EVENT_BYTES:
            raise ValueError("SSE event too large")
        return events

    def remainder(self) -> bytes:
        rest = bytes(self._buffer)
        self._buffer.clear()
        return rest


def event_data(event: bytes) -> str | None:
    """Return the event's joined data lines, or None if it has none (a comment)."""
    lines = []
    for line in _LINE_END.split(event):
        if line.startswith(b"data:"):
            value = line[5:]
            lines.append(value[1:] if value.startswith(b" ") else value)
    return b"\n".join(lines).decode("utf-8", errors="replace") if lines else None


def sse(payload: dict[str, Any] | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n".encode()


class ChoiceRedactors:
    """One redactor per choice index, since n>1 interleaves independent streams."""

    def __init__(self) -> None:
        self._redactors: dict[int, StreamRedactor] = {}
        self._template: dict[str, Any] = {}

    def apply(self, chunk: dict[str, Any]) -> list[bytes]:
        """Redact one provider chunk and return the frames to send on."""
        for key in ("id", "object", "created", "model"):
            if key in chunk:
                self._template[key] = chunk[key]

        choices = chunk.get("choices")
        if not choices:
            return [sse(chunk)]  # e.g. a usage-only chunk: no text to inspect

        kept = []
        for choice in choices:
            index = choice.get("index", 0)
            redactor = self._redactors.setdefault(index, StreamRedactor())
            delta = choice.get("delta") or {}
            text = delta.get("content")
            safe = redactor.feed(text) if isinstance(text, str) else ""

            finished = choice.get("finish_reason") is not None
            if finished:
                # Held text rides in this same chunk, so it stays in order and
                # reaches clients that stop reading at finish_reason.
                safe += redactor.flush()

            if isinstance(text, str) or safe:
                delta["content"] = safe
            choice["delta"] = delta

            # A delta that was entirely held back has nothing to say yet. The
            # role-bearing first chunk is kept, so the client still sees the
            # stream open right away.
            if finished or safe or set(delta) - {"content"}:
                kept.append(choice)

        if not kept:
            return []
        chunk["choices"] = kept
        return [sse(chunk)]

    def finish(self) -> list[bytes]:
        """Flush anything left when the stream ended with no finish_reason."""
        frames = []
        for index, redactor in self._redactors.items():
            tail = redactor.flush()
            if tail:
                frames.append(self._content_frame(index, tail))
        return frames

    def _content_frame(self, index: int, text: str) -> bytes:
        return sse(
            {
                **self._template,
                "choices": [{"index": index, "delta": {"content": text}, "finish_reason": None}],
            }
        )

    def counts(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for redactor in self._redactors.values():
            for kind, count in redactor.redactions.items():
                totals[kind] = totals.get(kind, 0) + count
        return totals


def gateway_error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


async def redact_stream(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Turn raw provider SSE bytes into redacted SSE bytes."""
    framer = SSEFramer()
    redactors = ChoiceRedactors()
    done = False

    def process(event: bytes) -> list[bytes]:
        nonlocal done
        data = event_data(event)
        if data is None:
            return []  # keep-alive comment
        if data.strip() == "[DONE]":
            done = True
            return redactors.finish() + [sse("[DONE]")]
        try:
            return redactors.apply(json.loads(data))
        except (ValueError, TypeError, AttributeError):
            # Fail closed: content we cannot inspect is never passed through.
            log.warning("dropped an upstream event that could not be parsed")
            return []

    try:
        async for chunk in chunks:
            for event in framer.feed(chunk):
                for frame in process(event):
                    yield frame
        rest = framer.remainder()
        if rest.strip():
            for frame in process(rest):
                yield frame
    except (httpx.HTTPError, ValueError) as exc:
        # Headers have already gone out, so the failure has to travel in-band.
        log.error("upstream stream failed: %s", type(exc).__name__)
        for frame in redactors.finish():
            yield frame
        yield sse({"error": {"code": "upstream_stream_error", "message": "Stream interrupted"}})
        yield sse("[DONE]")
        return

    if not done:
        for frame in redactors.finish():
            yield frame
        yield sse("[DONE]")
    if counts := redactors.counts():
        log.info("redacted %s", counts)  # counts only, never the matched values


def create_app(upstream_url: str, api_key: str = "", client: httpx.AsyncClient | None = None):
    # The read timeout is per chunk, so it catches a stalled stream without
    # limiting how long a healthy generation may run.
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    async def completions(request: Request) -> Response:
        try:
            payload = await request.json()
        except ValueError:
            return gateway_error("invalid_request", "Body must be valid JSON", 400)
        if not isinstance(payload, dict):
            return gateway_error("invalid_request", "Body must be a JSON object", 400)

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        outbound = http.build_request("POST", upstream_url, json=payload, headers=headers)
        try:
            upstream = await http.send(outbound, stream=True)
        except httpx.TimeoutException:
            return gateway_error("upstream_timeout", "The model provider timed out", 504)
        except httpx.TransportError:
            return gateway_error("upstream_unavailable", "The model provider is unavailable", 502)

        if upstream.status_code >= 400:
            await upstream.aclose()
            log.error("upstream returned HTTP %s", upstream.status_code)
            return gateway_error("upstream_error", "The model provider returned an error", 502)

        if not payload.get("stream"):
            try:
                body = json.loads(await upstream.aread())
                for choice in body.get("choices") or []:
                    message = choice.get("message") or {}
                    if isinstance(message.get("content"), str):
                        message["content"] = redact_text(message["content"])
            except (ValueError, AttributeError, TypeError, httpx.HTTPError):
                return gateway_error("upstream_error", "Invalid provider response", 502)
            finally:
                await upstream.aclose()
            return JSONResponse(body)

        async def body_stream() -> AsyncIterator[bytes]:
            try:
                async for frame in redact_stream(upstream.aiter_bytes()):
                    yield frame
            finally:
                await upstream.aclose()

        return StreamingResponse(
            body_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @asynccontextmanager
    async def lifespan(_: Starlette):
        yield
        await http.aclose()

    return Starlette(
        routes=[Route("/v1/chat/completions", completions, methods=["POST"])],
        lifespan=lifespan,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    url = os.environ.get("LLM_UPSTREAM_URL", "http://127.0.0.1:9101/v1/chat/completions")
    uvicorn.run(create_app(url, os.environ.get("LLM_API_KEY", "")), host="127.0.0.1", port=9100)


if __name__ == "__main__":
    main()
