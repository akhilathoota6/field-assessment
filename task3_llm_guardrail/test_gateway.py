from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(f"task3_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gateway = load("gateway")
R = "[REDACTED]"


def frame(content=None, finish=None, index=0, role=None):
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    body = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": index, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n".encode()


DONE = b"data: [DONE]\n\n"


async def agen(items):
    for item in items:
        yield item


async def collect(chunks):
    return [f async for f in gateway.redact_stream(agen(chunks))]


def parse(frames):
    return [
        json.loads(f[6:])
        for f in b"".join(frames).decode().split("\n\n")
        for f in [f]
        if f.startswith("data: {")
    ]


def text_of(frames, index=0):
    return "".join(
        c["delta"].get("content") or ""
        for event in parse(frames)
        for c in event.get("choices", [])
        if c.get("index", 0) == index
    )


# ---- SSE framing ----------------------------------------------------------
@pytest.mark.parametrize("size", [1, 2, 3, 7, 64])
async def test_byte_splits_at_any_offset_including_inside_utf8(size):
    raw = frame("Héllo 日本 ") + frame("mail a@b.") + frame("com done") + frame(finish="stop") + DONE
    pieces = [raw[i : i + size] for i in range(0, len(raw), size)]
    frames = await collect(pieces)
    assert text_of(frames) == f"Héllo 日本 mail {R} done"
    assert frames[-1] == DONE


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
async def test_all_sse_line_endings(newline):
    raw = (frame("hello ") + DONE).replace(b"\n", newline)
    assert text_of(await collect([raw])) == "hello "


async def test_data_field_without_space_and_comments():
    body = json.loads(frame("hi ")[6:])
    raw = b": keep-alive\n\n" + b"data:" + json.dumps(body).encode() + b"\n\n" + DONE
    assert text_of(await collect([raw])) == "hi "


# ---- Redaction through the pipeline ---------------------------------------
async def test_secret_split_across_frames():
    frames = await collect(
        [frame(t) for t in ["card ", "4111 ", "1111", " 1111 1", "111", " ok"]]
        + [frame(finish="stop"), DONE]
    )
    assert text_of(frames) == f"card {R} ok"
    assert b"4111" not in b"".join(frames)


async def test_held_text_is_flushed_before_finish_and_in_order():
    frames = await collect([frame("first second"), frame(" third", finish="stop"), DONE])
    assert text_of(frames) == "first second third"
    events = parse(frames)
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["choices"][0]["delta"]["content"]


async def test_secret_at_very_end_with_no_finish_reason_or_done():
    frames = await collect([frame("reach me at "), frame("a@b.com")])
    assert text_of(frames) == f"reach me at {R}"
    assert frames[-1] == DONE


async def test_choices_are_tracked_independently():
    frames = await collect(
        [
            frame("a@ex", index=0),
            frame("123-4", index=1),
            frame("ample.com ", index=0),
            frame("5-6789 ", index=1),
            frame(finish="stop", index=0),
            frame(finish="stop", index=1),
            DONE,
        ]
    )
    assert text_of(frames, 0) == f"{R} "
    assert text_of(frames, 1) == f"{R} "


async def test_role_chunk_passes_immediately():
    frames = await collect([frame("", role="assistant")])
    assert parse(frames)[0]["choices"][0]["delta"]["role"] == "assistant"


async def test_usage_only_chunk_is_preserved():
    usage = {"id": "c1", "choices": [], "usage": {"total_tokens": 5}}
    frames = await collect([f"data: {json.dumps(usage)}\n\n".encode(), DONE])
    assert parse(frames)[0]["usage"]["total_tokens"] == 5


@pytest.mark.parametrize(
    "bad",
    [b"data: not json a@b.com\n\n", b'data: ["a@b.com"]\n\n', b'data: {"choices":"a@b.com"}\n\n'],
)
async def test_uninspectable_events_are_dropped_not_passed_through(bad):
    frames = await collect([bad, frame("ok "), DONE])
    assert b"a@b.com" not in b"".join(frames)
    assert text_of(frames) == "ok "


# ---- Streaming behaviour --------------------------------------------------
async def test_output_is_incremental_not_buffered():
    """Frame 1 must come out while the upstream is still blocked on frame 2."""
    release = asyncio.Event()

    async def upstream():
        yield frame("hello world ")
        await release.wait()
        yield frame("more ")
        yield DONE

    stream = gateway.redact_stream(upstream())
    first = await asyncio.wait_for(anext(stream), timeout=1)
    assert b"hello world " in first
    release.set()
    rest = [f async for f in stream]
    assert rest[-1] == DONE


async def test_consumer_controls_the_pace():
    """Back-pressure: an idle consumer means no further upstream reads."""
    pulled = 0

    async def upstream():
        nonlocal pulled
        for _ in range(1000):
            pulled += 1
            yield frame("word ")

    stream = gateway.redact_stream(upstream())
    await anext(stream)
    await asyncio.sleep(0.01)
    assert pulled <= 2
    await stream.aclose()


async def test_mid_stream_failure_flushes_safely_and_reports_in_band():
    async def upstream():
        yield frame("partial a@b.com")
        raise httpx.ReadError("10.1.2.3 connection reset")

    frames = [f async for f in gateway.redact_stream(upstream())]
    joined = b"".join(frames)
    assert b"upstream_stream_error" in joined
    assert b"10.1.2.3" not in joined
    assert b"a@b.com" not in joined
    assert R.encode() in joined
    assert frames[-1] == DONE


async def test_oversized_event_is_cut_off():
    frames = await collect([b"data: " + b"x" * (1024 * 1024 + 10)])
    assert b"upstream_stream_error" in b"".join(frames)


# ---- HTTP endpoint --------------------------------------------------------
def gateway_client(handler, api_key="provider-secret"):
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = gateway.create_app("http://provider/v1/chat/completions", api_key, client=upstream)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


async def test_endpoint_streams_redacted_sse():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=agen([frame("ssn 123-"), frame("45-6789 "), frame(finish="stop"), DONE]),
        )

    async with gateway_client(handler) as http:
        response = await http.post(
            "/v1/chat/completions",
            json={"stream": True, "model": "m", "messages": []},
            headers={"Authorization": "Bearer client-token"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert R in response.text
    assert "6789" not in response.text
    assert seen["auth"] == "Bearer provider-secret"  # client's token is not relayed
    assert seen["body"]["model"] == "m"


async def test_endpoint_non_streaming_is_redacted():
    def handler(_):
        return httpx.Response(
            200, json={"choices": [{"index": 0, "message": {"content": "mail a@b.com now"}}]}
        )

    async with gateway_client(handler) as http:
        response = await http.post("/v1/chat/completions", json={"model": "m"})
    assert response.json()["choices"][0]["message"]["content"] == f"mail {R} now"


@pytest.mark.parametrize("status", [400, 401, 429, 500])
async def test_upstream_errors_are_sanitized(status):
    def handler(_):
        return httpx.Response(status, json={"error": "sk-secret at /srv/internal/app.py line 9"})

    async with gateway_client(handler) as http:
        response = await http.post("/v1/chat/completions", json={"stream": True})
    assert response.status_code == 502
    assert "sk-secret" not in response.text
    assert response.json()["error"]["code"] == "upstream_error"


@pytest.mark.parametrize(
    "exc, status, code",
    [
        (httpx.ConnectError("10.0.0.1"), 502, "upstream_unavailable"),
        (httpx.ReadTimeout("slow"), 504, "upstream_timeout"),
    ],
)
async def test_upstream_connection_failures(exc, status, code):
    def handler(_):
        raise exc

    async with gateway_client(handler) as http:
        response = await http.post("/v1/chat/completions", json={"stream": True})
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize("body", [b"{bad", b"[1]"])
async def test_bad_client_body(body):
    async with gateway_client(lambda _: httpx.Response(200)) as http:
        response = await http.post("/v1/chat/completions", content=body)
    assert response.status_code == 400
