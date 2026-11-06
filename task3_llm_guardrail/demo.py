"""Starts the mock provider and gateway, streams through both, and compares timing.

Run: python task3_llm_guardrail/demo.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))
import gateway  # noqa: E402
import mock_llm  # noqa: E402

DIRECT = "http://127.0.0.1:9101/v1/chat/completions"
GATEWAY = "http://127.0.0.1:9100/v1/chat/completions"
BODY = {"model": "mock-1", "stream": True, "messages": [{"role": "user", "content": "hi"}]}


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


def measure(url: str, show: bool) -> tuple[float, float, str]:
    start = time.perf_counter()
    first = 0.0
    text = []
    with httpx.stream("POST", url, json=BODY, timeout=30) as response:
        for line in response.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if delta:
                first = first or time.perf_counter() - start
                text.append(delta)
                if show:
                    print(delta, end="", flush=True)
    if show:
        print()
    return first, time.perf_counter() - start, "".join(text)


def main() -> None:
    with serve(mock_llm.app, 9101), serve(gateway.create_app(DIRECT), 9100):
        measure(GATEWAY, False)  # warm up connections so the timing is fair
        print("Streaming through the gateway, as it arrives:\n")
        gw_first, gw_total, gw_text = measure(GATEWAY, True)
        direct_first, direct_total, direct_text = measure(DIRECT, False)

    print(f"\nRaw provider text: {direct_text}")
    print(f"\n{'':10}{'first text':>12}{'total':>10}")
    print(f"{'direct':10}{direct_first * 1000:>10.0f}ms{direct_total * 1000:>8.0f}ms")
    print(f"{'gateway':10}{gw_first * 1000:>10.0f}ms{gw_total * 1000:>8.0f}ms")
    leaked = [s for s in ("john.doe", "123-45", "4111") if s in gw_text]
    print(f"\nPII found in gateway output: {leaked or 'none'}")


if __name__ == "__main__":
    main()
