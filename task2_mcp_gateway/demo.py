"""Starts the mock server and gateway on real ports, then sends requests through.

Run: python task2_mcp_gateway/demo.py
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
import mock_mcp_server  # noqa: E402
from mint_token import mint  # noqa: E402

SECRET = "demo-secret-demo-secret-demo-secret-1234"
GATEWAY_URL = "http://127.0.0.1:9000/mcp"


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


def show(label, token, payload):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = httpx.post(GATEWAY_URL, json=payload, headers=headers)
    print(f"\n{label}\n  HTTP {response.status_code}  {json.dumps(response.json())[:150]}")


def call(name, id_=1):
    return {"jsonrpc": "2.0", "id": id_, "method": "tools/call", "params": {"name": name}}


def main() -> None:
    downstream = mock_mcp_server.create_app()
    proxy = gateway.create_app("http://127.0.0.1:9001/mcp", SECRET)
    viewer, admin = mint("viewer", SECRET), mint("admin", SECRET)

    with serve(downstream, 9001), serve(proxy, 9000):
        show("tools/list as viewer (forwarded)", viewer, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        show("viewer calls get_status (forwarded)", viewer, call("get_status"))
        before = len(downstream.state.requests)
        show("viewer calls admin_reset_key (blocked)", viewer, call("admin_reset_key"))
        print(f"  requests that reached downstream: {len(downstream.state.requests) - before}")
        show("admin calls admin_reset_key (forwarded)", admin, call("admin_reset_key"))
        show("no token", None, call("get_status"))
        show("viewer smuggles an admin call in a batch", viewer, [call("get_status", 1), call("admin_reset_key", 2)])


if __name__ == "__main__":
    main()
