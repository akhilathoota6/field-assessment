"""Downstream mock MCP server for Task 2. Run: python task2_mcp_gateway/mock_mcp_server.py"""

from __future__ import annotations

import json
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

TOOLS = [
    {"name": "get_status", "description": "Read service status", "inputSchema": {"type": "object"}},
    {"name": "admin_reset_key", "description": "Rotate an API key", "inputSchema": {"type": "object"}},
]


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    if "id" not in message or method is None:
        return None  # a notification, or a client's reply to a server request
    request_id = message["id"]
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mock-downstream", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name = message["params"]["name"]
        result = {"content": [{"type": "text", "text": f"executed {name}"}]}
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "Method not found"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def create_app() -> Starlette:
    async def mcp(request: Request) -> Response:
        raw = await request.body()
        request.app.state.requests.append({"body": raw, "headers": dict(request.headers)})
        payload = json.loads(raw)
        if isinstance(payload, list):
            replies = [r for r in map(handle, payload) if r is not None]
            return JSONResponse(replies) if replies else Response(status_code=202)
        reply = handle(payload)
        return JSONResponse(reply) if reply else Response(status_code=202)

    app = Starlette(routes=[Route("/mcp", mcp, methods=["POST"])])
    app.state.requests = []  # lets tests prove what did, and did not, get through
    return app


if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1", port=9001)
