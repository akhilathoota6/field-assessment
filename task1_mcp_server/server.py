"""Task 1 - Customer support MCP server over stdio.

Run:  python task1_mcp_server/server.py

Design decisions
----------------
* Uses the SDK's low-level ``Server`` rather than ``MCPServer``. The high-level
  class turns handler exceptions into ``isError`` tool results, but the brief
  asks for invalid input to be rejected with a JSON-RPC error code, so handlers
  raise ``MCPError(-32602)`` directly.
* Two kinds of failure are kept apart, as the MCP spec does:
    - protocol errors (bad arguments, unknown tool) -> JSON-RPC ``error`` object
    - execution errors (customer not found, refund too large) -> a normal result
      with ``isError: true``, so the model can read it and recover.
* stdout belongs to JSON-RPC. Logging is configured onto stderr and nothing in
  this file writes to stdout. As a second layer, the SDK's ``stdio_server()``
  points fd 1 at stderr while serving, so a stray ``print`` in a handler or a
  child process cannot corrupt the wire. ``test_server.py`` proves both.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from typing import Annotated, Any, Final

import anyio
import mcp_types as types
from mcp import MCPError
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("customer-mcp")

# --------------------------------------------------------------------------
# Input schemas
# --------------------------------------------------------------------------
# [0-9], not \d: pydantic's regex engine treats \d as Unicode, which would let
# Arabic-Indic digits through. The pattern is also published in the tool's JSON
# Schema, where [0-9] behaves the same in every client's regex engine.
CustomerId = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^CUST-[0-9]{5}$"),
    Field(description="Customer identifier, e.g. CUST-10001"),
]


class StrictInput(BaseModel):
    # strict: no coercing "50" to 50.0 or true to 1. extra=forbid: a misspelled
    # argument is an error, not something silently ignored.
    model_config = ConfigDict(strict=True, extra="forbid")


class GetCustomerRecordInput(StrictInput):
    customer_id: CustomerId


class TriggerRefundInput(StrictInput):
    customer_id: CustomerId
    # allow_inf_nan=False rejects NaN and Infinity, which Python's JSON handling
    # would otherwise let through and which pass a naive "> 0" check (Infinity).
    amount: Annotated[
        float,
        Field(gt=0, allow_inf_nan=False, description="Refund amount, greater than 0"),
    ]
    # Whitespace is stripped first, so ten spaces do not count as a reason.
    reason: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=10, max_length=500),
        Field(description="Why the refund is being issued (at least 10 characters)"),
    ]


# --------------------------------------------------------------------------
# Mock data
# --------------------------------------------------------------------------
CUSTOMERS: Final[dict[str, dict[str, Any]]] = {
    "CUST-10001": {"name": "Asha Rao", "tier": "gold", "refundable_balance": 250.00},
    "CUST-10002": {"name": "Sam Lee", "tier": "silver", "refundable_balance": 80.50},
}


def _ok(payload: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload,
    )


def _tool_failure(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)],
        is_error=True,
    )


def get_customer_record(args: GetCustomerRecordInput) -> types.CallToolResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return _tool_failure(f"Customer {args.customer_id} not found")
    return _ok({"customer_id": args.customer_id, **record})


def trigger_refund(args: TriggerRefundInput) -> types.CallToolResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return _tool_failure(f"Customer {args.customer_id} not found")
    if args.amount > record["refundable_balance"]:
        return _tool_failure(
            f"Refund of {args.amount:.2f} exceeds refundable balance "
            f"of {record['refundable_balance']:.2f}"
        )
    record["refundable_balance"] = round(record["refundable_balance"] - args.amount, 2)
    refund_id = f"RF-{uuid.uuid4().hex[:12]}"
    log.info("refund issued id=%s customer=%s amount=%.2f", refund_id, args.customer_id, args.amount)
    return _ok(
        {
            "refund_id": refund_id,
            "customer_id": args.customer_id,
            "amount": args.amount,
            "status": "processed",
            "remaining_balance": record["refundable_balance"],
        }
    )


TOOLS: Final = {
    "get_customer_record": (
        GetCustomerRecordInput,
        get_customer_record,
        "Look up a customer's profile and refundable balance.",
        types.ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    ),
    "trigger_refund": (
        TriggerRefundInput,
        trigger_refund,
        "Issue a refund to a customer. This moves money and cannot be undone.",
        types.ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False),
    ),
}


# --------------------------------------------------------------------------
# MCP handlers
# --------------------------------------------------------------------------
def _validation_details(exc: ValidationError) -> list[dict[str, str]]:
    # Only field, reason and type: pydantic's "input" echo is left out so a
    # rejected value is never repeated back into logs or the model's context.
    return [
        {
            "field": ".".join(str(part) for part in err["loc"]) or "arguments",
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors(include_url=False, include_context=False, include_input=False)
    ]


async def handle_list_tools(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(
                name=name,
                description=description,
                input_schema=model.model_json_schema(),
                annotations=annotations,
            )
            for name, (model, _, description, annotations) in TOOLS.items()
        ]
    )


async def handle_call_tool(
    ctx: ServerRequestContext, params: types.CallToolRequestParams
) -> types.CallToolResult:
    entry = TOOLS.get(params.name)
    if entry is None:
        log.warning("rejected call to unknown tool")
        raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {params.name}")

    model, handler = entry[0], entry[1]
    try:
        args = model.model_validate(params.arguments or {})
    except ValidationError as exc:
        details = _validation_details(exc)
        log.warning("rejected %s: %s", params.name, details)
        raise MCPError(
            code=types.INVALID_PARAMS,
            message=f"Invalid arguments for {params.name}",
            data={"errors": details},
        ) from exc

    try:
        return handler(args)
    except Exception as exc:
        # Full detail goes to stderr; the client gets a generic -32603.
        log.exception("tool %s failed", params.name)
        raise MCPError(code=types.INTERNAL_ERROR, message="Internal error") from exc


server = Server(
    "customer-support",
    version="1.0.0",
    on_list_tools=handle_list_tools,
    on_call_tool=handle_call_tool,
)


class RejectMalformedLines:
    """Read-stream wrapper that answers lines the transport could not decode.

    The SDK's stdio transport hands undecodable lines to the server as Exception
    items, and the server drops them with only a debug log, so the client would
    wait forever. JSON-RPC 2.0 requires a reply with ``"id": null``: -32700 for
    invalid JSON, -32600 for valid JSON that is not a JSON-RPC message.
    """

    def __init__(self, inner: Any, write_stream: Any) -> None:
        self._inner = inner
        self._write = write_stream

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def _reject(self, exc: Exception) -> None:
        code, message = types.PARSE_ERROR, "Parse error"
        if isinstance(exc, ValidationError):
            if all(e["type"] != "json_invalid" for e in exc.errors(include_input=False)):
                code, message = types.INVALID_REQUEST, "Invalid Request"
        log.warning("rejected undecodable line with code %s", code)
        error = types.ErrorData(code=code, message=message)
        await self._write.send(SessionMessage(types.JSONRPCError(jsonrpc="2.0", id=None, error=error)))

    async def receive(self) -> Any:
        while True:
            item = await self._inner.receive()
            if not isinstance(item, Exception):
                return item
            await self._reject(item)

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __aiter__(self) -> RejectMalformedLines:
        return self

    async def __anext__(self) -> Any:
        while True:
            item = await self._inner.__anext__()
            if not isinstance(item, Exception):
                return item
            await self._reject(item)

    async def __aenter__(self) -> RejectMalformedLines:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> bool | None:
        return await self._inner.__aexit__(*exc_info)


async def main() -> None:
    log.info("starting customer-support MCP server on stdio")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            RejectMalformedLines(read_stream, write_stream),
            write_stream,
            server.create_initialization_options(),
        )
    log.info("stdin closed, shutting down")


if __name__ == "__main__":
    anyio.run(main)
