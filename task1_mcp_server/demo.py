"""Drives server.py with the official MCP client over real stdio.

Run: python task1_mcp_server/demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import Client, MCPError, StdioServerParameters

SERVER = str(Path(__file__).with_name("server.py"))


async def attempt(client: Client, label: str, tool: str, arguments: dict) -> None:
    print(f"\n{label}\n  call: {tool}({arguments})")
    try:
        result = await client.call_tool(tool, arguments)
    except MCPError as exc:
        print(f"  JSON-RPC error {exc.code}: {exc.message}")
        if exc.data:
            print(f"  data: {exc.data}")
        return
    kind = "tool error (isError)" if result.is_error else "ok"
    print(f"  {kind}: {result.content[0].text}")


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[SERVER])
    async with Client(params) as client:
        tools = await client.list_tools()
        print("Tools:", [tool.name for tool in tools.tools])

        await attempt(client, "Valid lookup", "get_customer_record", {"customer_id": "CUST-10001"})
        await attempt(client, "Bad ID format", "get_customer_record", {"customer_id": "CUST-1"})
        await attempt(client, "Unknown customer", "get_customer_record", {"customer_id": "CUST-99999"})
        await attempt(
            client,
            "Valid refund",
            "trigger_refund",
            {"customer_id": "CUST-10001", "amount": 25.5, "reason": "Item arrived damaged"},
        )
        await attempt(
            client,
            "Negative amount and short reason",
            "trigger_refund",
            {"customer_id": "CUST-10001", "amount": -5, "reason": "oops"},
        )
        await attempt(client, "Unknown tool", "drop_tables", {})


if __name__ == "__main__":
    asyncio.run(main())
