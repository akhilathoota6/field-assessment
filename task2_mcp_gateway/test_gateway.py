from __future__ import annotations

import base64
import importlib.util
import json
import sys
import time
from pathlib import Path

import httpx
import jwt
import pytest

HERE = Path(__file__).parent
SECRET = "test-secret-that-is-at-least-32-characters-long"


def load(name):
    spec = importlib.util.spec_from_file_location(f"task2_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gateway = load("gateway")
mock = load("mock_mcp_server")
mint = load("mint_token").mint


@pytest.fixture
def downstream():
    return mock.create_app()


@pytest.fixture
async def client(downstream):
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=downstream))
    app = gateway.create_app("http://downstream/mcp", SECRET, client=upstream)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    ) as http:
        yield http
    await upstream.aclose()


def auth(role="viewer"):
    return {"Authorization": f"Bearer {mint(role, SECRET)}"}


def tool_call(name, id_=1):
    return {"jsonrpc": "2.0", "id": id_, "method": "tools/call", "params": {"name": name}}


def sent_to_downstream(app):
    return [json.loads(r["body"]) for r in app.state.requests]


# ---- Core requirements ----------------------------------------------------
async def test_tools_list_forwarded_byte_for_byte(client, downstream):
    # Unusual spacing and key order would not survive a parse and re-serialize.
    raw = b'{ "id":7,   "method":"tools/list", "jsonrpc":"2.0" }'
    response = await client.post("/mcp", content=raw, headers=auth())
    assert response.status_code == 200
    assert response.json()["id"] == 7
    assert [t["name"] for t in response.json()["result"]["tools"]] == [
        "get_status",
        "admin_reset_key",
    ]
    assert downstream.state.requests[0]["body"] == raw


async def test_viewer_blocked_from_admin_tool_without_calling_downstream(client, downstream):
    response = await client.post("/mcp", json=tool_call("admin_reset_key", 42), headers=auth())
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": 42,
        "error": {"code": -32001, "message": "Unauthorized Tool Call"},
    }
    assert downstream.state.requests == []


async def test_admin_can_call_admin_tool(client, downstream):
    response = await client.post("/mcp", json=tool_call("admin_reset_key"), headers=auth("admin"))
    assert response.json()["result"]["content"][0]["text"] == "executed admin_reset_key"
    assert len(downstream.state.requests) == 1


async def test_viewer_can_call_regular_tool(client):
    response = await client.post("/mcp", json=tool_call("get_status"), headers=auth())
    assert "result" in response.json()


@pytest.mark.parametrize("id_", ["string-id", 0, 1.5])
async def test_request_id_is_preserved_on_denial(client, id_):
    response = await client.post("/mcp", json=tool_call("admin_x", id_), headers=auth())
    assert response.json()["id"] == id_


async def test_string_id_of_zero_length_and_prefix_only_name(client):
    response = await client.post("/mcp", json=tool_call("admin_", ""), headers=auth())
    assert response.json()["error"]["code"] == -32001


async def test_similar_name_is_not_treated_as_admin(client):
    response = await client.post("/mcp", json=tool_call("administrator_report"), headers=auth())
    assert "result" in response.json()


# ---- Authentication -------------------------------------------------------
def b64(data):
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


def claims(**overrides):
    base = {
        "sub": "x",
        "role": "admin",
        "aud": "mcp-gateway",
        "exp": int(time.time()) + 60,
    }
    return {k: v for k, v in {**base, **overrides}.items() if v is not None}


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "Bearer",
        "Bearer ",
        "Basic dXNlcjpwYXNz",
        "Bearer not-a-jwt",
        f"Bearer {mint('admin', 'a-different-secret-of-sufficient-length!!')}",
        f"Bearer {mint('admin', SECRET, ttl=-10)}",
        f"Bearer {jwt.encode(claims(role='superuser'), SECRET, algorithm='HS256')}",
        f"Bearer {jwt.encode(claims(role=None), SECRET, algorithm='HS256')}",
        f"Bearer {jwt.encode(claims(aud='other-service'), SECRET, algorithm='HS256')}",
        f"Bearer {jwt.encode(claims(exp=None), SECRET, algorithm='HS256')}",
        f"Bearer {b64({'alg': 'none', 'typ': 'JWT'})}.{b64(claims())}.",
    ],
    ids=[
        "absent",
        "empty",
        "no-token",
        "blank-token",
        "basic",
        "garbage",
        "wrong-key",
        "expired",
        "unknown-role",
        "no-role",
        "wrong-audience",
        "no-expiry",
        "alg-none",
    ],
)
async def test_bad_credentials_get_401_and_never_reach_downstream(client, downstream, header):
    headers = {"Authorization": header} if header is not None else {}
    response = await client.post("/mcp", json=tool_call("get_status"), headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == -32000
    assert downstream.state.requests == []


async def test_bearer_scheme_is_case_insensitive(client):
    token = mint("viewer", SECRET)
    response = await client.post(
        "/mcp", json=tool_call("get_status"), headers={"Authorization": f"bearer {token}"}
    )
    assert response.status_code == 200


# ---- Bypass attempts ------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    ["ADMIN_reset_key", "Admin_reset_key", " admin_reset_key", "admin_reset_key ", "ａｄｍｉｎ_x"],
)
async def test_name_variants_are_blocked(client, downstream, name):
    response = await client.post("/mcp", json=tool_call(name), headers=auth())
    assert response.json()["error"]["code"] == -32001
    assert downstream.state.requests == []


async def test_json_unicode_escape_is_blocked(client, downstream):
    raw = (
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        b'"params":{"name":"\\u0061dmin_reset_key"}}'
    )
    response = await client.post("/mcp", content=raw, headers=auth())
    assert response.json()["error"]["code"] == -32001
    assert downstream.state.requests == []


@pytest.mark.parametrize(
    "raw",
    [
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","method":"tools/list",'
        b'"params":{"name":"admin_x"}}',
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        b'"params":{"name":"admin_x","name":"get_status"}}',
    ],
    ids=["duplicate-method", "duplicate-name"],
)
async def test_duplicate_keys_are_rejected(client, downstream, raw):
    response = await client.post("/mcp", content=raw, headers=auth())
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32700
    assert downstream.state.requests == []


async def test_notification_style_tool_call_is_blocked(client, downstream):
    message = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "admin_x"}}
    response = await client.post("/mcp", json=message, headers=auth())
    assert response.json()["error"]["code"] == -32600
    assert downstream.state.requests == []


async def test_batch_smuggling_is_filtered_per_element(client, downstream):
    batch = [tool_call("get_status", 1), tool_call("admin_reset_key", 2)]
    response = await client.post("/mcp", json=batch, headers=auth())
    replies = {r["id"]: r for r in response.json()}
    assert "result" in replies[1]
    assert replies[2]["error"]["code"] == -32001
    forwarded = sent_to_downstream(downstream)[0]
    assert [m["params"]["name"] for m in forwarded] == ["get_status"]


async def test_batch_of_only_denied_calls_never_reaches_downstream(client, downstream):
    response = await client.post(
        "/mcp", json=[tool_call("admin_a", 1), tool_call("admin_b", 2)], headers=auth()
    )
    assert [r["error"]["code"] for r in response.json()] == [-32001, -32001]
    assert downstream.state.requests == []


async def test_admin_batch_passes_through(client, downstream):
    batch = [tool_call("get_status", 1), tool_call("admin_reset_key", 2)]
    response = await client.post("/mcp", json=batch, headers=auth("admin"))
    assert all("result" in r for r in response.json())


async def test_spoofed_identity_headers_are_replaced(client, downstream):
    headers = {
        **auth(),
        "X-Gateway-Role": "admin",
        "X-Gateway-Subject": "root",
        "X-Gateway-Tenant": "someone-else",  # not set by the gateway: must vanish
    }
    await client.post("/mcp", json=tool_call("get_status"), headers=headers)
    seen = downstream.state.requests[0]["headers"]
    assert seen["x-gateway-role"] == "viewer"
    assert seen["x-gateway-subject"] == "demo-user"
    assert "x-gateway-tenant" not in seen


async def test_client_token_is_not_passed_downstream(client, downstream):
    await client.post("/mcp", json=tool_call("get_status"), headers=auth())
    assert "authorization" not in downstream.state.requests[0]["headers"]


# ---- Wire format ----------------------------------------------------------
@pytest.mark.parametrize(
    "raw, status, code",
    [
        (b"{not json", 400, -32700),
        (b"", 400, -32700),
        (b"[" * 100_000, 400, -32700),
        (b"[]", 400, -32600),
        (b'"a string"', 200, -32600),
        (b"42", 200, -32600),
        (b'{"id":1,"method":"tools/list"}', 200, -32600),
        (b'{"jsonrpc":"1.0","id":1,"method":"tools/list"}', 200, -32600),
        (b'{"jsonrpc":"2.0","id":1}', 200, -32600),
        (b'{"jsonrpc":"2.0","id":1,"method":7}', 200, -32600),
        (b'{"jsonrpc":"2.0","id":{"a":1},"method":"tools/list"}', 200, -32600),
        (b'{"jsonrpc":"2.0","id":true,"method":"tools/list"}', 200, -32600),
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/call"}', 200, -32602),
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":[1]}', 200, -32602),
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":7}}', 200, -32602),
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":""}}', 200, -32602),
    ],
    ids=lambda v: v[:30] if isinstance(v, bytes) else None,
)
async def test_malformed_payloads(client, downstream, raw, status, code):
    response = await client.post("/mcp", content=raw, headers=auth())
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert downstream.state.requests == []


async def test_oversized_body_rejected(client, downstream):
    response = await client.post("/mcp", content=b" " * (1024 * 1024 + 1), headers=auth())
    assert response.status_code == 413
    assert downstream.state.requests == []


async def test_notifications_and_client_responses_pass_through(client, downstream):
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    reply = {"jsonrpc": "2.0", "id": 9, "result": {}}
    assert (await client.post("/mcp", json=note, headers=auth())).status_code == 202
    assert (await client.post("/mcp", json=reply, headers=auth())).status_code == 202
    assert len(downstream.state.requests) == 2


async def test_downstream_error_is_relayed(client):
    message = {"jsonrpc": "2.0", "id": 1, "method": "bogus/method"}
    response = await client.post("/mcp", json=message, headers=auth())
    assert response.json()["error"]["code"] == -32601


@pytest.mark.parametrize(
    "exc, status",
    [(httpx.ConnectError("10.0.0.5 refused"), 502), (httpx.ReadTimeout("slow"), 504)],
)
async def test_downstream_failure_is_sanitized(exc, status):
    def boom(_):
        raise exc

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    app = gateway.create_app("http://downstream/mcp", SECRET, client=upstream)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    ) as http:
        response = await http.post("/mcp", json=tool_call("get_status", 5), headers=auth())
    assert response.status_code == status
    assert response.json()["id"] == 5
    assert response.json()["error"]["code"] == -32002
    assert "10.0.0.5" not in response.text


async def test_partial_batch_with_downstream_down_is_a_clean_502():
    def boom(_):
        raise httpx.ConnectError("refused")

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    app = gateway.create_app("http://downstream/mcp", SECRET, client=upstream)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    ) as http:
        batch = [tool_call("get_status", 1), tool_call("admin_x", 2)]
        response = await http.post("/mcp", json=batch, headers=auth())
    assert response.status_code == 502
    assert response.json()["error"]["code"] == -32002


def test_short_secret_refused_at_startup():
    with pytest.raises(ValueError):
        gateway.create_app("http://x/mcp", "short")
