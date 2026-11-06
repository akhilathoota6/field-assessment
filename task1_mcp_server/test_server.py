"""Black-box tests: spawn the real server and speak raw JSON-RPC over its pipes.

Going through real pipes rather than an in-memory client is deliberate, because
stdout purity can only be observed from outside the process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import threading
from pathlib import Path

import pytest

SERVER = Path(__file__).with_name("server.py")
INIT = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def run_session(*messages, argv=None):
    """Send lines to a fresh server; return (parsed stdout messages, stderr, raw stdout).

    Each message is a dict, or a str sent verbatim (for malformed input).
    """
    lines = [json.dumps(INIT), json.dumps(INITIALIZED)]
    lines += [m if isinstance(m, str) else json.dumps(m) for m in messages]
    pending = {0} | {m["id"] for m in messages if isinstance(m, dict) and "id" in m}

    with tempfile.TemporaryFile("w+", encoding="utf-8") as stderr_file:
        proc = subprocess.Popen(
            argv or [sys.executable, str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            encoding="utf-8",
        )
        # A watchdog so a hung server fails the test instead of blocking it.
        watchdog = threading.Timer(30, proc.kill)
        watchdog.start()
        try:
            proc.stdin.write("\n".join(lines) + "\n")
            proc.stdin.flush()
            # Like a real client, stay connected until every reply has arrived:
            # the SDK cancels in-flight requests when stdin reaches EOF.
            raw = []
            while pending:
                line = proc.stdout.readline()
                if not line:
                    break
                raw.append(line)
                pending.discard(json.loads(line).get("id"))
            proc.stdin.close()
            raw.append(proc.stdout.read())
            proc.wait()
        finally:
            watchdog.cancel()
        assert not pending, f"no reply for ids {pending}"
        stderr_file.seek(0)
        stderr = stderr_file.read()

    stdout = "".join(raw)
    parsed = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    return parsed, stderr, stdout


def call(tool, arguments, id_=1):
    return {
        "jsonrpc": "2.0",
        "id": id_,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }


def response(messages, id_):
    return next(m for m in messages if m.get("id") == id_)


VALID_REFUND = {"customer_id": "CUST-10001", "amount": 10.5, "reason": "Damaged on arrival"}


# ---- STDIO isolation ------------------------------------------------------
def test_stdout_is_only_json_rpc_and_logs_go_to_stderr():
    _, stderr, raw = run_session(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        call("trigger_refund", VALID_REFUND, 2),
        call("get_customer_record", {"customer_id": "bad"}, 3),
    )
    lines = [line for line in raw.splitlines() if line.strip()]
    assert len(lines) == 4  # initialize + 3 calls: nothing extra
    for line in lines:
        assert json.loads(line)["jsonrpc"] == "2.0"
    assert "starting customer-support" in stderr
    assert "refund issued" in stderr
    assert "refund issued" not in raw


def test_stray_print_and_child_process_output_cannot_reach_the_wire(tmp_path):
    """Sabotage a handler with print() and a child process that writes to stdout."""
    rogue = tmp_path / "rogue.py"
    rogue.write_text(
        textwrap.dedent(
            f"""
            import subprocess, sys
            sys.path.insert(0, {str(SERVER.parent)!r})
            import anyio, server

            original = server.get_customer_record

            def noisy(args):
                print("ROGUE PRINT", flush=True)
                subprocess.run([sys.executable, "-c", "print('ROGUE CHILD')"])
                return original(args)

            name = "get_customer_record"
            server.TOOLS[name] = (server.TOOLS[name][0], noisy, *server.TOOLS[name][2:])
            anyio.run(server.main)
            """
        ),
        encoding="utf-8",
    )
    messages, stderr, raw = run_session(
        call("get_customer_record", {"customer_id": "CUST-10001"}),
        argv=[sys.executable, str(rogue)],
    )
    assert "ROGUE" not in raw
    assert "ROGUE PRINT" in stderr
    assert "ROGUE CHILD" in stderr
    assert response(messages, 1)["result"]["structuredContent"]["name"] == "Asha Rao"


# ---- Protocol flow --------------------------------------------------------
def test_tools_list_publishes_strict_schemas():
    messages, _, _ = run_session({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = {t["name"]: t for t in response(messages, 1)["result"]["tools"]}
    assert set(tools) == {"get_customer_record", "trigger_refund"}
    schema = tools["trigger_refund"]["inputSchema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"customer_id", "amount", "reason"}
    assert schema["properties"]["customer_id"]["pattern"] == "^CUST-[0-9]{5}$"
    assert schema["properties"]["amount"]["exclusiveMinimum"] == 0
    assert schema["properties"]["reason"]["minLength"] == 10
    assert tools["trigger_refund"]["annotations"]["destructiveHint"] is True


def test_get_customer_record_success():
    messages, _, _ = run_session(call("get_customer_record", {"customer_id": "CUST-10002"}))
    result = response(messages, 1)["result"]
    assert result["structuredContent"]["tier"] == "silver"
    assert not result.get("isError")


def test_trigger_refund_success_reduces_balance():
    messages, _, _ = run_session(call("trigger_refund", VALID_REFUND))
    body = response(messages, 1)["result"]["structuredContent"]
    assert body["status"] == "processed"
    assert body["remaining_balance"] == 239.5


def test_integer_amount_is_accepted():
    # JSON has one number type, so 10 is a valid float.
    messages, _, _ = run_session(call("trigger_refund", {**VALID_REFUND, "amount": 10}))
    assert response(messages, 1)["result"]["structuredContent"]["status"] == "processed"


@pytest.mark.parametrize(
    "tool, arguments, expected",
    [
        ("get_customer_record", {"customer_id": "CUST-99999"}, "not found"),
        ("trigger_refund", {**VALID_REFUND, "customer_id": "CUST-99999"}, "not found"),
        ("trigger_refund", {**VALID_REFUND, "amount": 9999.0}, "exceeds"),
    ],
)
def test_business_failures_are_tool_errors_not_protocol_errors(tool, arguments, expected):
    messages, _, _ = run_session(call(tool, arguments))
    reply = response(messages, 1)
    assert "error" not in reply
    assert reply["result"]["isError"] is True
    assert expected in reply["result"]["content"][0]["text"]


def test_unknown_tool_is_invalid_params():
    messages, _, _ = run_session(call("delete_everything", {}))
    assert response(messages, 1)["error"]["code"] == -32602


def test_unknown_method_is_method_not_found():
    messages, _, _ = run_session({"jsonrpc": "2.0", "id": 1, "method": "tools/explode"})
    assert response(messages, 1)["error"]["code"] == -32601


@pytest.mark.parametrize(
    "line, code",
    [
        ("{this is not json", -32700),
        ('{"jsonrpc":"2.0","id":1,', -32700),
        ('{"not":"json-rpc"}', -32600),
        ("[1, 2, 3]", -32600),
        ('"just a string"', -32600),
        ('{"jsonrpc":"1.0","id":1,"method":"ping"}', -32600),
    ],
)
def test_undecodable_line_gets_an_error_reply_and_server_keeps_running(line, code):
    messages, _, raw = run_session(
        line,
        call("get_customer_record", {"customer_id": "CUST-10001"}, 7),
    )
    # Exactly one reply with a null id, per JSON-RPC 2.0, and not silence.
    errors = [m for m in messages if m.get("id") is None]
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == code
    assert '"id":null' in raw
    assert response(messages, 7)["result"]["structuredContent"]["name"] == "Asha Rao"


# ---- Validation -----------------------------------------------------------
@pytest.mark.parametrize(
    "customer_id",
    [
        "CUST-1234",  # too short
        "CUST-123456",  # too long
        "cust-12345",  # wrong case
        "CUST12345",  # no dash
        "CUST-1234A",  # letter
        " CUST-12345",  # leading space
        "CUST-12345\n",  # trailing newline defeats a Python-style "$"
        "CUST-١٢٣٤٥",  # Arabic-Indic digits satisfy a Unicode \d
        "",
        12345,
        None,
        ["CUST-12345"],
    ],
)
def test_bad_customer_id_rejected(customer_id):
    messages, _, _ = run_session(call("get_customer_record", {"customer_id": customer_id}))
    error = response(messages, 1)["error"]
    assert error["code"] == -32602
    assert error["data"]["errors"][0]["field"] == "customer_id"


@pytest.mark.parametrize("amount", [0, -1, -0.01, "10.50", True, None, [10], {"v": 1}])
def test_bad_amount_rejected(amount):
    messages, _, _ = run_session(call("trigger_refund", {**VALID_REFUND, "amount": amount}))
    error = response(messages, 1)["error"]
    assert error["code"] == -32602
    assert error["data"]["errors"][0]["field"] == "amount"


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_non_finite_amount_rejected(token):
    # These are not valid JSON, so they are spliced in as raw text. Python's
    # json.dumps emits them too, which is how a real client could send them.
    raw = json.dumps(call("trigger_refund", {**VALID_REFUND, "amount": "@"})).replace('"@"', token)
    messages, _, _ = run_session(raw, call("get_customer_record", {"customer_id": "CUST-10001"}, 9))
    # The refund must not succeed, however the transport rejects it.
    refund = [m for m in messages if m.get("id") == 1]
    assert all("result" not in m for m in refund)
    assert "result" in response(messages, 9)


@pytest.mark.parametrize("reason", ["short", "123456789", " " * 30, "  short   ", 1234567890, None])
def test_bad_reason_rejected(reason):
    messages, _, _ = run_session(call("trigger_refund", {**VALID_REFUND, "reason": reason}))
    error = response(messages, 1)["error"]
    assert error["code"] == -32602
    assert error["data"]["errors"][0]["field"] == "reason"


def test_reason_of_exactly_ten_characters_is_accepted():
    messages, _, _ = run_session(call("trigger_refund", {**VALID_REFUND, "reason": "1234567890"}))
    assert "result" in response(messages, 1)


def test_missing_and_extra_fields_rejected():
    messages, _, _ = run_session(
        call("trigger_refund", {"customer_id": "CUST-10001"}, 1),
        call("get_customer_record", {"customer_id": "CUST-10001", "admin": True}, 2),
        call("get_customer_record", {}, 3),
    )
    missing = {e["field"] for e in response(messages, 1)["error"]["data"]["errors"]}
    assert missing == {"amount", "reason"}
    assert response(messages, 2)["error"]["data"]["errors"][0]["type"] == "extra_forbidden"
    assert response(messages, 3)["error"]["code"] == -32602


def test_rejected_values_are_not_echoed_back():
    secret = "4111111111111111"
    messages, stderr, _ = run_session(call("get_customer_record", {"customer_id": secret}))
    assert secret not in json.dumps(response(messages, 1)["error"])
    assert secret not in stderr
