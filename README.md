# AI Field Developer / Solutions Engineer Assessment

Four tasks in Python 3.14, each in its own folder with its own tests and a one-command demo.

| Task | Folder | What it is |
|---|---|---|
| 1 | [task1_mcp_server](task1_mcp_server) | MCP server over stdio with strict validation |
| 2 | [task2_mcp_gateway](task2_mcp_gateway) | JSON-RPC security proxy with JWT roles and tool authorization |
| 3 | [task3_llm_guardrail](task3_llm_guardrail) | LLM gateway that redacts PII from a live stream |
| 4 | [task4_model_router](task4_model_router) | Token rate limiter on SQLite with model failover |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Windows, keep the project in a short path such as `C:\ai-assessment`. A dependency of the
`mcp` package installs very deep folders and fails with `WinError 206` under long paths.

## Run

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe task1_mcp_server\demo.py
.venv\Scripts\python.exe task2_mcp_gateway\demo.py
.venv\Scripts\python.exe task3_llm_guardrail\demo.py
.venv\Scripts\python.exe task4_model_router\demo.py
```

The demos start everything they need, on real ports, using mock upstreams. No API keys are needed.

---

## Task 1: MCP server

**Stdio isolation has two layers.** Logging is configured onto stderr, and nothing writes to
stdout. Separately, the SDK's `stdio_server()` points file descriptor 1 at stderr while serving.
`test_stray_print_and_child_process_output_cannot_reach_the_wire` sabotages a handler with a
`print()` and a noisy child process, then confirms that stdout stays pure.

**Protocol errors and tool errors are kept apart, as the MCP spec does.**

| Situation | Reply |
|---|---|
| Invalid JSON on the wire | JSON-RPC `-32700`, `id: null` |
| Valid JSON that isn't JSON-RPC | JSON-RPC `-32600`, `id: null` |
| Unknown method | JSON-RPC `-32601` |
| Bad arguments or unknown tool | JSON-RPC `-32602` with per-field detail |
| Unexpected exception | JSON-RPC `-32603`, detail in stderr only |
| Customer not found, refund too large | Normal result with `isError: true` |

Two SDK behaviours shaped the design:

- The high-level `MCPServer` turns handler exceptions into `isError` results, so the low-level
  `Server` is used to return real `-32602` errors.
- The SDK silently drops undecodable stdin lines, which leaves a client waiting forever.
  `RejectMalformedLines` wraps the read stream to send the reply JSON-RPC requires.

**Validation edge cases covered:** trailing newline, Arabic-Indic digits (which a Unicode `\d`
would accept), `"10.50"` as a string, `true` as a number, NaN and Infinity, a whitespace-only
reason, and unknown extra fields. Rejected values are never echoed back or logged.

**Assumption:** the `X` in `CUST-XXXXX` means a digit. To allow letters, change the one pattern.

## Task 2: MCP gateway

`POST /mcp` authenticates the JWT, parses the JSON-RPC body, authorizes it, then forwards the
original bytes. Responses are streamed, so SSE keeps working.

Beyond the brief's prefix check, these bypasses are closed and tested:

| Attack | Defence |
|---|---|
| Admin call hidden in a JSON-RPC batch | Every element is authorized. Allowed ones go through, denied ones get `-32001` |
| Duplicate keys (`"name":"admin_x","name":"ok"`) | Rejected, so gateway and downstream can't parse differently |
| `ADMIN_x`, `" admin_x"`, full-width letters | Names are normalized (NFKC, trim, casefold) before the check |
| `tools/call` with no `id` | Rejected, so it can't run downstream as a notification |
| `alg: none` and forged JWTs | Algorithm pinned to HS256; `exp`, `sub` and `aud` required |
| Spoofed `X-Gateway-Role` header | Inbound `x-gateway-*` headers are stripped |
| Token passthrough | `Authorization` is never forwarded, as the MCP spec requires |

**Decisions worth reviewing:**

- `-32001` is returned with HTTP 200. The transport worked and the failure belongs to the
  JSON-RPC layer. Bad credentials are different and get HTTP 401.
- `tools/list` is forwarded unchanged because the brief says so. Viewers can therefore see
  admin tool names, though they can't call them. Filtering the list would be a natural next step.
- HS256 keeps the exercise self-contained. Production should use RS256 with JWKS from an
  identity provider, so the gateway holds no signing key.

**Verified with the official SDK:** an SDK client on protocol `2026-07-28` went through the
gateway to a real SDK streamable-HTTP server. The viewer got `-32001` and the admin got through.

## Task 3: Streaming PII guardrail

A secret can be split across deltas (`"4111 11"`, `"11 1111 1111"`), so chunks can't be released
just because they look clean. Every pattern is a contiguous run of email characters or
digits-and-separators, so **only the run touching the end of the buffer can still become a
match**. Everything before it is released immediately.

```
"Contact me at john.doe@exa"
 ^^^^^^^^^^^^^^               released now
               ^^^^^^^^^^^^   held: may still become an email
```

- **Memory:** the hold is capped at 320 characters however long the response is.
- **Latency:** measured in the demo on mock tokens 30ms apart, first text took 93ms through the
  gateway against 63ms direct, and the total took 1168ms against 1162ms. The extra first-text
  delay is about one token, because the first word might be the start of an email address.
- **Correctness:** tests check that streamed output equals one-shot output at every split point,
  character by character, and over 3,000 random chunkings. Another inspects each emitted piece
  for fragments of a secret.
- **Framing:** SSE is parsed as bytes and only complete events are decoded, so a UTF-8 character
  split across reads can't be corrupted.
- **Back-pressure:** the generator pulls upstream only when the client is ready.
- **Fail closed:** events that can't be parsed are dropped, not passed through.
- **Cards:** Luhn checking is off by default, because masking an order number costs less than
  leaking a mistyped card. `require_luhn=True` turns it on.

**Known limits:**

- Patterns can't catch obfuscated text such as `john at example dot com`, or separators other
  than a space or hyphen. A production system would add an NER model on a slower path.
- Only `delta.content` is scanned. `tool_calls` arguments and reasoning fields pass through
  unredacted, which is the first thing to close before production.
- OpenAI-compatible streams only. Other formats need their own adapter.

## Task 4: Rate limiter and router

**Limiter:** a sliding log in SQLite, one row per request. A request goes through
reserve, call the model, then settle:

1. `reserve` atomically checks the window and records an estimate.
2. `reconcile` replaces the estimate with the provider's real usage.
3. `release` removes the row if no model served the request, including on client disconnect.

Reserving first is what stops 100 concurrent requests from seeing the same free capacity.

**Concurrency:** each transaction is one synchronous function on a dedicated single-thread
executor, so coroutines can't interleave inside check-then-insert and the event loop never blocks
on disk. `BEGIN IMMEDIATE` gives the same guarantee across processes, tested with four real
processes racing for one tenant's budget in one file.

**Eviction:** a tenant's expired rows go on each `reserve`, and a background sweep removes rows
for tenants who went quiet.

**Failover:** attempts run in sequence rather than as a hedged race, so only one provider is
billed and a late reply can't produce a second answer. The deadline is `asyncio.timeout` around
the whole attempt, because httpx timeouts apply per read and a provider dripping one byte at a
time would never trip them.

| Upstream result | Action |
|---|---|
| 429, 408, 5xx, timeout, connection error, unusable body | Fail over |
| 401, 403, 404 | Fail over: this is our config for that provider, not the tenant's fault |
| 400, 413, 422 | Stop: the backup would only repeat the rejection |

**Errors:** every failure has the same shape, and details stay in the logs under `request_id`.

```json
{"error": {"type": "rate_limit_error", "code": "tenant_rate_limited",
           "message": "Token rate limit exceeded", "request_id": "req_ab12", "retryable": true}}
```

If SQLite fails, the gateway fails closed with 503, because it can't enforce quotas without it.
API keys are stored only as SHA-256 hashes.

**Known limits:**

- Non-streaming only. Streaming would reconcile usage at the end of the stream.
- There's no circuit breaker, so during a primary outage that hangs, every request waits the full
  3000ms before failing over. Adding one is the next improvement.
- The estimate is `characters / 4 + max_tokens`. A real tokenizer would be tighter, though
  `reconcile` corrects usage either way.
- SQLite suits one host. A multi-host gateway needs a shared store such as Redis.
- The brief sets 3000ms for the primary. The secondary's 10s default is my assumption, set with
  `SECONDARY_TIMEOUT_S`.
