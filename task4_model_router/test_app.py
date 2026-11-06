"""End-to-end through the HTTP app, with scripted providers behind httpx.MockTransport."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).parent


def load(name, alias=None):
    alias = alias or name
    spec = importlib.util.spec_from_file_location(alias, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


load("rate_limiter")
router_module = load("router")
app_module = load("app", "task4_app")
TokenRateLimiter = sys.modules["rate_limiter"].TokenRateLimiter

PRIMARY = "http://primary.internal/v1/chat/completions"
SECONDARY = "http://secondary.internal/v1/chat/completions"
KEY = {"Authorization": "Bearer tenant-a"}
BODY = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 100}


def ok(text, tokens=50):
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "model": "internal-model-v7",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            "usage": {"total_tokens": tokens},
        },
    )


class Upstreams:
    """Scripted providers that record who was called."""

    def __init__(self, primary, secondary=None):
        self.primary = primary
        self.secondary = secondary or (lambda: ok("from secondary"))
        self.calls = []
        self.requests = []

    async def __call__(self, request):
        target = "primary" if str(request.url) == PRIMARY else "secondary"
        self.calls.append(target)
        self.requests.append(json.loads(request.content))
        result = (self.primary if target == "primary" else self.secondary)()
        return await result if asyncio.iscoroutine(result) else result


async def hang():
    await asyncio.sleep(30)
    return ok("too late")


class Harness:
    def __init__(self, tmp_path, upstreams, primary_timeout=0.1, secondary_timeout=0.1, limit=50_000):
        self.limiter = TokenRateLimiter(tmp_path / "gw.db", limit=limit)
        self.upstreams = upstreams
        self.router = router_module.ModelRouter(
            [
                router_module.Provider("primary", PRIMARY, "m1", "pk", primary_timeout),
                router_module.Provider("secondary", SECONDARY, "m2", "sk", secondary_timeout),
            ],
            client=httpx.AsyncClient(transport=httpx.MockTransport(upstreams), timeout=None),
        )
        app = app_module.create_app(self.limiter, self.router, {"tenant-a", "tenant-b"})
        self.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gw", timeout=30
        )

    def post(self, body=BODY, headers=KEY):
        return self.http.post("/v1/chat/completions", json=body, headers=headers)

    async def close(self):
        await self.http.aclose()
        await self.router.close()
        await self.limiter.close()


@pytest.fixture
async def make(tmp_path):
    harnesses = []

    def build(*args, **kwargs):
        harness = Harness(tmp_path, *args, **kwargs)
        harnesses.append(harness)
        return harness

    yield build
    for harness in harnesses:
        await harness.close()


def assert_standard_error(response, status, code):
    assert response.status_code == status
    error = response.json()["error"]
    assert set(error) == {"type", "code", "message", "request_id", "retryable"}
    assert error["code"] == code
    assert error["request_id"].startswith("req_")
    assert response.headers["x-request-id"] == error["request_id"]


# ---- Routing --------------------------------------------------------------
async def test_primary_success_skips_secondary(make):
    gw = make(Upstreams(lambda: ok("from primary")))
    response = await gw.post()
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "from primary"
    assert response.headers["x-gateway-route"] == "primary"
    assert gw.upstreams.calls == ["primary"]


async def test_429_fails_over(make):
    gw = make(Upstreams(lambda: httpx.Response(429, json={"error": "slow down"})))
    response = await gw.post()
    assert response.status_code == 200
    assert response.headers["x-gateway-route"] == "secondary"
    assert gw.upstreams.calls == ["primary", "secondary"]


async def test_each_provider_gets_its_own_model(make):
    gw = make(Upstreams(lambda: httpx.Response(429)))
    await gw.post({**BODY, "model": "client-picked"})
    assert [r["model"] for r in gw.upstreams.requests] == ["m1", "m2"]


async def test_timeout_fails_over_near_the_deadline(make):
    gw = make(Upstreams(hang), primary_timeout=0.2)
    start = time.perf_counter()
    response = await gw.post()
    elapsed = time.perf_counter() - start
    assert response.headers["x-gateway-route"] == "secondary"
    assert 0.2 <= elapsed < 0.6


async def test_real_3000ms_default_deadline(make):
    assert router_module.Provider("p", "u", "m").timeout_s == 3.0
    gw = make(Upstreams(hang), primary_timeout=3.0)
    start = time.perf_counter()
    response = await gw.post()
    elapsed = time.perf_counter() - start
    assert response.headers["x-gateway-route"] == "secondary"
    assert 3.0 <= elapsed < 3.5


async def test_slow_drip_body_cannot_outrun_the_deadline(make):
    """Per-read timeouts would never fire here, because each read is quick."""

    async def drip():
        for _ in range(100):
            await asyncio.sleep(0.05)
            yield b" "

    gw = make(Upstreams(lambda: httpx.Response(200, content=drip())), primary_timeout=0.2)
    start = time.perf_counter()
    response = await gw.post()
    assert response.headers["x-gateway-route"] == "secondary"
    assert time.perf_counter() - start < 0.8


@pytest.mark.parametrize("status", [401, 403, 404, 408, 500, 502, 503, 504])
async def test_provider_side_failures_fail_over(make, status):
    gw = make(Upstreams(lambda: httpx.Response(status)))
    assert (await gw.post()).headers["x-gateway-route"] == "secondary"


@pytest.mark.parametrize(
    "bad",
    [
        lambda: httpx.Response(200, content=b"<html>proxy error</html>"),
        lambda: httpx.Response(200, json={"no": "choices"}),
        lambda: httpx.Response(200, json={"choices": []}),
        lambda: httpx.Response(200, json=["not", "an", "object"]),
    ],
)
async def test_unusable_success_body_fails_over(make, bad):
    gw = make(Upstreams(bad))
    assert (await gw.post()).headers["x-gateway-route"] == "secondary"


async def test_connection_error_fails_over(make):
    def refuse():
        raise httpx.ConnectError("connection refused")

    gw = make(Upstreams(refuse))
    assert (await gw.post()).headers["x-gateway-route"] == "secondary"


@pytest.mark.parametrize("status", [400, 413, 422])
async def test_bad_request_does_not_fail_over(make, status):
    gw = make(Upstreams(lambda: httpx.Response(status, text="context length exceeded at x.py:9")))
    response = await gw.post()
    assert_standard_error(response, 400, "request_rejected")
    assert gw.upstreams.calls == ["primary"]
    assert "x.py" not in response.text


async def test_both_providers_down(make):
    gw = make(Upstreams(lambda: httpx.Response(429), lambda: httpx.Response(500)))
    response = await gw.post()
    assert_standard_error(response, 503, "all_providers_unavailable")
    assert response.json()["error"]["retryable"] is True
    assert "retry-after" in response.headers


async def test_late_primary_reply_cannot_race_the_fallback(make):
    """Across a spread of delays around the deadline: one answer, one charge each."""

    async def around_deadline():
        await asyncio.sleep(0.05 + (len(gw.upstreams.calls) % 7) * 0.005)
        return ok("primary", tokens=10)

    gw = make(Upstreams(around_deadline, lambda: ok("secondary", tokens=10)), primary_timeout=0.065)
    responses = await asyncio.gather(*(gw.post() for _ in range(40)))
    assert all(r.status_code == 200 for r in responses)
    routes = {r.headers["x-gateway-route"] for r in responses}
    assert routes <= {"primary", "secondary"}
    assert await gw.limiter.row_count() == 40
    assert await gw.limiter.usage(app_module.hash_key("tenant-a")) == 400


# ---- Rate limiting --------------------------------------------------------
async def test_concurrent_requests_admit_exactly_the_limit(make):
    gw = make(Upstreams(lambda: ok("x", tokens=1000)), limit=10_000)
    body = {"messages": [{"role": "user", "content": "abcd"}], "max_tokens": 999}  # estimate 1000
    responses = await asyncio.gather(*(gw.post(body) for _ in range(30)))
    assert sum(r.status_code == 200 for r in responses) == 10
    limited = [r for r in responses if r.status_code == 429]
    assert len(limited) == 20
    for response in limited:
        assert_standard_error(response, 429, "tenant_rate_limited")
        assert int(response.headers["retry-after"]) >= 1
        assert response.headers["x-ratelimit-limit"] == "10000"
    assert len(gw.upstreams.calls) == 10  # denied requests cost no provider call


async def test_actual_usage_replaces_estimate(make):
    gw = make(Upstreams(lambda: ok("x", tokens=7)))
    response = await gw.post()
    assert await gw.limiter.usage(app_module.hash_key("tenant-a")) == 7
    assert response.headers["x-ratelimit-remaining"] == str(50_000 - 7)


async def test_estimate_is_kept_when_provider_omits_usage(make):
    reply = {"choices": [{"message": {"content": "x"}}]}
    gw = make(Upstreams(lambda: httpx.Response(200, json=reply)))
    await gw.post()
    assert await gw.limiter.usage(app_module.hash_key("tenant-a")) == 102  # ceil(5/4) + 100


@pytest.mark.parametrize(
    "primary, secondary",
    [
        (lambda: httpx.Response(429), lambda: httpx.Response(500)),
        (lambda: httpx.Response(400), None),
    ],
    ids=["all-down", "rejected"],
)
async def test_unserved_request_is_not_charged(make, primary, secondary):
    gw = make(Upstreams(primary, secondary))
    assert (await gw.post()).status_code != 200
    assert await gw.limiter.row_count() == 0


async def test_client_disconnect_releases_the_reservation(make):
    started = asyncio.Event()

    async def wait_forever():
        started.set()
        await asyncio.sleep(30)

    gw = make(Upstreams(wait_forever), primary_timeout=20)
    task = asyncio.create_task(gw.post())
    await asyncio.wait_for(started.wait(), 5)
    assert await gw.limiter.row_count() == 1
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    for _ in range(100):
        if await gw.limiter.row_count() == 0:
            break
        await asyncio.sleep(0.01)
    assert await gw.limiter.row_count() == 0


async def test_tenants_have_separate_budgets(make):
    gw = make(Upstreams(lambda: ok("x", tokens=1100)), limit=1200)
    assert (await gw.post()).status_code == 200
    assert (await gw.post()).status_code == 429
    other = await gw.post(headers={"X-API-Key": "tenant-b"})
    assert other.status_code == 200


async def test_request_above_the_limit_is_413_not_429(make):
    gw = make(Upstreams(lambda: ok("x")), limit=100)
    response = await gw.post({**BODY, "max_tokens": 5000})
    assert_standard_error(response, 413, "request_exceeds_token_limit")
    assert "retry-after" not in response.headers


async def test_api_keys_are_not_stored_in_plain_text(make):
    gw = make(Upstreams(lambda: ok("x")))
    await gw.post()
    path = gw.limiter._conn.execute("PRAGMA database_list").fetchone()[2]
    with sqlite3.connect(path) as conn:
        tenants = [row[0] for row in conn.execute("SELECT tenant FROM token_usage")]
    assert tenants == [app_module.hash_key("tenant-a")]
    assert "tenant-a" not in tenants


# ---- Error sanitization ---------------------------------------------------
async def test_success_body_hides_internal_model_and_fields(make):
    gw = make(Upstreams(lambda: ok("x")))
    response = await gw.post()
    assert "internal-model-v7" not in response.text
    assert set(response.json()) == {"id", "object", "model", "choices", "usage"}


async def test_failure_details_never_reach_the_client(make):
    leak = "Traceback (most recent call last): File /srv/app/secret.py sk-live-abc 10.2.3.4"

    def explode():
        raise httpx.ConnectError(leak)

    gw = make(Upstreams(lambda: httpx.Response(500, text=leak), explode))
    response = await gw.post()
    assert_standard_error(response, 503, "all_providers_unavailable")
    for fragment in ("Traceback", "secret.py", "sk-live", "10.2.3.4", "internal", "httpx"):
        assert fragment not in response.text


async def test_unexpected_exception_is_a_clean_500(make, monkeypatch):
    gw = make(Upstreams(lambda: ok("x")))

    async def explode(_):
        raise RuntimeError("password=hunter2 at /srv/app.py")

    monkeypatch.setattr(gw.router, "complete", explode)
    response = await gw.post()
    assert_standard_error(response, 500, "internal_error")
    assert "hunter2" not in response.text
    assert await gw.limiter.row_count() == 0


async def test_limiter_outage_fails_closed(make, monkeypatch):
    gw = make(Upstreams(lambda: ok("x")))

    async def explode(*_):
        raise sqlite3.OperationalError("unable to open database file C:/secret/gw.db")

    monkeypatch.setattr(gw.limiter, "reserve", explode)
    response = await gw.post()
    assert_standard_error(response, 503, "rate_limiter_unavailable")
    assert "secret" not in response.text
    assert gw.upstreams.calls == []


@pytest.mark.parametrize(
    "headers, code",
    [
        ({}, "missing_api_key"),
        ({"Authorization": "Basic abc"}, "missing_api_key"),
        ({"Authorization": "Bearer "}, "missing_api_key"),
        ({"Authorization": "Bearer nope"}, "invalid_api_key"),
        ({"X-API-Key": "nope"}, "invalid_api_key"),
    ],
)
async def test_auth_failures(make, headers, code):
    gw = make(Upstreams(lambda: ok("x")))
    assert_standard_error(await gw.post(headers=headers), 401, code)
    assert gw.upstreams.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"messages": []},
        {"messages": "hi"},
        {"messages": [{"role": "root", "content": "x"}]},
        {"messages": [{"role": "user", "content": ""}]},
        {**BODY, "max_tokens": 0},
        {**BODY, "max_tokens": -1},
        {**BODY, "stream": True},
        {**BODY, "unknown_field": 1},
    ],
)
async def test_invalid_bodies(make, body):
    gw = make(Upstreams(lambda: ok("x")))
    assert_standard_error(await gw.post(body), 400, "invalid_request")
    assert gw.upstreams.calls == []
    assert await gw.limiter.row_count() == 0


async def test_invalid_json_and_oversized_body(make):
    gw = make(Upstreams(lambda: ok("x")))
    bad = await gw.http.post("/v1/chat/completions", content=b"{oops", headers=KEY)
    assert_standard_error(bad, 400, "invalid_request")
    big = await gw.http.post(
        "/v1/chat/completions", content=b"x" * (1024 * 1024 + 1), headers=KEY
    )
    assert_standard_error(big, 413, "request_too_large")


async def test_validation_errors_do_not_echo_input(make):
    gw = make(Upstreams(lambda: ok("x")))
    secret = "my-private-prompt-text"
    response = await gw.post({"messages": [{"role": secret, "content": "x"}]})
    assert secret not in response.text


async def test_unknown_route_and_method_use_the_standard_shape(make):
    gw = make(Upstreams(lambda: ok("x")))
    assert_standard_error(await gw.http.get("/nope"), 404, "not_found")
    assert_standard_error(await gw.http.get("/v1/chat/completions"), 405, "method_not_allowed")
