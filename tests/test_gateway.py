"""End-to-end gateway tests over the ASGI transport (no network, no keys)."""

import json

import httpx
import pytest

from mul.conductor.llm import MockConductorLLM
from mul.config import Settings
from mul.gateway.app import create_app


@pytest.fixture
def app(config, settings):
    return create_app(config=config, settings=settings, conductor_llm=MockConductorLLM())


@pytest.fixture
def secured_app(config, tmp_path):
    settings = Settings(
        _env_file=None,
        mul_api_keys="sk-normal",
        mul_admin_keys="sk-admin",
        mul_db_url=f"sqlite:///{tmp_path.as_posix()}/usage.db",
    )
    return create_app(config=config, settings=settings, conductor_llm=MockConductorLLM())


def client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://mul.test"
    )


async def test_models_lists_only_mul_tiers(app):
    async with client(app) as c:
        r = await c.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert ids == ["mul", "mul-ultra"]  # worker model ids must never appear


async def test_unknown_model_is_openai_shaped_404(app):
    async with client(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


async def test_non_streaming_completion_with_usage_and_mul_ext(app):
    async with client(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "mul", "messages": [{"role": "user", "content": "hello"}]},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["total_tokens"] > 0
    assert body["mul"]["billed_tokens"] > 0
    assert body["mul"]["billed_rate_tier"].startswith("rank-")
    # routing must not leak to normal responses
    assert body["mul"].get("trace") is None


async def test_streaming_completion_sse(app):
    async with client(app) as c:
        async with c.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "mul",
                "stream": True,
                "messages": [{"role": "user", "content": "stream me a poem"}],
            },
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            lines = [line async for line in r.aiter_lines() if line.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[5:]) for line in lines[:-1]]
    text = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if c.get("choices")
    )
    assert text  # streamed content arrived
    finals = [c for c in chunks if c.get("usage")]
    assert finals and finals[-1]["mul"]["billed_tokens"] > 0


async def test_orchestrated_request_through_gateway(app):
    # "[orchestrate]" forces the mock conductor's triage to orchestrate
    async with client(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            headers={"X-Mul-Trace": "true"},  # open dev mode → admin
            json={
                "model": "mul-ultra",
                "messages": [
                    {"role": "user", "content": "[orchestrate] 한국어로 코드를 작성해 주세요"}
                ],
            },
        )
    body = r.json()
    trace = body["mul"]["trace"]
    assert trace is not None
    kinds = [e["kind"] for e in trace["events"]]
    assert "plan" in kinds
    providers = {
        c["provider"] for c in trace["bill_detail"]["per_call"] if c["provider"] != "conductor"
    }
    assert len(providers) >= 2


async def test_normal_key_never_gets_trace(secured_app):
    async with client(secured_app) as c:
        r = await c.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-normal", "X-Mul-Trace": "true"},
            json={"model": "mul", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert r.json()["mul"].get("trace") is None


async def test_admin_key_with_header_gets_trace(secured_app):
    async with client(secured_app) as c:
        r = await c.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-admin", "X-Mul-Trace": "true"},
            json={"model": "mul", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.json()["mul"]["trace"] is not None


async def test_bad_key_rejected(secured_app):
    async with client(secured_app) as c:
        r = await c.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-wrong"},
            json={"model": "mul", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


async def test_per_request_provider_opt_out(app):
    async with client(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            headers={"X-Mul-Trace": "true"},
            json={
                "model": "mul-ultra",
                "messages": [{"role": "user", "content": "[orchestrate] write code please"}],
                "mul_exclude_providers": ["openai", "lg"],
            },
        )
    per_call = r.json()["mul"]["trace"]["bill_detail"]["per_call"]
    used = {c["provider"] for c in per_call}
    assert "openai" not in used and "lg" not in used
