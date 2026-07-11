"""Mul Gateway — FastAPI app exposing the OpenAI-compatible surface.

POST /v1/chat/completions   models: "mul", "mul-ultra"; streaming + non-streaming
GET  /v1/models             only the two Mul tiers — worker models are never listed
GET  /healthz

The multi-agent machinery is invisible: responses carry standard usage plus a
namespaced `mul` extension (billed tokens + rate tier). Per-worker routing is
returned only to admin keys sending X-Mul-Trace: true.
"""

from __future__ import annotations

import contextlib
import json
from typing import AsyncIterator

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from mul.billing.pricing import blended_bill
from mul.billing.usage import UsageStore
from mul.conductor.conductor import Conductor
from mul.conductor.llm import build_conductor_llm
from mul.conductor.types import ConductorOutcome
from mul.config import MulConfig, Settings, load_config
from mul.gateway.auth import Authenticator, KeyInfo
from mul.gateway.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    ModelInfo,
    ModelList,
    MulExtension,
    StreamChoice,
    Usage,
    new_completion_id,
    now_ts,
)
from mul.workers.base import WorkerClient
from mul.workers.registry import WorkerRegistry

log = structlog.get_logger()


def create_app(
    config: MulConfig | None = None,
    settings: Settings | None = None,
    conductor_llm: WorkerClient | None = None,
) -> FastAPI:
    settings = settings or Settings()
    config = config or load_config(settings.mul_config_path)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.registry.aclose()
        await app.state.conductor_llm.aclose()

    app = FastAPI(title="Mul AI (물 AI)", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def openai_shaped_errors(_request: Request, exc: HTTPException):
        from fastapi.responses import JSONResponse

        detail = exc.detail
        if not (isinstance(detail, dict) and "error" in detail):
            detail = {"error": {"message": str(detail), "type": "invalid_request_error"}}
        return JSONResponse(status_code=exc.status_code, content=detail)

    app.state.config = config
    app.state.settings = settings
    app.state.registry = WorkerRegistry(config, settings)
    app.state.conductor_llm = conductor_llm or build_conductor_llm(config.conductor)
    app.state.conductor = Conductor(app.state.conductor_llm, app.state.registry, config)
    app.state.auth = Authenticator(settings)
    app.state.usage_store = UsageStore(settings.mul_db_url)

    def get_key(request: Request) -> KeyInfo:
        return app.state.auth.authenticate(request)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "service": "mul-gateway"}

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        created = now_ts()
        return ModelList(
            data=[ModelInfo(id=t, created=created) for t in config.tiers]
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest, request: Request, key: KeyInfo = Depends(get_key)
    ):
        if body.model not in config.tiers:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": f"The model '{body.model}' does not exist. "
                        f"Available models: {', '.join(config.tiers)}.",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )
        if not body.messages:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "messages must be non-empty",
                        "type": "invalid_request_error",
                        "code": "invalid_messages",
                    }
                },
            )

        exclude = set(body.mul_exclude_providers)
        header_exclude = request.headers.get("x-mul-exclude-providers", "")
        exclude |= {p.strip() for p in header_exclude.split(",") if p.strip()}
        trace_on = key.is_admin and request.headers.get("x-mul-trace", "").lower() == "true"

        messages = [
            {"role": m.role, "content": m.flat_content()} for m in body.messages
        ]
        completion_id = new_completion_id()
        created = now_ts()
        log.info(
            "request",
            request_id=completion_id,
            tier=body.model,
            stream=body.stream,
            excluded=sorted(exclude) or None,
        )

        outcome: ConductorOutcome = await app.state.conductor.handle(
            messages,
            body.model,
            exclude_providers=exclude,
            allow_providers=key.allowed_providers,
            stream=body.stream,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )

        async def finalize() -> tuple[Usage, MulExtension]:
            records = outcome.ctx.usage_records
            bill = blended_bill(records)
            usage = Usage(
                prompt_tokens=sum(r.prompt_tokens for r in records),
                completion_tokens=sum(r.completion_tokens for r in records),
                total_tokens=sum(r.total_tokens for r in records),
            )
            trace_dict = outcome.ctx.trace.to_dict()
            trace_dict["bill_detail"] = {
                "top_provider": bill.top_provider,
                "top_model_id": bill.top_model_id,
                "per_call": [
                    {
                        "provider": r.provider,
                        "model_id": r.model_id,
                        "purpose": r.purpose,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "cost_usd": round(r.cost_usd, 8),
                    }
                    for r in records
                ],
            }
            mul_ext = MulExtension(
                billed_tokens=bill.billed_tokens,
                billed_rate_tier=bill.rate_tier,
                billed_cost_usd=bill.cost_usd,
                trace=trace_dict if trace_on else None,
            )
            await app.state.usage_store.record(
                request_id=completion_id,
                api_key=key.key,
                tier=body.model,
                fast_path=outcome.fast_path,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                billed_tokens=bill.billed_tokens,
                billed_rate_tier=bill.rate_tier,
                billed_cost_usd=bill.cost_usd,
                trace=trace_dict,
            )
            return usage, mul_ext

        if not body.stream:
            usage, mul_ext = await finalize()
            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=body.model,
                choices=[Choice(message=ChoiceMessage(content=outcome.text or ""))],
                usage=usage,
                mul=mul_ext,
            )

        async def sse() -> AsyncIterator[str]:
            def chunk(**kwargs) -> str:
                c = ChatCompletionChunk(
                    id=completion_id, created=created, model=body.model, **kwargs
                )
                return f"data: {c.model_dump_json(exclude_none=True)}\n\n"

            yield chunk(choices=[StreamChoice(delta=DeltaMessage(role="assistant"))])
            if outcome.stream is not None:
                async for piece in outcome.stream:
                    if piece:
                        yield chunk(choices=[StreamChoice(delta=DeltaMessage(content=piece))])
            elif outcome.text:
                yield chunk(choices=[StreamChoice(delta=DeltaMessage(content=outcome.text))])
            yield chunk(choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")])
            usage, mul_ext = await finalize()
            yield chunk(choices=[], usage=usage, mul=mul_ext)
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


def main() -> None:
    """Entry point: uvicorn mul.gateway.app:main is wired via __main__ below."""
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
