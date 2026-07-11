"""Conductor entry point: triage → (fast path | plan → execute → synthesize → verify).

Only the final synthesised answer ever leaves this module; routing detail lives
exclusively in ctx.trace (admin-only).
"""

from __future__ import annotations

from typing import AsyncIterator

import structlog

from mul.config import MulConfig
from mul.conductor.executor import execute_plan
from mul.conductor.planner import extract_json, make_plan
from mul.conductor.prompt_loader import detect_lang, fill, load_prompt
from mul.conductor.synthesizer import synthesize
from mul.conductor.types import (
    Budget,
    ConductorOutcome,
    RequestContext,
    UsageRecord,
)
from mul.conductor.verifier import verify
from mul.tracing import Trace
from mul.workers.base import Message, WorkerClient, WorkerUsage
from mul.workers.registry import WorkerRegistry

log = structlog.get_logger()

_STREAM_CHUNK = 24  # chars per pseudo-stream chunk of an orchestrated answer


class Conductor:
    def __init__(self, llm: WorkerClient, registry: WorkerRegistry, config: MulConfig):
        self.llm = llm
        self.registry = registry
        self.config = config

    def _recorder(self, ctx: RequestContext):
        cfg = self.config.conductor

        def record(usage: WorkerUsage, purpose: str) -> None:
            record = UsageRecord(
                provider="conductor",
                model_id=cfg.model_id,
                tier_rank=cfg.tier_rank,
                cost_per_1m_input=cfg.cost_per_1m_input,
                cost_per_1m_output=cfg.cost_per_1m_output,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                purpose=purpose,
            )
            ctx.usage_records.append(record)
            ctx.budget.add_spend(record.cost_usd)

        return record

    @staticmethod
    def _render_task(messages: list[Message]) -> str:
        turns = [m for m in messages if m.get("role") != "system"]
        if len(turns) <= 1:
            return str(turns[0].get("content", "")) if turns else ""
        return "\n".join(f"{m['role']}: {m.get('content', '')}" for m in turns[-6:])

    async def handle(
        self,
        messages: list[Message],
        tier_name: str,
        *,
        exclude_providers: set[str] | None = None,
        allow_providers: set[str] | None = None,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        ctx: RequestContext | None = None,
        depth: int = 0,
    ) -> ConductorOutcome:
        tier = self.config.tiers[tier_name]
        if ctx is None:
            ctx = RequestContext(
                budget=Budget.from_tier(tier),
                trace=Trace(),
                lang=detect_lang(messages),
                exclude_providers=exclude_providers or set(),
                allow_providers=allow_providers,
            )
        record_usage = self._recorder(ctx)
        task = self._render_task(messages)

        # 1. Triage — fast-path gate
        triage_system = fill(load_prompt("triage", ctx.lang), bias=tier.orchestrate_bias)
        triage_result = await self.llm.chat(
            [
                {"role": "system", "content": triage_system},
                {"role": "user", "content": task},
            ]
        )
        record_usage(triage_result.usage, purpose="triage")
        decision = extract_json(triage_result.text) or {}
        mode = decision.get("mode", "direct")
        ctx.trace.add("triage", depth=depth, mode=mode, reason=decision.get("reason", ""))

        if mode != "orchestrate":
            return await self._direct(messages, ctx, record_usage, stream, temperature, max_tokens)

        # 2. Plan
        plan = await make_plan(self.llm, task, tier, ctx.lang, record_usage)
        if plan is None or not plan.subtasks:
            ctx.trace.add("plan_failed_fallback_direct", depth=depth)
            return await self._direct(messages, ctx, record_usage, stream, temperature, max_tokens)
        ctx.trace.add(
            "plan",
            depth=depth,
            strategy=plan.strategy,
            subtasks=[
                {
                    "id": s.id,
                    "role": s.role,
                    "capability": s.preferred_capability,
                    "depends_on": s.depends_on,
                    "recurse": s.recurse,
                }
                for s in plan.subtasks
            ],
        )

        # 3. Execute the DAG
        results = await execute_plan(
            plan,
            conductor=self,
            ctx=ctx,
            tier=tier,
            tier_name=tier_name,
            task=task,
            depth=depth,
        )

        # 4. Synthesise, then 5. verify with one bounded repair round
        final = await synthesize(self.llm, task, results, ctx.lang, record_usage)
        checks = repairs = 0
        verified = True
        while checks < tier.verify_passes + repairs:
            verified, feedback = await verify(self.llm, task, final, ctx.lang, record_usage)
            checks += 1
            ctx.trace.add("verify", depth=depth, passed=verified, feedback=feedback[:500])
            if verified:
                continue
            if repairs == 0:
                repairs = 1
                ctx.trace.add("repair", depth=depth)
                final = await synthesize(
                    self.llm, task, results, ctx.lang, record_usage, feedback=feedback
                )
            else:
                ctx.trace.add("verify_gave_up", depth=depth)  # best answer so far
                break

        outcome = ConductorOutcome(ctx=ctx, text=final, fast_path=False)
        if stream and depth == 0:
            outcome.stream = self._chunk(final)
        return outcome

    async def _direct(
        self,
        messages: list[Message],
        ctx: RequestContext,
        record_usage,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> ConductorOutcome:
        msgs = [{"role": "system", "content": load_prompt("direct", ctx.lang)}] + [
            m for m in messages if m.get("role") != "system"
        ]
        if not stream:
            result = await self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            record_usage(result.usage, purpose="direct")
            return ConductorOutcome(ctx=ctx, text=result.text, fast_path=True)

        async def gen() -> AsyncIterator[str]:
            async for kind, payload in self.llm.chat_stream(
                msgs, temperature=temperature, max_tokens=max_tokens
            ):
                if kind == "delta":
                    yield payload  # type: ignore[misc]
                else:
                    record_usage(payload, purpose="direct")  # type: ignore[arg-type]

        return ConductorOutcome(ctx=ctx, stream=gen(), fast_path=True)

    @staticmethod
    async def _chunk(text: str) -> AsyncIterator[str]:
        for i in range(0, len(text), _STREAM_CHUNK):
            yield text[i : i + _STREAM_CHUNK]
