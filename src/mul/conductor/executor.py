"""Executor: runs the coordination plan as a DAG.

Independent subtasks are dispatched in parallel under a bounded semaphore;
dependent subtasks wait for their dependencies. `role` shapes the system prompt
handed to the worker. A subtask marked recurse spins up a fresh conductor pass
(sharing the same budget and trace) while depth allows.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog

from mul.config import TierConfig
from mul.conductor.planner import CoordinationPlan, Subtask
from mul.conductor.prompt_loader import load_prompt
from mul.conductor.types import BudgetExceeded, RequestContext, UsageRecord

log = structlog.get_logger()

_OUTPUT_SNIPPET = 4000  # chars of a dependency's output forwarded as context


@dataclass
class SubtaskResult:
    id: str
    role: str
    status: str  # ok | error | skipped
    output: str
    provider: str = ""
    model_id: str = ""


async def execute_plan(
    plan: CoordinationPlan,
    *,
    conductor,  # Conductor — untyped to avoid a circular import
    ctx: RequestContext,
    tier: TierConfig,
    tier_name: str,
    task: str,
    depth: int,
) -> dict[str, SubtaskResult]:
    results: dict[str, SubtaskResult] = {}
    pending: dict[str, Subtask] = {st.id: st for st in plan.subtasks}
    known_ids = set(pending)
    sem = asyncio.Semaphore(tier.max_workers)

    async def run_one(st: Subtask) -> SubtaskResult:
        async with sem:
            reason = ctx.budget.check()
            if reason:
                ctx.trace.add("subtask_skipped", id=st.id, reason=reason)
                return SubtaskResult(st.id, st.role, "skipped", f"[budget: {reason}]")

            dep_context = "\n\n".join(
                f"[{d} — {results[d].status}]\n{results[d].output[:_OUTPUT_SNIPPET]}"
                for d in st.depends_on
                if d in results
            )

            if st.recurse and depth < tier.max_depth:
                ctx.trace.add("recurse", id=st.id, depth=depth + 1)
                sub_messages = [
                    {
                        "role": "user",
                        "content": (
                            f"{st.instruction}\n\nContext from earlier subtasks:\n{dep_context}"
                            if dep_context
                            else st.instruction
                        ),
                    }
                ]
                try:
                    ctx.budget.acquire()
                    outcome = await conductor.handle(
                        sub_messages, tier_name, ctx=ctx, depth=depth + 1, stream=False
                    )
                    return SubtaskResult(
                        st.id, st.role, "ok", outcome.text or "", "conductor", "recursive"
                    )
                except BudgetExceeded as e:
                    return SubtaskResult(st.id, st.role, "skipped", f"[budget: {e}]")

            worker = conductor.registry.resolve(
                st.preferred_capability, ctx.exclude_providers, ctx.allow_providers
            )
            if worker is None:
                ctx.trace.add("subtask_no_worker", id=st.id)
                return SubtaskResult(st.id, st.role, "error", "[no eligible worker in pool]")

            role_prompt = load_prompt(f"role_{st.role}", ctx.lang)
            user_content = f"Original task:\n{task}\n"
            if dep_context:
                user_content += f"\nContext from earlier subtasks:\n{dep_context}\n"
            user_content += f"\nYour subtask:\n{st.instruction}"
            messages = [
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": user_content},
            ]
            try:
                ctx.budget.acquire()
                result = await worker.client.chat(messages)
            except BudgetExceeded as e:
                ctx.trace.add("subtask_skipped", id=st.id, reason=str(e))
                return SubtaskResult(st.id, st.role, "skipped", f"[budget: {e}]")
            except Exception as e:
                log.warning("worker call failed", id=st.id, error=str(e))
                ctx.trace.add("subtask_error", id=st.id, error=str(e))
                return SubtaskResult(st.id, st.role, "error", f"[worker error: {e}]")

            entry = worker.entry
            record = UsageRecord(
                provider=entry.provider,
                model_id=entry.model_id,
                tier_rank=entry.tier_rank,
                cost_per_1m_input=entry.cost_per_1m_input,
                cost_per_1m_output=entry.cost_per_1m_output,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                purpose=f"subtask:{st.id}",
            )
            ctx.usage_records.append(record)
            ctx.budget.add_spend(record.cost_usd)
            ctx.trace.add(
                "subtask_done",
                id=st.id,
                role=st.role,
                provider=entry.provider,
                model_id=entry.model_id,
                capability=st.preferred_capability,
                tokens=record.total_tokens,
            )
            return SubtaskResult(
                st.id, st.role, "ok", result.text, entry.provider, entry.model_id
            )

    while pending:
        ready = [
            st
            for st in pending.values()
            # dependencies outside the plan are treated as satisfied
            if all(d in results for d in st.depends_on if d in known_ids)
        ]
        if not ready:  # dependency cycle — fail the remainder softly
            for st in pending.values():
                results[st.id] = SubtaskResult(
                    st.id, st.role, "error", "[unresolvable dependency cycle]"
                )
            ctx.trace.add("plan_cycle", ids=list(pending))
            break
        wave = await asyncio.gather(*(run_one(st) for st in ready))
        for r in wave:
            results[r.id] = r
            del pending[r.id]
    return results
