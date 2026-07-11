"""Planner: ask the conductor LLM for a structured coordination plan (strict JSON)."""

from __future__ import annotations

import json
from typing import Literal

import structlog
from pydantic import BaseModel, Field, ValidationError

from mul.config import CAPABILITIES, TierConfig
from mul.conductor.prompt_loader import fill, load_prompt
from mul.workers.base import WorkerClient

log = structlog.get_logger()


class Subtask(BaseModel):
    id: str
    role: Literal["thinker", "worker", "verifier"] = "worker"
    instruction: str
    depends_on: list[str] = Field(default_factory=list)
    preferred_capability: str | None = None
    recurse: bool = False


class CoordinationPlan(BaseModel):
    strategy: str = ""
    subtasks: list[Subtask]
    recurse: bool = False


def extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


async def make_plan(
    llm: WorkerClient,
    task: str,
    tier: TierConfig,
    lang: str,
    record_usage,
) -> CoordinationPlan | None:
    """Returns a validated plan, or None (caller falls back to the fast path)."""
    max_subtasks = min(tier.max_worker_calls, tier.max_workers * 2)
    system = fill(
        load_prompt("plan", lang),
        capabilities=", ".join(sorted(CAPABILITIES)),
        max_subtasks=max_subtasks,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    for attempt in range(2):
        result = await llm.chat(messages)
        record_usage(result.usage, purpose="plan")
        obj = extract_json(result.text)
        if obj is not None:
            try:
                plan = CoordinationPlan.model_validate(obj)
            except ValidationError as e:
                log.warning("plan failed validation", attempt=attempt, error=str(e))
                plan = None
            if plan is not None:
                # de-duplicate ids, clamp size — never trust the LLM to respect caps
                seen: set[str] = set()
                unique = []
                for st in plan.subtasks:
                    if st.id not in seen:
                        seen.add(st.id)
                        unique.append(st)
                plan.subtasks = unique[:max_subtasks]
                return plan
        messages += [
            {"role": "assistant", "content": result.text},
            {
                "role": "user",
                "content": "That was not valid JSON matching the schema. Reply with the JSON object only.",
            },
        ]
    return None
