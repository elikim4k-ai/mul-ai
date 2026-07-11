"""Verification pass over the candidate final answer."""

from __future__ import annotations

from mul.conductor.planner import extract_json
from mul.conductor.prompt_loader import load_prompt
from mul.workers.base import WorkerClient


async def verify(
    llm: WorkerClient,
    task: str,
    candidate: str,
    lang: str,
    record_usage,
) -> tuple[bool, str]:
    """Returns (passed, feedback). Fails open on unparseable verifier output."""
    messages = [
        {"role": "system", "content": load_prompt("verify", lang)},
        {
            "role": "user",
            "content": f"Original task:\n{task}\n\nCandidate answer:\n{candidate}",
        },
    ]
    result = await llm.chat(messages)
    record_usage(result.usage, purpose="verify")
    obj = extract_json(result.text)
    if obj is None:
        return True, ""
    return bool(obj.get("pass", True)), str(obj.get("feedback") or "")
