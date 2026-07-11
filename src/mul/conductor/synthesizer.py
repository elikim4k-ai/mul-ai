"""Synthesizer: merge worker outputs into one coherent final answer."""

from __future__ import annotations

from mul.conductor.executor import SubtaskResult
from mul.conductor.prompt_loader import load_prompt
from mul.workers.base import WorkerClient


async def synthesize(
    llm: WorkerClient,
    task: str,
    results: dict[str, SubtaskResult],
    lang: str,
    record_usage,
    feedback: str | None = None,
) -> str:
    parts = "\n\n".join(
        f"### {r.id} ({r.role}, {r.status})\n{r.output}" for r in results.values()
    )
    content = f"Original task:\n{task}\n\nTeam outputs:\n{parts}"
    if feedback:
        content += f"\n\nVerifier feedback to address:\n{feedback}"
    messages = [
        {"role": "system", "content": load_prompt("synthesize", lang)},
        {"role": "user", "content": content},
    ]
    result = await llm.chat(messages)
    record_usage(result.usage, purpose="synthesize")
    return result.text
