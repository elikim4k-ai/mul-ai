"""Conductor LLM factory.

serving: vllm → OpenAI-compatible client pointed at the local vLLM server.
serving: mock → deterministic scripted conductor so the whole system runs with
no GPU and no keys (dev + tests). The mock recognises the stage from the
"MUL-STAGE:" marker each prompt file begins with.
"""

from __future__ import annotations

import json
import re

from mul.config import ConductorConfig
from mul.workers.base import Message, WorkerClient
from mul.workers.providers.mock import MockWorkerClient
from mul.workers.providers.openai_compat import OpenAICompatClient

_STAGE = re.compile(r"MUL-STAGE:\s*(\w+)")
_BIAS = re.compile(r"(?:Tier bias|티어 성향):\s*(\w+)")
_HANGUL = re.compile(r"[가-힣]")


def build_conductor_llm(cfg: ConductorConfig) -> WorkerClient:
    if cfg.serving == "mock":
        return MockConductorLLM(model_id=cfg.model_id)
    return OpenAICompatClient(
        provider="conductor", model_id=cfg.model_id, base_url=cfg.base_url
    )


class MockConductorLLM(MockWorkerClient):
    """Deterministic conductor behaviour for keyless dev/demo runs."""

    def __init__(self, model_id: str = "mock-conductor"):
        super().__init__(provider="conductor", model_id=model_id)

    def _respond(self, messages: list[Message]) -> str:
        system = next((str(m["content"]) for m in messages if m["role"] == "system"), "")
        user = next(
            (str(m["content"]) for m in reversed(messages) if m["role"] == "user"), ""
        )
        stage_m = _STAGE.search(system)
        stage = stage_m.group(1) if stage_m else "direct"

        if stage == "triage":
            bias_m = _BIAS.search(system)
            bias = bias_m.group(1) if bias_m else "low"
            orchestrate = (
                "[orchestrate]" in user
                or (bias == "high" and len(user) > 60)
                or len(user) > 400
            )
            return json.dumps(
                {
                    "mode": "orchestrate" if orchestrate else "direct",
                    "reason": "mock triage heuristic",
                }
            )

        if stage == "plan":
            lowered = user.lower()
            if "code" in lowered or "코드" in user:
                capability = "code"
            elif _HANGUL.search(user):
                capability = "korean"
            else:
                capability = "reasoning"
            return json.dumps(
                {
                    "strategy": "mock decomposition: think, execute, verify",
                    "subtasks": [
                        {
                            "id": "t1",
                            "role": "thinker",
                            "instruction": f"Outline an approach for: {user[:160]}",
                            "depends_on": [],
                            "preferred_capability": "reasoning",
                        },
                        {
                            "id": "t2",
                            "role": "worker",
                            "instruction": f"Execute the task: {user[:160]}",
                            "depends_on": ["t1"],
                            "preferred_capability": capability,
                        },
                        {
                            "id": "t3",
                            "role": "verifier",
                            "instruction": "Check t2's output against the task.",
                            "depends_on": ["t2"],
                            # a different capability from t2 so demo traces show
                            # the plan genuinely fanning out across providers
                            "preferred_capability": "korean",
                        },
                    ],
                    "recurse": False,
                }
            )

        if stage == "verify":
            return json.dumps({"pass": True, "feedback": ""})

        if stage == "synthesize":
            return f"[mul:mock-synthesis] Final merged answer based on the team's work.\n\n{user[-400:]}"

        # direct
        return f"[mul:mock-direct] {user[:300]}"
