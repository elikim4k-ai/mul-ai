from __future__ import annotations

import re
from collections import defaultdict

import pytest

from mul.config import ConductorConfig, MulConfig, PoolEntry, Settings, TierConfig
from mul.workers.providers.mock import MockWorkerClient
from mul.workers.registry import WorkerRegistry

_STAGE = re.compile(r"MUL-STAGE:\s*(\w+)")


class ScriptedConductorLLM(MockWorkerClient):
    """Conductor LLM whose per-stage responses are scripted by tests.

    script maps stage -> str, or -> callable(messages, nth_call_for_stage) -> str.
    """

    def __init__(self, script: dict):
        super().__init__(provider="conductor", model_id="scripted-conductor")
        self.script = script
        self.calls: dict[str, int] = defaultdict(int)

    def _respond(self, messages):
        system = next((str(m["content"]) for m in messages if m["role"] == "system"), "")
        m = _STAGE.search(system)
        stage = m.group(1) if m else "direct"
        self.calls[stage] += 1
        if stage not in self.script:
            raise AssertionError(f"unscripted conductor stage: {stage}")
        resp = self.script[stage]
        return resp(messages, self.calls[stage]) if callable(resp) else resp


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        mul_db_url=f"sqlite:///{tmp_path.as_posix()}/usage.db",
    )


@pytest.fixture
def config() -> MulConfig:
    return MulConfig(
        tiers={
            "mul": TierConfig(
                orchestrate_bias="low",
                max_workers=4,
                max_depth=1,
                verify_passes=1,
                max_worker_calls=8,
                wall_clock_seconds=30,
                max_spend_usd=1.0,
            ),
            "mul-ultra": TierConfig(
                orchestrate_bias="high",
                max_workers=8,
                max_depth=2,
                verify_passes=2,
                max_worker_calls=24,
                wall_clock_seconds=60,
                max_spend_usd=5.0,
            ),
        },
        conductor=ConductorConfig(serving="mock", model_id="mock-conductor", tier_rank=1),
        # no provider keys in test settings → all of these register as mock stubs,
        # which is exactly the keyless-run behaviour the build brief requires
        pool=[
            PoolEntry(
                provider="upstage",
                model_id="solar-test",
                capabilities=["korean", "reasoning"],
                cost_per_1m_input=1.0,
                cost_per_1m_output=2.0,
                tier_rank=3,
            ),
            PoolEntry(
                provider="lg",
                model_id="exaone-test",
                capabilities=["korean", "code"],
                cost_per_1m_input=1.0,
                cost_per_1m_output=2.0,
                tier_rank=3,
            ),
            PoolEntry(
                provider="openai",
                model_id="gpt-test",
                capabilities=["code", "reasoning"],
                cost_per_1m_input=5.0,
                cost_per_1m_output=15.0,
                tier_rank=5,
            ),
            PoolEntry(
                provider="vllm",
                model_id="qwen-test",
                capabilities=["code", "korean", "math"],
                cost_per_1m_input=0.0,
                cost_per_1m_output=0.0,
                tier_rank=1,
            ),
            PoolEntry(
                provider="google",
                model_id="gemini-test",
                capabilities=["vision"],
                tier_rank=4,
                enabled=False,  # must never be registered
            ),
        ],
    )


@pytest.fixture
def registry(config, settings) -> WorkerRegistry:
    return WorkerRegistry(config, settings)
