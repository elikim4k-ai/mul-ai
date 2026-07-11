"""Orchestration-loop tests, driven by a scripted conductor LLM and mock workers."""

import json

import pytest

from mul.conductor.conductor import Conductor
from tests.conftest import ScriptedConductorLLM

DIRECT = {"mode": "direct", "reason": "trivial"}
ORCH = {"mode": "orchestrate", "reason": "decomposable"}


def plan(subtasks, strategy="test plan"):
    return json.dumps({"strategy": strategy, "subtasks": subtasks, "recurse": False})


def msgs(text):
    return [{"role": "user", "content": text}]


async def test_fast_path_on_trivial_task(config, registry):
    llm = ScriptedConductorLLM(
        {"triage": json.dumps(DIRECT), "direct": "The answer is 4."}
    )
    conductor = Conductor(llm, registry, config)
    outcome = await conductor.handle(msgs("What is 2+2?"), "mul")
    assert outcome.fast_path
    assert outcome.text == "The answer is 4."
    # only the conductor itself was used — no worker records
    assert {r.provider for r in outcome.ctx.usage_records} == {"conductor"}


async def test_orchestration_uses_multiple_workers_and_synthesises(config, registry):
    llm = ScriptedConductorLLM(
        {
            "triage": json.dumps(ORCH),
            "plan": plan(
                [
                    {"id": "t1", "role": "thinker", "instruction": "think",
                     "preferred_capability": "reasoning"},
                    {"id": "t2", "role": "worker", "instruction": "write code",
                     "depends_on": ["t1"], "preferred_capability": "code"},
                    {"id": "t3", "role": "verifier", "instruction": "check t2",
                     "depends_on": ["t2"], "preferred_capability": "korean"},
                ]
            ),
            "synthesize": "FINAL SYNTHESISED ANSWER",
            "verify": json.dumps({"pass": True, "feedback": ""}),
        }
    )
    conductor = Conductor(llm, registry, config)
    outcome = await conductor.handle(msgs("Build a Korean web scraper"), "mul-ultra")
    assert not outcome.fast_path
    assert outcome.text == "FINAL SYNTHESISED ANSWER"
    worker_providers = {
        r.provider for r in outcome.ctx.usage_records if r.provider != "conductor"
    }
    assert len(worker_providers) >= 2  # acceptance: ≥2 workers used
    done = [e for e in outcome.ctx.trace.events if e["kind"] == "subtask_done"]
    assert {e["id"] for e in done} == {"t1", "t2", "t3"}


async def test_recursion_respects_depth_cap(config, registry):
    recursing_plan = plan(
        [{"id": "r1", "role": "worker", "instruction": "big subproblem",
          "recurse": True, "preferred_capability": "reasoning"}]
    )
    llm = ScriptedConductorLLM(
        {
            "triage": json.dumps(ORCH),
            "plan": recursing_plan,  # every planning round asks to recurse again
            "synthesize": lambda m, n: f"SYNTH-{n}",
            "verify": json.dumps({"pass": True, "feedback": ""}),
        }
    )
    conductor = Conductor(llm, registry, config)
    outcome = await conductor.handle(msgs("recurse forever"), "mul")  # max_depth=1
    assert outcome.text  # terminated with an answer instead of recursing forever
    # depth 0 plans + depth 1 plans, then the depth-1 recurse request runs as a leaf
    assert llm.calls["plan"] == 2
    leaf_calls = [r for r in outcome.ctx.usage_records if r.provider != "conductor"]
    assert leaf_calls  # the capped recursion was downgraded to a real worker call


async def test_verify_failure_triggers_one_repair_round(config, registry):
    llm = ScriptedConductorLLM(
        {
            "triage": json.dumps(ORCH),
            "plan": plan(
                [{"id": "t1", "role": "worker", "instruction": "do it",
                  "preferred_capability": "code"}]
            ),
            "synthesize": lambda m, n: f"DRAFT-{n}",
            "verify": lambda m, n: json.dumps(
                {"pass": n > 1, "feedback": "" if n > 1 else "wrong constant"}
            ),
        }
    )
    conductor = Conductor(llm, registry, config)
    outcome = await conductor.handle(msgs("task"), "mul")
    assert outcome.text == "DRAFT-2"  # repaired synthesis
    assert llm.calls["synthesize"] == 2
    assert llm.calls["verify"] == 2
    kinds = [e["kind"] for e in outcome.ctx.trace.events]
    assert "repair" in kinds


async def test_verify_gives_up_gracefully_after_one_repair(config, registry):
    llm = ScriptedConductorLLM(
        {
            "triage": json.dumps(ORCH),
            "plan": plan(
                [{"id": "t1", "role": "worker", "instruction": "do it",
                  "preferred_capability": "code"}]
            ),
            "synthesize": lambda m, n: f"DRAFT-{n}",
            "verify": json.dumps({"pass": False, "feedback": "never good enough"}),
        }
    )
    conductor = Conductor(llm, registry, config)
    outcome = await conductor.handle(msgs("task"), "mul-ultra")
    assert outcome.text == "DRAFT-2"  # best answer so far, never a hang
    assert llm.calls["synthesize"] == 2  # exactly ONE bounded repair round
    assert "verify_gave_up" in [e["kind"] for e in outcome.ctx.trace.events]


async def test_planner_clamps_oversized_plans(config, registry):
    config.tiers["mul"].max_worker_calls = 2
    many = [
        {"id": f"t{i}", "role": "worker", "instruction": f"part {i}",
         "preferred_capability": "code"}
        for i in range(6)
    ]
    llm = ScriptedConductorLLM(
        {
            "triage": json.dumps(ORCH),
            "plan": plan(many),
            "synthesize": "CAPPED FINAL",
            "verify": json.dumps({"pass": True, "feedback": ""}),
        }
    )
    conductor = Conductor(llm, registry, config)
    outcome = await conductor.handle(msgs("huge task"), "mul")
    assert outcome.text == "CAPPED FINAL"  # degraded softly, still answered
    workers = [r for r in outcome.ctx.usage_records if r.provider != "conductor"]
    assert len(workers) <= 2


async def test_executor_enforces_worker_call_cap(config, registry):
    """Even a plan that slips past the planner clamp cannot exceed the budget."""
    from mul.conductor.executor import execute_plan
    from mul.conductor.planner import CoordinationPlan, Subtask
    from mul.conductor.types import Budget, RequestContext
    from mul.tracing import Trace

    llm = ScriptedConductorLLM({})
    conductor = Conductor(llm, registry, config)
    oversized = CoordinationPlan(
        strategy="oversized",
        subtasks=[
            Subtask(id=f"t{i}", role="worker", instruction=f"part {i}",
                    preferred_capability="code")
            for i in range(6)
        ],
    )
    ctx = RequestContext(
        budget=Budget(max_calls=2, wall_clock_seconds=30, max_spend_usd=1.0),
        trace=Trace(),
    )
    results = await execute_plan(
        oversized,
        conductor=conductor,
        ctx=ctx,
        tier=config.tiers["mul"],
        tier_name="mul",
        task="huge task",
        depth=0,
    )
    ok = [r for r in results.values() if r.status == "ok"]
    skipped = [r for r in results.values() if r.status == "skipped"]
    assert len(ok) == 2  # exactly the budget
    assert len(skipped) == 4  # the excess was skipped, not silently dropped
    assert any(e["kind"] == "subtask_skipped" for e in ctx.trace.events)


async def test_invalid_plan_falls_back_to_direct(config, registry):
    llm = ScriptedConductorLLM(
        {
            "triage": json.dumps(ORCH),
            "plan": "I refuse to emit JSON",
            "direct": "fallback direct answer",
        }
    )
    conductor = Conductor(llm, registry, config)
    outcome = await conductor.handle(msgs("task"), "mul")
    assert outcome.fast_path
    assert outcome.text == "fallback direct answer"
    assert llm.calls["plan"] == 2  # one retry before giving up


async def test_streaming_fast_path_yields_chunks_and_usage(config, registry):
    llm = ScriptedConductorLLM(
        {"triage": json.dumps(DIRECT), "direct": "streamed answer " * 10}
    )
    conductor = Conductor(llm, registry, config)
    outcome = await conductor.handle(msgs("hi"), "mul", stream=True)
    assert outcome.stream is not None
    text = "".join([piece async for piece in outcome.stream])
    assert "streamed answer" in text
    # usage recorded after the stream drained
    assert any(r.purpose == "direct" for r in outcome.ctx.usage_records)
