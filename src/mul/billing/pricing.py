"""Blended-rate billing (Fugu-style).

⚠️ OWNER SIGN-OFF REQUIRED (§6 of the build brief) — formula implemented as
specified there, pending confirmation:

  billed tokens = SUM of real token usage across every model actually called
                  for the request (conductor calls included, recursion included)
  billed rate   = the per-token rate of the SINGLE top-tier_rank model that
                  participated in the request (ties broken by higher price)
  billed cost   = billed prompt tokens  × top rate (input)
                + billed completion tokens × top rate (output)

Fees are never stacked per model; one blended rate per request.
"""

from __future__ import annotations

from dataclasses import dataclass

from mul.conductor.types import UsageRecord


@dataclass
class Bill:
    billed_prompt_tokens: int
    billed_completion_tokens: int
    billed_tokens: int
    rate_tier: str  # opaque label, e.g. "rank-5" — never leaks provider/model
    cost_usd: float
    # admin-only detail (kept out of normal API responses)
    top_provider: str
    top_model_id: str


def blended_bill(records: list[UsageRecord]) -> Bill:
    if not records:
        return Bill(0, 0, 0, "rank-0", 0.0, "", "")
    top = max(
        records,
        key=lambda r: (r.tier_rank, r.cost_per_1m_output, r.cost_per_1m_input),
    )
    prompt = sum(r.prompt_tokens for r in records)
    completion = sum(r.completion_tokens for r in records)
    cost = (
        prompt / 1_000_000 * top.cost_per_1m_input
        + completion / 1_000_000 * top.cost_per_1m_output
    )
    return Bill(
        billed_prompt_tokens=prompt,
        billed_completion_tokens=completion,
        billed_tokens=prompt + completion,
        rate_tier=f"rank-{top.tier_rank}",
        cost_usd=round(cost, 8),
        top_provider=top.provider,
        top_model_id=top.model_id,
    )
