"""Billing-formula tests for the cases required by §6 of the build brief:
single-worker, multi-worker, recursion, and opt-out-changes-top-tier."""

from mul.billing.pricing import blended_bill
from mul.conductor.types import UsageRecord


def rec(provider, rank, cin, cout, p, c, purpose="subtask:t1", model=None):
    return UsageRecord(
        provider=provider,
        model_id=model or f"{provider}-model",
        tier_rank=rank,
        cost_per_1m_input=cin,
        cost_per_1m_output=cout,
        prompt_tokens=p,
        completion_tokens=c,
        purpose=purpose,
    )


def test_empty_records():
    bill = blended_bill([])
    assert bill.billed_tokens == 0 and bill.cost_usd == 0.0


def test_single_worker():
    bill = blended_bill([rec("upstage", 3, 1.0, 2.0, 1_000_000, 500_000)])
    assert bill.billed_tokens == 1_500_000
    assert bill.rate_tier == "rank-3"
    assert bill.cost_usd == 1.0 * 1 + 2.0 * 0.5  # $2.00


def test_multi_worker_blended_rate_is_top_rank():
    records = [
        rec("conductor", 1, 0.0, 0.0, 200_000, 100_000, purpose="plan"),
        rec("upstage", 3, 1.0, 2.0, 1_000_000, 0),
        rec("openai", 5, 5.0, 15.0, 0, 1_000_000),
    ]
    bill = blended_bill(records)
    # tokens summed across ALL calls; rate = openai's (rank 5); fees never stacked
    assert bill.billed_prompt_tokens == 1_200_000
    assert bill.billed_completion_tokens == 1_100_000
    assert bill.rate_tier == "rank-5"
    assert bill.cost_usd == 1.2 * 5.0 + 1.1 * 15.0
    assert bill.top_provider == "openai"


def test_recursion_records_aggregate_the_same_way():
    # a recursive sub-conductor just appends more records to the same request
    records = [
        rec("conductor", 1, 0.0, 0.0, 100_000, 50_000, purpose="triage"),
        rec("conductor", 1, 0.0, 0.0, 100_000, 50_000, purpose="plan"),
        rec("lg", 3, 1.0, 2.0, 400_000, 200_000, purpose="subtask:t1"),
        rec("lg", 3, 1.0, 2.0, 400_000, 200_000, purpose="subtask:nested"),
    ]
    bill = blended_bill(records)
    assert bill.billed_tokens == 1_500_000
    assert bill.rate_tier == "rank-3"
    assert bill.cost_usd == 1.0 * 1.0 + 2.0 * 0.5


def test_opt_out_changes_top_tier():
    full = [
        rec("upstage", 3, 1.0, 2.0, 1_000_000, 0),
        rec("openai", 5, 5.0, 15.0, 100_000, 100_000),
    ]
    # caller opted out of openai → it never participates → cheaper top tier
    opted_out = [r for r in full if r.provider != "openai"]
    assert blended_bill(full).rate_tier == "rank-5"
    bill = blended_bill(opted_out)
    assert bill.rate_tier == "rank-3"
    assert bill.cost_usd == 1.0  # 1M prompt tokens at $1/1M


def test_rank_tie_broken_by_higher_price():
    records = [
        rec("upstage", 3, 1.0, 2.0, 500_000, 500_000),
        rec("naver", 3, 2.0, 4.0, 500_000, 500_000),
    ]
    bill = blended_bill(records)
    assert bill.top_provider == "naver"
    # totals: 1M prompt at $2/1M + 1M completion at $4/1M
    assert bill.cost_usd == 1.0 * 2.0 + 1.0 * 4.0
