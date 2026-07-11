"""Shared conductor types: usage records, per-request budget, request context."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from mul.config import TierConfig
from mul.tracing import Trace


@dataclass
class UsageRecord:
    provider: str
    model_id: str
    tier_rank: int
    cost_per_1m_input: float
    cost_per_1m_output: float
    prompt_tokens: int
    completion_tokens: int
    purpose: str  # triage | plan | subtask:<id> | verify | synthesize | direct

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1_000_000 * self.cost_per_1m_input
            + self.completion_tokens / 1_000_000 * self.cost_per_1m_output
        )


class BudgetExceeded(Exception):
    pass


class Budget:
    """Hard caps on a single request: total worker calls, wall clock, spend.

    Shared across recursion so a nested conductor cannot escape the caps.
    A tripped budget degrades the request to 'best answer so far' — never a hang.
    """

    def __init__(self, max_calls: int, wall_clock_seconds: float, max_spend_usd: float):
        self.max_calls = max_calls
        self.deadline = time.monotonic() + wall_clock_seconds
        self.max_spend_usd = max_spend_usd
        self.calls = 0
        self.spend_usd = 0.0

    def check(self) -> str | None:
        if self.calls >= self.max_calls:
            return f"max worker calls reached ({self.max_calls})"
        if time.monotonic() >= self.deadline:
            return "wall clock exhausted"
        if self.spend_usd >= self.max_spend_usd:
            return f"spend cap reached (${self.max_spend_usd:.2f})"
        return None

    def acquire(self) -> None:
        reason = self.check()
        if reason:
            raise BudgetExceeded(reason)
        self.calls += 1

    def add_spend(self, usd: float) -> None:
        self.spend_usd += usd

    @classmethod
    def from_tier(cls, tier: TierConfig) -> "Budget":
        return cls(tier.max_worker_calls, tier.wall_clock_seconds, tier.max_spend_usd)


@dataclass
class RequestContext:
    """Mutable state shared across one request, including recursive sub-conductors."""

    budget: Budget
    trace: Trace
    lang: str = "en"
    exclude_providers: set[str] = field(default_factory=set)
    allow_providers: set[str] | None = None
    usage_records: list[UsageRecord] = field(default_factory=list)


@dataclass
class ConductorOutcome:
    ctx: RequestContext
    text: str | None = None
    stream: AsyncIterator[str] | None = None
    fast_path: bool = True
