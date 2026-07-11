"""Eval harness (Phase 6 skeleton — not yet the deliverable).

Runs task suites from evals/suites/*.yaml against a Mul gateway and any baseline
OpenAI-compatible endpoints, scores keyword coverage, and prints a comparison
table with cost + call counts.

Usage:
    python evals/harness.py --gateway http://localhost:8080/v1 --model mul
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import httpx
import yaml

SUITES_DIR = Path(__file__).parent / "suites"


async def run_case(client: httpx.AsyncClient, model: str, case: dict) -> dict:
    t0 = time.perf_counter()
    r = await client.post(
        "/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": case["prompt"]}]},
    )
    r.raise_for_status()
    body = r.json()
    text = body["choices"][0]["message"]["content"].lower()
    keywords = [k.lower() for k in case.get("expect_keywords", [])]
    hits = sum(1 for k in keywords if k in text)
    return {
        "id": case["id"],
        "score": hits / len(keywords) if keywords else None,
        "latency_s": round(time.perf_counter() - t0, 2),
        "total_tokens": body.get("usage", {}).get("total_tokens", 0),
        "billed_cost_usd": body.get("mul", {}).get("billed_cost_usd"),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="mul")
    parser.add_argument("--api-key", default="dev")
    args = parser.parse_args()

    cases = []
    for suite_file in sorted(SUITES_DIR.glob("*.yaml")):
        suite = yaml.safe_load(suite_file.read_text(encoding="utf-8")) or []
        cases.extend(suite)
    if not cases:
        print("No suites found in evals/suites/ — add YAML files with "
              "[{id, prompt, expect_keywords}] entries.")
        return

    async with httpx.AsyncClient(
        base_url=args.gateway.rstrip("/"),
        headers={"Authorization": f"Bearer {args.api_key}"},
        timeout=300,
    ) as client:
        results = [await run_case(client, args.model, c) for c in cases]

    print(f"\n{'case':<24}{'score':>8}{'latency':>10}{'tokens':>10}{'cost':>12}")
    for r in results:
        score = f"{r['score']:.2f}" if r["score"] is not None else "-"
        cost = f"${r['billed_cost_usd']:.4f}" if r["billed_cost_usd"] is not None else "-"
        print(f"{r['id']:<24}{score:>8}{r['latency_s']:>9}s{r['total_tokens']:>10}{cost:>12}")


if __name__ == "__main__":
    asyncio.run(main())
