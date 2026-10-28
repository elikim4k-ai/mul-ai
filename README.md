# Mul AI (물 AI)

**A conductor-orchestrated multi-agent system delivered as a single model, behind one
OpenAI-compatible API.** Mul (물) means *water* — fluid, adaptive, taking the shape of
the problem. A small conductor model triages each request: simple ones flow straight to
an answer (fast path); complex ones are decomposed and delegated to a team of expert
worker LLMs (Korean-first pool + international frontier), verified, and synthesised into
one answer. Callers only ever see two models: `mul` and `mul-ultra`.

**물 AI는 하나의 모델처럼 보이는 멀티 에이전트 오케스트레이션 시스템입니다.**
지휘자(conductor) 모델이 요청을 분류하여 단순한 요청은 직접 답하고, 복잡한 요청은
전문가 워커 LLM 팀에게 분해·위임한 뒤 검증을 거쳐 하나의 답으로 통합합니다.
호출자는 `mul`과 `mul-ultra` 두 모델만 봅니다.

> Status: Phases 1–4 complete (gateway, worker registry, orchestration loop, billing/traces).
> The conductor is **prompt-orchestrated** — it is not "trained" (that is Phase 7 research).

## Quickstart (keyless, no GPU) · 빠른 시작

Prerequisites: Git and Python 3.11+. Run from the repository root.

```bash
git clone https://github.com/elikim4k-ai/mul-ai.git
cd mul-ai
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q                             # no API keys needed
python -m uvicorn --factory mul.gateway.app:create_app --host 127.0.0.1 --port 8080
```

`mul.config.yaml` ships with `conductor.serving: mock`, and any pool provider without an
API key in `.env` runs as a deterministic mock stub — so the full system (triage → plan →
parallel workers → verify → synthesise → blended billing) runs end-to-end with zero keys.
Add real keys to `.env` (copy `.env.example`) and providers activate automatically.

The example client requires an extra package: `python -m pip install openai`.
Run it in another terminal while the server is running.

Talk to it with any stock OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="dev")
stream = client.chat.completions.create(
    model="mul",  # or "mul-ultra"
    messages=[{"role": "user", "content": "안녕하세요!"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## API

- `POST /v1/chat/completions` — OpenAI schema, streaming + non-streaming. `model` ∈ {`mul`, `mul-ultra`}.
- `GET /v1/models` — the two tiers only; worker models are never exposed.
- Responses carry standard `usage` plus a namespaced `mul` object:
  `{"billed_tokens", "billed_rate_tier", "billed_cost_usd"}`.
- Per-request pool opt-out: `"mul_exclude_providers": ["openai", "google"]` in the body
  (via `extra_body` in stock SDKs) or header `X-Mul-Exclude-Providers: openai,google`.
  Intersected with the account-level allow-list (`MUL_KEY_PROVIDER_ALLOW`).
- Routing is hidden by design. Admin keys may send `X-Mul-Trace: true` to receive the
  full decision trace (triage, plan, per-worker calls, verification, repair) in `mul.trace`.

## Billing (blended rate — ⚠️ formula pending owner sign-off)

Per request: token usage is summed across **every** model actually called (conductor,
workers, recursion). The caller is billed at the single rate of the **top-`tier_rank`
model that participated** — fees are never stacked. Implemented in
`src/mul/billing/pricing.py` with unit tests for single-worker, multi-worker, recursion,
and opt-out-changes-top-tier cases. Prices in `mul.config.yaml` are TODO placeholders —
fill real numbers before charging anyone.

## Tiers

| tier | bias | max workers | recursion depth | verify passes |
|---|---|---|---|---|
| `mul` | fast path | 4 | 1 | 1 |
| `mul-ultra` | orchestrate | 8 | 2 | 2 |

Hard caps per request (worker calls, wall clock, spend) are enforced in the executor;
a tripped cap degrades to "best answer so far", never a hang.

## Vultr GPU deployment

Only the **conductor** needs the GPU — worker calls go out to hosted provider APIs.

| conductor model | precision | VRAM needed | Vultr SKU |
|---|---|---|---|
| Qwen2.5-7B / EXAONE-7.8B | bf16 | ~18 GB | L40S 48GB |
| Qwen2.5-14B | bf16 | ~32 GB | L40S 48GB |
| Qwen2.5-32B (default) | bf16 | ~68 GB | A100 80GB |
| Qwen2.5-32B | AWQ 4-bit | ~24 GB | L40S 48GB |

On a fresh Ubuntu 24.04 GPU instance: `bash deploy/vultr/provision.sh` (installs vLLM,
systemd units `mul-vllm` + `mul`, flips `conductor.serving` to `vllm`). Useful vLLM flags:
`--max-model-len` (KV-cache budget), `--gpu-memory-utilization` (default 0.92 here),
`--quantization awq` for tight VRAM. Or use Docker: `docker compose -f deploy/docker-compose.yml --profile gpu up`.

Before committing to a region: confirm GPU stock there, and that outbound calls to your
chosen worker providers (Upstage/LG/Naver/OpenAI/Anthropic/Google) are reachable. Vultr
bundles generous egress, so worker API traffic cost is a non-issue at MVP scale.

## Repository map

```
src/mul/gateway/     FastAPI app, OpenAI schemas, auth (keys, allow-lists, rate limit)
src/mul/conductor/   triage → plan → execute (DAG) → verify → synthesize; prompts/ (KR+EN)
src/mul/workers/     WorkerClient adapters + registry (capability → best worker)
src/mul/billing/     blended pricing + usage persistence (SQLite/Postgres)
src/mul/tracing/     admin-only decision traces
evals/               Phase-6 harness skeleton (score Mul vs baselines)
deploy/              Dockerfile, docker-compose, Vultr provision + systemd
```

## Roadmap

- **Phase 5** — Next.js chat UI (`web/`): streaming, sidebar, tier switcher, KR-first.
- **Phase 6** — eval harness: coding / reasoning / KR suites, Mul vs baselines, cost report.
- **Phase 7 (research)** — learned coordinator (evolutionary / RL routing policy). The
  prompt-orchestrated conductor stays the stable default.

## Validate and reproduce the keyless demo

On Windows PowerShell, create the environment with `python -m venv .venv`
and activate it with `.\.venv\Scripts\Activate.ps1`; then run the same
install, pytest, and server commands above. Use a fresh checkout without a
`.env` file and with provider API-key environment variables unset for mock mode.

With the server running, visit `http://127.0.0.1:8080/healthz` and
`http://127.0.0.1:8080/v1/models`. The models response must expose `mul` and
`mul-ultra`. The supplied Python streaming example should print nonempty text.
The gateway tests exercise non-streaming responses, SSE termination, auth,
provider exclusion, and admin trace visibility without making provider calls.

Mock replies validate software behavior, not language-model quality. The
`evals/` directory is a harness skeleton; no comparative quality, cost, or
latency benchmark is established by passing unit tests. Billing rates remain
placeholders. Live-provider output and latency vary.

Record `git rev-parse HEAD`, `python --version`, `python -m pip freeze`,
`mul.config.yaml`, the request, and the provider/mock mode with any reported
result. Dependencies use lower bounds rather than a lockfile. Usage is stored
in `mul_usage.db` by default; use a fresh working copy for a fresh demo ledger.
Never include provider secrets in a shared reproduction bundle.

## License and contributions

See [LICENSE](LICENSE). Include a minimal request, configuration with secrets
removed, expected/actual response, and dependency versions in bug reports.
Run `python -m pytest -q` before submitting changes.
