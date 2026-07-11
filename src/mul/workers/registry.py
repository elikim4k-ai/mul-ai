"""Worker registry: loads the pool from config, applies opt-outs, resolves
capability → best available worker (highest tier_rank wins).

A pool entry whose provider key is missing from the environment is registered
as a mock *stub* so the whole system runs keyless; real providers activate
automatically once their key appears in .env.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from mul.config import MulConfig, PoolEntry, Settings
from mul.workers.base import WorkerClient
from mul.workers.providers.anthropic import AnthropicClient
from mul.workers.providers.google import GoogleClient
from mul.workers.providers.mock import MockWorkerClient
from mul.workers.providers.naver import NaverClovaClient
from mul.workers.providers.openai_compat import OpenAICompatClient

log = structlog.get_logger()

# provider name -> (default base_url, Settings attr holding the key)
_OPENAI_COMPAT = {
    "openai": ("https://api.openai.com/v1", "openai_api_key"),
    "upstage": ("https://api.upstage.ai/v1", "upstage_api_key"),
    "lg": (None, "lg_exaone_api_key"),  # base_url from entry or LG_EXAONE_BASE_URL
    "vllm": ("http://localhost:8001/v1", None),  # local — no key required
}


@dataclass
class RegisteredWorker:
    entry: PoolEntry
    client: WorkerClient


class WorkerRegistry:
    def __init__(self, config: MulConfig, settings: Settings):
        self.workers: list[RegisteredWorker] = []
        for entry in config.pool:
            if not entry.enabled:
                continue
            client = self._build_client(entry, settings)
            if client.is_stub:
                log.warning(
                    "provider key missing — registered as mock stub",
                    provider=entry.provider,
                    model_id=entry.model_id,
                )
            self.workers.append(RegisteredWorker(entry=entry, client=client))

    @staticmethod
    def _build_client(entry: PoolEntry, settings: Settings) -> WorkerClient:
        def stub() -> WorkerClient:
            return MockWorkerClient(
                provider=entry.provider, model_id=entry.model_id, is_stub=True
            )

        p = entry.provider
        if p == "mock":
            return MockWorkerClient(provider="mock", model_id=entry.model_id)
        if p in _OPENAI_COMPAT:
            default_base, key_attr = _OPENAI_COMPAT[p]
            api_key = getattr(settings, key_attr) if key_attr else None
            base_url = entry.base_url or (
                settings.lg_exaone_base_url if p == "lg" else default_base
            )
            if key_attr and not api_key:
                return stub()
            if not base_url:
                return stub()
            return OpenAICompatClient(
                provider=p,
                model_id=entry.model_id,
                base_url=base_url,
                api_key=api_key or None,
                supports_stream_options=p in ("openai", "vllm"),
            )
        if p == "anthropic":
            if not settings.anthropic_api_key:
                return stub()
            return AnthropicClient(model_id=entry.model_id, api_key=settings.anthropic_api_key)
        if p == "google":
            if not settings.google_api_key:
                return stub()
            return GoogleClient(model_id=entry.model_id, api_key=settings.google_api_key)
        if p == "naver":
            if not settings.naver_clova_api_key:
                return stub()
            kwargs = {"base_url": entry.base_url} if entry.base_url else {}
            return NaverClovaClient(
                model_id=entry.model_id, api_key=settings.naver_clova_api_key, **kwargs
            )
        log.warning("unknown provider — registered as mock stub", provider=p)
        return stub()

    def eligible(
        self,
        exclude_providers: set[str] | None = None,
        allow_providers: set[str] | None = None,
    ) -> list[RegisteredWorker]:
        exclude = exclude_providers or set()
        out = []
        for w in self.workers:
            if w.entry.provider in exclude:
                continue
            if allow_providers is not None and w.entry.provider not in allow_providers:
                continue
            out.append(w)
        return out

    def resolve(
        self,
        capability: str | None,
        exclude_providers: set[str] | None = None,
        allow_providers: set[str] | None = None,
    ) -> RegisteredWorker | None:
        pool = self.eligible(exclude_providers, allow_providers)
        if capability:
            capable = [w for w in pool if capability in w.entry.capabilities]
            pool = capable or pool  # fall back to any eligible worker
        if not pool:
            return None
        return max(pool, key=lambda w: w.entry.tier_rank)

    async def aclose(self) -> None:
        for w in self.workers:
            await w.client.aclose()
