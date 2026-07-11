"""Generic adapter for any OpenAI-compatible endpoint.

Covers: OpenAI, Upstage Solar (api.upstage.ai/v1), vLLM-served open-weight models,
EXAONE behind an OpenAI-compatible host, and the local conductor itself.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from mul.workers.base import (
    Message,
    StreamEvent,
    WorkerClient,
    WorkerResult,
    WorkerUsage,
    estimate_tokens,
)


class OpenAICompatClient(WorkerClient):
    def __init__(
        self,
        provider: str,
        model_id: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 180.0,
        supports_stream_options: bool = True,
    ):
        self.provider = provider
        self.model_id = model_id
        self._supports_stream_options = supports_stream_options
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    def _payload(self, messages, temperature, max_tokens) -> dict:
        payload: dict = {"model": self.model_id, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> WorkerResult:
        r = await self._client.post(
            "/chat/completions", json=self._payload(messages, temperature, max_tokens)
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"].get("content") or ""
        u = data.get("usage") or {}
        prompt_text = "".join(str(m.get("content") or "") for m in messages)
        usage = WorkerUsage(
            prompt_tokens=u.get("prompt_tokens") or estimate_tokens(prompt_text),
            completion_tokens=u.get("completion_tokens") or estimate_tokens(text),
        )
        return WorkerResult(text=text, usage=usage, provider=self.provider, model_id=self.model_id)

    async def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        payload = self._payload(messages, temperature, max_tokens)
        payload["stream"] = True
        if self._supports_stream_options:
            payload["stream_options"] = {"include_usage": True}
        usage: WorkerUsage | None = None
        emitted = ""
        async with self._client.stream("POST", "/chat/completions", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    u = obj["usage"]
                    usage = WorkerUsage(
                        prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0),
                    )
                for choice in obj.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        emitted += content
                        yield ("delta", content)
        if usage is None:
            prompt_text = "".join(str(m.get("content") or "") for m in messages)
            usage = WorkerUsage(
                prompt_tokens=estimate_tokens(prompt_text),
                completion_tokens=estimate_tokens(emitted),
            )
        yield ("usage", usage)

    async def aclose(self) -> None:
        await self._client.aclose()
