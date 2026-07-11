"""Anthropic Messages API adapter, normalised to OpenAI-style messages."""

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

_API_VERSION = "2023-06-01"


class AnthropicClient(WorkerClient):
    provider = "anthropic"

    def __init__(self, model_id: str, api_key: str, timeout: float = 180.0):
        self.model_id = model_id
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={"x-api-key": api_key, "anthropic-version": _API_VERSION},
            timeout=timeout,
        )

    def _payload(self, messages, temperature, max_tokens) -> dict:
        system_parts = [str(m["content"]) for m in messages if m["role"] == "system"]
        chat = [
            {"role": m["role"], "content": str(m["content"])}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        payload: dict = {
            "model": self.model_id,
            "max_tokens": max_tokens or 4096,
            "messages": chat,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> WorkerResult:
        r = await self._client.post(
            "/v1/messages", json=self._payload(messages, temperature, max_tokens)
        )
        r.raise_for_status()
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage") or {}
        usage = WorkerUsage(
            prompt_tokens=u.get("input_tokens", 0),
            completion_tokens=u.get("output_tokens", 0),
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
        usage = WorkerUsage()
        emitted = ""
        async with self._client.stream("POST", "/v1/messages", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                obj = json.loads(line[len("data:") :].strip())
                kind = obj.get("type")
                if kind == "message_start":
                    usage.prompt_tokens = (obj["message"].get("usage") or {}).get("input_tokens", 0)
                elif kind == "content_block_delta":
                    text = (obj.get("delta") or {}).get("text")
                    if text:
                        emitted += text
                        yield ("delta", text)
                elif kind == "message_delta":
                    usage.completion_tokens = (obj.get("usage") or {}).get("output_tokens", 0)
        if usage.completion_tokens == 0:
            usage.completion_tokens = estimate_tokens(emitted)
        yield ("usage", usage)

    async def aclose(self) -> None:
        await self._client.aclose()
