"""Google Gemini (generativelanguage.googleapis.com) adapter."""

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


class GoogleClient(WorkerClient):
    provider = "google"

    def __init__(self, model_id: str, api_key: str, timeout: float = 180.0):
        self.model_id = model_id
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com/v1beta", timeout=timeout
        )

    def _payload(self, messages, temperature, max_tokens) -> dict:
        system_parts = [str(m["content"]) for m in messages if m["role"] == "system"]
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": str(m["content"])}],
            }
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        payload: dict = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        generation: dict = {}
        if temperature is not None:
            generation["temperature"] = temperature
        if max_tokens is not None:
            generation["maxOutputTokens"] = max_tokens
        if generation:
            payload["generationConfig"] = generation
        return payload

    @staticmethod
    def _extract_text(obj: dict) -> str:
        candidates = obj.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts)

    @staticmethod
    def _extract_usage(obj: dict) -> WorkerUsage | None:
        meta = obj.get("usageMetadata")
        if not meta:
            return None
        return WorkerUsage(
            prompt_tokens=meta.get("promptTokenCount", 0),
            completion_tokens=meta.get("candidatesTokenCount", 0),
        )

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> WorkerResult:
        r = await self._client.post(
            f"/models/{self.model_id}:generateContent",
            params={"key": self._api_key},
            json=self._payload(messages, temperature, max_tokens),
        )
        r.raise_for_status()
        data = r.json()
        text = self._extract_text(data)
        usage = self._extract_usage(data) or WorkerUsage(
            prompt_tokens=estimate_tokens("".join(str(m.get("content") or "") for m in messages)),
            completion_tokens=estimate_tokens(text),
        )
        return WorkerResult(text=text, usage=usage, provider=self.provider, model_id=self.model_id)

    async def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        usage: WorkerUsage | None = None
        emitted = ""
        async with self._client.stream(
            "POST",
            f"/models/{self.model_id}:streamGenerateContent",
            params={"key": self._api_key, "alt": "sse"},
            json=self._payload(messages, temperature, max_tokens),
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                obj = json.loads(line[len("data:") :].strip())
                text = self._extract_text(obj)
                if text:
                    emitted += text
                    yield ("delta", text)
                usage = self._extract_usage(obj) or usage
        if usage is None:
            usage = WorkerUsage(
                prompt_tokens=estimate_tokens("".join(str(m.get("content") or "") for m in messages)),
                completion_tokens=estimate_tokens(emitted),
            )
        yield ("usage", usage)

    async def aclose(self) -> None:
        await self._client.aclose()
