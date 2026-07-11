"""Naver HyperCLOVA X via CLOVA Studio chat completions.

TODO(owner): verify the endpoint path + model id against current CLOVA Studio docs
before enabling in production — this adapter follows the v3 chat-completions shape.
"""

from __future__ import annotations

import httpx

from mul.workers.base import (
    Message,
    WorkerClient,
    WorkerResult,
    WorkerUsage,
    estimate_tokens,
)


class NaverClovaClient(WorkerClient):
    provider = "naver"

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str = "https://clovastudio.stream.ntruss.com",
        timeout: float = 180.0,
    ):
        self.model_id = model_id
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> WorkerResult:
        payload: dict = {
            "messages": [
                {"role": m["role"], "content": str(m["content"])} for m in messages
            ]
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["maxTokens"] = max_tokens
        r = await self._client.post(f"/v3/chat-completions/{self.model_id}", json=payload)
        r.raise_for_status()
        result = (r.json() or {}).get("result") or {}
        text = (result.get("message") or {}).get("content") or ""
        usage_raw = result.get("usage") or {}
        usage = WorkerUsage(
            prompt_tokens=usage_raw.get("promptTokens")
            or estimate_tokens("".join(str(m.get("content") or "") for m in messages)),
            completion_tokens=usage_raw.get("completionTokens") or estimate_tokens(text),
        )
        return WorkerResult(text=text, usage=usage, provider=self.provider, model_id=self.model_id)

    # Streaming: inherits the non-streaming fallback from WorkerClient.

    async def aclose(self) -> None:
        await self._client.aclose()
