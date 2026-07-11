"""Deterministic mock worker — lets the whole system run and test with zero API keys."""

from __future__ import annotations

from typing import AsyncIterator, Callable

from mul.workers.base import (
    Message,
    StreamEvent,
    WorkerClient,
    WorkerResult,
    WorkerUsage,
    estimate_tokens,
)


class MockWorkerClient(WorkerClient):
    def __init__(
        self,
        provider: str = "mock",
        model_id: str = "mock-1",
        responder: Callable[[list[Message]], str] | None = None,
        is_stub: bool = False,
    ):
        self.provider = provider
        self.model_id = model_id
        self.is_stub = is_stub
        self._responder = responder

    def _respond(self, messages: list[Message]) -> str:
        if self._responder is not None:
            return self._responder(messages)
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return f"[{self.provider}/{self.model_id}] mock answer to: {str(last_user)[:200]}"

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> WorkerResult:
        text = self._respond(messages)
        prompt_text = "".join(str(m.get("content") or "") for m in messages)
        usage = WorkerUsage(
            prompt_tokens=estimate_tokens(prompt_text),
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
        result = await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        text = result.text
        for i in range(0, len(text), 12):
            yield ("delta", text[i : i + 12])
        yield ("usage", result.usage)
