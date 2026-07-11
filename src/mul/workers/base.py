"""WorkerClient — the common interface every provider adapter implements.

Everything internal stays OpenAI-shaped: messages are ``{"role": ..., "content": ...}``
dicts, and adapters normalise their provider's native API to this shape.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


def estimate_tokens(text: str) -> int:
    """Crude fallback when a provider doesn't return usage (~4 chars/token)."""
    return max(1, len(text) // 4)


@dataclass
class WorkerUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class WorkerResult:
    text: str
    usage: WorkerUsage
    provider: str
    model_id: str


# Stream events: ("delta", str) chunks followed by exactly one ("usage", WorkerUsage).
StreamEvent = tuple[Literal["delta", "usage"], object]


class WorkerClient(abc.ABC):
    provider: str
    model_id: str
    is_stub: bool = False  # True when this stands in for a provider missing its API key

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> WorkerResult: ...

    async def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Default: emulate streaming with a single non-streamed call."""
        result = await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        yield ("delta", result.text)
        yield ("usage", result.usage)

    async def aclose(self) -> None:
        pass
