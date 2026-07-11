"""OpenAI-compatible request/response models plus the namespaced `mul` extension."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any = None

    def flat_content(self) -> str:
        """Normalise OpenAI content (str or list-of-parts) to plain text."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "".join(
                p.get("text", "") for p in self.content if isinstance(p, dict)
            )
        return "" if self.content is None else str(self.content)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolerate any OpenAI SDK extras

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    # per-request pool opt-out (also accepted via extra_body by stock SDKs)
    mul_exclude_providers: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class MulExtension(BaseModel):
    billed_tokens: int
    billed_rate_tier: str
    billed_cost_usd: float
    trace: dict | None = None  # only populated for admin keys with X-Mul-Trace: true


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str = ""


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    mul: MulExtension


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
    usage: Usage | None = None
    mul: MulExtension | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "mul-ai"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def now_ts() -> int:
    return int(time.time())
