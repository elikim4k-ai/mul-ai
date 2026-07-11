"""Loads stage prompts from conductor/prompts/ (Korean + English variants).

Placeholders use {name} but are substituted with str.replace, never str.format,
because the prompts contain literal JSON braces.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
_HANGUL = re.compile(r"[가-힣]")


def detect_lang(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return "ko" if _HANGUL.search(str(m.get("content") or "")) else "en"
    return "en"


@lru_cache(maxsize=None)
def load_prompt(stage: str, lang: str = "en") -> str:
    path = PROMPTS_DIR / f"{stage}_{lang}.txt"
    if not path.exists():
        path = PROMPTS_DIR / f"{stage}_en.txt"
    return path.read_text(encoding="utf-8")


def fill(prompt: str, **placeholders: str) -> str:
    for key, value in placeholders.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt
