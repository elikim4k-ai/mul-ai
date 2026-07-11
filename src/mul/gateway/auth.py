"""API-key auth, per-key provider allow-lists, and a simple per-key rate limit.

Open dev mode: when no keys are configured at all, any caller is accepted with
admin rights (loudly logged) so local development works out of the box.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import structlog
from fastapi import HTTPException, Request

from mul.config import Settings

log = structlog.get_logger()


@dataclass
class KeyInfo:
    key: str
    is_admin: bool
    allowed_providers: set[str] | None  # None = all providers allowed


class Authenticator:
    def __init__(self, settings: Settings):
        self._keys = {k.strip() for k in settings.mul_api_keys.split(",") if k.strip()}
        self._admin_keys = {
            k.strip() for k in settings.mul_admin_keys.split(",") if k.strip()
        }
        self._open_mode = not self._keys and not self._admin_keys
        if self._open_mode:
            log.warning("no MUL_API_KEYS configured — running in OPEN DEV MODE")
        self._allow: dict[str, set[str]] = {}
        if settings.mul_key_provider_allow:
            raw = json.loads(settings.mul_key_provider_allow)
            self._allow = {k: set(v) for k, v in raw.items()}
        self._rpm = settings.mul_rate_limit_rpm
        self._hits: dict[str, deque] = defaultdict(deque)

    def authenticate(self, request: Request) -> KeyInfo:
        auth = request.headers.get("authorization", "")
        key = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""

        if self._open_mode:
            info = KeyInfo(key=key or "dev", is_admin=True, allowed_providers=None)
        elif key in self._admin_keys:
            info = KeyInfo(key=key, is_admin=True, allowed_providers=self._allow.get(key))
        elif key in self._keys:
            info = KeyInfo(key=key, is_admin=False, allowed_providers=self._allow.get(key))
        else:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Invalid API key.",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            )

        self._rate_limit(info.key)
        return info

    def _rate_limit(self, key: str) -> None:
        if self._rpm <= 0:
            return
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self._rpm:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": "Rate limit exceeded.",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
            )
        hits.append(now)
