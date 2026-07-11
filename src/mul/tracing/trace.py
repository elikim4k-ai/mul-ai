"""Internal decision traces. Admin-only: never returned to normal callers."""

from __future__ import annotations

import time
import uuid


class Trace:
    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex
        self.events: list[dict] = []

    def add(self, kind: str, **data) -> None:
        self.events.append({"kind": kind, "t": round(time.time(), 3), **data})

    def to_dict(self) -> dict:
        return {"trace_id": self.trace_id, "events": self.events}
