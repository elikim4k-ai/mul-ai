"""Per-request usage persistence. SQLite for dev, Postgres-ready via SQLAlchemy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

from sqlalchemy import Boolean, Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class RequestLog(Base):
    __tablename__ = "request_log"

    id = Column(String, primary_key=True)
    created = Column(Float, nullable=False)
    api_key_hash = Column(String, index=True)
    tier = Column(String, nullable=False)
    fast_path = Column(Boolean, default=True)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    billed_tokens = Column(Integer, default=0)
    billed_rate_tier = Column(String, default="")
    billed_cost_usd = Column(Float, default=0.0)
    trace_json = Column(Text, nullable=True)  # admin-only decision trace


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:16] if api_key else "anonymous"


class UsageStore:
    def __init__(self, db_url: str):
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self._engine = create_engine(db_url, connect_args=connect_args)
        Base.metadata.create_all(self._engine)
        self._session = sessionmaker(bind=self._engine)

    def _record_sync(self, **fields) -> None:
        with self._session() as s:
            s.add(RequestLog(**fields))
            s.commit()

    async def record(
        self,
        *,
        request_id: str,
        api_key: str,
        tier: str,
        fast_path: bool,
        prompt_tokens: int,
        completion_tokens: int,
        billed_tokens: int,
        billed_rate_tier: str,
        billed_cost_usd: float,
        trace: dict | None,
    ) -> None:
        await asyncio.to_thread(
            self._record_sync,
            id=request_id,
            created=time.time(),
            api_key_hash=hash_key(api_key),
            tier=tier,
            fast_path=fast_path,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            billed_tokens=billed_tokens,
            billed_rate_tier=billed_rate_tier,
            billed_cost_usd=billed_cost_usd,
            trace_json=json.dumps(trace, ensure_ascii=False) if trace else None,
        )
