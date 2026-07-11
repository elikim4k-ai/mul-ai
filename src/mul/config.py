"""Configuration: mul.config.yaml (pool + tiers) and .env secrets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CAPABILITIES = {"code", "math", "korean", "long_context", "reasoning", "vision"}


class TierConfig(BaseModel):
    orchestrate_bias: Literal["low", "high"] = "low"
    max_workers: int = 4
    max_depth: int = 1
    verify_passes: int = 1
    max_worker_calls: int = 12
    wall_clock_seconds: float = 120.0
    max_spend_usd: float = 1.0


class ConductorConfig(BaseModel):
    serving: Literal["mock", "vllm"] = "vllm"
    base_url: str = "http://localhost:8000/v1"
    model_id: str = "Qwen/Qwen2.5-32B-Instruct"
    gpu: str = "A100-80GB"
    tier_rank: int = 1
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0


class PoolEntry(BaseModel):
    provider: str
    model_id: str
    capabilities: list[str] = Field(default_factory=list)
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    tier_rank: int = 1
    enabled: bool = True
    base_url: str | None = None  # overrides the provider's default endpoint


class MulConfig(BaseModel):
    tiers: dict[str, TierConfig]
    conductor: ConductorConfig
    pool: list[PoolEntry]


def load_config(path: str | Path) -> MulConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return MulConfig.model_validate(raw)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    mul_config_path: str = "mul.config.yaml"
    mul_db_url: str = "sqlite:///mul_usage.db"
    mul_api_keys: str = ""  # comma-separated caller keys; empty = open dev mode
    mul_admin_keys: str = ""  # comma-separated admin keys
    mul_key_provider_allow: str = ""  # JSON: {"<key>": ["upstage", ...]}
    mul_rate_limit_rpm: int = 0  # 0 = disabled

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    upstage_api_key: str = ""
    lg_exaone_api_key: str = ""
    lg_exaone_base_url: str = ""
    naver_clova_api_key: str = ""
