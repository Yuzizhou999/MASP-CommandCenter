from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str
    host: str
    port: int
    engine_root: Path
    engine_commit: str
    allow_dirty_development: bool
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    deepseek_timeout_seconds: float
    deepseek_max_retries: int = 2
    deepseek_circuit_failure_threshold: int = 3
    deepseek_circuit_reset_seconds: float = 30
    deepseek_input_cost_per_million: float = 0.27
    deepseek_output_cost_per_million: float = 1.10
    llm_provider: str = "deepseek"
    local_llm_enabled: bool = False
    local_llm_api_key: str = "local"
    local_llm_base_url: str = "http://127.0.0.1:8000/v1"
    local_llm_model: str = "masp-intent-lora"
    local_llm_timeout_seconds: float = 30
    local_llm_model_card: Path | None = None
    agent_model_id: str = "masp-ppo-priority"
    agent_model_version: str = "1.0.0"
    agent_checkpoint: Path | None = None
    agent_device: str = "cpu"
    agent_torch_threads: int = 1
    agent_runtime_mode: str = "linear"
    agent_max_decisions: int = 8
    agent_max_tool_calls: int = 6
    agent_max_repair_attempts: int = 2
    agent_max_total_tokens: int = 8192
    agent_max_estimated_cost_usd: float = 0.25
    agent_max_latency_ms: int = 30000
    agent_max_steps: int = 48
    root: Path = ROOT

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @classmethod
    def load(cls) -> "Settings":
        lock = json.loads((ROOT / "engine.lock.json").read_text(encoding="utf-8"))
        default_engine = ROOT.parent / "MASP"
        engine_root = Path(os.getenv("MASP_ENGINE_ROOT", str(default_engine))).resolve()
        checkpoint_value = os.getenv("MASP_AGENT_CHECKPOINT", "").strip()
        checkpoint = None
        if checkpoint_value:
            checkpoint = Path(checkpoint_value)
            if not checkpoint.is_absolute():
                checkpoint = ROOT / checkpoint
            checkpoint = checkpoint.resolve()
        else:
            bundled_checkpoint = ROOT / "models" / "ppo-priority-v1.pt"
            if bundled_checkpoint.is_file():
                checkpoint = bundled_checkpoint.resolve()
        model_card_value = os.getenv("LOCAL_LLM_MODEL_CARD", "").strip()
        model_card = None
        if model_card_value:
            model_card = Path(model_card_value)
            if not model_card.is_absolute():
                model_card = ROOT / model_card
            model_card = model_card.resolve()
        else:
            default_model_card = ROOT / "models" / "masp-intent-lora" / "model-card.json"
            if default_model_card.is_file():
                model_card = default_model_card.resolve()
        llm_provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
        if llm_provider not in {"deepseek", "local", "auto"}:
            raise ValueError("LLM_PROVIDER 必须是 deepseek、local 或 auto")
        runtime_mode = os.getenv("AGENT_RUNTIME_MODE", "linear").strip().lower()
        if runtime_mode not in {"linear", "loop"}:
            raise ValueError("AGENT_RUNTIME_MODE 必须是 linear 或 loop")
        return cls(
            app_env=os.getenv("APP_ENV", "development").strip().lower(),
            host=os.getenv("APP_HOST", "127.0.0.1"),
            port=int(os.getenv("APP_PORT", "8877")),
            engine_root=engine_root,
            engine_commit=str(lock["commit"]),
            allow_dirty_development=_as_bool(
                os.getenv("MASP_ALLOW_DIRTY_DEVELOPMENT"),
                bool(lock.get("allowDirtyInDevelopment", False)),
            ),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            deepseek_timeout_seconds=float(
                os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "30")
            ),
            deepseek_max_retries=max(
                0, min(5, int(os.getenv("DEEPSEEK_MAX_RETRIES", "2")))
            ),
            deepseek_circuit_failure_threshold=max(
                1,
                int(os.getenv("DEEPSEEK_CIRCUIT_FAILURE_THRESHOLD", "3")),
            ),
            deepseek_circuit_reset_seconds=max(
                1, float(os.getenv("DEEPSEEK_CIRCUIT_RESET_SECONDS", "30"))
            ),
            deepseek_input_cost_per_million=max(
                0, float(os.getenv("DEEPSEEK_INPUT_COST_PER_MILLION", "0.27"))
            ),
            deepseek_output_cost_per_million=max(
                0, float(os.getenv("DEEPSEEK_OUTPUT_COST_PER_MILLION", "1.10"))
            ),
            llm_provider=llm_provider,
            local_llm_enabled=_as_bool(os.getenv("LOCAL_LLM_ENABLED")),
            local_llm_api_key=os.getenv("LOCAL_LLM_API_KEY", "local"),
            local_llm_base_url=os.getenv(
                "LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1"
            ).rstrip("/"),
            local_llm_model=os.getenv(
                "LOCAL_LLM_MODEL", "masp-intent-lora"
            ).strip(),
            local_llm_timeout_seconds=max(
                1, float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "30"))
            ),
            local_llm_model_card=model_card,
            agent_model_id=os.getenv(
                "MASP_AGENT_MODEL_ID", "masp-ppo-priority"
            ).strip(),
            agent_model_version=os.getenv(
                "MASP_AGENT_MODEL_VERSION", "1.0.0"
            ).strip(),
            agent_checkpoint=checkpoint,
            agent_device=os.getenv("MASP_AGENT_DEVICE", "cpu").strip().lower(),
            agent_torch_threads=max(
                1, min(16, int(os.getenv("MASP_AGENT_TORCH_THREADS", "1")))
            ),
            agent_runtime_mode=runtime_mode,
            agent_max_decisions=max(
                1, min(32, int(os.getenv("AGENT_MAX_DECISIONS", "8")))
            ),
            agent_max_tool_calls=max(
                1, min(24, int(os.getenv("AGENT_MAX_TOOL_CALLS", "6")))
            ),
            agent_max_repair_attempts=max(
                0, min(5, int(os.getenv("AGENT_MAX_REPAIR_ATTEMPTS", "2")))
            ),
            agent_max_total_tokens=max(
                128, int(os.getenv("AGENT_MAX_TOTAL_TOKENS", "8192"))
            ),
            agent_max_estimated_cost_usd=max(
                0, float(os.getenv("AGENT_MAX_ESTIMATED_COST_USD", "0.25"))
            ),
            agent_max_latency_ms=max(
                100, int(os.getenv("AGENT_MAX_LATENCY_MS", "30000"))
            ),
            agent_max_steps=max(
                8, min(256, int(os.getenv("AGENT_MAX_STEPS", "48")))
            ),
        )
