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
        )
