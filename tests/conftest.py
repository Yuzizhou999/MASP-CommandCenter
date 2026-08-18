from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from command_center.settings import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = Path(os.getenv("MASP_TEST_ENGINE_ROOT", PROJECT_ROOT.parent / "MASP"))


@pytest.fixture()
def isolated_settings(tmp_path: Path) -> Settings:
    lock = json.loads((PROJECT_ROOT / "engine.lock.json").read_text(encoding="utf-8"))
    return Settings(
        app_env="development",
        host="127.0.0.1",
        port=8877,
        engine_root=ENGINE_ROOT,
        engine_commit=lock["commit"],
        allow_dirty_development=True,
        deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        deepseek_timeout_seconds=2,
        root=tmp_path,
    )
