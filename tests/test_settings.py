from __future__ import annotations

import json
from pathlib import Path

import command_center.settings as settings_module
import pytest
from command_center.settings import Settings


def _prepare_root(root: Path) -> None:
    (root / "models").mkdir(parents=True)
    (root / "engine.lock.json").write_text(
        json.dumps({"commit": "a" * 40}), encoding="utf-8"
    )


def test_settings_discovers_bundled_agent_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    _prepare_root(tmp_path)
    checkpoint = tmp_path / "models" / "ppo-priority-v1.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(settings_module, "ROOT", tmp_path)
    monkeypatch.delenv("MASP_AGENT_CHECKPOINT", raising=False)

    settings = Settings.load()

    assert settings.agent_checkpoint == checkpoint.resolve()
    assert settings.agent_model_version == "1.0.0"
    assert settings.agent_torch_threads == 1


def test_explicit_agent_checkpoint_overrides_bundled_model(
    tmp_path: Path, monkeypatch
) -> None:
    _prepare_root(tmp_path)
    (tmp_path / "models" / "ppo-priority-v1.pt").write_bytes(b"bundled")
    alternate = tmp_path / "models" / "alternate.pt"
    alternate.write_bytes(b"alternate")
    monkeypatch.setattr(settings_module, "ROOT", tmp_path)
    monkeypatch.setenv("MASP_AGENT_CHECKPOINT", "models/alternate.pt")

    settings = Settings.load()

    assert settings.agent_checkpoint == alternate.resolve()


def test_settings_discovers_local_llm_model_card(tmp_path: Path, monkeypatch) -> None:
    _prepare_root(tmp_path)
    model_dir = tmp_path / "models" / "masp-intent-lora"
    model_dir.mkdir()
    card = model_dir / "model-card.json"
    card.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings_module, "ROOT", tmp_path)
    monkeypatch.delenv("LOCAL_LLM_MODEL_CARD", raising=False)

    settings = Settings.load()

    assert settings.local_llm_model_card == card.resolve()


def test_settings_rejects_unknown_llm_provider(tmp_path: Path, monkeypatch) -> None:
    _prepare_root(tmp_path)
    monkeypatch.setattr(settings_module, "ROOT", tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        Settings.load()
