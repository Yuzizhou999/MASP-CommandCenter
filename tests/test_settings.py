from __future__ import annotations

import json
from pathlib import Path

import command_center.settings as settings_module
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
