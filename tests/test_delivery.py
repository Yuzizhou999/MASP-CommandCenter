from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from command_center.delivery import verify_delivery_manifest
from command_center.engine_adapter import MaspAdapter
from command_center.settings import Settings


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_settings(tmp_path: Path, engine_root: Path, commit: str) -> Settings:
    return Settings(
        app_env="production",
        host="127.0.0.1",
        port=8877,
        engine_root=engine_root,
        engine_commit=commit,
        allow_dirty_development=False,
        deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        deepseek_timeout_seconds=2,
        root=tmp_path,
    )


def test_verified_engine_bundle_is_allowed_without_git(tmp_path: Path) -> None:
    engine_root = tmp_path / "engine" / "MASP"
    required = (
        "masp/online.py",
        "generated/xiate-unified-map-model.json",
        "scenarios/interactive-multi-fleet.json",
    )
    hashes: dict[str, str] = {}
    for relative in required:
        target = engine_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")
        hashes[relative] = _sha256(target)
    commit = "a" * 40
    (engine_root / "engine.bundle.json").write_text(
        json.dumps({"schemaVersion": 1, "commit": commit, "files": hashes}),
        encoding="utf-8",
    )

    status = MaspAdapter(_bundle_settings(tmp_path, engine_root, commit)).engine_status()

    assert status["allowed"] is True
    assert status["source"] == "verified-bundle"
    assert status["bundleVerified"] is True

    (engine_root / "masp" / "online.py").write_text("changed", encoding="utf-8")
    changed = MaspAdapter(
        _bundle_settings(tmp_path, engine_root, commit)
    ).engine_status()
    assert changed["allowed"] is False
    assert changed["dirtyFileCount"] == 1


def test_engine_bundle_does_not_inherit_parent_git_worktree(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    engine_root = repository / "package" / "engine" / "MASP"
    required = (
        "masp/online.py",
        "generated/xiate-unified-map-model.json",
        "scenarios/interactive-multi-fleet.json",
    )
    hashes: dict[str, str] = {}
    for relative in required:
        target = engine_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")
        hashes[relative] = _sha256(target)
    commit = "a" * 40
    (engine_root / "engine.bundle.json").write_text(
        json.dumps({"schemaVersion": 1, "commit": commit, "files": hashes}),
        encoding="utf-8",
    )
    repository.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)

    status = MaspAdapter(
        _bundle_settings(tmp_path, engine_root, commit)
    ).engine_status()
    assert status["allowed"] is True
    assert status["source"] == "verified-bundle"


def test_delivery_manifest_detects_changes_and_prohibited_files(tmp_path: Path) -> None:
    payload = tmp_path / "README.md"
    payload.write_text("demo", encoding="utf-8")
    environment = tmp_path / ".env"
    environment.write_text("DEEPSEEK_API_KEY=secret", encoding="utf-8")
    manifest = {
        "schemaVersion": 1,
        "sourceCommit": "b" * 40,
        "engineCommit": "a" * 40,
        "files": {"README.md": _sha256(payload)},
    }
    manifest_path = tmp_path / "delivery-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_delivery_manifest(tmp_path)["ok"] is True

    manifest["files"][".env"] = _sha256(environment)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_delivery_manifest(tmp_path)["prohibited"] == [".env"]

    payload.write_text("modified", encoding="utf-8")
    report = verify_delivery_manifest(tmp_path)
    assert report["ok"] is False
    assert report["mismatched"] == ["README.md"]
    assert report["prohibited"] == [".env"]
