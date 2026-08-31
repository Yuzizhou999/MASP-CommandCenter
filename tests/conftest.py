from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from command_center.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = Path(os.getenv("MASP_TEST_ENGINE_ROOT", PROJECT_ROOT.parent / "MASP"))

INTEGRATION_MODULES = {
    "test_agent_loop.py",
    "test_agent_policy.py",
    "test_agent_run_manager.py",
    "test_agent_runtime.py",
    "test_agent_stage_two.py",
    "test_api.py",
    "test_benchmark.py",
    "test_clarification_and_explanation.py",
    "test_incidents.py",
    "test_safety_flow.py",
    "test_scenario_drafts.py",
    "test_scenario_package.py",
    "test_task_stream.py",
}


def _engine_gate() -> tuple[bool, str]:
    lock = json.loads((PROJECT_ROOT / "engine.lock.json").read_text(encoding="utf-8"))
    if not ENGINE_ROOT.is_dir():
        return False, f"MASP integration engine does not exist: {ENGINE_ROOT}"
    try:
        commit = subprocess.run(
            ["git", "-C", str(ENGINE_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ENGINE_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return False, f"MASP integration engine cannot be inspected: {error}"
    if commit != lock["commit"]:
        return False, f"MASP integration engine commit {commit} != {lock['commit']}"
    if dirty:
        return False, "MASP integration engine is dirty"
    return True, ""


ENGINE_AVAILABLE, ENGINE_SKIP_REASON = _engine_gate()


def pytest_collection_modifyitems(items) -> None:
    for item in items:
        if Path(str(item.fspath)).name in INTEGRATION_MODULES:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


def pytest_runtest_setup(item) -> None:
    if item.get_closest_marker("integration") and not ENGINE_AVAILABLE:
        pytest.skip(ENGINE_SKIP_REASON)


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
