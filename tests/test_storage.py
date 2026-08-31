from __future__ import annotations

import json
from pathlib import Path

import pytest

from command_center.approvals import ApprovalStore
from command_center.contracts import (
    DispatchIntent,
    IntentType,
    IntentValidation,
    RiskLevel,
    TaskDraft,
)
from command_center.storage import atomic_write_json, atomic_write_text


def test_atomic_write_creates_parent_and_writes_content(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "state.json"

    atomic_write_json(target, {"a": 1})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


def test_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "state.json"

    atomic_write_json(target, {"a": 1})

    assert [path.name for path in tmp_path.iterdir()] == ["state.json"]


def test_failed_write_keeps_previous_content(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"revision": 1})

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("command_center.storage.os.replace", explode)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_json(target, {"revision": 2})

    # 目标文件必须仍是替换前的完整内容，而不是被截断的半个 JSON。
    assert json.loads(target.read_text(encoding="utf-8")) == {"revision": 1}
    assert [path.name for path in tmp_path.iterdir()] == ["state.json"]


def test_failed_write_removes_temp_file(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "state.json"

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr("command_center.storage.os.fsync", explode)
    with pytest.raises(OSError, match="fsync failed"):
        atomic_write_text(target, "partial")

    assert list(tmp_path.iterdir()) == []


def _intent() -> DispatchIntent:
    return DispatchIntent(
        intentType=IntentType.CREATE_TASK,
        environment="simulation",
        basedOnWorldRevision=1,
        task=TaskDraft(
            pickupNodeId="AP1",
            dropoffNodeId="AP2",
            requiredRobotGroup="fork",
        ),
    )


def _validation(intent: DispatchIntent) -> IntentValidation:
    return IntentValidation(
        intentId=intent.intent_id,
        valid=True,
        riskLevel=RiskLevel.R3_HIGH,
        approvalRequired=True,
        policyCode="policy.r3.approval_required",
        issues=[],
    )


def test_approval_store_survives_failed_save(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "approvals.json"
    store = ApprovalStore(path)
    first = _intent()
    store.create(first, _validation(first))
    original = path.read_text(encoding="utf-8")

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("command_center.storage.os.replace", explode)
    second = _intent()
    with pytest.raises(OSError, match="disk full"):
        store.create(second, _validation(second))

    # 审批单文件仍可被解析，重新加载不会因截断 JSON 而失败。
    assert path.read_text(encoding="utf-8") == original
    assert len(ApprovalStore(path).list()) == 1
