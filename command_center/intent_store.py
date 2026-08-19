from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from .contracts import ApprovalStatus, DispatchIntent, new_id, utc_now


class IntentStore:
    """Stores simulation commitments only; it never writes to a field controller."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def commit(
        self,
        intent: DispatchIntent,
        *,
        current_world_revision: int,
        approval: Any | None,
        actor: str,
    ) -> dict[str, Any]:
        if intent.environment != "simulation":
            raise ValueError("当前版本只允许提交到simulation环境。")
        if intent.based_on_world_revision != current_world_revision:
            raise ValueError("世界状态已经变化，请重新仿真。")
        if approval is not None:
            if approval.status is not ApprovalStatus.APPROVED:
                raise ValueError("高风险意图尚未批准。")
            if approval.intent.intent_id != intent.intent_id:
                raise ValueError("审批单与调度意图不匹配。")
        record = {
            "commitId": new_id("commit"),
            "status": "SIMULATION_COMMITTED",
            "environment": "simulation",
            "intent": intent.model_dump(by_alias=True, mode="json"),
            "approvalId": approval.approval_id if approval is not None else None,
            "actor": actor,
            "committedAt": utc_now().isoformat(),
            "notice": "该提交仅写入仿真环境，不会向真实车辆下发指令。",
        }
        with self._lock:
            rows: list[dict[str, Any]] = []
            if self.path.exists():
                rows = json.loads(self.path.read_text(encoding="utf-8"))
            rows.append(record)
            self.path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return record

    def list(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            return json.loads(self.path.read_text(encoding="utf-8"))[::-1]

