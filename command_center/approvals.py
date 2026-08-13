from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from .contracts import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    DispatchIntent,
    IntentValidation,
    utc_now,
)


class ApprovalStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._items = self._load()

    def _load(self) -> dict[str, ApprovalRequest]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        rows = [ApprovalRequest.model_validate(item) for item in raw]
        return {item.approval_id: item for item in rows}

    def _save(self) -> None:
        payload = [
            item.model_dump(by_alias=True, mode="json")
            for item in sorted(self._items.values(), key=lambda row: row.created_at)
        ]
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def create(
        self,
        intent: DispatchIntent,
        validation: IntentValidation,
        run_ids: list[str] | None = None,
    ) -> ApprovalRequest:
        request = ApprovalRequest(
            intent=intent,
            validation=validation,
            simulationRunIds=run_ids or [],
            requestedBy=intent.requested_by,
        )
        with self._lock:
            self._items[request.approval_id] = request
            self._save()
        return request

    def decide(self, approval_id: str, decision: ApprovalDecision) -> ApprovalRequest:
        with self._lock:
            current = self._items.get(approval_id)
            if current is None:
                raise KeyError(approval_id)
            if current.status is not ApprovalStatus.PENDING:
                raise ValueError(f"approval {approval_id} is already {current.status.value}")
            updated = current.model_copy(
                update={
                    "status": (
                        ApprovalStatus.APPROVED
                        if decision.approved
                        else ApprovalStatus.REJECTED
                    ),
                    "decided_by": decision.decided_by,
                    "decision_reason": decision.reason,
                    "decided_at": utc_now(),
                }
            )
            self._items[approval_id] = updated
            self._save()
            return updated

    def get(self, approval_id: str) -> ApprovalRequest:
        item = self._items.get(approval_id)
        if item is None:
            raise KeyError(approval_id)
        return item

    def list(self) -> list[ApprovalRequest]:
        return sorted(self._items.values(), key=lambda row: row.created_at, reverse=True)

