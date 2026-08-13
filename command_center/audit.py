from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from .contracts import AuditEvent


class AuditStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(
        self,
        *,
        trace_id: str,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
    ) -> AuditEvent:
        event = AuditEvent(
            traceId=trace_id,
            eventType=event_type,
            actor=actor,
            payload=payload,
        )
        encoded = json.dumps(
            event.model_dump(by_alias=True, mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
        return event

    def latest(self, limit: int = 100) -> list[AuditEvent]:
        if not self.path.exists():
            return []
        with self._lock:
            rows = self.path.read_text(encoding="utf-8").splitlines()
        return [AuditEvent.model_validate_json(row) for row in rows[-max(1, limit) :]][::-1]

