from __future__ import annotations

import json
import os
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
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
            # 审计是安全叙事的证据链，落盘前不能停留在用户态缓冲里。
            stream.flush()
            os.fsync(stream.fileno())
        return event

    def latest(self, limit: int = 100) -> list[AuditEvent]:
        if not self.path.exists():
            return []
        with self._lock:
            rows = self.path.read_text(encoding="utf-8").splitlines()
        return [AuditEvent.model_validate_json(row) for row in rows[-max(1, limit) :]][
            ::-1
        ]
