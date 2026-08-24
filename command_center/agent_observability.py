from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any

from .contracts import AgentExecutionTrace, IntentValidation, utc_now


class AgentObservabilityStore:
    """Append-only Agent telemetry without storing prompts or model responses."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def record(
        self,
        *,
        trace_id: str,
        conversation_id: str,
        scenario_id: str,
        trace: AgentExecutionTrace,
        model: str,
        fallback_used: bool,
        validation: IntentValidation | None,
    ) -> None:
        event = {
            "traceId": trace_id,
            "conversationId": conversation_id,
            "scenarioId": scenario_id,
            "createdAt": utc_now().isoformat(),
            "status": trace.status,
            "strategy": trace.strategy,
            "plannerModel": trace.planner_model,
            "intentModel": model,
            "fallbackUsed": fallback_used,
            "durationMs": trace.duration_ms,
            "stepCount": len(trace.steps),
            "toolNames": [
                step.tool_name for step in trace.steps if step.tool_name is not None
            ],
            "validationPassed": validation.valid if validation else None,
            "riskLevel": validation.risk_level.value if validation else None,
        }
        encoded = json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")

    def _events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def summary(self, *, recent_limit: int = 20) -> dict[str, Any]:
        events = self._events()
        total = len(events)
        durations = sorted(float(row.get("durationMs", 0)) for row in events)
        tool_counts = Counter(
            str(name)
            for row in events
            for name in (row.get("toolNames") or [])
        )

        def rate(predicate) -> float:
            return round(sum(1 for row in events if predicate(row)) / total, 4) if total else 0.0

        p95_index = max(0, math.ceil(len(durations) * 0.95) - 1)
        return {
            "generatedAt": utc_now().isoformat(),
            "requestCount": total,
            "completedCount": sum(row.get("status") == "COMPLETED" for row in events),
            "clarificationCount": sum(
                row.get("status") == "CLARIFICATION_REQUIRED" for row in events
            ),
            "taskCompletionRate": rate(lambda row: row.get("status") == "COMPLETED"),
            "modelToolPlanningRate": rate(
                lambda row: row.get("strategy") == "MODEL_TOOL_CALLING"
            ),
            "fallbackRate": rate(lambda row: bool(row.get("fallbackUsed"))),
            "safetyBlockRate": rate(
                lambda row: row.get("validationPassed") is False
            ),
            "averageDurationMs": round(sum(durations) / len(durations), 3)
            if durations
            else 0.0,
            "p95DurationMs": round(durations[p95_index], 3) if durations else 0.0,
            "averageStepCount": round(
                sum(int(row.get("stepCount", 0)) for row in events) / total, 3
            )
            if total
            else 0.0,
            "toolCallCounts": dict(sorted(tool_counts.items())),
            "recent": list(reversed(events[-max(1, recent_limit) :])),
        }
