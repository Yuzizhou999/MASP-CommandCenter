from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .approvals import ApprovalStore
from .audit import AuditStore
from .contracts import DatasetExportRequest, new_id
from .engine_adapter import MaspAdapter
from .incidents import IncidentStore
from .intent_store import IntentStore

ACTOR_KEYS = {
    "actor",
    "requestedBy",
    "decidedBy",
    "createdBy",
    "publishedBy",
}
PATH_KEYS = {
    "manifestPath",
    "evidencePath",
    "directory",
    "buildDirectory",
}
FREE_TEXT_KEYS = {
    "reason",
    "decisionReason",
    "query",
    "request",
    "message",
    "detail",
    "fact",
    "explanation",
    "rationale",
    "summary",
}
SENSITIVE_KEY_PATTERN = re.compile(
    r"password|secret|api.?key|access.?token|refresh.?token|email|phone|mobile",
    flags=re.IGNORECASE,
)
PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class DatasetExporter:
    """Create a de-identified, versioned evaluation dataset from local evidence."""

    def __init__(
        self,
        root: Path,
        *,
        engine: MaspAdapter,
        audit: AuditStore,
        approvals: ApprovalStore,
        intents: IntentStore,
        incidents: IncidentStore,
    ) -> None:
        self.root = root / "dataset-exports"
        self.engine = engine
        self.audit = audit
        self.approvals = approvals
        self.intents = intents
        self.incidents = incidents
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(path: Path, document: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _safe_id(export_id: str) -> str:
        if not re.fullmatch(r"dataset-[a-z0-9]+", export_id):
            raise ValueError("invalid exportId")
        return export_id

    @staticmethod
    def _pseudonym(export_id: str, value: str) -> str:
        digest = hashlib.sha256(f"{export_id}:{value}".encode()).hexdigest()
        return f"user-{digest[:12]}"

    def _sanitize(
        self,
        value: Any,
        *,
        export_id: str,
        include_evidence_text: bool,
        key: str | None = None,
    ) -> Any:
        if key in PATH_KEYS:
            return None
        if key in ACTOR_KEYS and isinstance(value, str):
            return self._pseudonym(export_id, value)
        if key in FREE_TEXT_KEYS and not include_evidence_text:
            return "[TEXT_REMOVED]"
        if isinstance(value, dict):
            return {
                child_key: self._sanitize(
                    child_value,
                    export_id=export_id,
                    include_evidence_text=include_evidence_text,
                    key=child_key,
                )
                for child_key, child_value in value.items()
                if child_key not in PATH_KEYS
                and not SENSITIVE_KEY_PATTERN.search(child_key)
            }
        if isinstance(value, list):
            return [
                self._sanitize(
                    item,
                    export_id=export_id,
                    include_evidence_text=include_evidence_text,
                )
                for item in value
            ]
        if isinstance(value, str) and PHONE_PATTERN.search(value):
            return PHONE_PATTERN.sub("[PHONE_REMOVED]", value)
        return value

    @staticmethod
    def _split(source_id: str) -> str:
        bucket = (
            int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) % 100
        )
        if bucket < 70:
            return "train"
        if bucket < 85:
            return "validation"
        return "test"

    def _record(
        self,
        *,
        export_id: str,
        record_type: str,
        source_id: str,
        document: dict[str, Any],
        include_evidence_text: bool,
    ) -> dict[str, Any]:
        record_id = f"{record_type}:{source_id}"
        return {
            "recordId": record_id,
            "recordType": record_type,
            "sourceId": source_id,
            "split": self._split(record_id),
            "data": self._sanitize(
                document,
                export_id=export_id,
                include_evidence_text=include_evidence_text,
            ),
        }

    @staticmethod
    def _sensitive_findings(value: Any, path: str = "$") -> list[str]:
        findings: list[str] = []
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if SENSITIVE_KEY_PATTERN.search(key):
                    findings.append(child_path)
                findings.extend(DatasetExporter._sensitive_findings(child, child_path))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                findings.extend(
                    DatasetExporter._sensitive_findings(child, f"{path}[{index}]")
                )
        elif isinstance(value, str) and PHONE_PATTERN.search(value):
            findings.append(path)
        return findings

    @staticmethod
    def _quality(records: list[dict[str, Any]]) -> dict[str, Any]:
        record_ids = [row.get("recordId") for row in records]
        duplicates = sorted(
            key for key, count in Counter(record_ids).items() if key and count > 1
        )
        missing_required = [
            index
            for index, row in enumerate(records)
            if not all(
                row.get(key) for key in ("recordId", "recordType", "sourceId", "split")
            )
        ]
        sensitive = [
            {"recordId": row["recordId"], "paths": paths}
            for row in records
            if (paths := DatasetExporter._sensitive_findings(row))
        ]
        type_counts = Counter(row["recordType"] for row in records)
        split_counts = Counter(row["split"] for row in records)
        return {
            "passed": not duplicates and not missing_required and not sensitive,
            "recordCount": len(records),
            "recordTypeCounts": dict(sorted(type_counts.items())),
            "splitCounts": dict(sorted(split_counts.items())),
            "duplicateRecordIds": duplicates,
            "missingRequiredRecordIndexes": missing_required,
            "sensitiveFieldFindings": sensitive,
            "checks": {
                "uniqueRecordId": not duplicates,
                "requiredFieldsComplete": not missing_required,
                "sensitiveFieldScan": not sensitive,
                "deterministicSplit": True,
                "freeTextRemovedByDefault": True,
            },
        }

    def _source_documents(
        self, request: DatasetExportRequest
    ) -> Iterable[tuple[str, str, dict[str, Any]]]:
        for run in self.engine.list_runs():
            document = run.model_dump(by_alias=True, mode="json")
            yield "simulation", run.run_id, document
            if run.intent_id:
                yield (
                    "intent-link",
                    f"{run.run_id}:{run.intent_id}",
                    {
                        "runId": run.run_id,
                        "intentId": run.intent_id,
                        "scenarioId": run.scenario_id,
                    },
                )
        for approval in self.approvals.list():
            yield (
                "approval",
                approval.approval_id,
                approval.model_dump(by_alias=True, mode="json"),
            )
        for commit in self.intents.list():
            yield "commit", str(commit["commitId"]), commit
        if request.include_incidents:
            for incident in self.incidents.list():
                yield (
                    "incident",
                    incident.incident_id,
                    incident.model_dump(by_alias=True, mode="json"),
                )
        if request.include_audit:
            for event in self.audit.latest(100000):
                yield (
                    "audit",
                    event.event_id,
                    event.model_dump(by_alias=True, mode="json"),
                )

    def create(self, request: DatasetExportRequest) -> dict[str, Any]:
        export_id = new_id("dataset")
        output_dir = self.root / export_id
        output_dir.mkdir(parents=True, exist_ok=False)
        records = [
            self._record(
                export_id=export_id,
                record_type=record_type,
                source_id=source_id,
                document=document,
                include_evidence_text=request.include_evidence_text,
            )
            for record_type, source_id, document in self._source_documents(request)
        ]
        records.sort(key=lambda row: (row["recordType"], row["sourceId"]))
        quality = self._quality(records)
        dataset_path = output_dir / "dataset.jsonl"
        dataset_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in records
            ),
            encoding="utf-8",
        )
        self._write_json(output_dir / "quality-report.json", quality)
        manifest = {
            "schemaVersion": 1,
            "exportId": export_id,
            "name": request.name,
            "createdAt": _utc_now(),
            "createdBy": self._pseudonym(export_id, request.requested_by),
            "classification": "INTERNAL_EVALUATION",
            "simulationOnly": True,
            "includeEvidenceText": request.include_evidence_text,
            "recordCount": len(records),
            "quality": quality,
            "splits": {
                "method": "sha256(recordId) modulo 100",
                "train": "0-69",
                "validation": "70-84",
                "test": "85-99",
            },
            "usageRestrictions": [
                "仅用于仿真验证、回归测试和内部评测。",
                "不得据此推定可用于外部模型训练。",
                "生产数据导出前必须完成授权、分级和个人信息影响评估。",
            ],
            "artifacts": {
                "dataset": "dataset.jsonl",
                "qualityReport": "quality-report.json",
                "bundle": "dataset-bundle.zip",
            },
        }
        self._write_json(output_dir / "manifest.json", manifest)
        bundle_path = output_dir / "dataset-bundle.zip"
        with zipfile.ZipFile(
            bundle_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as bundle:
            for name in ("manifest.json", "quality-report.json", "dataset.jsonl"):
                bundle.write(output_dir / name, arcname=name)
        self.audit.append(
            trace_id=new_id("trace"),
            event_type="DATASET_EXPORTED",
            actor=request.requested_by,
            payload={
                "exportId": export_id,
                "recordCount": len(records),
                "qualityPassed": quality["passed"],
                "includeEvidenceText": request.include_evidence_text,
            },
        )
        return manifest

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in self.root.glob("dataset-*/manifest.json"):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: row["createdAt"], reverse=True)

    def get(self, export_id: str) -> dict[str, Any]:
        path = self.root / self._safe_id(export_id) / "manifest.json"
        if not path.exists():
            raise KeyError(export_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def bundle_path(self, export_id: str) -> Path:
        path = self.root / self._safe_id(export_id) / "dataset-bundle.zip"
        if not path.exists():
            raise KeyError(export_id)
        return path
