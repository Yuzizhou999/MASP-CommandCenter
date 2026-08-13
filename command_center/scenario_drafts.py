from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .audit import AuditStore
from .contracts import new_id
from .engine_adapter import MaspAdapter


class ScenarioDraftConflict(ValueError):
    pass


class ScenarioDraftStore:
    """Filesystem-backed editable package lifecycle for simulation scenarios."""

    def __init__(self, root: Path, engine: MaspAdapter, audit: AuditStore) -> None:
        self.root = root
        self.engine = engine
        self.audit = audit
        self.drafts_dir = root / "scenario-drafts"
        self.builds_dir = root / "scenario-builds"
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        self.builds_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write(path: Path, document: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _path(self, package_id: str) -> Path:
        if not package_id or "/" in package_id or "\\" in package_id or package_id in {".", ".."}:
            raise ValueError("packageId contains an invalid path")
        return self.drafts_dir / f"{package_id}.json"

    @staticmethod
    def _envelope(document: dict[str, Any], *, revision: int, status: str) -> dict[str, Any]:
        value = deepcopy(document)
        value["status"] = status
        value.setdefault("metadata", {})
        value["metadata"]["revision"] = revision
        return value

    def _record(self, document: dict[str, Any]) -> dict[str, Any]:
        metadata = document.get("metadata", {})
        return {
            "packageId": document["packageId"],
            "version": document["version"],
            "status": document["status"],
            "revision": int(metadata.get("revision", 0)),
            "sceneId": document["warehouseScene"]["sceneId"],
            "streamId": document["taskStream"]["streamId"],
            "taskCount": len(document["taskStream"]["tasks"]),
            "updatedAt": metadata.get("updatedAt"),
            "build": metadata.get("build"),
        }

    def create(self, document: dict[str, Any], actor: str) -> dict[str, Any]:
        value = deepcopy(document)
        value.setdefault("packageId", new_id("scenario"))
        value.setdefault("version", "0.1.0")
        value["status"] = "draft"
        value.setdefault("metadata", {})
        value["metadata"]["createdBy"] = actor
        value["metadata"]["updatedAt"] = datetime.now(timezone.utc).isoformat()
        value["metadata"]["revision"] = 1
        self.engine.validate_scenario_package(value)
        path = self._path(str(value["packageId"]))
        with self._lock:
            if path.exists():
                raise ScenarioDraftConflict(f"场景包 {value['packageId']} 已存在。")
            self._write(path, value)
        self._audit("SCENARIO_DRAFT_CREATED", actor, value)
        return self._record(value)

    def create_from_runtime(
        self,
        scenario_id: str,
        package_id: str,
        version: str,
        actor: str,
    ) -> dict[str, Any]:
        document = self.engine.scenario_package_from_runtime(
            scenario_id,
            package_id=package_id,
            version=version,
            created_by=actor,
        )
        return self.create(document, actor)

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.drafts_dir.glob("*.json")):
            try:
                rows.append(self._record(self._read(path)))
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        return rows

    def get(self, package_id: str) -> dict[str, Any]:
        path = self._path(package_id)
        if not path.exists():
            raise KeyError(package_id)
        return self._read(path)

    def update(
        self, package_id: str, document: dict[str, Any], expected_revision: int, actor: str
    ) -> dict[str, Any]:
        with self._lock:
            current = self.get(package_id)
            actual = int(current.get("metadata", {}).get("revision", 0))
            if actual != expected_revision:
                raise ScenarioDraftConflict(
                    f"场景包版本已变化，当前revision={actual}，请求revision={expected_revision}。"
                )
            if current.get("status") != "draft":
                raise ScenarioDraftConflict("已发布场景包不可直接修改，请创建新版本。")
            value = deepcopy(document)
            value["packageId"] = package_id
            value["status"] = "draft"
            value.setdefault("metadata", {})
            value["metadata"]["createdBy"] = current.get("metadata", {}).get("createdBy", actor)
            value["metadata"]["updatedAt"] = datetime.now(timezone.utc).isoformat()
            value["metadata"]["revision"] = actual + 1
            self.engine.validate_scenario_package(value)
            self._write(self._path(package_id), value)
        self._audit("SCENARIO_DRAFT_UPDATED", actor, value)
        return self._record(value)

    def validate(self, package_id: str, actor: str) -> dict[str, Any]:
        value = self.get(package_id)
        report = self.engine.validate_scenario_package(value)
        self._audit("SCENARIO_DRAFT_VALIDATED", actor, {"packageId": package_id, "report": report})
        return report

    def generate_tasks(
        self, package_id: str, generation: dict[str, Any], expected_revision: int, actor: str
    ) -> dict[str, Any]:
        with self._lock:
            current = self.get(package_id)
            actual = int(current.get("metadata", {}).get("revision", 0))
            if actual != expected_revision:
                raise ScenarioDraftConflict(f"revision不匹配，当前为{actual}。")
            if current.get("status") != "draft":
                raise ScenarioDraftConflict("已发布场景包不可修改任务流。")
            stream = self.engine.generate_scenario_tasks(current, generation)
            value = deepcopy(current)
            value["taskStream"] = stream
            value["metadata"]["revision"] = actual + 1
            value["metadata"]["updatedAt"] = datetime.now(timezone.utc).isoformat()
            self.engine.validate_scenario_package(value)
            self._write(self._path(package_id), value)
        self._audit("SCENARIO_TASK_STREAM_GENERATED", actor, {"packageId": package_id, "generation": generation, "taskCount": len(stream["tasks"])})
        return self._record(value)

    def compile(self, package_id: str, actor: str) -> dict[str, Any]:
        value = self.get(package_id)
        report = self.engine.validate_scenario_package(value)
        if not report["valid"]:
            return {"packageId": package_id, "compiled": False, "validation": report}
        revision = int(value.get("metadata", {}).get("revision", 0))
        output = self.builds_dir / package_id / str(value["version"]) / f"draft-revision-{revision}"
        result = self.engine.compile_scenario_package(value, output)
        value["metadata"]["build"] = {"directory": str(output), "manifest": result["manifest"]}
        self._write(self._path(package_id), value)
        self._audit("SCENARIO_DRAFT_COMPILED", actor, {"packageId": package_id, "manifest": result["manifest"]})
        return {"packageId": package_id, "compiled": True, **result}

    def publish(self, package_id: str, actor: str) -> dict[str, Any]:
        value = self.get(package_id)
        if value.get("status") != "draft":
            raise ScenarioDraftConflict("只有draft场景包可以发布。")
        report = self.engine.validate_scenario_package(value)
        if not report["valid"]:
            raise ValueError("场景包未通过确定性校验，不能发布。")
        output = self.builds_dir / package_id / str(value["version"]) / "published"
        value["status"] = "published"
        value["metadata"]["publishedBy"] = actor
        value["metadata"]["publishedAt"] = datetime.now(timezone.utc).isoformat()
        result = self.engine.compile_scenario_package(value, output)
        value["metadata"]["build"] = {
            "directory": str(output),
            "immutable": True,
            "manifest": result["manifest"],
        }
        self._write(self._path(package_id), value)
        self._audit("SCENARIO_DRAFT_PUBLISHED", actor, {"packageId": package_id, "version": value["version"], "buildDirectory": str(output)})
        return self._record(value)

    def _audit(self, event_type: str, actor: str, payload: dict[str, Any]) -> None:
        self.audit.append(trace_id=new_id("trace"), event_type=event_type, actor=actor, payload=payload)
