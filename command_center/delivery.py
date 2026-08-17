from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import SimulationRequest
from .engine_adapter import MaspAdapter
from .provider import DeepSeekProvider
from .settings import Settings


PROHIBITED_PACKAGE_PATHS = {
    ".env",
    "docs/LLM_DISPATCH_COPILOT_DESIGN.md",
}
PROHIBITED_PACKAGE_PREFIXES = (
    ".git/",
    "data/evaluations/",
    "data/dataset-exports/",
    "runs/",
    "submission/",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_delivery_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "delivery-manifest.json"
    if not manifest_path.is_file():
        return {
            "ok": False,
            "error": "缺少 delivery-manifest.json",
            "missing": [],
            "mismatched": [],
            "prohibited": [],
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schemaVersion") != 1:
            raise ValueError("不支持的清单版本")
        files = dict(manifest["files"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return {
            "ok": False,
            "error": f"交付清单不可读取：{error}",
            "missing": [],
            "mismatched": [],
            "prohibited": [],
        }

    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected_hash in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            mismatched.append(str(relative))
            continue
        candidate = (root / relative_path).resolve()
        if not candidate.is_file():
            missing.append(str(relative))
        elif _sha256(candidate) != expected_hash:
            mismatched.append(str(relative))

    prohibited = sorted(
        str(relative)
        for relative in files
        if str(relative).replace("\\", "/") in PROHIBITED_PACKAGE_PATHS
        or any(
            str(relative).replace("\\", "/").startswith(prefix)
            for prefix in PROHIBITED_PACKAGE_PREFIXES
        )
    )

    return {
        "ok": not missing and not mismatched and not prohibited,
        "error": None,
        "missing": sorted(missing),
        "mismatched": sorted(mismatched),
        "prohibited": prohibited,
        "fileCount": len(files),
        "sourceCommit": manifest.get("sourceCommit"),
        "engineCommit": manifest.get("engineCommit"),
        "wheelhouseIncluded": bool(manifest.get("wheelhouseIncluded")),
    }


def run_delivery_check(smoke_run: bool = False) -> dict[str, Any]:
    settings = Settings.load()
    root = settings.root
    engine = MaspAdapter(settings)
    engine_status = engine.engine_status()
    provider_status = DeepSeekProvider(settings).status()
    manifest = (
        verify_delivery_manifest(root)
        if (root / "delivery-manifest.json").exists()
        else {"ok": True, "skipped": True}
    )
    frontend_ready = (root / "frontend" / "dist" / "index.html").is_file()
    scenarios = engine.scenarios() if engine_status["allowed"] else []
    checks: dict[str, Any] = {
        "manifest": manifest,
        "engine": {"ok": bool(engine_status["allowed"]), **engine_status},
        "frontend": {"ok": frontend_ready},
        "model": {
            "ok": provider_status["mode"] in {"api", "deterministic-fallback"},
            "mode": provider_status["mode"],
            "model": provider_status["model"],
        },
        "scenarios": {"ok": bool(scenarios), "count": len(scenarios)},
    }

    if smoke_run and engine_status["allowed"]:
        summary = engine.simulate(
            SimulationRequest(
                scenarioId="explicit-single-vehicle",
                label="离线交付自检",
                policy="top_k",
                seed=0,
            )
        )
        checks["smokeSimulation"] = {
            "ok": summary.status == "COMPLETED",
            "runId": summary.run_id,
            "status": summary.status,
        }

    ok = all(bool(check.get("ok")) for check in checks.values())
    return {
        "status": "ok" if ok else "failed",
        "mode": "simulation-only",
        "fieldExecutionEnabled": False,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="验证灵枢离线演示交付包")
    parser.add_argument("--smoke-run", action="store_true", help="运行最小确定性仿真")
    arguments = parser.parse_args()
    report = run_delivery_check(smoke_run=arguments.smoke_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
