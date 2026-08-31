from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
from typing import Any

from command_center.agent_run_manager import AgentRunManager
from command_center.audit import AuditStore
from command_center.contracts import AgentRunCreateRequest
from command_center.engine_adapter import MaspAdapter
from command_center.knowledge import KnowledgeBase
from command_center.orchestrator import DispatchOrchestrator
from command_center.provider import DeepSeekProvider
from command_center.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对照 JSON 与 SQLite Agent run 持久化")
    parser.add_argument("--baseline-ref", default="96406f7")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--output", type=Path, default=Path("results/store-benchmark/latest.json")
    )
    return parser.parse_args()


def _baseline_manager_class(ref: str):
    completed = subprocess.run(
        ["git", "show", f"{ref}:command_center/agent_run_manager.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    module = types.ModuleType("command_center._json_agent_run_manager_baseline")
    module.__package__ = "command_center"
    exec(
        compile(completed.stdout, f"{ref}:agent_run_manager.py", "exec"),
        module.__dict__,
    )
    return module.AgentRunManager


def _manager(manager_class, settings: Settings, path: Path, workers: int):
    provider = DeepSeekProvider(settings)
    engine = MaspAdapter(settings)
    orchestrator = DispatchOrchestrator(
        engine=engine,
        provider=provider,
        knowledge=KnowledgeBase(PROJECT_ROOT / "knowledge"),
        audit=AuditStore(settings.data_dir / "audit.jsonl"),
        runtime_mode="loop",
    )
    return manager_class(
        path,
        orchestrator=orchestrator,
        provider=provider,
        max_workers=workers,
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))
    return ordered[index]


def _run_variant(
    *,
    label: str,
    manager_class,
    settings: Settings,
    path: Path,
    runs: int,
    workers: int,
    timeout: float,
) -> dict[str, Any]:
    manager = _manager(manager_class, settings, path, workers)
    started = perf_counter()

    def create(index: int):
        return manager.create(
            AgentRunCreateRequest(
                message="当前车辆和任务状态怎么样？",
                scenarioId="interactive-multi-fleet",
                conversationId=f"store-benchmark-{label}-{index}",
                timeoutSeconds=max(5, min(300, int(timeout))),
            ),
            idempotency_key=f"store-benchmark-{label}-{index}",
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        created = list(executor.map(create, range(runs)))
    completion_ms: list[float] = []
    pending = {row.run_id for row in created}
    deadline = perf_counter() + timeout
    while pending and perf_counter() < deadline:
        for run_id in list(pending):
            if manager.get(run_id).status in {"COMPLETED", "FAILED"}:
                completion_ms.append((perf_counter() - started) * 1000)
                pending.remove(run_id)
        if pending:
            sleep(0.01)
    records = [manager.get(row.run_id) for row in created]
    elapsed_ms = (perf_counter() - started) * 1000
    manager.shutdown()
    storage_bytes = sum(
        item.stat().st_size
        for item in path.parent.glob(f"{path.stem}*")
        if item.is_file()
    )
    return {
        "label": label,
        "runCount": runs,
        "completed": sum(row.status == "COMPLETED" for row in records),
        "failed": sum(row.status == "FAILED" for row in records),
        "timedOut": len(pending),
        "totalMs": round(elapsed_ms, 3),
        "throughputRunsPerSecond": round(runs / max(elapsed_ms / 1000, 1e-9), 3),
        "meanCompletionMs": round(statistics.fmean(completion_ms), 3),
        "p95CompletionMs": round(_percentile(completion_ms, 0.95), 3),
        "storageBytes": storage_bytes,
    }


def main() -> None:
    args = _arguments()
    if args.runs < 1 or args.workers < 1:
        raise ValueError("runs 和 workers 必须大于 0")
    baseline_class = _baseline_manager_class(args.baseline_ref)
    base_settings = replace(Settings.load(), deepseek_api_key=None)
    with TemporaryDirectory(prefix="masp-agent-store-benchmark-") as directory:
        root = Path(directory)
        baseline_settings = replace(base_settings, root=root / "json")
        sqlite_settings = replace(base_settings, root=root / "sqlite")
        baseline = _run_variant(
            label="json-baseline",
            manager_class=baseline_class,
            settings=baseline_settings,
            path=baseline_settings.data_dir / "agent-runs.json",
            runs=args.runs,
            workers=args.workers,
            timeout=args.timeout,
        )
        candidate = _run_variant(
            label="sqlite-wal",
            manager_class=AgentRunManager,
            settings=sqlite_settings,
            path=sqlite_settings.data_dir / "agent-runs.json",
            runs=args.runs,
            workers=args.workers,
            timeout=args.timeout,
        )
    result = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "baselineRef": args.baseline_ref,
        "configuration": {"runs": args.runs, "workers": args.workers},
        "baseline": baseline,
        "candidate": candidate,
        "speedup": round(baseline["totalMs"] / candidate["totalMs"], 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
