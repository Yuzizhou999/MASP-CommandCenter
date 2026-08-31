from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from command_center.clarifications import ClarificationResolver, ClarificationStore
from command_center.contracts import EvidenceItem
from command_center.engine_adapter import MaspAdapter
from command_center.llm_provider import create_llm_provider
from command_center.provider import DeepSeekProvider
from command_center.settings import Settings
from training.intent_dataset import read_jsonl


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测基座、微调或降级意图模型")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--provider", choices=("deterministic", "deepseek", "local"), default="local"
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _expected_fields(payload: dict[str, Any]) -> dict[str, Any]:
    result = {"intentType": payload["intentType"]}
    if payload.get("task") is not None:
        result["task"] = {
            key: payload["task"][key]
            for key in (
                "pickupNodeId",
                "dropoffNodeId",
                "requiredRobotGroup",
                "payloadType",
            )
        }
    if payload.get("resourceBlock") is not None:
        result["resourceBlock"] = {
            key: payload["resourceBlock"][key]
            for key in ("resourceIds", "startMs", "endMs")
        }
    return result


def _observed_fields(intent) -> dict[str, Any] | None:
    if intent is None:
        return None
    payload = intent.model_dump(by_alias=True, mode="json")
    return _expected_fields(payload)


def main() -> None:
    args = _arguments()
    root = args.dataset_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    settings = Settings.load()
    if args.provider == "deterministic":
        provider = DeepSeekProvider(replace(settings, deepseek_api_key=None))
    else:
        selected = replace(
            settings,
            llm_provider=args.provider,
            local_llm_enabled=args.provider == "local",
            local_llm_base_url=args.base_url or settings.local_llm_base_url,
            local_llm_model=args.model or settings.local_llm_model,
        )
        provider = create_llm_provider(selected)
    engine = MaspAdapter(settings)
    test_rows = read_jsonl(root / manifest["files"]["test"]["path"])
    cases: list[dict[str, Any]] = []
    started_all = perf_counter()
    for row in test_rows:
        request = json.loads(row["messages"][1]["content"])
        expected_payload = json.loads(row["messages"][2]["content"])
        authoritative = request.get("authoritativeParameters") or {}
        evidence = [
            EvidenceItem.model_validate(item) for item in request["retrievedContext"]
        ]
        started = perf_counter()
        result = provider.parse_intent(
            request["request"],
            world_revision=int(request["worldRevision"]),
            requested_by=str(request["requestedBy"]),
            resolved_task=authoritative.get("task"),
            resolved_resource_block=authoritative.get("resourceBlock"),
            context_evidence=evidence,
        )
        latency_ms = (perf_counter() - started) * 1000
        expected = _expected_fields(expected_payload)
        observed = _observed_fields(result.intent)
        schema_valid = result.intent is not None
        exact_match = observed == expected
        masp_valid = False
        if result.intent is not None:
            masp_valid = engine.validate_intent(
                result.intent, row["metadata"]["scenarioId"]
            ).valid
        cases.append(
            {
                "exampleId": row["metadata"]["exampleId"],
                "category": row["metadata"]["category"],
                "providerOutputUsed": not result.fallback_used,
                "schemaValid": schema_valid,
                "exactMatch": exact_match,
                "maspValid": masp_valid,
                "latencyMs": round(latency_ms, 3),
                "expected": expected,
                "observed": observed,
            }
        )

    scenario_id = manifest["scenarioIds"][0]
    world_revision = engine.world_revision(scenario_id)
    safety_cases = []
    for row in read_jsonl(root / manifest["files"]["safetyHoldout"]["path"]):
        result = provider.parse_intent(
            row["message"],
            world_revision=world_revision,
            requested_by="finetune-evaluator",
        )
        observed_type = result.intent.intent_type.value if result.intent else None
        allowed = set(row["expected"]["allowedIntentTypes"])
        passed = (
            result.fallback_used or observed_type in allowed or result.intent is None
        )
        safety_cases.append(
            {
                "message": row["message"],
                "passed": passed,
                "fallbackUsed": result.fallback_used,
                "observedIntentType": observed_type,
            }
        )

    with TemporaryDirectory(prefix="masp-clarification-eval-") as temp_dir:
        resolver = ClarificationResolver(
            ClarificationStore(Path(temp_dir) / "clarifications.json"), engine
        )
        clarification_cases = []
        for index, row in enumerate(
            read_jsonl(root / manifest["files"]["clarificationHoldout"]["path"])
        ):
            resolved = resolver.resolve(row["message"], f"eval-{index}")
            clarification_cases.append(
                {
                    "message": row["message"],
                    "passed": resolved.clarification is not None,
                    "observedState": (
                        "CLARIFICATION_REQUIRED"
                        if resolved.clarification is not None
                        else "READY"
                    ),
                }
            )

    total = len(cases)
    latencies = sorted(float(row["latencyMs"]) for row in cases)
    p95_index = max(0, min(len(latencies) - 1, int(len(latencies) * 0.95) - 1))
    category_counts = Counter(row["category"] for row in cases)
    report = {
        "schemaVersion": 1,
        "evaluationId": f"intent-eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "createdAt": datetime.now(UTC).isoformat(),
        "datasetId": manifest["datasetId"],
        "provider": provider.status(),
        "counts": {"test": total, "categories": dict(category_counts)},
        "metrics": {
            "providerOutputRate": round(
                sum(row["providerOutputUsed"] for row in cases) / max(1, total), 4
            ),
            "schemaValidRate": round(
                sum(row["schemaValid"] for row in cases) / max(1, total), 4
            ),
            "exactMatchRate": round(
                sum(row["exactMatch"] for row in cases) / max(1, total), 4
            ),
            "maspValidRate": round(
                sum(row["maspValid"] for row in cases) / max(1, total), 4
            ),
            "safetyPassRate": round(
                sum(row["passed"] for row in safety_cases) / max(1, len(safety_cases)),
                4,
            ),
            "clarificationPassRate": round(
                sum(row["passed"] for row in clarification_cases)
                / max(1, len(clarification_cases)),
                4,
            ),
            "averageLatencyMs": round(sum(latencies) / max(1, len(latencies)), 3),
            "p95LatencyMs": round(latencies[p95_index] if latencies else 0, 3),
        },
        "cases": cases,
        "safetyCases": safety_cases,
        "clarificationCases": clarification_cases,
        "durationMs": round((perf_counter() - started_all) * 1000, 3),
    }
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output = args.output or (root / f"evaluation-{args.provider}-{stamp}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output), **report["metrics"]}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
