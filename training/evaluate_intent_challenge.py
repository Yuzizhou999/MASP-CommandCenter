from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import httpx

from command_center.agent_protocol import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    action_messages,
    agent_action_response_schema,
)
from command_center.agent_tools import CurrentWorldSnapshotInput, SearchSopInput
from command_center.clarifications import ClarificationResolver, ClarificationStore
from command_center.contracts import DispatchIntent, EvidenceItem, IntentType
from command_center.engine_adapter import MaspAdapter
from command_center.model_safety import model_request_violation
from command_center.provider import intent_training_messages
from command_center.settings import Settings


MODEL_INTENT_LABELS = tuple(
    row.value
    for row in (
        IntentType.QUERY_STATUS,
        IntentType.EXPLAIN_DECISION,
        IntentType.CREATE_TASK,
        IntentType.BLOCK_RESOURCE,
        IntentType.GENERATE_REPORT,
    )
)
TASK_FIELDS = (
    "pickupNodeId",
    "dropoffNodeId",
    "requiredRobotGroup",
    "payloadType",
)
BLOCK_FIELDS = ("resourceIds", "startMs", "endMs")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测模型未经修正的原始调度意图")
    parser.add_argument("suite", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--protocol",
        choices=("dispatch_intent", "agent_action"),
        default="dispatch_intent",
    )
    return parser.parse_args()


def load_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if suite.get("schemaVersion") != 1:
        raise ValueError("挑战集 schemaVersion 必须为 1")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("挑战集必须包含非空 cases")
    ids = [str(row.get("caseId") or "") for row in cases]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("caseId 必须存在且唯一")
    for row in cases:
        expected = row.get("expected") or {}
        if expected.get("intentType") not in MODEL_INTENT_LABELS:
            raise ValueError(f"{row['caseId']} 的 intentType 无效")
    qualification = suite.get("qualification") or {}
    if not qualification.get("modelThresholds"):
        raise ValueError("挑战集必须配置 modelThresholds")
    if not qualification.get("systemThresholds"):
        raise ValueError("挑战集必须配置 systemThresholds")
    return suite


def expected_fields(case: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    result: dict[str, Any] = {"intentType": expected["intentType"]}
    authoritative = case.get("authoritativeParameters") or {}
    if authoritative.get("task") is not None:
        result["task"] = {
            key: authoritative["task"][key] for key in TASK_FIELDS
        }
    if authoritative.get("resourceBlock") is not None:
        result["resourceBlock"] = {
            key: authoritative["resourceBlock"][key] for key in BLOCK_FIELDS
        }
    return result


def observed_fields(intent: DispatchIntent | None) -> dict[str, Any] | None:
    if intent is None:
        return None
    payload = intent.model_dump(by_alias=True, mode="json")
    result: dict[str, Any] = {"intentType": payload["intentType"]}
    if payload.get("task") is not None:
        result["task"] = {key: payload["task"].get(key) for key in TASK_FIELDS}
    if payload.get("resourceBlock") is not None:
        result["resourceBlock"] = {
            key: payload["resourceBlock"].get(key) for key in BLOCK_FIELDS
        }
    return result


def parse_raw_intent(
    content: str, *, world_revision: int, requested_by: str
) -> tuple[bool, DispatchIntent | None, str | None]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        return False, None, str(error)
    if not isinstance(payload, dict):
        return True, None, "模型输出 JSON 必须是对象"
    payload["basedOnWorldRevision"] = world_revision
    payload["requestedBy"] = requested_by
    payload["environment"] = "simulation"
    try:
        return True, DispatchIntent.model_validate(payload), None
    except ValueError as error:
        return True, None, str(error)


def parse_raw_agent_intent(
    content: str, *, world_revision: int, requested_by: str
) -> tuple[bool, DispatchIntent | None, str | None]:
    try:
        action = AgentAction.from_content(content)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        try:
            json.loads(content)
            json_valid = True
        except json.JSONDecodeError:
            json_valid = False
        return json_valid, None, str(error)
    if action.action is not AgentActionType.PROPOSE_INTENT or action.intent is None:
        return True, None, f"expected PROPOSE_INTENT, got {action.action.value}"
    return parse_raw_intent(
        json.dumps(action.intent, ensure_ascii=False),
        world_revision=world_revision,
        requested_by=requested_by,
    )


def _agent_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_world_snapshot",
                "parameters": CurrentWorldSnapshotInput.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_sop",
                "parameters": SearchSopInput.model_json_schema(),
            },
        },
    ]


def agent_intent_challenge_messages(
    case: dict[str, Any], *, world_revision: int
) -> list[dict[str, str]]:
    request = str(case["message"])
    expected_intent = str((case.get("expected") or {}).get("intentType") or "")
    requires_sop = expected_intent in {"EXPLAIN_DECISION", "BLOCK_RESOURCE"}
    observations = [
        AgentObservation(
            sequence=1,
            kind="INITIAL",
            code="request.received",
            summary="意图保持评测已收到请求",
            data={"hasMemory": False},
        ),
        AgentObservation(
            sequence=2,
            kind="TOOL_RESULT",
            code="tool.ok",
            summary=f"读取权威世界版本 {world_revision}",
            data={"value": {"worldRevision": world_revision, "counts": {}}},
            toolName="get_world_snapshot",
        ),
    ]
    action_history = [
        {"action": "CALL_TOOL", "tool": "get_world_snapshot", "arguments": {}}
    ]
    if requires_sop:
        action_history.append(
            {
                "action": "CALL_TOOL",
                "tool": "search_sop",
                "arguments": {"query": request, "limit": 2},
            }
        )
        observations.append(
            AgentObservation(
                sequence=3,
                kind="TOOL_RESULT",
                code="tool.ok",
                summary="已检索适用的调度与安全 SOP",
                data={
                    "value": {
                        "items": [
                            {
                                "title": "调度安全制度",
                                "detail": "当前请求所需的制度依据已就绪。",
                            }
                        ]
                    }
                },
                toolName="search_sop",
            )
        )
    return action_messages(
        request=request,
        observations=observations,
        tool_definitions=_agent_tool_definitions(),
        authoritative_parameters=dict(case.get("authoritativeParameters") or {}),
        action_history=action_history,
    )


def classification_metrics(
    expected: list[str], observed: list[str | None]
) -> dict[str, Any]:
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in MODEL_INTENT_LABELS:
        true_positive = sum(
            expected_value == label and observed_value == label
            for expected_value, observed_value in zip(expected, observed)
        )
        false_positive = sum(
            expected_value != label and observed_value == label
            for expected_value, observed_value in zip(expected, observed)
        )
        false_negative = sum(
            expected_value == label and observed_value != label
            for expected_value, observed_value in zip(expected, observed)
        )
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0
        )
        f1_values.append(f1)
        per_label[label] = {
            "support": sum(value == label for value in expected),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return {
        "accuracy": round(
            sum(left == right for left, right in zip(expected, observed))
            / max(1, len(expected)),
            4,
        ),
        "macroF1": round(sum(f1_values) / len(f1_values), 4),
        "perIntent": per_label,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))
    return ordered[index]


def _threshold_results(
    metrics: dict[str, float], thresholds: dict[str, float]
) -> dict[str, bool]:
    results = {
        name: metrics.get(name, 0) >= float(value)
        for name, value in thresholds.items()
        if name != "p95LatencyMs"
    }
    if "p95LatencyMs" in thresholds:
        results["p95LatencyMs"] = metrics.get("p95LatencyMs", float("inf")) <= float(
            thresholds["p95LatencyMs"]
        )
    return results


def _model_call(
    *,
    base_url: str,
    model: str,
    timeout: float,
    messages: list[dict[str, str]],
    response_schema: dict[str, Any],
) -> tuple[str, dict[str, int]]:
    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": "Bearer local"},
        json={
            "model": model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "intent_challenge",
                    "strict": True,
                    "schema": response_schema,
                },
            },
            "messages": messages,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["choices"][0]["message"]["content"]), dict(
        payload.get("usage") or {}
    )


def evaluate(
    suite: dict[str, Any],
    *,
    base_url: str,
    model: str,
    timeout: float,
    protocol: str = "dispatch_intent",
) -> dict[str, Any]:
    settings = Settings.load()
    engine = MaspAdapter(settings)
    scenario_id = str(suite["scenarioId"])
    world_revision = engine.world_revision(scenario_id)
    snapshot = engine.world_snapshot(scenario_id)
    counts = snapshot["counts"]
    evidence = [
        EvidenceItem(
            source=f"MASP:{scenario_id}",
            title="挑战集世界快照",
            detail=(
                f"revision {world_revision}，{counts['vehicles']} 辆车，"
                f"{counts['tasks']} 个任务，{counts['conflictPairs']} 对冲突资源。"
            ),
        )
    ]
    cases: list[dict[str, Any]] = []
    latencies: list[float] = []
    started_all = perf_counter()
    for case in suite["cases"]:
        authoritative = case.get("authoritativeParameters") or {}
        messages = (
            agent_intent_challenge_messages(case, world_revision=world_revision)
            if protocol == "agent_action"
            else intent_training_messages(
                str(case["message"]),
                world_revision=world_revision,
                requested_by="intent-challenge-evaluator",
                resolved_task=authoritative.get("task"),
                resolved_resource_block=authoritative.get("resourceBlock"),
                context_evidence=evidence,
            )
        )
        started = perf_counter()
        content = ""
        usage: dict[str, int] = {}
        request_error = None
        try:
            content, usage = _model_call(
                base_url=base_url,
                model=model,
                timeout=timeout,
                messages=messages,
                response_schema=(
                    agent_action_response_schema()
                    if protocol == "agent_action"
                    else DispatchIntent.model_json_schema(by_alias=True)
                ),
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            request_error = str(error)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        json_valid, intent, parse_error = (
            (
                parse_raw_agent_intent
                if protocol == "agent_action"
                else parse_raw_intent
            )(
                content,
                world_revision=world_revision,
                requested_by="intent-challenge-evaluator",
            )
            if request_error is None
            else (False, None, request_error)
        )
        expected = expected_fields(case)
        observed = observed_fields(intent)
        masp_valid = (
            engine.validate_intent(intent, scenario_id).valid
            if intent is not None
            else False
        )
        cases.append(
            {
                "caseId": case["caseId"],
                "category": case["expected"]["intentType"],
                "message": case["message"],
                "requestSucceeded": request_error is None,
                "jsonValid": json_valid,
                "schemaValid": intent is not None,
                "intentCorrect": (
                    intent is not None
                    and intent.intent_type.value == expected["intentType"]
                ),
                "exactMatch": observed == expected,
                "maspValid": masp_valid,
                "latencyMs": round(latency_ms, 3),
                "usage": usage,
                "expected": expected,
                "observed": observed,
                "error": parse_error,
                "rawOutput": content,
            }
        )

    safety_cases: list[dict[str, Any]] = []
    for case in suite.get("safetyCases", []):
        messages = (
            agent_intent_challenge_messages(case, world_revision=world_revision)
            if protocol == "agent_action"
            else intent_training_messages(
                str(case["message"]),
                world_revision=world_revision,
                requested_by="intent-challenge-evaluator",
                context_evidence=evidence,
            )
        )
        started = perf_counter()
        content = ""
        request_error = None
        try:
            content, _ = _model_call(
                base_url=base_url,
                model=model,
                timeout=timeout,
                messages=messages,
                response_schema=(
                    agent_action_response_schema()
                    if protocol == "agent_action"
                    else DispatchIntent.model_json_schema(by_alias=True)
                ),
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            request_error = str(error)
        latency_ms = (perf_counter() - started) * 1000
        json_valid, intent, parse_error = (
            (
                parse_raw_agent_intent
                if protocol == "agent_action"
                else parse_raw_intent
            )(
                content,
                world_revision=world_revision,
                requested_by="intent-challenge-evaluator",
            )
            if request_error is None
            else (False, None, request_error)
        )
        observed_type = intent.intent_type.value if intent else None
        allowed = set(case["expected"]["allowedIntentTypes"])
        safety_cases.append(
            {
                "caseId": case["caseId"],
                "message": case["message"],
                "rawSafe": observed_type in allowed,
                "systemGateBlocked": model_request_violation(case["message"])
                is not None,
                "jsonValid": json_valid,
                "observedIntentType": observed_type,
                "latencyMs": round(latency_ms, 3),
                "error": parse_error,
                "rawOutput": content,
            }
        )

    with TemporaryDirectory(prefix="masp-intent-challenge-") as temp_dir:
        resolver = ClarificationResolver(
            ClarificationStore(Path(temp_dir) / "clarifications.json"), engine
        )
        clarification_cases = []
        for index, case in enumerate(suite.get("clarificationCases", [])):
            result = resolver.resolve(case["message"], f"challenge-{index}")
            observed_state = (
                "CLARIFICATION_REQUIRED"
                if result.clarification is not None
                else "READY"
            )
            clarification_cases.append(
                {
                    "caseId": case["caseId"],
                    "message": case["message"],
                    "expectedState": case["expected"]["state"],
                    "observedState": observed_state,
                    "passed": observed_state == case["expected"]["state"],
                }
            )

    expected_labels = [row["category"] for row in cases]
    observed_labels = [
        row["observed"]["intentType"] if row["observed"] else None
        for row in cases
    ]
    classification = classification_metrics(expected_labels, observed_labels)
    slot_cases = [
        row
        for row in cases
        if row["category"] in {"CREATE_TASK", "BLOCK_RESOURCE"}
    ]
    total = len(cases)
    metrics = {
        "requestSuccessRate": round(
            sum(row["requestSucceeded"] for row in cases) / max(1, total), 4
        ),
        "rawJsonValidRate": round(
            sum(row["jsonValid"] for row in cases) / max(1, total), 4
        ),
        "rawSchemaValidRate": round(
            sum(row["schemaValid"] for row in cases) / max(1, total), 4
        ),
        "rawIntentAccuracy": classification["accuracy"],
        "rawIntentMacroF1": classification["macroF1"],
        "rawExactMatchRate": round(
            sum(row["exactMatch"] for row in cases) / max(1, total), 4
        ),
        "rawSlotExactMatchRate": round(
            sum(row["exactMatch"] for row in slot_cases)
            / max(1, len(slot_cases)),
            4,
        ),
        "rawMaspValidRate": round(
            sum(row["maspValid"] for row in cases) / max(1, total), 4
        ),
        "rawSafetyPassRate": round(
            sum(row["rawSafe"] for row in safety_cases)
            / max(1, len(safety_cases)),
            4,
        ),
        "systemSafetyGateRecall": round(
            sum(row["systemGateBlocked"] for row in safety_cases)
            / max(1, len(safety_cases)),
            4,
        ),
        "clarificationAccuracy": round(
            sum(row["passed"] for row in clarification_cases)
            / max(1, len(clarification_cases)),
            4,
        ),
        "averageLatencyMs": round(sum(latencies) / max(1, len(latencies)), 3),
        "p95LatencyMs": round(_percentile(latencies, 0.95), 3),
    }
    qualification_config = suite["qualification"]
    model_thresholds = qualification_config["modelThresholds"]
    system_thresholds = qualification_config["systemThresholds"]
    model_threshold_results = _threshold_results(metrics, model_thresholds)
    system_threshold_results = _threshold_results(metrics, system_thresholds)
    model_passed = bool(model_threshold_results) and all(
        model_threshold_results.values()
    )
    system_passed = bool(system_threshold_results) and all(
        system_threshold_results.values()
    )
    return {
        "schemaVersion": 1,
        "evaluationId": f"intent-challenge-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "suiteId": suite["suiteId"],
        "model": model,
        "protocol": protocol,
        "baseUrl": base_url,
        "scenarioId": scenario_id,
        "counts": {
            "normal": total,
            "byIntent": dict(Counter(expected_labels)),
            "safety": len(safety_cases),
            "clarification": len(clarification_cases),
        },
        "metrics": metrics,
        "classification": classification,
        "qualification": {
            "model": {
                "thresholds": model_thresholds,
                "thresholdResults": model_threshold_results,
                "passed": model_passed,
            },
            "system": {
                "thresholds": system_thresholds,
                "thresholdResults": system_threshold_results,
                "passed": system_passed,
            },
            "passed": model_passed and system_passed,
        },
        "diagnostics": {
            "rawSafetyPassRate": metrics["rawSafetyPassRate"],
            "note": (
                "模型原始安全率仅用于诊断；部署安全由模型调用前的确定性安全门保证。"
            ),
        },
        "passed": model_passed and system_passed,
        "cases": cases,
        "safetyCases": safety_cases,
        "clarificationCases": clarification_cases,
        "durationMs": round((perf_counter() - started_all) * 1000, 3),
    }


def main() -> None:
    args = _arguments()
    suite = load_suite(args.suite.resolve())
    report = evaluate(
        suite,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        protocol=args.protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": report["passed"],
                **report["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
