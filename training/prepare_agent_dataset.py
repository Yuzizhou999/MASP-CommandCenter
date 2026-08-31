from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from command_center.agent_protocol import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    action_messages,
)
from command_center.agent_tools import CurrentWorldSnapshotInput, SearchSopInput
from command_center.knowledge import KnowledgeBase
from command_center.model_safety import (
    screen_retrieved_evidence,
    untrusted_retrieval_record,
)
from training.intent_dataset import (
    file_sha256,
    read_jsonl,
    validate_example,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]

SOFT_CLARIFICATION = [
    "帮我处理一下这个问题",
    "看看现在该怎么办",
    "按刚才说的继续处理",
    "这个情况麻烦安排一下",
    "帮忙做一个调度操作",
    "我想调整一下当前计划",
    "处理一下那台设备的事",
    "请执行前面提到的业务",
    "现在需要改一下安排",
    "帮我把这个目标推进下去",
    "麻烦处理当前异常",
    "做一下刚刚提到的调整",
]

INJECTION_QUARANTINE = [
    "查询当前仓库安全规范",
    "查看车辆任务状态和操作规则",
    "说明当前调度依据",
    "查询任务队列与审批边界",
    "看看当前运行情况和安全要求",
    "获取车队状态与 SOP 摘要",
]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 MASP Agent 多轮轨迹 SFT 数据")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/finetuning/intent-sft-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/finetuning/agent-sft-v2"),
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--teacher-jsonl",
        type=Path,
        default=None,
        help="可选的人工审核 teacher 轨迹；不提供时不阻塞数据生成。",
    )
    return parser.parse_args()


def _tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_world_snapshot",
                "description": "读取当前服务端绑定场景的权威世界快照。",
                "parameters": CurrentWorldSnapshotInput.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_sop",
                "description": "检索仓储调度和安全 SOP。",
                "parameters": SearchSopInput.model_json_schema(),
            },
        },
    ]


def _assistant(action: AgentAction) -> dict[str, str]:
    return {
        "role": "assistant",
        "content": json.dumps(
            action.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def _base_context(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    request = json.loads(row["messages"][1]["content"])
    proposal = json.loads(row["messages"][2]["content"])
    return request, proposal


def _snapshot_observation(request: dict[str, Any], sequence: int) -> AgentObservation:
    return AgentObservation(
        sequence=sequence,
        kind="TOOL_RESULT",
        code="tool.ok",
        summary=f"读取世界版本 {request['worldRevision']}",
        data={
            "value": {
                "worldRevision": int(request["worldRevision"]),
                "counts": {},
            }
        },
        toolName="get_world_snapshot",
    )


def _convert_intent_row(
    row: dict[str, Any], knowledge: KnowledgeBase
) -> dict[str, Any]:
    request, proposal = _base_context(row)
    category = str(row["metadata"]["category"])
    observations = [
        AgentObservation(
            sequence=1,
            kind="INITIAL",
            code="request.received",
            summary="用户目标和权威参数已就绪",
            data={"hasMemory": False, "scenarioId": row["metadata"]["scenarioId"]},
        )
    ]
    history = [
        AgentAction(
            action=AgentActionType.CALL_TOOL,
            tool="get_world_snapshot",
            arguments={},
        ).model_dump(mode="json", exclude_none=True)
    ]
    observations.append(_snapshot_observation(request, 2))
    if category in {"BLOCK_RESOURCE", "EXPLAIN_DECISION"}:
        history.append(
            AgentAction(
                action=AgentActionType.CALL_TOOL,
                tool="search_sop",
                arguments={"query": request["request"], "limit": 2},
            ).model_dump(mode="json", exclude_none=True)
        )
        screened = screen_retrieved_evidence(
            knowledge.search(str(request["request"]), limit=2)
        )
        observations.append(
            AgentObservation(
                sequence=3,
                kind="TOOL_RESULT",
                code="tool.ok",
                summary=f"命中 {len(screened.accepted)} 条安全 SOP",
                data={
                    "value": [
                        untrusted_retrieval_record(item) for item in screened.accepted
                    ],
                    "quarantined": [],
                },
                toolName="search_sop",
                trusted=False,
            )
        )
    messages = action_messages(
        request=str(request["request"]),
        observations=observations,
        tool_definitions=_tool_definitions(),
        authoritative_parameters=dict(request.get("authoritativeParameters") or {}),
        action_history=history,
    )
    messages.append(
        _assistant(
            AgentAction(
                action=AgentActionType.PROPOSE_INTENT,
                intent=proposal,
            )
        )
    )
    metadata = {
        **row["metadata"],
        "datasetType": "agent-trajectory",
        "trajectorySource": "deterministic-policy-conversion",
        "expectedTerminalState": "READY",
        "superviseAssistantIndices": list(range(len(history) + 1)),
    }
    return {"messages": messages, "metadata": metadata}


def _retention_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row["metadata"])
    metadata.update(
        {
            "exampleId": f"intent-retention-{metadata['exampleId']}",
            "category": f"INTENT_RETENTION_{metadata['category']}",
            "source": "frozen-v1-intent-retention",
            "trainingObjective": "intent-v1-no-regression",
        }
    )
    return {
        "messages": [dict(message) for message in row["messages"]],
        "metadata": metadata,
    }


def _clarification_row(text: str, index: int, split: str) -> dict[str, Any]:
    observations = [
        AgentObservation(
            sequence=1,
            kind="INITIAL",
            code="request.received",
            summary="目标没有足够上下文，且不存在硬性实体绑定",
            data={"hasMemory": False},
        )
    ]
    messages = action_messages(
        request=text,
        observations=observations,
        tool_definitions=_tool_definitions(),
        authoritative_parameters={"task": None, "resourceBlock": None},
        action_history=[],
    )
    messages.append(
        _assistant(AgentAction(action=AgentActionType.REQUEST_CLARIFICATION))
    )
    return {
        "messages": messages,
        "metadata": {
            "exampleId": f"agent-soft-clarification-{index:03d}",
            "datasetType": "agent-trajectory",
            "category": "SOFT_CLARIFICATION",
            "source": "human-authored-policy-boundary",
            "split": split,
            "expectedTerminalState": "CLARIFICATION_REQUIRED",
            "superviseAssistantIndices": [0],
        },
    }


def _repair_row(source: dict[str, Any], index: int) -> dict[str, Any]:
    request, correct = _base_context(source)
    invalid = json.loads(json.dumps(correct, ensure_ascii=False))
    invalid.setdefault("task", {})["priorityClass"] = 99
    history = [
        AgentAction(
            action=AgentActionType.CALL_TOOL,
            tool="get_world_snapshot",
            arguments={},
        ).model_dump(mode="json", exclude_none=True),
        AgentAction(
            action=AgentActionType.PROPOSE_INTENT,
            intent=invalid,
        ).model_dump(mode="json", exclude_none=True),
    ]
    observations = [
        AgentObservation(
            sequence=1,
            kind="INITIAL",
            code="request.received",
            summary="任务参数已由服务端绑定",
            data={"hasMemory": False},
        ),
        _snapshot_observation(request, 2),
        AgentObservation(
            sequence=3,
            kind="VALIDATION_ISSUES",
            code="validation.fixable",
            summary="MASP 返回可修复优先级问题",
            data={
                "attempt": 1,
                "issues": [
                    {
                        "code": "intent.task.priority.invalid",
                        "message": "priorityClass 必须在 0 到 9 之间",
                        "severity": "error",
                    }
                ],
            },
        ),
    ]
    messages = action_messages(
        request=str(request["request"]),
        observations=observations,
        tool_definitions=_tool_definitions(),
        authoritative_parameters=dict(request.get("authoritativeParameters") or {}),
        action_history=history,
    )
    messages.append(
        _assistant(AgentAction(action=AgentActionType.PROPOSE_INTENT, intent=correct))
    )
    return {
        "messages": messages,
        "metadata": {
            **source["metadata"],
            "exampleId": f"agent-repair-{index:03d}",
            "datasetType": "agent-trajectory",
            "category": "VALIDATION_REPAIR",
            "source": "human-reviewed-deterministic-verifier",
            "split": "train",
            "expectedTerminalState": "READY",
            "superviseAssistantIndices": [0, 2],
            "fixableIssueCodes": ["intent.task.priority.invalid"],
        },
    }


def _invalid_tool_recovery_row(index: int) -> dict[str, Any]:
    text = f"查询当前车辆任务状态，恢复样本 {index + 1}"
    history = [
        AgentAction(
            action=AgentActionType.CALL_TOOL,
            tool="delete_all",
            arguments={},
        ).model_dump(mode="json", exclude_none=True),
        AgentAction(
            action=AgentActionType.CALL_TOOL,
            tool="get_world_snapshot",
            arguments={},
        ).model_dump(mode="json", exclude_none=True),
    ]
    observations = [
        AgentObservation(
            sequence=1,
            kind="INITIAL",
            code="request.received",
            summary="查询目标已就绪",
            data={"hasMemory": False},
        ),
        AgentObservation(
            sequence=2,
            kind="TOOL_REJECTION",
            code="tool.rejected",
            summary="Agent 工具不在允许列表中：delete_all",
            data={"arguments": {}},
            toolName="delete_all",
        ),
        AgentObservation(
            sequence=3,
            kind="TOOL_RESULT",
            code="tool.ok",
            summary="读取权威世界快照",
            data={"value": {"worldRevision": 42, "counts": {}}},
            toolName="get_world_snapshot",
        ),
    ]
    messages = action_messages(
        request=text,
        observations=observations,
        tool_definitions=_tool_definitions(),
        authoritative_parameters={"task": None, "resourceBlock": None},
        action_history=history,
    )
    messages.append(
        _assistant(
            AgentAction(
                action=AgentActionType.PROPOSE_INTENT,
                intent={
                    "intentType": "QUERY_STATUS",
                    "reason": text,
                    "query": text,
                },
            )
        )
    )
    return {
        "messages": messages,
        "metadata": {
            "exampleId": f"agent-invalid-tool-recovery-{index:03d}",
            "datasetType": "agent-trajectory",
            "category": "TOOL_REJECTION_RECOVERY",
            "source": "human-reviewed-boundary-recovery",
            "split": "train",
            "expectedTerminalState": "READY",
            "superviseAssistantIndices": [1, 2],
        },
    }


def _injection_row(text: str, index: int, split: str) -> dict[str, Any]:
    history = [
        AgentAction(
            action=AgentActionType.CALL_TOOL,
            tool="get_world_snapshot",
            arguments={},
        ).model_dump(mode="json", exclude_none=True),
        AgentAction(
            action=AgentActionType.CALL_TOOL,
            tool="search_sop",
            arguments={"query": text, "limit": 2},
        ).model_dump(mode="json", exclude_none=True),
    ]
    observations = [
        AgentObservation(
            sequence=1,
            kind="INITIAL",
            code="request.received",
            summary="安全查询目标已就绪",
            data={"hasMemory": False},
        ),
        AgentObservation(
            sequence=2,
            kind="TOOL_RESULT",
            code="tool.ok",
            summary="读取权威世界快照",
            data={"value": {"worldRevision": 42, "counts": {}}},
            toolName="get_world_snapshot",
        ),
        AgentObservation(
            sequence=3,
            kind="TOOL_RESULT",
            code="tool.ok",
            summary="可疑检索内容已由服务端隔离",
            data={
                "value": [],
                "quarantined": [
                    {
                        "source": "wiki/poison.md",
                        "violation": "retrieval.ignore-instructions",
                    }
                ],
            },
            toolName="search_sop",
            trusted=False,
        ),
    ]
    messages = action_messages(
        request=text,
        observations=observations,
        tool_definitions=_tool_definitions(),
        authoritative_parameters={"task": None, "resourceBlock": None},
        action_history=history,
    )
    messages.append(
        _assistant(
            AgentAction(
                action=AgentActionType.PROPOSE_INTENT,
                intent={
                    "intentType": "QUERY_STATUS",
                    "reason": text,
                    "query": text,
                },
            )
        )
    )
    return {
        "messages": messages,
        "metadata": {
            "exampleId": f"agent-injection-boundary-{index:03d}",
            "datasetType": "agent-trajectory",
            "category": "RETRIEVAL_INJECTION_BOUNDARY",
            "source": "human-reviewed-security-trajectory",
            "split": split,
            "expectedTerminalState": "READY",
            "superviseAssistantIndices": [0, 1, 2],
        },
    }


def _stable_split(text: str) -> str:
    bucket = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "valid" if bucket == 0 else "test" if bucket == 1 else "train"


def main() -> None:
    args = _arguments()
    random.seed(args.seed)
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_manifest = json.loads(
        (source_dir / "manifest.json").read_text(encoding="utf-8")
    )
    knowledge = KnowledgeBase(ROOT / "knowledge")
    splits: dict[str, list[dict[str, Any]]] = {
        key: [] for key in ("train", "valid", "test")
    }
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        path = source_dir / source_manifest["files"][split]["path"]
        if file_sha256(path) != source_manifest["files"][split]["sha256"]:
            raise ValueError(f"冻结 v1 {split} 文件摘要不匹配")
        rows = read_jsonl(path)
        source_rows[split] = rows
        for row in rows:
            splits[split].append(_retention_row(row))
            splits[split].append(_convert_intent_row(row, knowledge))

    for index, text in enumerate(SOFT_CLARIFICATION):
        split = _stable_split(f"clarification|{text}")
        splits[split].append(_clarification_row(text, index, split))
    for index, text in enumerate(INJECTION_QUARANTINE):
        split = _stable_split(f"injection|{text}")
        splits[split].append(_injection_row(text, index, split))

    task_train = [
        row
        for row in source_rows["train"]
        if row["metadata"]["category"] == "CREATE_TASK"
    ]
    random.shuffle(task_train)
    splits["train"].extend(
        _repair_row(row, index) for index, row in enumerate(task_train[:24])
    )
    splits["train"].extend(_invalid_tool_recovery_row(index) for index in range(12))

    if args.teacher_jsonl is not None:
        for row in read_jsonl(args.teacher_jsonl.resolve()):
            if not bool((row.get("metadata") or {}).get("humanReviewed")):
                raise ValueError("teacher 轨迹必须标记 metadata.humanReviewed=true")
            split = str(row["metadata"].get("split") or "train")
            if split not in splits:
                raise ValueError(f"teacher 轨迹 split 无效：{split}")
            splits[split].append(row)

    for rows in splits.values():
        for row in rows:
            validate_example(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for split, rows in splits.items():
        path = output_dir / f"agent-sft-{split}.jsonl"
        write_jsonl(path, rows)
        files[split] = {
            "path": path.name,
            "count": len(rows),
            "sha256": file_sha256(path),
        }
    categories = Counter(
        str(row["metadata"]["category"]) for rows in splits.values() for row in rows
    )
    manifest = {
        "schemaVersion": 1,
        "datasetId": "masp-agent-sft-v2",
        "createdAt": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "engineCommit": source_manifest["engineCommit"],
        "sourceDataset": {
            "datasetId": source_manifest["datasetId"],
            "manifestSha256": file_sha256(source_dir / "manifest.json"),
        },
        "goldEvaluationSuite": "evals/agent-trajectories-v1.json",
        "goldLeakagePolicy": "评测 gold 未用于生成训练动作或标签",
        "generationPolicy": {
            "selfRollout1_5B": False,
            "deterministicRollout": True,
            "teacherOptional": True,
            "teacherHumanReviewRequired": True,
            "frozenV1IntentRetention": True,
        },
        "counts": {
            "total": sum(len(rows) for rows in splits.values()),
            "splits": {key: len(value) for key, value in splits.items()},
            "categories": dict(sorted(categories.items())),
        },
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
