from __future__ import annotations

from pathlib import Path

from command_center.audit import AuditStore
from command_center.contracts import ChatRequest
from command_center.engine_adapter import MaspAdapter
from command_center.knowledge import KnowledgeBase
from command_center.orchestrator import DispatchOrchestrator
from command_center.provider import DeepSeekProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _orchestrator(isolated_settings) -> DispatchOrchestrator:
    engine = MaspAdapter(isolated_settings)
    return DispatchOrchestrator(
        engine=engine,
        provider=DeepSeekProvider(isolated_settings),
        knowledge=KnowledgeBase(PROJECT_ROOT / "knowledge"),
        audit=AuditStore(isolated_settings.data_dir / "audit.jsonl"),
    )


def test_hybrid_retrieval_ranks_colloquial_fault_query_with_metadata() -> None:
    knowledge = KnowledgeBase(PROJECT_ROOT / "knowledge")

    rows = knowledge.search("坏车以后怎么安全处理和重新安排任务", limit=3)

    assert rows
    assert rows[0].source == "sop/vehicle-fault.md"
    assert rows[0].chunk_id and rows[0].chunk_id.startswith("kb-")
    assert rows[0].score is not None and rows[0].score > 0.5
    assert rows[0].retrieval_method == "hybrid-bm25-char-vector-v1"
    assert knowledge.stats() == {
        "chunkCount": 14,
        "sourceCount": 4,
        "retrievalMethod": "hybrid-bm25-char-vector-v1",
    }


def test_second_turn_recalls_structured_conversation_memory(isolated_settings) -> None:
    orchestrator = _orchestrator(isolated_settings)
    conversation_id = "conversation-memory"
    first = orchestrator.chat(
        ChatRequest(
            message="创建紧急叉车任务，从 AP1123 运到 AP2121",
            scenarioId="interactive-multi-fleet",
            conversationId=conversation_id,
        )
    )
    assert first.state == "READY"

    second = orchestrator.chat(
        ChatRequest(
            message="刚才那轮使用了哪些工具？",
            scenarioId="interactive-multi-fleet",
            conversationId=conversation_id,
        )
    )

    memory = orchestrator.memory.get(conversation_id)
    assert memory is not None
    assert memory.confirmed_entities["nodeIds"] == ["fork:AP1123", "fork:AP2121"]
    assert memory.confirmed_entities["robotGroups"] == ["fork"]
    assert len(memory.turns) == 2
    assert second.agent_trace is not None
    assert any(
        step.tool_name == "recall_conversation_memory"
        for step in second.agent_trace.steps
    )
    assert any(
        row.retrieval_method == "structured-memory-v1" for row in second.evidence
    )
    assert "fork:AP1123" in second.message
    assert "get_world_snapshot" in second.message


def test_agent_observability_aggregates_requests_without_prompts(
    isolated_settings,
) -> None:
    orchestrator = _orchestrator(isolated_settings)
    orchestrator.chat(
        ChatRequest(
            message="当前车辆状态怎么样？",
            scenarioId="interactive-multi-fleet",
            conversationId="conversation-metrics-ready",
        )
    )
    orchestrator.chat(
        ChatRequest(
            message="创建一个紧急叉车任务",
            scenarioId="interactive-multi-fleet",
            conversationId="conversation-metrics-clarify",
        )
    )

    metrics = orchestrator.observability.summary()

    assert metrics["requestCount"] == 2
    assert metrics["completedCount"] == 1
    assert metrics["clarificationCount"] == 1
    assert metrics["taskCompletionRate"] == 0.5
    assert metrics["toolCallCounts"]["get_world_snapshot"] == 2
    assert all("message" not in row for row in metrics["recent"])
