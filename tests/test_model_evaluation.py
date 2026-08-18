from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import command_center.api as api_module
from command_center.audit import AuditStore
from command_center.contracts import ModelEvaluationRequest
from command_center.knowledge import KnowledgeBase
from command_center.model_evaluation import ModelSafetyEvaluator
from command_center.provider import DeepSeekProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_safety_suite_generates_traceable_report(isolated_settings) -> None:
    audit = AuditStore(isolated_settings.data_dir / "audit.jsonl")
    evaluator = ModelSafetyEvaluator(
        isolated_settings.data_dir,
        suite_path=PROJECT_ROOT / "evals" / "model-safety-v1.json",
        provider=DeepSeekProvider(isolated_settings),
        knowledge=KnowledgeBase(PROJECT_ROOT / "knowledge"),
        audit=audit,
    )

    report = evaluator.run(
        ModelEvaluationRequest(suiteName="离线模型安全回归", requestedBy="tester")
    )

    assert report["status"] == "PASSED"
    assert report["safetyGate"]["passed"] is True
    assert report["passedCaseCount"] == report["coverage"]["caseCount"] == 12
    assert report["liveProviderEvaluated"] is False
    assert report["fallbackCaseCount"] >= 1
    assert len(report["suiteSha256"]) == 64
    assert {row["category"] for row in report["cases"]} >= {
        "意图识别",
        "提示注入",
        "证据约束",
        "动作授权",
        "知识检索",
        "降级韧性",
    }

    evaluation_id = report["evaluationId"]
    root = isolated_settings.data_dir / "model-evaluations" / evaluation_id
    assert (root / "request.json").is_file()
    assert (root / "report.json").is_file()
    assert (root / "report.md").is_file()
    assert evaluator.get(evaluation_id)["suiteSha256"] == report["suiteSha256"]
    assert evaluator.list()[0]["evaluationId"] == evaluation_id
    assert audit.latest(1)[0].event_type == "MODEL_SAFETY_EVALUATED"


def test_model_safety_report_rejects_invalid_identifier(isolated_settings) -> None:
    evaluator = ModelSafetyEvaluator(
        isolated_settings.data_dir,
        suite_path=PROJECT_ROOT / "evals" / "model-safety-v1.json",
        provider=DeepSeekProvider(isolated_settings),
        knowledge=KnowledgeBase(PROJECT_ROOT / "knowledge"),
    )

    try:
        evaluator.get("../report")
    except KeyError:
        pass
    else:
        raise AssertionError("path traversal identifier must be rejected")


def test_model_safety_api_uses_versioned_suite(isolated_settings, monkeypatch) -> None:
    evaluator = ModelSafetyEvaluator(
        isolated_settings.data_dir,
        suite_path=PROJECT_ROOT / "evals" / "model-safety-v1.json",
        provider=DeepSeekProvider(isolated_settings),
        knowledge=KnowledgeBase(PROJECT_ROOT / "knowledge"),
    )
    monkeypatch.setattr(api_module, "model_evaluations", evaluator)
    client = TestClient(api_module.app)

    created = client.post(
        "/api/v1/evaluations/model-safety",
        json={"suiteName": "接口模型安全回归", "requestedBy": "api-tester"},
    )
    assert created.status_code == 200
    report = created.json()
    assert report["safetyGate"]["passed"] is True

    rows = client.get("/api/v1/evaluations/model-safety")
    assert rows.status_code == 200
    assert rows.json()[0]["evaluationId"] == report["evaluationId"]

    detail = client.get(
        f"/api/v1/evaluations/model-safety/{report['evaluationId']}"
    )
    assert detail.status_code == 200
    assert detail.json()["suiteSha256"] == report["suiteSha256"]
