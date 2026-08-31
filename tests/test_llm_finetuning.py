from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from command_center.contracts import IntentType
from command_center.llm_provider import (
    OpenAICompatibleLocalProvider,
    create_llm_provider,
)
from command_center.model_registry import model_registration
from command_center.provider import DeepSeekProvider
from training.intent_dataset import build_example, stable_split, validate_example


def _write_model_card(root: Path, adapter_bytes: bytes = b"adapter") -> Path:
    root.mkdir(parents=True)
    adapter = root / "adapter_model.safetensors"
    adapter.write_bytes(adapter_bytes)
    card = root / "model-card.json"
    card.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "modelId": "masp-intent-lora",
                "version": "0.1.0",
                "baseModel": "Qwen/Qwen2.5-1.5B-Instruct",
                "trainingMethod": "QLoRA-SFT-completion-only",
                "datasetVersion": "masp-intent-sft-v1",
                "adapterFile": adapter.name,
                "adapterSha256": hashlib.sha256(adapter_bytes).hexdigest(),
                "status": "candidate",
                "runScope": "full",
                "sampleLimits": {"train": None, "valid": None, "maxSteps": -1},
                "createdAt": "2026-08-24T00:00:00+00:00",
                "metrics": {"eval_loss": 0.5},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return card


def test_provider_factory_selects_deepseek_local_and_auto(
    isolated_settings,
) -> None:
    assert isinstance(
        create_llm_provider(isolated_settings), OpenAICompatibleLocalProvider
    )

    deepseek = create_llm_provider(replace(isolated_settings, llm_provider="deepseek"))
    assert isinstance(deepseek, DeepSeekProvider)
    assert not isinstance(deepseek, OpenAICompatibleLocalProvider)

    local = create_llm_provider(
        replace(
            isolated_settings,
            llm_provider="local",
            local_llm_base_url="http://127.0.0.1:9000/v1",
            local_llm_model="registered-intent-model",
        )
    )
    assert isinstance(local, OpenAICompatibleLocalProvider)
    assert local.settings.deepseek_base_url == "http://127.0.0.1:9000/v1"
    assert local.settings.deepseek_model == "registered-intent-model"

    automatic = create_llm_provider(
        replace(isolated_settings, llm_provider="auto", local_llm_enabled=True)
    )
    assert isinstance(automatic, OpenAICompatibleLocalProvider)


def test_local_provider_reports_registration_and_keeps_tools_deterministic(
    isolated_settings, tmp_path: Path
) -> None:
    card = _write_model_card(tmp_path / "model")
    provider = OpenAICompatibleLocalProvider(
        replace(isolated_settings, local_llm_model_card=card)
    )

    status = provider.status()
    plan = provider.plan_context_tools("封闭通道前需要检查什么？", [], has_memory=True)

    assert status["provider"] == "local-openai-compatible"
    assert status["capability"] == "dispatch-intent-and-agent-actions"
    assert status["agentCapability"] == "single-action-protocol"
    assert status["registration"]["valid"] is True
    assert status["registration"]["adapterSha256Matches"] is True
    assert status["registration"]["runScope"] == "full"
    assert status["registration"]["sampleLimits"]["maxSteps"] == -1
    assert plan.strategy == "DETERMINISTIC_POLICY"
    assert [row.name for row in plan.calls] == [
        "get_world_snapshot",
        "recall_conversation_memory",
        "search_sop",
    ]
    assert plan.calls[-1].arguments["query"] == "封闭通道前需要检查什么？"


def test_model_registration_rejects_changed_adapter(tmp_path: Path) -> None:
    card = _write_model_card(tmp_path / "model")
    (card.parent / "adapter_model.safetensors").write_bytes(b"changed")

    registration = model_registration(card)

    assert registration is not None
    assert registration["valid"] is False
    assert registration["adapterSha256Matches"] is False


def test_model_registration_accepts_training_metadata(tmp_path: Path) -> None:
    card = _write_model_card(tmp_path / "model")
    payload = json.loads(card.read_text(encoding="utf-8"))
    payload.update(
        {
            "checkpointing": {"intervalSteps": 20, "resumedFrom": None},
            "tokenization": {
                "maxLength": 2048,
                "maxObservedTokens": 1403,
                "p50": 948.0,
                "p99": 1391.0,
                "truncatedExamples": 0,
                "preflightPassed": True,
            },
            "trainingEnvironment": {
                "python": "3.12.3",
                "torch": "2.8.0+cu128",
                "cuda": "12.8",
                "transformers": "4.57.6",
                "peft": "0.20.0",
                "bitsandbytes": "0.50.1",
                "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
            },
        }
    )
    card.write_text(json.dumps(payload), encoding="utf-8")

    registration = model_registration(card)

    assert registration is not None
    assert registration["valid"] is True
    assert registration["tokenization"]["truncatedExamples"] == 0
    assert registration["trainingEnvironment"]["cuda"] == "12.8"


def test_dataset_split_is_stable_at_entity_level() -> None:
    entity = "scenario|CREATE_TASK|fork:AP1|fork:AP2|fork"

    assert stable_split(entity) == stable_split(entity)
    assert stable_split(entity) in {"train", "valid", "test"}


def test_dataset_example_uses_runtime_prompt_and_validates() -> None:
    resolved_task = {
        "pickupNodeId": "fork:AP1123",
        "dropoffNodeId": "fork:AP2121",
        "requiredRobotGroup": "fork",
        "payloadType": "pallet",
    }
    example = build_example(
        example_id="example-1",
        text="新增叉车搬运：fork:AP1123 到 fork:AP2121",
        scenario_id="interactive-multi-fleet",
        world_revision=7,
        intent_type=IntentType.CREATE_TASK,
        source="test",
        split_key="entity-1",
        template_id="task-test",
        resolved_task=resolved_task,
    ).as_dict()

    request = json.loads(example["messages"][1]["content"])
    result = validate_example(example)

    assert request["authoritativeParameters"]["task"] == resolved_task
    assert "schema" in request
    assert result == {"intentType": "CREATE_TASK", "valid": True}
