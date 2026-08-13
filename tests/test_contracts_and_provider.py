from __future__ import annotations

import pytest
from pydantic import ValidationError

from command_center.contracts import DispatchIntent, IntentType, ResourceBlockDraft
from command_center.provider import DeepSeekProvider


def test_resource_block_rejects_invalid_time_window() -> None:
    with pytest.raises(ValidationError):
        ResourceBlockDraft(
            resourceIds=["zone:zone-jack-pp363-pp365"],
            startMs=1000,
            endMs=1000,
        )


def test_create_task_requires_structured_task() -> None:
    with pytest.raises(ValidationError):
        DispatchIntent(intentType=IntentType.CREATE_TASK)


def test_provider_falls_back_to_deterministic_intent(isolated_settings) -> None:
    provider = DeepSeekProvider(isolated_settings)
    result = provider.parse_intent(
        "创建紧急叉车任务，从 AP1123 运到 AP2121",
        world_revision=123,
        requested_by="tester",
    )
    assert result.fallback_used is True
    assert result.intent.intent_type is IntentType.CREATE_TASK
    assert result.intent.task is not None
    assert result.intent.task.pickup_node_id == "fork:AP1123"
    assert result.intent.task.dropoff_node_id == "fork:AP2121"
    assert result.intent.based_on_world_revision == 123
