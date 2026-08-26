from __future__ import annotations

import pytest

from command_center.clarifications import ClarificationResolver
from command_center.contracts import IntentType


@pytest.mark.parametrize(
    "message",
    [
        "通道封闭前需要遵循哪些审批和安全流程？",
        "共享窄路停用有什么安全要求？",
        "如何处理通道检修？",
    ],
)
def test_read_only_resource_questions_do_not_become_block_commands(message: str) -> None:
    assert ClarificationResolver._intent_type(message, pending=None) is None


@pytest.mark.parametrize(
    "message",
    [
        "共享窄路需要检修，请封闭三分钟并评估任务影响",
        "按照安全要求，请封闭共享窄路",
    ],
)
def test_explicit_resource_change_remains_a_block_command(message: str) -> None:
    assert (
        ClarificationResolver._intent_type(message, pending=None)
        is IntentType.BLOCK_RESOURCE
    )


@pytest.mark.parametrize(
    "message",
    [
        "创建紧急任务并说明要求",
        "请安排运输任务，怎么走由 MASP 规划",
    ],
)
def test_query_vocabulary_does_not_hide_explicit_task_commands(message: str) -> None:
    assert (
        ClarificationResolver._intent_type(message, pending=None)
        is IntentType.CREATE_TASK
    )
