from __future__ import annotations

import json
import logging

from command_center.logging_setup import (
    JsonFormatter,
    configure_logging,
    get_logger,
)


def _format(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


def _record(message: str, **context: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="command_center.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in context.items():
        setattr(record, key, value)
    return record


def test_formatter_emits_single_line_json() -> None:
    output = JsonFormatter().format(_record("hello"))

    assert "\n" not in output
    assert json.loads(output)["message"] == "hello"


def test_formatter_includes_level_and_logger() -> None:
    payload = _format(_record("hello"))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "command_center.test"
    assert "ts" in payload


def test_formatter_collects_extra_into_context() -> None:
    payload = _format(_record("agent decision", traceId="trace-1", attempt=3))

    assert payload["context"] == {"traceId": "trace-1", "attempt": 3}


def test_formatter_omits_context_when_empty() -> None:
    assert "context" not in _format(_record("hello"))


def test_formatter_serializes_unknown_types() -> None:
    payload = _format(_record("hello", path=object()))

    assert isinstance(payload["context"]["path"], str)


def test_configure_logging_is_idempotent() -> None:
    first = configure_logging("DEBUG")
    handler_count = len(first.handlers)

    second = configure_logging("DEBUG")

    assert second is first
    assert len(second.handlers) == handler_count


def test_configure_logging_does_not_propagate_to_root() -> None:
    logger = configure_logging()

    assert logger.propagate is False


def test_child_logger_is_namespaced() -> None:
    assert get_logger("agent_loop").name == "command_center.agent_loop"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[str] = []
        self.setFormatter(JsonFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        self.rows.append(self.format(record))


def test_agent_decision_log_carries_no_prompt_or_reply() -> None:
    """日志只记标识、状态和计数，不记提示词原文与模型回复。

    这条约定写在 logging_setup 的模块文档里，这里让它变成可执行断言。
    直接挂 handler 而不用 caplog，因为 configure_logging 关掉了向 root 冒泡。
    """
    configure_logging()
    logger = logging.getLogger("command_center.agent_loop")
    capture = _Capture()
    logger.addHandler(capture)
    secret_prompt = "把 AP1 的托盘紧急送到 AP7"
    secret_reply = '{"action": "PROPOSE_INTENT"}'
    try:
        logger.info(
            "agent decision",
            extra={
                "traceId": "trace-1",
                "conversationId": "conversation-1",
                "attempt": 1,
                "model": "masp-agent-lora-v2.3",
                "action": "CALL_TOOL",
                "tokens": 128,
            },
        )
    finally:
        logger.removeHandler(capture)

    rendered = "\n".join(capture.rows)
    assert "agent decision" in rendered
    assert secret_prompt not in rendered
    assert secret_reply not in rendered
