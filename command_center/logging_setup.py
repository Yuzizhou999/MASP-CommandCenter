"""结构化日志。

此前整个 command_center 没有任何 logging 调用：有审计 JSONL 和指标 JSONL，
但那是业务事件，回答不了"这个 run 卡在哪一轮""为什么降级"这类运维问题。

日志与审计的分工：

* 审计（``audit.jsonl``）是安全叙事的证据链，写什么、写多少由合规决定，
  格式稳定，不可丢。
* 日志是可运维信号，用于定位问题，允许按级别过滤，允许丢。

因此日志里**不写**提示词原文、模型回复、token 和任何密钥；只写标识、状态、
计数和耗时。这与 ``agent-metrics.jsonl`` 不含用户提示词的既有约定一致。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

LOGGER_NAME = "command_center"

# 这些键在结构化日志里保留原名，其余 extra 会被收进 context。
_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """把日志记录序列化为单行 JSON，便于 grep 和后续接入日志系统。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and not key.startswith("_")
        }
        if context:
            payload["context"] = context
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str | None = None) -> logging.Logger:
    """配置 ``command_center`` logger，重复调用安全。"""
    configured = level if level else os.environ.get("COMMAND_CENTER_LOG_LEVEL", "INFO")
    resolved = configured.upper()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, resolved, logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    # 不向 root 冒泡，避免 uvicorn 的 handler 再打印一遍。
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.LoggerAdapter | logging.Logger:
    """取得子 logger，``name`` 用点号分隔的模块名。"""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
