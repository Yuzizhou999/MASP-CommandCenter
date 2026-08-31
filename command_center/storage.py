"""可变状态存储的原子落盘工具。

审批单、已提交意图、澄清、会话记忆、故障记录和场景草稿都是被反复整体重写的
共享状态。直接 ``Path.write_text`` 在写入中途崩溃会留下被截断的 JSON，下次
``_load`` 直接抛 ``JSONDecodeError``，整个存储不可读。

这里统一走「同目录临时文件 → fsync → os.replace」：``os.replace`` 在 POSIX 和
Windows 上都是原子替换，因此任何时刻磁盘上的目标文件要么是替换前的完整内容，
要么是替换后的完整内容，不存在中间态。临时文件与目标同目录，保证在同一文件
系统上，``os.replace`` 才有原子性保证。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    """把 ``text`` 原子写入 ``path``。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件名带 pid，避免同目录并发写入互相覆盖。
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        # 失败时不留下残留临时文件，目标文件保持替换前的完整内容。
        temp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    """把 ``payload`` 序列化为缩进 JSON 后原子写入 ``path``。"""
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
