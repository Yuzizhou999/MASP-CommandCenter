"""确认 MASP 引擎门禁真的满足，避免 integration 用例被静默跳过。

tests/conftest.py 在引擎缺失、提交不匹配或工作区脏时会 skip 全部 integration
用例。CI 里这种 skip 会表现为「绿色但什么都没测」，所以这里显式失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 未以 editable 方式安装时也要能导入 command_center 和 tests/conftest。
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import conftest  # noqa: E402


def main() -> int:
    if not conftest.ENGINE_AVAILABLE:
        print(
            f"引擎门禁未满足，integration 用例会被跳过：{conftest.ENGINE_SKIP_REASON}"
        )
        return 1
    print(f"引擎门禁已满足，integration 用例将真实执行：{conftest.ENGINE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
