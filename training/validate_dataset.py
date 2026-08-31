from __future__ import annotations

import argparse
import json
from pathlib import Path

from command_center.engine_adapter import MaspAdapter
from command_center.settings import Settings
from training.intent_dataset import file_sha256, read_jsonl, validate_example


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 MASP 意图微调数据集")
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    engine = MaspAdapter(Settings.load())
    checked = 0
    for split in ("train", "valid", "test"):
        item = manifest["files"][split]
        path = root / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"数据文件摘要不匹配：{path}")
        for row in read_jsonl(path):
            validate_example(row, engine)
            checked += 1
    print(
        json.dumps({"status": "PASSED", "checkedExamples": checked}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
