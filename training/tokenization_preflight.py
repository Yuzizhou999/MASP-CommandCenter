from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .tokenization import chat_token_ids, tokenize_conversation


def require_transformers_4(version: str) -> None:
    try:
        major = int(version.split(".", 1)[0])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"无法识别 transformers 版本：{version}") from error
    if major != 4:
        raise RuntimeError(
            f"当前 transformers={version}，预检和训练只允许项目锁定的 4.x；"
            "请使用 masp-lora 环境"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def inspect_dataset_tokenization(
    dataset_dir: Path,
    config: dict[str, Any],
    tokenizer: Any,
    *,
    transformers_version: str,
    tokenizer_source: str,
) -> dict[str, Any]:
    require_transformers_4(transformers_version)
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    max_length = int(config["maxLength"])
    lengths: list[int] = []
    violations: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}

    for split in ("train", "valid", "test"):
        item = manifest["files"][split]
        path = dataset_dir / item["path"]
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"{split} 数据文件摘要不匹配：{path}")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != int(item["count"]):
            raise ValueError(
                f"{split} 数据数量不匹配：manifest={item['count']} actual={len(rows)}"
            )
        split_counts[split] = len(rows)
        for row in rows:
            metadata = row.get("metadata") or {}
            token_ids = chat_token_ids(
                tokenizer,
                row["messages"],
                add_generation_prompt=False,
            )
            observed = len(token_ids)
            lengths.append(observed)
            if observed > max_length:
                violations.append(
                    {
                        "split": split,
                        "exampleId": metadata.get("exampleId"),
                        "category": metadata.get("category"),
                        "observedTokens": observed,
                        "maxLength": max_length,
                    }
                )
                continue
            tokenize_conversation(
                tokenizer,
                row["messages"],
                max_length=max_length,
                supervise_assistant_indices=metadata.get(
                    "superviseAssistantIndices"
                ),
            )

    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "datasetId": manifest["datasetId"],
        "datasetManifestSha256": _sha256(manifest_path),
        "tokenizerSource": tokenizer_source,
        "transformersVersion": transformers_version,
        "maxLength": max_length,
        "splitCounts": split_counts,
        "lengthDistribution": {
            "count": len(lengths),
            "min": min(lengths) if lengths else 0,
            "p50": round(_percentile(lengths, 0.50), 3),
            "p95": round(_percentile(lengths, 0.95), 3),
            "p99": round(_percentile(lengths, 0.99), 3),
            "max": max(lengths) if lengths else 0,
        },
        "maxObservedTokens": max(lengths) if lengths else 0,
        "overMaxLengthExamples": len(violations),
        "truncatedExamples": 0,
        "violations": violations,
        "passed": not violations,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在训练前全量检查对话 token 长度和监督目标"
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-source", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("缺少 transformers，请使用 masp-lora 环境") from error

    require_transformers_4(transformers.__version__)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer_source = args.tokenizer_source or str(config["baseModel"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=False,
    )
    report = inspect_dataset_tokenization(
        args.dataset_dir,
        config,
        tokenizer,
        transformers_version=transformers.__version__,
        tokenizer_source=tokenizer_source,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
