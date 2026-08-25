from __future__ import annotations

import argparse
import csv
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from command_center.knowledge import KnowledgeBase


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测知识检索 recall@k 并搜索权重")
    parser.add_argument(
        "--suite", type=Path, default=Path("evals/knowledge-retrieval-v1.json")
    )
    parser.add_argument("--knowledge", type=Path, default=Path("knowledge"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/retrieval-eval")
    )
    parser.add_argument(
        "--enforce-thresholds",
        action="store_true",
        help="配置检索指标未达到冻结门槛时以非零状态退出。",
    )
    parser.add_argument(
        "--keep-history",
        action="store_true",
        help="额外保留带时间戳的报告；默认只覆盖 latest 产物。",
    )
    return parser.parse_args()


def _evaluate(
    cases: list[dict[str, Any]],
    knowledge_root: Path,
    weights: tuple[float, float, float],
    threshold: float,
) -> dict[str, Any]:
    kb = KnowledgeBase(
        knowledge_root,
        sparse_weight=weights[0],
        vector_weight=weights[1],
        title_weight=weights[2],
        score_threshold=threshold,
    )
    recall1 = 0.0
    recall3 = 0.0
    reciprocal_rank = 0.0
    rows: list[dict[str, Any]] = []
    for case in cases:
        retrieved = kb.search(case["query"], limit=3)
        sources = [row.source for row in retrieved]
        expected = set(case["expectedSources"])
        recall1_case = len(set(sources[:1]) & expected) / len(expected)
        recall3_case = len(set(sources) & expected) / len(expected)
        rank = next(
            (index + 1 for index, source in enumerate(sources) if source in expected),
            None,
        )
        reciprocal = 1 / rank if rank else 0.0
        recall1 += recall1_case
        recall3 += recall3_case
        reciprocal_rank += reciprocal
        rows.append(
            {
                "caseId": case["caseId"],
                "query": case["query"],
                "expectedSources": sorted(expected),
                "retrievedSources": sources,
                "recallAt1": recall1_case,
                "recallAt3": recall3_case,
                "reciprocalRank": reciprocal,
            }
        )
    count = len(cases)
    return {
        "weights": {
            "sparse": weights[0],
            "vector": weights[1],
            "title": weights[2],
            "threshold": threshold,
        },
        "recallAt1": round(recall1 / count, 6),
        "recallAt3": round(recall3 / count, 6),
        "mrr": round(reciprocal_rank / count, 6),
        "cases": rows,
    }


def main() -> None:
    args = _arguments()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    cases = list(suite["cases"])
    candidates: list[tuple[float, float, float]] = []
    for sparse, vector in itertools.product(
        (0.45, 0.55, 0.65, 0.68, 0.75, 0.85),
        (0.10, 0.18, 0.24, 0.30, 0.40),
    ):
        title = round(1.0 - sparse - vector, 2)
        if title >= 0:
            candidates.append((sparse, vector, title))
    results = [
        _evaluate(cases, args.knowledge.resolve(), weights, threshold)
        for weights, threshold in itertools.product(
            candidates, (0.02, 0.04, 0.06, 0.08)
        )
    ]
    ranked = sorted(
        results,
        key=lambda row: (
            -row["recallAt1"],
            -row["mrr"],
            -row["recallAt3"],
            abs(row["weights"]["threshold"] - 0.04),
            -row["weights"]["sparse"],
        ),
    )
    baseline = next(
        row
        for row in results
        if row["weights"]
        == {"sparse": 0.68, "vector": 0.24, "title": 0.08, "threshold": 0.04}
    )
    configured = next(
        row
        for row in results
        if row["weights"]
        == {"sparse": 0.45, "vector": 0.10, "title": 0.45, "threshold": 0.04}
    )
    thresholds = suite.get("qualification") or {}
    checks = {
        name: {
            "minimum": float(minimum),
            "actual": float(configured[name]),
            "passed": float(configured[name]) >= float(minimum),
        }
        for name, minimum in thresholds.items()
    }
    generated = datetime.now(timezone.utc)
    output = {
        "schemaVersion": 1,
        "suiteId": suite["suiteId"],
        "goldSource": suite["goldSource"],
        "generatedAt": generated.isoformat(),
        "caseCount": len(cases),
        "baseline": baseline,
        "configured": configured,
        "recommended": ranked[0],
        "qualification": {
            "checks": checks,
            "passed": bool(checks) and all(row["passed"] for row in checks.values()),
        },
        "gridSize": len(results),
        "ranking": [
            {key: row[key] for key in ("weights", "recallAt1", "recallAt3", "mrr")}
            for row in ranked[:10]
        ],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = generated.strftime("%Y%m%d-%H%M%S")
    latest = output_dir / "knowledge-retrieval-eval-latest.json"
    encoded = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    latest.write_text(encoded, encoding="utf-8")
    if args.keep_history:
        (output_dir / f"knowledge-retrieval-eval-{stamp}.json").write_text(
            encoded, encoding="utf-8"
        )
    csv_path = output_dir / "knowledge-retrieval-grid-latest.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sparse",
                "vector",
                "title",
                "threshold",
                "recallAt1",
                "recallAt3",
                "mrr",
            ],
        )
        writer.writeheader()
        for row in ranked:
            writer.writerow({**row["weights"], **{key: row[key] for key in ("recallAt1", "recallAt3", "mrr")}})
    if args.keep_history:
        history_csv = output_dir / f"knowledge-retrieval-grid-{stamp}.csv"
        history_csv.write_bytes(csv_path.read_bytes())
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.enforce_thresholds and not output["qualification"]["passed"]:
        raise SystemExit("配置检索指标未达到冻结门槛")


if __name__ == "__main__":
    main()
