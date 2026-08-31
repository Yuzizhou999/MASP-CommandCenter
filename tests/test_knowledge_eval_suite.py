from __future__ import annotations

import json
from pathlib import Path

from training.evaluate_knowledge_retrieval import _evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_gold_has_balanced_independent_labels() -> None:
    suite = json.loads(
        (ROOT / "evals" / "knowledge-retrieval-v1.json").read_text(encoding="utf-8")
    )

    assert suite["goldSource"] == "manual-independent-annotation"
    assert len(suite["cases"]) >= 20
    assert all(row["expectedSources"] for row in suite["cases"])


def test_default_retrieval_weights_meet_frozen_quality_gate() -> None:
    suite = json.loads(
        (ROOT / "evals" / "knowledge-retrieval-v1.json").read_text(encoding="utf-8")
    )

    result = _evaluate(
        suite["cases"],
        ROOT / "knowledge",
        (0.45, 0.10, 0.45),
        0.04,
    )

    assert suite["qualification"] == {
        "recallAt1": 0.79,
        "recallAt3": 0.97,
        "mrr": 0.91,
    }
    assert all(
        result[name] >= minimum for name, minimum in suite["qualification"].items()
    )
