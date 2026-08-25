from __future__ import annotations

from training.compare_agent_replays import _paired_interval, _prompt_sha


def test_paired_interval_preserves_direction() -> None:
    interval = _paired_interval([1.0, 0.0, 1.0, -1.0])

    assert interval["meanDifference"] == 0.25
    assert interval["ci95Low"] < interval["ci95High"]


def test_replay_prefers_mode_specific_prompt_hash() -> None:
    report = {
        "promptSha256": "legacy",
        "promptSha256ByMode": {"linear_v1": "linear", "loop_local": "loop"},
    }

    assert _prompt_sha(report, "linear_v1") == "linear"
    assert _prompt_sha(report, "missing") == "legacy"
