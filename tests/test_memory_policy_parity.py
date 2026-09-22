import json

import numpy as np
import pandas as pd
import pytest

from memory import TradingMemory


def test_memory_backfill_uses_canonical_policy(tmp_path):
    rows = 720
    dates = pd.date_range("2026-01-05 09:30", periods=rows, freq="min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "trade_dt": dates.strftime("%Y-%m-%d"),
            "close": 100 + np.sin(np.arange(rows) / 30),
            "Pred_Ret": np.sin(np.arange(rows) / 20),
            "Target_Ret": np.cos(np.arange(rows) / 25) * 0.001,
            "Market_Regime": np.zeros(rows, dtype=int),
            "Model_Used": "fake",
        }
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    frame.to_pickle(output_dir / "df_predictions.pkl")

    memory = TradingMemory(tmp_path)
    count = memory.backfill_from_predictions()

    records = [
        json.loads(line)
        for line in (output_dir / "trade_memory.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    evaluated = memory.policy.evaluate_frame(frame)
    expected = evaluated.groupby("trade_dt").tail(1)

    assert count == len(expected) == len(records)
    assert [record["direction"] for record in records] == expected[
        "Direction"
    ].astype(int).tolist()
    assert all(
        record["strategy_fingerprint"] == memory.strategy_spec.fingerprint()
        for record in records
    )
    assert all(record["decision_id"].startswith("decision:") for record in records)
    expected_ids = [
        memory.policy.decision_id_for_evaluated_row(
            row, model_release_id="legacy-unversioned"
        )
        for _, row in expected.iterrows()
    ]
    assert [record["decision_id"] for record in records] == expected_ids


def test_backfill_replaces_only_current_strategy_slice(tmp_path):
    rows = 720
    dates = pd.date_range("2026-01-05 09:30", periods=rows, freq="min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "trade_dt": dates.strftime("%Y-%m-%d"),
            "close": np.linspace(100, 101, rows),
            "Pred_Ret": np.sin(np.arange(rows) / 20),
            "Target_Ret": 0.001,
            "Market_Regime": 0,
            "Model_Used": "fake",
        }
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    frame.to_pickle(output_dir / "df_predictions.pkl")
    preserved = {
        "decision_id": "decision:other",
        "strategy_fingerprint": "sha256:other-strategy",
        "direction": -1,
    }
    (output_dir / "trade_memory.jsonl").write_text(
        json.dumps(preserved) + "\n", encoding="utf-8"
    )

    memory = TradingMemory(tmp_path)
    memory.backfill_from_predictions()

    records = [
        json.loads(line)
        for line in (output_dir / "trade_memory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert preserved in records
    assert any(
        record.get("strategy_fingerprint") == memory.strategy_spec.fingerprint()
        for record in records
    )


def test_shadow_research_accuracy_is_separate_from_executed_accuracy(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    memory = TradingMemory(tmp_path)
    fingerprint = memory.strategy_spec.fingerprint()
    records = [
        {
            "strategy_fingerprint": fingerprint,
            "direction": 1,
            "fused_direction": 1,
            "research_proposed_direction": -1,
            "fusion_level": "rag",
            "actual_return": 0.01,
            "is_correct": True,
        },
        {
            "strategy_fingerprint": fingerprint,
            "direction": -1,
            "fused_direction": -1,
            "research_proposed_direction": 1,
            "fusion_level": "rag",
            "actual_return": -0.01,
            "is_correct": True,
        },
    ]
    (output_dir / "trade_memory.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    stats = memory.reflection_stats()["by_shadow_context"]["rag"]

    assert stats["executed_accuracy"] == 1.0
    assert stats["proposal_accuracy"] == 0.0
    assert stats["executed_acted"] == 2
    assert stats["proposal_acted"] == 2


def test_backfill_rejects_mixed_model_releases(tmp_path):
    rows = 720
    dates = pd.date_range("2026-01-05 09:30", periods=rows, freq="min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "trade_dt": dates.strftime("%Y-%m-%d"),
            "close": np.linspace(100, 101, rows),
            "Pred_Ret": np.sin(np.arange(rows) / 20),
            "Target_Ret": 0.001,
            "Market_Regime": 0,
            "Model_Used": "fake",
            "Model_Release_ID": ["release-a"] * (rows // 2)
            + ["release-b"] * (rows - rows // 2),
        }
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    frame.to_pickle(output_dir / "df_predictions.pkl")

    with pytest.raises(ValueError, match="mixes multiple model releases"):
        TradingMemory(tmp_path).backfill_from_predictions()


def test_actual_backfill_does_not_cross_model_release_boundaries(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05 15:00")],
            "trade_dt": ["2026-01-05"],
            "Target_Ret": [0.02],
            "Model_Release_ID": ["release-current"],
        }
    )
    frame.to_pickle(output_dir / "df_predictions.pkl")
    memory = TradingMemory(tmp_path)
    fingerprint = memory.strategy_spec.fingerprint()
    records = [
        {
            "trade_dt": "2026-01-05",
            "strategy_fingerprint": fingerprint,
            "model_release_id": "release-old",
            "direction": 1,
            "actual_return": None,
        },
        {
            "trade_dt": "2026-01-05",
            "strategy_fingerprint": fingerprint,
            "model_release_id": "release-current",
            "direction": 1,
            "actual_return": None,
        },
    ]
    (output_dir / "trade_memory.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    memory.update_actuals()

    updated = [
        json.loads(line)
        for line in (output_dir / "trade_memory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert updated[0]["actual_return"] is None
    assert updated[1]["actual_return"] == 0.02
    assert updated[1]["is_correct"] is True
