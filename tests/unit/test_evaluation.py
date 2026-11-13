import pytest

from moderationkit import ModGuard
from moderationkit.evaluation import (
    DatasetRecord,
    category_metrics,
    evaluate_dataset,
    threshold_sweep,
)
from moderationkit.exceptions import ConfigurationError, InputError
from tests.fakes import FakeProvider


def test_hand_calculated_metrics_and_missingness():
    metrics = category_metrics(
        [True, True, False, False, True, None], [True, False, True, False, None, False]
    )
    assert [metrics[k] for k in ("tp", "tn", "fp", "fn")] == [1, 1, 1, 1]
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert metrics["coverage"] == 0.8
    assert metrics["missing_labels"] == metrics["unavailable"] == 1
    assert category_metrics([False], [None])["precision"] is None


def test_report_splits_and_threshold_sweep(policy):
    guard = ModGuard(provider=FakeProvider({"harassment": 0.7}), policy=policy)
    records = [
        DatasetRecord(id="val", text="x", labels={"harassment": True}, split="validation"),
        DatasetRecord(id="test", text="x", labels={"harassment": False}),
    ]
    report = evaluate_dataset(guard, records)
    assert report["count"] == 1 and report["categories"]["harassment"]["fp"] == 1
    assert report["cost"] is None
    assert report["dataset_sha256"] == evaluate_dataset(guard, records)["dataset_sha256"]
    sweep = threshold_sweep(guard, records, category="harassment", thresholds=[0.5, 0.9])
    assert sweep["recommended_threshold"] is None
    recommended = threshold_sweep(
        guard, records, category="harassment", thresholds=[0.5, 0.9], min_recall=0.9
    )
    assert recommended["recommended_threshold"] == 0.5
    with pytest.raises(ConfigurationError):
        threshold_sweep(guard, records[1:], category="harassment", thresholds=[0.5])


def test_jsonl_duplicate_ids(tmp_path):
    from moderationkit.evaluation import load_jsonl

    path = tmp_path / "data.jsonl"
    path.write_text('{"id":"a","text":"x","labels":{}}\n' * 2)
    with pytest.raises(InputError):
        load_jsonl(path)


def test_incomplete_results_cannot_improve_metrics_or_thresholds(policy):
    guard = ModGuard(provider=FakeProvider(coverage=False), policy=policy)
    records = [
        DatasetRecord(id="truncated", text="x", labels={"harassment": False}, split="validation")
    ]
    report = evaluate_dataset(guard, records, split="validation")
    metrics = report["categories"]["harassment"]
    assert metrics["evaluated"] == 0 and metrics["tn"] == 0
    assert metrics["coverage"] == 0 and metrics["unavailable"] == 1
    sweep = threshold_sweep(guard, records, category="harassment", thresholds=[0.5])
    assert sweep["rows"][0]["evaluated"] == 0
