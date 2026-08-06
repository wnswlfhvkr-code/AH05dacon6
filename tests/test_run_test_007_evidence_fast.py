from __future__ import annotations

import numpy as np

from src.test_007 import finalize_test_007_specialization as finalizer
from src.test_007 import run_test_007_evidence_fast as fast


def _probability(prediction: np.ndarray, classes: int = 3) -> np.ndarray:
    result = np.full((len(prediction), classes), 0.01, dtype=float)
    result[np.arange(len(prediction)), prediction] = 0.98
    return result / result.sum(axis=1, keepdims=True)


def test_vectorized_bootstrap_is_deterministic_and_preserves_exact_seed_delta(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fast, "CHECKPOINT_ROOT", tmp_path)
    labels = np.tile(np.arange(3), 30)
    baseline = np.stack([_probability(labels) for _ in finalizer.SEEDS])
    candidate = baseline.copy()
    candidate[:, [0, 3, 6], :] = _probability(np.array([1, 1, 1]))
    masks = np.ones((3, len(labels)), dtype=bool)
    first = fast.fast_masked_paired_bootstrap(
        labels, candidate, baseline, masks, iterations=100, seed=17
    )
    second = fast.fast_masked_paired_bootstrap(
        labels, candidate, baseline, masks, iterations=100, seed=17
    )
    expected = finalizer._masked_paired_bootstrap(
        labels, candidate, baseline, masks, iterations=2, seed=99
    )["seed_delta"]
    assert first == second
    assert first["seed_delta"] == expected
    assert np.isfinite(
        [first[key] for key in ("delta", "se_seed", "se_boot", "epsilon", "ci_lower", "ci_upper")]
    ).all()


def test_vectorized_bootstrap_respects_seed_specific_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, "CHECKPOINT_ROOT", tmp_path)
    labels = np.tile(np.arange(3), 20)
    baseline = np.stack([_probability(labels) for _ in finalizer.SEEDS])
    candidate = baseline.copy()
    candidate[:, 0, :] = _probability(np.array([1]))
    masks = np.zeros((3, len(labels)), dtype=bool)
    masks[0, :20] = True
    masks[1, 20:40] = True
    masks[2, 40:] = True
    result = fast.fast_masked_paired_bootstrap(
        labels, candidate, baseline, masks, iterations=50, seed=5
    )
    assert result["support"] == {"42": 20, "2026": 20, "777": 20}
    assert result["seed_delta"]["42"] < 0
    assert result["seed_delta"]["2026"] == 0
    assert result["seed_delta"]["777"] == 0


def test_fold_safe_group_checkpoint_resumes_and_rejects_corruption(
    tmp_path, monkeypatch
):
    import pandas as pd

    monkeypatch.setattr(fast, "CHECKPOINT_ROOT", tmp_path)
    monkeypatch.setattr(fast, "GROUP_TRAIN_HASH", "a" * 64)
    calls = []
    original = fast.vectorized_fold_safe_groups

    def compute(train_x, valid_x):
        calls.append((len(train_x), len(valid_x)))
        return original(train_x, valid_x)

    monkeypatch.setattr(fast, "vectorized_fold_safe_groups", compute)
    frame = pd.DataFrame({"x": ["WT", "A", "B", "C"], "y": ["WT", "WT", "D", "E"]})
    first = fast.checkpointed_fold_safe_groups(frame.iloc[:3], frame.iloc[3:])
    second = fast.checkpointed_fold_safe_groups(frame.iloc[:3], frame.iloc[3:])
    assert calls == [(3, 1)]
    assert np.array_equal(first[0]["burden"], second[0]["burden"])
    npz = next((tmp_path / "fold_safe_groups").glob("*.npz"))
    npz.write_bytes(b"corrupt")
    fast.checkpointed_fold_safe_groups(frame.iloc[:3], frame.iloc[3:])
    assert calls == [(3, 1), (3, 1)]


def test_complete_group_checkpoint_set_is_detected_without_dense_context(
    tmp_path, monkeypatch
):
    import pandas as pd

    monkeypatch.setattr(fast, "CHECKPOINT_ROOT", tmp_path)
    monkeypatch.setattr(fast, "GROUP_TRAIN_HASH", "b" * 64)
    frame = pd.DataFrame(
        {
            "x": ["WT" if index % 4 == 0 else f"X{index}" for index in range(25)],
            "y": ["WT" if index % 3 == 0 else f"Y{index}" for index in range(25)],
        }
    )
    assignment_rows = []
    for seed in finalizer.SEEDS:
        permutation = np.random.default_rng(seed).permutation(len(frame))
        folds = np.empty(len(frame), dtype=int)
        folds[permutation] = np.arange(len(frame)) % len(finalizer.FOLDS)
        assignment_rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "row_index": np.arange(len(frame)),
                    "ID": [f"I{index}" for index in range(len(frame))],
                    "fold": folds,
                }
            )
        )
    assignments = pd.concat(assignment_rows, ignore_index=True)
    for seed in finalizer.SEEDS:
        seeded = assignments.loc[assignments.seed == seed]
        for fold in finalizer.FOLDS:
            valid = seeded.loc[seeded.fold == fold, "row_index"].to_numpy(int)
            train = seeded.loc[seeded.fold != fold, "row_index"].to_numpy(int)
            fast.checkpointed_fold_safe_groups(frame.iloc[train], frame.iloc[valid])

    assert len(list((tmp_path / "fold_safe_groups").glob("*.npz"))) == 15
    assert fast.fold_safe_group_checkpoints_complete(frame, assignments) is True
    checkpoint = next((tmp_path / "fold_safe_groups").glob("*.npz"))
    checkpoint.write_bytes(b"corrupt")
    assert fast.fold_safe_group_checkpoints_complete(frame, assignments) is False


def test_group_checkpoint_rejects_threshold_receipt_corruption(tmp_path, monkeypatch):
    import json
    import pandas as pd

    monkeypatch.setattr(fast, "CHECKPOINT_ROOT", tmp_path)
    monkeypatch.setattr(fast, "GROUP_TRAIN_HASH", "c" * 64)
    frame = pd.DataFrame({"x": ["WT", "A", "B", "C"], "y": ["WT", "D", "E", "F"]})
    train, valid = frame.iloc[:3], frame.iloc[3:]
    fast.checkpointed_fold_safe_groups(train, valid)
    receipt_path = next((tmp_path / "fold_safe_groups").glob("*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["thresholds"] = {"corrupt": "still-a-dict"}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert fast._valid_fold_safe_group_checkpoint(train, valid) is False
    assert fast._valid_group_thresholds(
        {"burden": [[0.0], [1.0]], "novelty": [0.0, 1.0]}
    ) is False


def test_vectorized_fold_safe_groups_matches_reference_exactly():
    import pandas as pd

    train = pd.DataFrame(
        {
            "a": ["WT", "A", "A", "B", None, "C"],
            "b": ["WT", "X", "Y", "X", "WT", "Z"],
            "c": ["WT", "WT", "M", "N", "M", "WT"],
        }
    )
    valid = pd.DataFrame(
        {
            "a": ["A", "Q", "WT"],
            "b": ["Z", "P", "WT"],
            "c": ["M", "R", "WT"],
        }
    )
    expected_groups, expected_thresholds = fast.ORIGINAL_FOLD_SAFE_GROUPS(train, valid)
    actual_groups, actual_thresholds = fast.vectorized_fold_safe_groups(train, valid)
    assert actual_thresholds == expected_thresholds
    assert all(
        np.array_equal(actual_groups[name], expected_groups[name])
        for name in ("burden", "novelty")
    )
    old_active = fast.GROUP_CONTEXT_ACTIVE
    old_values = fast.GROUP_CONTEXT_VALUES
    old_columns = fast.GROUP_CONTEXT_COLUMNS
    try:
        full = pd.concat([train, valid], ignore_index=True)
        fast.initialize_group_context(full)
        context_groups, context_thresholds = fast.vectorized_fold_safe_groups(
            full.iloc[: len(train)], full.iloc[len(train) :]
        )
        assert context_thresholds == expected_thresholds
        assert all(
            np.array_equal(context_groups[name], expected_groups[name])
            for name in ("burden", "novelty")
        )
    finally:
        fast.GROUP_CONTEXT_ACTIVE = old_active
        fast.GROUP_CONTEXT_VALUES = old_values
        fast.GROUP_CONTEXT_COLUMNS = old_columns
