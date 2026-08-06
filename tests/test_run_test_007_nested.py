from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import yaml

import src.test_007.run_test_007_nested as nested


CLASSES = tuple(f"C{index:02d}" for index in range(26))
SEEDS = (42, 2026, 777)


def _write_yaml(path: Path, value) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _loaded_candidates(goal: dict) -> dict[str, nested.Candidate]:
    return nested._load_candidates(nested._candidate_paths(goal), nested.Audit())[0]


@pytest.fixture()
def synthetic_nested(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = 260
    y = np.resize(np.arange(26), rows)
    train = pd.DataFrame(
        {
            "ID": [f"R{index:04d}" for index in range(rows)],
            "signal": y,
            "noise": np.arange(rows) % 7,
            "SUBCLASS": np.asarray(CLASSES)[y],
        }
    )
    train.to_csv(raw / "train.csv", index=False)
    pd.DataFrame({"ID": ["T0"], "signal": [0], "noise": [0]}).to_csv(
        raw / "test.csv", index=False
    )
    pd.DataFrame({"ID": ["T0"], "SUBCLASS": ["C00"]}).to_csv(
        raw / "sample_submission.csv", index=False
    )

    candidates = []
    for alias in ("backbone", "robust", "expert"):
        candidate_path = tmp_path / f"{alias}.yaml"
        _write_yaml(
            candidate_path,
            {
                "project": {"experiment_name": alias, "seed": 42},
                "data": {
                    "raw_dir": str(raw),
                    "train_file": "train.csv",
                    "test_file": "test.csv",
                    "submission_file": "sample_submission.csv",
                },
                "preprocessing": {"name": "pipeComb_v3"},
                "model": {
                    "name": "logistic_regression",
                    "class_names": list(CLASSES),
                },
            },
        )
        candidates.append({"alias": alias, "config": str(candidate_path), "family": alias})

    goal_path = tmp_path / "goal.yaml"
    _write_yaml(
        goal_path,
        {
            "data": {
                "raw_dir": str(raw),
                "train_file": "train.csv",
                "test_file": "test.csv",
                "submission_file": "sample_submission.csv",
                "target_column": "SUBCLASS",
                "id_column": "ID",
            },
            "validation": {
                "final_seeds": list(SEEDS),
                "outer_folds": 5,
                "inner_folds": 4,
            },
            "search": {
                "random_trials": 20000,
                "random_seeds": [1001, 1002, 1003],
                "min_active_weight": 0.05,
                "max_model_weight": 0.80,
                "early_stop_patience": 20,
                "max_models": 2,
            },
            "candidates": candidates,
        },
    )
    universe_path = tmp_path / "universe.yaml"
    _write_yaml(
        universe_path,
        {
            "provenance": {
                "kind": "predeclared_search_universe",
                "label_independent": True,
                "global_oof_selection_used": False,
                "declared_before_outer_labels": True,
            },
            "search_universe": {
                "models": ["backbone", "robust", "expert"],
                "roles": {
                    "backbone": ["GLOBAL_BACKBONE"],
                    "robust": ["ROBUSTNESS_ANCHOR"],
                    "expert": ["COLLISION_EXPERT"],
                },
                "runtime_exclusions": {},
                "allowed_kinds": ["single", "blend", "collision_gate"],
                "max_models": 2,
                "temperature_values": [1.0],
                "trial_seeds": [1001, 1002, 1003],
                "trial_count_per_seed": 20000,
                "min_active_weight": 0.05,
                "max_model_weight": 0.80,
                "early_stop_patience": 20,
                "role_balanced_fraction": 0.80,
                "robustness_candidate_cap": 16,
                "collision_candidate_cap": 32,
                "margin_thresholds": [0.1],
                "expert_confidence_thresholds": [0.5],
                "expert_weights": [0.3],
                "minimum_directed_errors": 1,
                "minimum_rescues": 1,
                "robustness_constraints": {
                    "anchor_role": "ROBUSTNESS_ANCHOR",
                    "dimensions": ["burden", "novelty"],
                    "evidence_source": "inner_fold_train_only",
                    "collapse_tolerance": 0.05,
                    "maximum_repeated_collapse_folds": 1,
                },
            },
        },
    )
    fold_path = tmp_path / "folds.csv"
    fold_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "ID": train.ID,
                    "row_index": np.arange(rows),
                    "seed": seed,
                    "fold": np.arange(rows) % 5,
                }
            )
            for seed in SEEDS
        ],
        ignore_index=True,
    )
    # Deliberately reorder file rows. Validation must sort by row_index per seed.
    fold_frame.sample(frac=1.0, random_state=11).to_csv(fold_path, index=False)
    return train, raw, goal_path, universe_path, fold_path


def _predictor(calls, protected_path: Path | None = None):
    attempted_protected_read = False

    def predict(candidate, train_x, train_y, valid_x, class_names, model_seed, context):
        nonlocal attempted_protected_read
        del train_y
        if protected_path is not None and not attempted_protected_read:
            attempted_protected_read = True
            with pytest.raises(RuntimeError, match="runtime protected read blocked"):
                protected_path.read_text(encoding="utf-8")
        calls.append(
            {
                "alias": candidate.alias,
                "stage": context["stage"],
                "seed": context["seed"],
                "model_seed": model_seed,
                "outer_fold": context["outer_fold"],
                "inner_fold": context["inner_fold"],
                "train": tuple(context["train_row_indices"]),
                "valid": tuple(context["valid_row_indices"]),
                "train_size": len(train_x),
                "valid_size": len(valid_x),
            }
        )
        truth = valid_x["signal"].to_numpy(dtype=int)
        probability = np.full((len(valid_x), len(class_names)), 1e-5)
        probability[np.arange(len(valid_x)), truth] = 0.98
        if candidate.alias == "backbone":
            mask = truth == 0
            probability[mask, 0] = 0.48
            probability[mask, 1] = 0.51
        elif candidate.alias == "robust":
            mask = truth == 0
            probability[mask, 0] = 0.48
            probability[mask, 1] = 0.51
        probability /= probability.sum(axis=1, keepdims=True)
        return probability

    return predict


def test_complete_nested_selection_refit_seed_and_actual_io_audit(
    synthetic_nested, tmp_path
):
    train, raw, goal_path, universe_path, fold_path = synthetic_nested
    calls = []
    output = nested.run_nested(
        goal_path,
        universe_path,
        fold_path,
        tmp_path / "artifacts",
        predictor=_predictor(calls, raw / "test.csv"),
    )

    primary = np.load(output / "nested_oof_probability.npy")
    assert primary.shape == (3, len(train), 26)
    np.testing.assert_allclose(primary.sum(axis=2), 1.0, atol=1e-12)
    lanes = np.load(output / "nested_strategy_probability.npz")
    assert set(lanes.files) == set(nested.LANES)
    lanes.close()

    assert len(calls) == 3 * 5 * 3 * 5
    assert all(call["model_seed"] == call["seed"] for call in calls)
    outer_calls = [call for call in calls if call["stage"] == "outer_refit"]
    assert len(outer_calls) == 3 * 5 * 3
    assert all(set(call["train"]).isdisjoint(call["valid"]) for call in calls)
    assert all(call["train_size"] + call["valid_size"] == len(train) for call in outer_calls)

    selections = json.loads(
        (output / "nested_selection_records.json").read_text(encoding="utf-8")
    )["selections"]
    assert len(selections) == 3 * 5 * 3
    specialist = [row for row in selections if row["strategy"] == "specialization_challenger"]
    assert all(row["selected_variant"]["ordered_pair"] == [0, 1] for row in specialist)
    assert all(row["gate_evidence"]["net_rescue"] > 0 for row in specialist)
    assert all(row["outer_validation_used_for_selection"] is False for row in selections)

    audit = json.loads((output / "nested_io_audit.json").read_text(encoding="utf-8"))
    roles = [event["role"] for event in audit["events"]]
    assert roles.count("train") == 1
    assert roles.count("fold_assignments") == 1
    assert roles.count("search_universe") == 1
    assert audit["test_reads"] == audit["submission_reads"] == 0
    assert audit["denied_runtime_protected_reads"] == 1
    assert any(event["action"] == "denied_read" for event in audit["events"])
    exclusion_events = [event for event in audit["events"] if event["action"] == "excluded"]
    assert exclusion_events == []
    manifest = json.loads((output / "nested_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["search_universe_provenance"]["label_independent"] is True
    assert manifest["preselected_structure_input_rejected"] is True
    seed_rows = manifest["model_seed_matrix"]
    inner_seed_rows = [row for row in seed_rows if row["stage"] == "inner"]
    assert len(inner_seed_rows) == 3 * 5 * 4 * 3
    assert all(row["model_seed"] == row["seed"] for row in seed_rows)
    assert all(
        row["inner_split_seed"] == nested._inner_split_seed(row["seed"], row["outer_fold"])
        for row in seed_rows
    )
    assert manifest["model_seed_matrix_contract"]["stage_alias_outer_inner_coverage_complete"]
    contract = manifest["bounded_search_contract"]
    assert contract["trial_count_per_seed"] == 20000
    assert contract["role_balanced_fraction"] == 0.8
    assert contract["peak_materialized_base_variants"] == 1
    for row in manifest["search_diagnostics_by_outer_fold"]:
        assert row["base"]["peak_materialized_base_variant"] == 1
        assert row["base"]["peak_retained_robustness_variants"] <= 16
        assert row["collision"]["evaluated"] <= 32
    transcript_result = nested.validate_inner_selection_transcript(output, train)
    assert transcript_result == {"verified_records": 45, "exact_key_coverage": True}
    transcript = json.loads(
        (output / "nested_inner_selection_records.json").read_text(encoding="utf-8")
    )
    assert len(transcript["records"]) == 45
    assert manifest["inner_selection_transcript"]["key_count"] == 45
    assert "nested_inner_selection_probability.npz" in manifest["artifact_hashes"]
    assert "nested_inner_selection_records.json" in manifest["artifact_hashes"]
    assert manifest["trusted_input_hashes"]["goal_config"]
    assert manifest["trusted_input_hashes"]["search_universe"]
    assert manifest["trusted_input_hashes"]["source_bundle"]
    assert manifest["runtime_execution_safety"]["runtime_exclusions"] == {}
    assert manifest["runtime_execution_safety"]["excluded_aliases_never_fit"] is True
    assert manifest["source_unchanged"] is True
    assert manifest["source_bundle_before"] == manifest["source_bundle_after"]
    assert manifest["run_identity_payload"]["inner_selection_array_hashes"]
    assert manifest["io_audit"]["events"] == audit["events"]
    assert any(event["role"].startswith("source:") for event in audit["events"])

    # Array tamper, score forgery, and coverage forgery must each be detected.
    probability_path = output / "nested_inner_selection_probability.npz"
    with np.load(probability_path, allow_pickle=False) as archive:
        original_arrays = {key: archive[key].copy() for key in archive.files}
    first_key = transcript["records"][0]["array_key"]
    tampered_arrays = {key: value.copy() for key, value in original_arrays.items()}
    tampered_arrays[first_key][0] = np.roll(tampered_arrays[first_key][0], 1)
    np.savez_compressed(probability_path, **tampered_arrays)
    with pytest.raises(ValueError, match="array hash mismatch"):
        nested.validate_inner_selection_transcript(output, train)
    np.savez_compressed(probability_path, **original_arrays)

    transcript_path = output / "nested_inner_selection_records.json"
    original_transcript = transcript_path.read_text(encoding="utf-8")
    forged = json.loads(original_transcript)
    forged["records"][0]["inner_selection_macro_f1"] += 0.1
    transcript_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="Macro F1 transcript mismatch"):
        nested.validate_inner_selection_transcript(output, train)
    transcript_path.write_text(original_transcript, encoding="utf-8")

    forged = json.loads(original_transcript)
    forged["records"][0]["inner_oof_coverage"][0] = 0
    transcript_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage must equal one"):
        nested.validate_inner_selection_transcript(output, train)
    transcript_path.write_text(original_transcript, encoding="utf-8")

    forged = json.loads(original_transcript)
    forged["records"].pop()
    transcript_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 45 records"):
        nested.validate_inner_selection_transcript(output, train)
    transcript_path.write_text(original_transcript, encoding="utf-8")

    forged = json.loads(original_transcript)
    record = forged["records"][0]
    outer_valid_row = next(index for index in range(len(train)) if index not in record["row_indices"])
    record["row_indices"][0] = outer_valid_row
    transcript_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="exact ordered outer-train complement"):
        nested.validate_inner_selection_transcript(output, train)
    transcript_path.write_text(original_transcript, encoding="utf-8")

    forged = json.loads(original_transcript)
    fold_ids = forged["records"][0]["inner_fold_ids"]
    left = fold_ids.index(0)
    right = fold_ids.index(1)
    fold_ids[left], fold_ids[right] = fold_ids[right], fold_ids[left]
    transcript_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic live split"):
        nested.validate_inner_selection_transcript(output, train)
    transcript_path.write_text(original_transcript, encoding="utf-8")

    forged = json.loads(original_transcript)
    forged["records"][0]["strategy"] = "robustness_challenger"
    transcript_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        nested.validate_inner_selection_transcript(output, train)
    transcript_path.write_text(original_transcript, encoding="utf-8")

    live_fold_copy = output / "tampered_folds.csv"
    live_folds = pd.read_csv(fold_path)
    live_folds.loc[0, "fold"] = (int(live_folds.loc[0, "fold"]) + 1) % 5
    live_folds.to_csv(live_fold_copy, index=False)
    with pytest.raises(ValueError, match="live fold assignment hash"):
        nested.validate_inner_selection_transcript(
            output, train, fold_assignments_path=live_fold_copy
        )


def test_rejects_preselected_top3_before_train_read(synthetic_nested, tmp_path, monkeypatch):
    _, _, goal_path, universe_path, fold_path = synthetic_nested
    _write_yaml(
        universe_path,
        {
            "primary_strategy": "leaked",
            "strategies": [{"name": "leaked", "models": ["expert"]}],
            "provenance": {
                "kind": "predeclared_search_universe",
                "label_independent": True,
                "global_oof_selection_used": False,
            },
        },
    )
    reads = []
    original = Path.read_bytes

    def tracked(path):
        reads.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    with pytest.raises(ValueError, match="preselected/top-3"):
        nested.run_nested(goal_path, universe_path, fold_path, tmp_path / "out")
    assert "train.csv" not in reads


def test_rejects_nested_preselection_key_and_candidate_pruning(synthetic_nested):
    _, _, goal_path, universe_path, _ = synthetic_nested
    goal = yaml.safe_load(goal_path.read_text(encoding="utf-8"))
    candidates = _loaded_candidates(goal)
    raw = yaml.safe_load(universe_path.read_text(encoding="utf-8"))
    raw["search_universe"]["constraints"] = {"selected_models": ["expert"]}
    with pytest.raises(ValueError, match="preselected/top-3"):
        nested._load_universe(raw, candidates, goal["search"])
    raw = yaml.safe_load(universe_path.read_text(encoding="utf-8"))
    raw["search_universe"]["models"] = ["backbone", "expert"]
    with pytest.raises(ValueError, match="every goal candidate"):
        nested._load_universe(raw, candidates, goal["search"])


@pytest.mark.parametrize("aliased_role", ["train", "fold", "universe"])
def test_direct_test_alias_is_blocked_before_test_bytes(
    synthetic_nested, tmp_path, monkeypatch, aliased_role
):
    _, raw, goal_path, universe_path, fold_path = synthetic_nested
    if aliased_role == "train":
        goal = yaml.safe_load(goal_path.read_text(encoding="utf-8"))
        goal["data"]["train_file"] = "test.csv"
        _write_yaml(goal_path, goal)
    elif aliased_role == "fold":
        fold_path = raw / "test.csv"
    else:
        universe_path = raw / "test.csv"
    test_reads = 0
    original = Path.read_bytes

    def tracked(path):
        nonlocal test_reads
        if Path(path).resolve() == (raw / "test.csv").resolve():
            test_reads += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    with pytest.raises(ValueError, match="alias blocked"):
        nested.run_nested(goal_path, universe_path, fold_path, tmp_path / "out")
    assert test_reads == 0


def test_hardlink_and_symlink_test_aliases_are_blocked(synthetic_nested, tmp_path):
    _, raw, goal_path, universe_path, _ = synthetic_nested
    hardlink = tmp_path / "fold-hardlink.csv"
    os.link(raw / "test.csv", hardlink)
    with pytest.raises(ValueError, match="alias blocked"):
        nested.run_nested(goal_path, universe_path, hardlink, tmp_path / "hard-out")

    symlink = tmp_path / "fold-symlink.csv"
    try:
        symlink.symlink_to(raw / "test.csv")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="alias blocked"):
        nested.run_nested(goal_path, universe_path, symlink, tmp_path / "sym-out")


def test_fold_id_swap_rejected_after_row_index_sort(synthetic_nested, tmp_path):
    _, _, goal_path, universe_path, fold_path = synthetic_nested
    frame = pd.read_csv(fold_path)
    seed_rows = frame.index[frame.seed == 42][:2]
    frame.loc[seed_rows, "ID"] = frame.loc[seed_rows[::-1], "ID"].to_numpy()
    frame.to_csv(fold_path, index=False)
    with pytest.raises(ValueError, match="ID does not match Train"):
        nested.run_nested(
            goal_path, universe_path, fold_path, tmp_path / "out", predictor=_predictor([])
        )


def test_ordered_gate_does_not_merge_reverse_direction():
    base = np.full((2, 26), 1e-6)
    expert = np.full((2, 26), 1e-6)
    base[:, 0], base[:, 1] = 0.48, 0.51
    expert[:, 0] = 0.98
    base /= base.sum(axis=1, keepdims=True)
    expert /= expert.sum(axis=1, keepdims=True)
    common = {
        "kind": "collision_gate",
        "backbone_variant": {
            "kind": "blend",
            "models": ("base", "base2"),
            "weights": (0.5, 0.5),
            "temperatures": (1.0, 1.0),
        },
        "expert": "expert",
        "expert_temperature": 1.0,
        "margin_threshold": 0.1,
        "expert_confidence_threshold": 0.5,
        "expert_weight": 0.3,
        "global_weight": 0.7,
    }
    probabilities = {"base": base, "base2": base, "expert": expert}
    forward = nested._apply_gate({**common, "ordered_pair": (0, 1)}, probabilities)
    reverse = nested._apply_gate({**common, "ordered_pair": (1, 0)}, probabilities)
    assert not np.allclose(forward, base)
    np.testing.assert_allclose(reverse, base, atol=1e-12)


def test_deterministic_results(synthetic_nested, tmp_path):
    _, _, goal_path, universe_path, fold_path = synthetic_nested
    first = nested.run_nested(
        goal_path, universe_path, fold_path, tmp_path / "first", predictor=_predictor([])
    )
    second = nested.run_nested(
        goal_path, universe_path, fold_path, tmp_path / "second", predictor=_predictor([])
    )
    np.testing.assert_array_equal(
        np.load(first / "nested_oof_probability.npy"),
        np.load(second / "nested_oof_probability.npy"),
    )
    assert (first / "nested_selection_records.json").read_bytes() == (
        second / "nested_selection_records.json"
    ).read_bytes()


def test_cli_accepts_search_universe_alias():
    args = nested.parse_args(
        [
            "--config",
            "goal.yaml",
            "--search-universe",
            "universe.yaml",
            "--fold-assignments",
            "folds.csv",
        ]
    )
    assert args.strategies == Path("universe.yaml")


def test_actual_goal_and_all_registered_builders_dry_validate_without_data_read(monkeypatch):
    goal = (
        nested._repo_root()
        / "data"
        / "backup"
        / "yaml"
        / "test_007_specialization_goal.yaml"
    )
    goal_config = yaml.safe_load(goal.read_text(encoding="utf-8"))
    expected_candidate_count = len(goal_config["candidates"])

    def forbidden_csv(*args, **kwargs):
        raise AssertionError("dry validation must not read Train/Test CSV")

    monkeypatch.setattr(nested.pd, "read_csv", forbidden_csv)
    result = nested.dry_validate_goal(goal)
    assert result["candidate_count"] == expected_candidate_count
    assert result["builders_instantiated"] == expected_candidate_count
    assert len(result["registered_model_builders"]) == expected_candidate_count
    assert result["data_csv_reads"] == 0
    assert result["common_data_paths"]["test"].endswith("data\\raw\\test.csv")


@pytest.mark.parametrize("alias_kind", ["direct", "hardlink", "symlink"])
def test_goal_config_alias_blocked_before_config_reader(
    synthetic_nested, tmp_path, monkeypatch, alias_kind
):
    _, raw, _, _, _ = synthetic_nested
    canonical_test = nested._repo_root() / "data" / "raw" / "test.csv"
    goal_alias = canonical_test
    if alias_kind == "hardlink":
        goal_alias = tmp_path / "goal-hardlink.yaml"
        os.link(canonical_test, goal_alias)
    elif alias_kind == "symlink":
        goal_alias = tmp_path / "goal-symlink.yaml"
        try:
            goal_alias.symlink_to(canonical_test)
        except OSError:
            pytest.skip("symlink creation is unavailable")
    reads = 0
    original = Path.read_bytes

    def tracked(path):
        nonlocal reads
        if Path(path).resolve() == canonical_test.resolve():
            reads += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    with pytest.raises(ValueError, match="goal config alias blocked"):
        nested.dry_validate_goal(goal_alias)
    assert reads == 0


def test_robustness_role_and_fold_train_evidence_are_mandatory(synthetic_nested):
    _, _, goal_path, universe_path, _ = synthetic_nested
    goal = yaml.safe_load(goal_path.read_text(encoding="utf-8"))
    candidates = _loaded_candidates(goal)
    raw = yaml.safe_load(universe_path.read_text(encoding="utf-8"))
    del raw["search_universe"]["robustness_constraints"]
    with pytest.raises(ValueError, match="robustness_constraints"):
        nested._load_universe(raw, candidates, goal["search"])
    raw = yaml.safe_load(universe_path.read_text(encoding="utf-8"))
    raw["search_universe"]["roles"]["robust"] = []
    with pytest.raises(ValueError, match="ROBUSTNESS_ANCHOR"):
        nested._load_universe(raw, candidates, goal["search"])


def test_fold_safe_group_thresholds_do_not_depend_on_validation_values():
    train = pd.DataFrame({"G1": ["WT", "a", "a", "b"], "G2": ["WT", "WT", "c", "WT"]})
    first = pd.DataFrame({"G1": ["new", "WT"], "G2": ["new2", "WT"]})
    second = pd.DataFrame({"G1": ["WT", "WT"], "G2": ["WT", "WT"]})
    _, first_thresholds = nested._fold_safe_groups(train, first)
    _, second_thresholds = nested._fold_safe_groups(train, second)
    assert first_thresholds == second_thresholds


def test_modelwise_temperature_cartesian_and_global_blend_gate(synthetic_nested):
    _, _, goal_path, universe_path, _ = synthetic_nested
    goal = yaml.safe_load(goal_path.read_text(encoding="utf-8"))
    candidates = _loaded_candidates(goal)
    raw = yaml.safe_load(universe_path.read_text(encoding="utf-8"))
    raw["search_universe"]["temperature_values"] = [0.8, 1.2]
    universe = nested._load_universe(raw, candidates, goal["search"])
    rng = np.random.default_rng(1001)
    observed = {
        tuple(nested._random_base_variant(universe, rng, False)["temperatures"])
        for _ in range(200)
    }
    assert (0.8, 1.2) in observed
    assert (1.2, 0.8) in observed


def test_role_balanced_subset_counts_dual_role_global_as_anchor():
    aliases = ("dual", "anchor", "neutral", "expert")
    roles = {
        "dual": ("GLOBAL_BACKBONE", "ROBUSTNESS_ANCHOR"),
        "anchor": ("ROBUSTNESS_ANCHOR",),
        "neutral": (),
        "expert": ("COLLISION_EXPERT",),
    }
    universe = nested.SearchUniverse(
        aliases=aliases,
        executable_aliases=aliases,
        runtime_exclusions={},
        roles=roles,
        allowed_kinds=("single", "blend", "collision_gate"),
        max_models=3,
        temperatures=(0.85, 1.0, 1.15),
        trial_seeds=(1001,),
        trial_count_per_seed=20000,
        min_active_weight=0.05,
        max_model_weight=0.80,
        early_stop_patience=5000,
        role_balanced_fraction=0.80,
        robustness_candidate_cap=128,
        collision_candidate_cap=5000,
        margin_thresholds=(0.05, 0.10),
        expert_confidence_thresholds=(0.5,),
        expert_weights=(0.1, 0.2, 0.3),
        minimum_directed_errors=1,
        minimum_rescues=1,
        collapse_tolerance=0.05,
        maximum_repeated_collapse_folds=1,
        provenance={"label_independent": True},
    )
    for size in (2, 3):
        models = nested._role_balanced_subset(
            universe, size, np.random.default_rng(0)
        )
        assert len(models) == size
        assert len(set(models)) == size
        assert any("GLOBAL_BACKBONE" in roles[alias] for alias in models)
        assert sum("ROBUSTNESS_ANCHOR" in roles[alias] for alias in models) <= 1
        assert sum("COLLISION_EXPERT" in roles[alias] for alias in models) <= 2


def test_production_role_max5_stream_60k_is_bounded_deterministic_and_exact(
    monkeypatch,
):
    aliases = (
        "lightgbm",
        "tabfm_expert",
        "tabpfn_expert",
        "catboost",
        "tabm",
        "xgboost",
        "tabicl_expert",
        "tabicl_global",
        "realtabr_expert",
        "realmlp",
        "random_forest",
        "extra_trees",
        "balanced_random_forest",
        "modernnca_expert",
        "modernnca_global",
        "logistic_gpu",
        "torch_linear",
        "torch_mlp",
        "xrfm",
    )
    roles = {alias: () for alias in aliases}
    roles.update(
        {
            "lightgbm": ("GLOBAL_BACKBONE",),
            "tabfm_expert": ("COLLISION_EXPERT",),
            "tabpfn_expert": ("COLLISION_EXPERT",),
            "catboost": ("GLOBAL_BACKBONE", "ROBUSTNESS_ANCHOR"),
            "tabm": ("GLOBAL_BACKBONE", "STABILITY_ANCHOR"),
            "xgboost": ("GLOBAL_BACKBONE",),
            "tabicl_expert": ("COLLISION_EXPERT",),
            "tabicl_global": ("GLOBAL_BACKBONE", "STABILITY_ANCHOR"),
            "realtabr_expert": ("COLLISION_EXPERT",),
            "realmlp": ("GLOBAL_BACKBONE",),
            "random_forest": ("ROBUSTNESS_ANCHOR",),
            "extra_trees": ("ROBUSTNESS_ANCHOR",),
            "balanced_random_forest": ("ROBUSTNESS_ANCHOR",),
            "modernnca_expert": ("COLLISION_EXPERT",),
            "modernnca_global": ("GLOBAL_BACKBONE",),
            "logistic_gpu": ("GLOBAL_BACKBONE",),
            "torch_linear": ("GLOBAL_BACKBONE",),
            "torch_mlp": ("GLOBAL_BACKBONE",),
            "xrfm": ("GLOBAL_BACKBONE",),
        }
    )
    universe = nested.SearchUniverse(
        aliases=aliases,
        executable_aliases=aliases,
        runtime_exclusions={},
        roles=roles,
        allowed_kinds=("single", "blend", "collision_gate"),
        max_models=5,
        temperatures=(0.85, 1.0, 1.15),
        trial_seeds=(1001, 1002, 1003),
        trial_count_per_seed=20000,
        min_active_weight=0.05,
        max_model_weight=0.80,
        early_stop_patience=5000,
        role_balanced_fraction=0.80,
        robustness_candidate_cap=128,
        collision_candidate_cap=5000,
        margin_thresholds=(0.05, 0.10),
        expert_confidence_thresholds=(0.5,),
        expert_weights=(0.1, 0.2, 0.3),
        minimum_directed_errors=1,
        minimum_rescues=1,
        collapse_tolerance=0.05,
        maximum_repeated_collapse_folds=1,
        provenance={"label_independent": True},
    )
    for name in ("_read_csv", "_macro_f1", "_apply_base"):
        monkeypatch.setattr(
            nested,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"trial stream must not call {_name}"
            ),
        )

    anchor_roles = {"STABILITY_ANCHOR", "ROBUSTNESS_ANCHOR", "DIVERSITY_MODEL"}
    expert_roles = {"COLLISION_EXPERT", "RARE_CLASS_RESCUER", "NOVELTY_EXPERT"}

    def consume(seed: int) -> tuple[str, int, int, int]:
        digest = hashlib.sha256()
        count = balanced = unconstrained = 0
        for trial, is_balanced, variant in nested._stream_random_variants(universe, seed):
            assert trial == count
            assert 2 <= len(variant["models"]) <= 5
            assert len(set(variant["models"])) == len(variant["models"])
            assert set(variant["models"]).issubset(universe.executable_aliases)
            assert all(0.05 <= weight <= 0.80 for weight in variant["weights"])
            if is_balanced:
                assert any(
                    "GLOBAL_BACKBONE" in roles[alias]
                    for alias in variant["models"]
                )
                assert sum(
                    bool(set(roles[alias]).intersection(anchor_roles))
                    for alias in variant["models"]
                ) <= 1
                assert sum(
                    bool(set(roles[alias]).intersection(expert_roles))
                    for alias in variant["models"]
                ) <= 2
            digest.update(nested._variant_fingerprint(variant).encode())
            count += 1
            balanced += int(is_balanced)
            unconstrained += int(not is_balanced)
        return digest.hexdigest(), count, balanced, unconstrained

    tracemalloc.start()
    first = tuple(consume(seed) for seed in universe.trial_seeds)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    second = tuple(consume(seed) for seed in universe.trial_seeds)
    assert first == second
    assert sum(item[1] for item in first) == 60000
    assert sum(item[2] for item in first) == 48000
    assert sum(item[3] for item in first) == 12000
    assert all(item[1:] == (20000, 16000, 4000) for item in first)
    assert peak < 8 * 1024 * 1024
    assert nested._exhaustive_subset_temperature_upper_bound(universe) > 3_000_000


def test_trial_contract_must_match_goal_search(synthetic_nested):
    _, _, goal_path, universe_path, _ = synthetic_nested
    goal = yaml.safe_load(goal_path.read_text(encoding="utf-8"))
    candidates = _loaded_candidates(goal)
    raw = yaml.safe_load(universe_path.read_text(encoding="utf-8"))
    raw["search_universe"]["trial_count_per_seed"] = 19999
    with pytest.raises(ValueError, match="trial contract"):
        nested._load_universe(raw, candidates, goal["search"])


def test_execution_source_closure_and_bundle_change(synthetic_nested, tmp_path):
    _, _, goal_path, _, _ = synthetic_nested
    goal = yaml.safe_load(goal_path.read_text(encoding="utf-8"))
    audit = nested.Audit()
    candidates, _ = nested._load_candidates(nested._candidate_paths(goal), audit)
    paths = nested._execution_source_paths(candidates)
    repo_root = nested._repo_root()
    relative = {
        path.relative_to(repo_root).as_posix()
        for path in paths
        if path.is_relative_to(repo_root)
    }
    assert "src/models/__init__.py" in relative
    assert "src/models/torch_tabular_base.py" in relative
    assert "src/pipelines/preprocessing_registry.py" in relative
    assert "src/ensembles/common.py" in relative

    dependency = tmp_path / "shared_dependency.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    first = nested._hash_source_paths([dependency], nested.Audit())
    dependency.write_text("VALUE = 2\n", encoding="utf-8")
    second = nested._hash_source_paths([dependency], nested.Audit())
    assert nested._canonical_hash(first) != nested._canonical_hash(second)
    with pytest.raises(RuntimeError, match="execution source drift"):
        nested._assert_source_unchanged([dependency], first, nested.Audit())


def test_runtime_guard_portable_os_open_read_modes(tmp_path, monkeypatch):
    ordinary = tmp_path / "ordinary.bin"
    protected = tmp_path / "test.csv"
    submission = tmp_path / "sample_submission.csv"
    ordinary.write_bytes(b"ordinary")
    protected.write_bytes(b"protected")
    submission.write_bytes(b"submission")
    audit = nested.Audit()
    monkeypatch.delattr(nested.os, "O_ACCMODE", raising=False)

    with nested._runtime_read_guard(protected, submission, audit, "inner", "probe"):
        descriptor = nested.os.open(ordinary, nested.os.O_RDONLY)
        try:
            assert nested.os.read(descriptor, 8) == b"ordinary"
        finally:
            nested.os.close(descriptor)

        with pytest.raises(RuntimeError, match="runtime protected read blocked"):
            nested.os.open(protected, nested.os.O_RDONLY)

        descriptor = nested.os.open(ordinary, nested.os.O_RDWR)
        nested.os.close(descriptor)
        with pytest.raises(RuntimeError, match="runtime protected read blocked"):
            nested.os.open(protected, nested.os.O_RDWR)

    denied = [event for event in audit.events if event["action"] == "denied_read"]
    assert len(denied) == 2
    assert all(event["role"] == "test" for event in denied)
