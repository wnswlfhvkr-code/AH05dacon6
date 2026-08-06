from __future__ import annotations

import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
import yaml

import src.test_007.analyze_test_007_specialization as analysis


CLASSES = (
    "ACC", "BLCA", "BRCA", "CESC", "COAD", "DLBC", "GBMLGG", "HNSC",
    "KIPAN", "KIRC", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "OV",
    "PAAD", "PCPG", "PRAD", "SARC", "SKCM", "STES", "TGCT", "THCA",
    "THYM", "UCEC",
)
ALIASES = ("leader", "robust", "specialist")
SEEDS = (42, 2026, 777)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _replace_probability(run_dir: Path, probability: np.ndarray) -> None:
    probability_path = run_dir / "oof_probability_by_seed.npy"
    np.save(probability_path, probability[None])
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["oof_probability_by_seed.npy"] = {
        "sha256": _sha(probability_path),
        "size_bytes": probability_path.stat().st_size,
    }
    _write_json(manifest_path, manifest)


def _read_trial_chunks(output: Path, manifest: dict) -> pd.DataFrame:
    frames = []
    for chunk in manifest["chunks"]:
        path = output / chunk["path"]
        assert _sha(path) == chunk["sha256"]
        frame = pd.read_parquet(path) if manifest["format"] == "parquet" else pd.read_csv(path)
        assert len(frame) == chunk["rows"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _scalar_search_reference(bank_root: Path, train: pd.DataFrame) -> pd.DataFrame:
    y = train.SUBCLASS.map({name: index for index, name in enumerate(CLASSES)}).to_numpy()
    probability_by_alias = {
        alias: np.stack([
            np.load(next(bank_root.glob(f"exp_{alias}/*/{seed}/oof_probability_by_seed.npy")))[0]
            for seed in SEEDS
        ])
        for alias in ALIASES
    }
    fold_by_seed = np.stack([np.arange(len(y)) % 5] * 3)
    y_grid = np.broadcast_to(y, fold_by_seed.shape)
    rows, trial_id = [], 0
    for seed_index, held_seed in enumerate(SEEDS):
        for held_fold in range(5):
            selection_mask, _, _ = analysis._holdout_masks(fold_by_seed, seed_index, held_fold)
            role_map = analysis._derive_selection_role_map(
                list(ALIASES), probability_by_alias, y_grid, selection_mask
            )
            best = None
            for random_seed in (1001, 1002, 1003):
                rng = np.random.default_rng(
                    np.random.SeedSequence([random_seed, held_seed, held_fold])
                )
                for local_id in range(5):
                    constrained = local_id % 5 != 4
                    selected = analysis._draw_model_trial(
                        rng, list(ALIASES), role_map, 3, constrained
                    )
                    weight = analysis._draw_constrained_weights(rng, len(selected), 0.05, 0.80)
                    probability = np.tensordot(
                        weight,
                        np.stack([probability_by_alias[alias] for alias in selected]),
                        axes=(0, 0),
                    )
                    score = analysis._masked_macro_f1(y_grid, probability, selection_mask)
                    row = {
                        "trial_id": trial_id,
                        "holdout_id": f"{held_seed}:{held_fold}",
                        "models": "+".join(selected),
                        "model_weights": json.dumps(
                            dict(zip(selected, map(float, weight))), sort_keys=True
                        ),
                        "selection_macro_f1": score,
                    }
                    if best is None or score > best["selection_macro_f1"]:
                        best = row
                    trial_id += 1
            rows.append(best)
    return pd.DataFrame(rows)


def _scalar_bootstrap_reference(y, candidate, baseline, repeats, seed, classes):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == value) for value in np.unique(y)]
    deltas = []
    for _ in range(repeats):
        sample = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in groups
        ])
        deltas.append(
            analysis._macro_f1(y[sample], candidate[sample], classes)
            - analysis._macro_f1(y[sample], baseline[sample], classes)
        )
    return (
        float(np.std(deltas, ddof=1)),
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
    )


def _probabilities(y: np.ndarray, alias: str, seed: int) -> np.ndarray:
    probability = np.full((len(y), 26), 0.1 / 25, dtype=np.float64)
    prediction = y.copy()
    if alias == "leader":
        collision = np.flatnonzero(y == CLASSES.index("GBMLGG"))
        prediction[collision] = CLASSES.index("LGG")
    elif alias == "robust":
        prediction[np.arange(len(y)) % 11 == 0] = (prediction[np.arange(len(y)) % 11 == 0] + 1) % 26
    elif alias == "specialist":
        prediction[np.arange(len(y)) % 7 == 0] = (prediction[np.arange(len(y)) % 7 == 0] + 2) % 26
        for index, value in enumerate(y):
            if CLASSES[value] in {"GBMLGG", "LGG", "KIPAN", "KIRC"}:
                prediction[index] = value
    probability[np.arange(len(y)), prediction] = 0.9
    probability += (seed % 5) * 1e-7
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def _write_bank_run(
    bank_root: Path,
    candidate_config: Path,
    alias: str,
    seed: int,
    train: pd.DataFrame,
) -> Path:
    config = yaml.safe_load(candidate_config.read_text(encoding="utf-8"))
    config_hash = _sha(candidate_config)
    run = bank_root / config["project"]["experiment_name"] / config_hash / str(seed)
    run.mkdir(parents=True)
    y = train["SUBCLASS"].map({name: index for index, name in enumerate(CLASSES)}).to_numpy()
    probability = _probabilities(y, alias, seed)
    np.save(run / "oof_probability_by_seed.npy", probability[None])
    pd.DataFrame(
        {
            "row_index": np.arange(len(train)),
            "ID": train.ID,
            "seed": seed,
            "fold": np.arange(len(train)) % 5,
        }
    ).to_csv(run / "fold_assignments.csv", index=False)
    _write_json(run / "class_names.json", list(CLASSES))
    _write_json(
        run / "model_passport.json",
        {"probability_mode": "native_proba", "oof_macro_f1": 0.5},
    )
    _write_json(
        run / "runtime_metrics.json",
        {"fit_seconds": 1.0, "inference_seconds": 0.1, "total_seconds": 1.1},
    )
    pd.DataFrame(
        {
            "seed": seed,
            "fold": np.arange(5),
            "train_f1": np.linspace(0.7, 0.74, 5),
            "valid_f1": np.linspace(0.5, 0.54, 5),
        }
    ).to_csv(run / "fold_metrics.csv", index=False)
    _write_json(run / "io_audit.json", {"phase": "oof", "test_accessed": False})
    artifacts = {}
    for name in (
        "oof_probability_by_seed.npy",
        "fold_assignments.csv",
        "class_names.json",
        "model_passport.json",
        "runtime_metrics.json",
        "fold_metrics.csv",
        "io_audit.json",
    ):
        path = run / name
        artifacts[name] = {"sha256": _sha(path), "size_bytes": path.stat().st_size}
    protected = {"src/train.py": "a" * 64}
    _write_json(
        run / "run_manifest.json",
        {
            "schema_version": 1,
            "config_hash": config_hash,
            "seed": seed,
            "train_rows": len(train),
            "test_accessed": False,
            "score_source": "train_only_outer_oof",
            "probability_mode": "native_proba",
            "class_names": list(CLASSES),
            "protected_hashes_before": protected,
            "protected_hashes_after": protected,
            "artifacts": artifacts,
        },
    )
    return run


@pytest.fixture()
def synthetic_project(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = 104
    train = pd.DataFrame(
        {
            "ID": [f"R{index:03d}" for index in range(rows)],
            "G1": ["WT" if index % 3 else f"p.A{index}V" for index in range(rows)],
            "G2": ["synonymous" if index % 5 == 0 else "WT" for index in range(rows)],
            "SUBCLASS": np.resize(np.asarray(CLASSES), rows),
        }
    )
    train.to_csv(raw / "train.csv", index=False)
    (raw / "test.csv").write_text("must never be read", encoding="utf-8")
    (raw / "sample_submission.csv").write_text("must never be read", encoding="utf-8")
    bank_root = tmp_path / "bank"
    artifact_root = tmp_path / "artifacts"
    candidate_entries = []
    roles = ("GLOBAL_BACKBONE", "ROBUSTNESS_ANCHOR", "COLLISION_EXPERT")
    for alias, role in zip(ALIASES, roles):
        candidate_config = tmp_path / f"{alias}.yaml"
        candidate_config.write_text(
            yaml.safe_dump(
                {
                    "project": {"experiment_name": f"exp_{alias}", "seed": 42},
                    "model": {"name": alias, "class_names": list(CLASSES)},
                    "preprocessing": {"name": "pipeComb_v3"},
                }
            ),
            encoding="utf-8",
        )
        candidate_entries.append(
            {
                "alias": alias,
                "config": str(candidate_config),
                "family": f"family_{alias}",
                "expected_role": role,
            }
        )
        for seed in SEEDS:
            _write_bank_run(bank_root, candidate_config, alias, seed, train)
    config = {
        "project": {"experiment_name": "synthetic"},
        "data": {
            "raw_dir": str(raw),
            "train_file": "train.csv",
            "target_column": "SUBCLASS",
            "id_column": "ID",
            "artifact_root": str(artifact_root),
            "oof_bank_root": str(bank_root),
        },
        "validation": {
            "screening_seed": 42,
            "final_seeds": list(SEEDS),
            "outer_folds": 5,
            "bootstrap_repeats": 24,
            "bootstrap_seed": 730,
        },
        "screening": {
            "global_f1_within_best": 1.0,
            "blend_gain_minimum": -1.0,
            "unique_rescue_minimum": 0,
            "specialist_delta_minimum": -1.0,
        },
        "candidates": candidate_entries,
        "subgroups": {
            "burden_quantiles": [0.25, 0.75],
            "functional_ratio_quantiles": [0.25, 0.75],
            "novelty_quantiles": [0.25, 0.75],
            "confidence_bins": [0, 0.4, 0.6, 0.8, 1.0],
        },
        "search": {
            "random_trials": 5,
            "random_seeds": [1001, 1002, 1003],
            "min_active_weight": 0.05,
            "max_model_weight": 0.80,
            "max_models": 3,
            "early_stop_patience": 50,
        },
    }
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, artifact_root, bank_root


def test_screen_calculates_metrics_alignment_and_all_equal_blends(synthetic_project):
    config_path, output_root, _ = synthetic_project
    assert analysis.main(["--config", str(config_path), "--stage", "screen"]) == 0
    output = output_root / "screen"
    leaderboard = pd.read_csv(output / "model_leaderboard.csv")
    classes = pd.read_csv(output / "class_specialization.csv")
    pairs = pd.read_csv(output / "equal_pair_blends.csv")
    triples = pd.read_csv(output / "equal_triple_blends.csv")
    calibration = pd.read_csv(output / "calibration_metrics.csv")
    assert len(leaderboard) == len(ALIASES)
    assert len(classes) == len(ALIASES) * 26
    assert len(pairs) == len(list(combinations(ALIASES, 2)))
    assert len(triples) == len(list(combinations(ALIASES, 3)))
    assert np.isfinite(calibration[["logloss", "brier", "ece", "high_confidence_error"]]).all().all()
    audit = json.loads((output / "io_audit.json").read_text(encoding="utf-8"))
    assert audit["summary"]["test_reads"] == 0
    assert audit["summary"]["submission_reads"] == 0


def test_class_order_mismatch_is_rejected(synthetic_project):
    config_path, _, bank_root = synthetic_project
    run_manifest = next(bank_root.glob("exp_robust/*/42/run_manifest.json"))
    run_dir = run_manifest.parent
    classes = list(CLASSES)
    classes[0], classes[1] = classes[1], classes[0]
    _write_json(run_dir / "class_names.json", classes)
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    manifest["class_names"] = classes
    artifact = run_dir / "class_names.json"
    manifest["artifacts"]["class_names.json"] = {
        "sha256": _sha(artifact),
        "size_bytes": artifact.stat().st_size,
    }
    _write_json(run_manifest, manifest)
    with pytest.raises(ValueError, match="class order differs"):
        analysis.main(["--config", str(config_path), "--stage", "screen"])


def test_test_file_alias_is_blocked_before_read(synthetic_project):
    config_path, _, _ = synthetic_project
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["train_file"] = "test.csv"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden train_file"):
        analysis.main(["--config", str(config_path), "--stage", "screen"])


def test_config_hardlink_to_test_is_blocked_before_reader_call(synthetic_project, monkeypatch):
    config_path, _, _ = synthetic_project
    test_path = config_path.parent / "raw" / "test.csv"
    alias = config_path.parent / "goal.yaml"
    os.link(test_path, alias)
    reads = []
    original = Path.read_text

    def tracked(path, *args, **kwargs):
        reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked)
    with pytest.raises(RuntimeError, match="analysis config alias"):
        analysis.main(["--config", str(alias), "--stage", "screen"])
    assert reads == []


def test_train_hardlink_to_test_is_blocked_before_csv_reader(synthetic_project, monkeypatch):
    config_path, _, _ = synthetic_project
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    alias = config_path.parent / "raw" / "train_alias.csv"
    os.link(config_path.parent / "raw" / "test.csv", alias)
    config["data"]["train_file"] = alias.name
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    reads = []
    original = pd.read_csv

    def tracked(path, *args, **kwargs):
        reads.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", tracked)
    with pytest.raises(RuntimeError, match="train_file alias"):
        analysis.main(["--config", str(config_path), "--stage", "screen"])
    assert reads == []


def test_candidate_config_hardlink_to_test_is_blocked_before_read(synthetic_project, monkeypatch):
    config_path, _, _ = synthetic_project
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    alias = config_path.parent / "candidate_alias.yaml"
    os.link(config_path.parent / "raw" / "test.csv", alias)
    config["candidates"][0]["config"] = str(alias)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    reads = []
    original = Path.read_text

    def tracked(path, *args, **kwargs):
        reads.append(Path(path).resolve())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked)
    with pytest.raises(RuntimeError, match="candidate_config_presence"):
        analysis.main(["--config", str(config_path), "--stage", "screen"])
    assert alias.resolve() not in reads


def test_oof_hardlink_to_test_is_blocked_before_numpy_reader(synthetic_project, monkeypatch):
    config_path, _, bank_root = synthetic_project
    run_dir = next(bank_root.glob("exp_leader/*/42"))
    probability = run_dir / "oof_probability_by_seed.npy"
    probability.unlink()
    os.link(config_path.parent / "raw" / "test.csv", probability)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][probability.name] = {
        "sha256": _sha(probability),
        "size_bytes": probability.stat().st_size,
    }
    _write_json(manifest_path, manifest)
    reads = []
    original = np.load

    def tracked(path, *args, **kwargs):
        reads.append(Path(path).resolve())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", tracked)
    with pytest.raises(RuntimeError, match="bank_artifact:oof_probability"):
        analysis.main(["--config", str(config_path), "--stage", "screen"])
    assert reads == []


def test_exact_three_seed_contract_is_required(synthetic_project):
    config_path, _, _ = synthetic_project
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["validation"]["final_seeds"] = [42, 2026]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        analysis.main(["--config", str(config_path), "--stage", "screen"])


def test_screen_rejects_missing_required_seed_bank(synthetic_project):
    config_path, _, bank_root = synthetic_project
    missing = next(bank_root.glob("exp_robust/*/777/run_manifest.json"))
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="robust:777"):
        analysis.main(["--config", str(config_path), "--stage", "screen"])


def test_three_seed_outputs_fold_local_subgroups_and_stability(synthetic_project):
    config_path, output_root, _ = synthetic_project
    analysis.main(["--config", str(config_path), "--stage", "screen"])
    assert analysis.main(["--config", str(config_path), "--stage", "three-seed"]) == 0
    output = output_root / "three-seed"
    classes = pd.read_csv(output / "class_metrics.csv")
    folds = pd.read_csv(output / "fold_metrics.csv")
    thresholds = pd.read_csv(output / "subgroup_thresholds.csv")
    stability = pd.read_csv(output / "seed_stability.csv")
    assert set(classes.seed) == set(SEEDS)
    assert len(folds) == len(ALIASES) * len(SEEDS) * 5
    assert set(thresholds.threshold_fit_scope) == {"outer_train_only"}
    assert set(thresholds.dimension) == {"burden", "functional_ratio", "novelty"}
    assert len(stability) == len(ALIASES)
    assert np.isfinite(stability[["seed_std_macro_f1", "seed_min_macro_f1"]]).all().all()
    role_map = yaml.safe_load((output / "model_role_map.yaml").read_text(encoding="utf-8"))
    assert role_map["evidence_scope"] == "three_seed_train_oof_only"
    assert role_map["expected_role_used"] is False
    for alias in ALIASES:
        evidence = role_map["models"][alias]["evidence"]
        assert {"seed_mean_macro_f1", "seed_min_macro_f1", "fold_std_macro_f1"} <= set(evidence)
        assert evidence["gap_used_for_role_or_objective"] is False
    collision_seed = pd.read_csv(output / "collision_direction_by_seed.csv")
    collision_summary = pd.read_csv(output / "collision_direction_evidence.csv")
    assert set(collision_seed.direction) == {"GBMLGG->LGG"}
    assert set(collision_seed.direction_source) == {"per_seed_train_oof_baseline_confusion"}
    assert collision_seed.fold_evidence.map(json.loads).map(len).eq(5).all()
    specialist = collision_summary.loc[collision_summary.alias == "specialist"].iloc[0]
    assert specialist.positive_net_rescue_seeds == 3
    assert bool(specialist.repeated_direction_2_of_3)
    assert specialist.unique_rescue_count >= 3
    assert specialist.outside_epsilon == max(specialist.outside_se_seed, specialist.outside_se_boot)


def test_collision_outside_noninferiority_uses_all_seeds_and_blocks_repeated_collapse(
    synthetic_project,
):
    config_path, output_root, bank_root = synthetic_project
    train = pd.read_csv(config_path.parent / "raw" / "train.csv")
    y = train.SUBCLASS.map({name: index for index, name in enumerate(CLASSES)}).to_numpy()
    leader_777 = next(bank_root.glob("exp_leader/*/777"))
    _replace_probability(leader_777, _probabilities(y, "perfect", 777))
    pair = np.isin(y, [CLASSES.index("GBMLGG"), CLASSES.index("LGG")])
    for seed in (2026, 777):
        probability = _probabilities(y, "specialist", seed)
        bad_prediction = (y + 1) % len(CLASSES)
        probability[~pair] = 0.1 / 25
        probability[np.flatnonzero(~pair), bad_prediction[~pair]] = 0.9
        probability /= probability.sum(axis=1, keepdims=True)
        _replace_probability(next(bank_root.glob(f"exp_specialist/*/{seed}")), probability)
    analysis.main(["--config", str(config_path), "--stage", "screen"])
    analysis.main(["--config", str(config_path), "--stage", "three-seed"])
    evidence = pd.read_csv(output_root / "three-seed" / "collision_direction_evidence.csv")
    row = evidence.loc[
        (evidence.alias == "specialist") & (evidence.direction == "GBMLGG->LGG")
    ].iloc[0]
    assert row.observed_seed_count == 2
    assert row.outside_evaluated_seed_count == 3
    assert row.outside_collapse_seed_count >= 2
    assert bool(row.repeated_outside_collapse)
    assert not bool(row.eligible_collision_expert)


def test_ordered_collisions_are_derived_from_baseline_oof_confusion():
    y = np.asarray([0, 0, 0, 0, 1, 1, 2])
    prediction = np.asarray([1, 1, 1, 0, 1, 0, 2])
    rows = analysis._ordered_collision_directions(y, prediction, ("A", "B", "C"), 2)
    assert rows == [
        {
            "source": "A",
            "target": "B",
            "source_support": 4,
            "confusion_support": 3,
            "confusion_rate": 0.75,
        }
    ]


def test_dynamic_epsilon_is_maximum_of_seed_and_bootstrap_se():
    y = np.resize(np.arange(3), 60)
    baseline = y.copy()
    baseline[::5] = (baseline[::5] + 1) % 3
    candidate = y.copy()
    candidate[::7] = (candidate[::7] + 1) % 3
    evidence = analysis._paired_evidence(
        [0.01, -0.02, 0.03], y, candidate, baseline, repeats=64, bootstrap_seed=9
    )
    assert evidence["epsilon"] == max(evidence["se_seed"], evidence["se_boot"])
    assert evidence["noninferiority_pass"] == (evidence["delta"] >= -evidence["epsilon"])


def test_vectorized_bootstrap_is_deterministic_shared_row_and_fast():
    rng = np.random.default_rng(19)
    rows, classes = 6201, 26
    y = np.resize(np.arange(classes), rows)
    candidate = np.stack([y.copy() for _ in range(3)])
    baseline = candidate.copy()
    for seed_index in range(3):
        mask = rng.random(rows) < 0.2
        candidate[seed_index, mask] = rng.integers(0, classes, mask.sum())
        mask = rng.random(rows) < 0.22
        baseline[seed_index, mask] = rng.integers(0, classes, mask.sum())
    started = time.perf_counter()
    first = analysis._stratified_paired_bootstrap_se(
        np.tile(y, 3),
        candidate.reshape(-1),
        baseline.reshape(-1),
        repeats=2000,
        seed=730,
        classes=classes,
    )
    elapsed = time.perf_counter() - started
    second = analysis._stratified_paired_bootstrap_se(
        np.tile(y, 3),
        candidate.reshape(-1),
        baseline.reshape(-1),
        repeats=2000,
        seed=730,
        classes=classes,
    )
    assert first == second
    assert elapsed < 0.25
    assert first[1] <= first[2]


def test_vectorized_bootstrap_agrees_with_scalar_reference_and_decision():
    rng = np.random.default_rng(31)
    y = np.resize(np.arange(5), 250)
    baseline = y.copy()
    candidate = y.copy()
    baseline[rng.choice(len(y), 70, replace=False)] = rng.integers(0, 5, 70)
    candidate[rng.choice(len(y), 40, replace=False)] = rng.integers(0, 5, 40)
    vectorized = analysis._stratified_paired_bootstrap_se(
        y, candidate, baseline, repeats=500, seed=88, classes=5
    )
    scalar = _scalar_bootstrap_reference(y, candidate, baseline, 500, 88, 5)
    np.testing.assert_allclose(vectorized, scalar, rtol=0, atol=0.02)
    delta = analysis._macro_f1(y, candidate, 5) - analysis._macro_f1(y, baseline, 5)
    assert (delta >= -vectorized[0]) == (delta >= -scalar[0])
    shared = analysis._stratified_paired_bootstrap_se(
        np.tile(y, 3),
        np.tile(candidate, 3),
        np.tile(baseline, 3),
        repeats=500,
        seed=88,
        classes=5,
    )
    assert shared == vectorized


def test_batched_gemm_scores_match_scalar_and_memory_budget():
    rng = np.random.default_rng(11)
    models, rows, classes, batch = 20, 240, 26, 25
    bank = rng.random((models, rows, classes))
    bank /= bank.sum(axis=2, keepdims=True)
    dense = np.zeros((batch, models), dtype=np.float64)
    for index in range(batch):
        selected = rng.choice(models, size=3, replace=False)
        dense[index, selected] = rng.dirichlet(np.ones(3))
    y = rng.integers(0, classes, rows)
    batched = analysis._score_weight_batch(dense, bank.reshape(models, -1), y, classes)
    scalar = np.asarray([
        analysis._macro_f1(y, (weight @ bank.reshape(models, -1)).reshape(rows, classes).argmax(axis=1))
        for weight in dense
    ])
    np.testing.assert_allclose(batched, scalar, rtol=0, atol=1e-15)
    estimate = analysis._estimated_search_peak_bytes(20, 3, 6201, 26, 256)
    assert estimate < 2 * 1024**3


@pytest.mark.skipif(
    os.environ.get("RUN_SPECIALIZATION_PERF_GATE") != "1",
    reason="opt-in actual-shape performance gate",
)
def test_actual_shape_kernel_benchmark_meets_speed_and_projection_gate():
    benchmark = analysis._benchmark_search_kernel(batch_size=128, repeats=3)
    assert benchmark["scores_equal"]
    assert benchmark["speedup"] >= 3.0
    assert benchmark["projected_900k_minutes"] < 30.0


def test_crossfit_selection_never_uses_held_fold():
    rows = pd.DataFrame(
        [
            {"trial_id": trial, "fold": fold, "macro_f1": trial + fold / 100}
            for trial in range(3)
            for fold in range(5)
        ]
    )
    selected = analysis._select_best_trial(rows, held_fold=3)
    assert 3 not in json.loads(selected.selection_folds)
    assert set(json.loads(selected.selection_folds)) == {0, 1, 2, 4}


def test_dirichlet_search_seed_is_reproducible():
    first = analysis._dirichlet_weights(1001, 20, 3)
    second = analysis._dirichlet_weights(1001, 20, 3)
    third = analysis._dirichlet_weights(1002, 20, 3)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_gap_is_not_search_objective():
    rows = pd.DataFrame(
        [
            {"trial_id": 1, "fold": fold, "macro_f1": 0.6, "overfit_gap": 0.9}
            for fold in range(5)
        ]
        + [
            {"trial_id": 2, "fold": fold, "macro_f1": 0.5, "overfit_gap": 0.0}
            for fold in range(5)
        ]
    )
    selected = analysis._select_best_trial(rows, held_fold=4)
    assert int(selected.trial_id) == 1


def test_search_writes_reproducible_trials_and_crossfit_outputs(synthetic_project):
    config_path, output_root, bank_root = synthetic_project
    analysis.main(["--config", str(config_path), "--stage", "screen"])
    analysis.main(["--config", str(config_path), "--stage", "three-seed"])
    assert analysis.main(["--config", str(config_path), "--stage", "search"]) == 0
    output = output_root / "search"
    manifest = json.loads((output / "random_search_trials_manifest.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(output / "crossfit_selected_trials.csv")
    equal_selected = pd.read_csv(output / "crossfit_equal_selected.csv")
    assert manifest["rows"] == 5 * 3 * 15
    assert manifest["objective_fields"] == ["selection_macro_f1"]
    assert manifest["gap_usage"] == "not_used_in_search_or_promotion"
    assert manifest["subset_size"] == [2, 3]
    assert manifest["early_stop_patience"] == 50
    assert len(selected) == 15
    assert len(equal_selected) == 30
    assert set(selected.holdout_id) == {f"{seed}:{fold}" for seed in SEEDS for fold in range(5)}
    assert not selected.gap_used_in_objective.astype(bool).any()
    assert selected.selection_id_overlap.max() == 0
    assert equal_selected.selection_id_overlap.max() == 0
    assert manifest["held_id_overlap_max"] == 0
    assert manifest["holdout_count"] == 15
    trials = _read_trial_chunks(output, manifest)
    assert "held_macro_f1" not in trials
    assert "held_row_ids" not in trials
    assert trials.model_count.between(2, 3).all()
    assert set(trials.role_constrained.astype(bool)) == {False, True}
    assert trials.role_constrained.astype(bool).mean() == pytest.approx(0.8)
    assert manifest["role_constrained_fraction_actual"] == pytest.approx(0.8)
    assert manifest["metadata_accumulation"] == "streamed_chunks_top20_and_counters_only"
    registry = json.loads((output / "holdout_registry.json").read_text(encoding="utf-8"))
    assert {item["selection_id_overlap"] for item in registry.values()} == {0}
    crossfit_roles = json.loads((output / "crossfit_role_maps.json").read_text(encoding="utf-8"))
    assert set(crossfit_roles) == {f"{seed}:{fold}" for seed in SEEDS for fold in range(5)}
    assert {
        item["evidence_scope"]
        for mapping in crossfit_roles.values()
        for item in mapping.values()
    } == {"holdout_selection_rows_only"}
    for held_seed_index, held_seed in enumerate(SEEDS):
        folds = np.arange(104) % 5
        fold_by_seed = np.stack([folds] * 3)
        for held_fold in range(5):
            selection, _, held_rows = analysis._holdout_masks(
                fold_by_seed, held_seed_index, held_fold
            )
            selection_ids = set(np.flatnonzero(selection.any(axis=0)))
            assert selection_ids.isdisjoint(set(held_rows))
    for row in trials.itertuples():
        weights = json.loads(row.model_weights)
        assert set(weights) == set(row.models.split("+"))
        assert sum(weights.values()) == pytest.approx(1.0)
        if bool(row.eligible):
            assert min(weights.values()) >= 0.05
            assert max(weights.values()) <= 0.80
    promotion = json.loads((output / "promotion_evidence.json").read_text(encoding="utf-8"))
    assert promotion["epsilon"] == max(promotion["se_seed"], promotion["se_boot"])
    assert promotion["gap_used_in_objective"] is False
    scalar = _scalar_search_reference(
        bank_root, pd.read_csv(config_path.parent / "raw" / "train.csv")
    ).sort_values("holdout_id").reset_index(drop=True)
    actual = selected.sort_values("holdout_id").reset_index(drop=True)
    assert actual.trial_id.tolist() == scalar.trial_id.tolist()
    assert actual.models.tolist() == scalar.models.tolist()
    assert actual.model_weights.tolist() == scalar.model_weights.tolist()
    np.testing.assert_allclose(
        actual.selection_macro_f1, scalar.selection_macro_f1, rtol=0, atol=1e-15
    )


def test_random_search_applies_early_stop_patience(synthetic_project, monkeypatch):
    config_path, output_root, _ = synthetic_project
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["search"].update({"random_trials": 20, "early_stop_patience": 3})
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    analysis.main(["--config", str(config_path), "--stage", "screen"])
    analysis.main(["--config", str(config_path), "--stage", "three-seed"])
    monkeypatch.setattr(analysis, "_macro_f1", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(
        analysis,
        "_batch_macro_f1",
        lambda y, predictions, classes=26: np.full(predictions.shape[0], 0.5),
    )
    analysis.main(["--config", str(config_path), "--stage", "search"])
    manifest = json.loads(
        (output_root / "search" / "random_search_trials_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["executed_trials_by_seed"] == {str(seed): 75 for seed in (1001, 1002, 1003)}
    assert set(manifest["executed_trials_by_holdout_search_seed"].values()) == {5}


def test_trial_stream_transaction_cleans_temporary_chunks(synthetic_project, monkeypatch):
    config_path, output_root, _ = synthetic_project
    analysis.main(["--config", str(config_path), "--stage", "screen"])
    analysis.main(["--config", str(config_path), "--stage", "three-seed"])
    original = analysis._TrialChunkStream.write
    calls = 0

    def fail_after_first(self, rows):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("synthetic chunk failure")
        return original(self, rows)

    monkeypatch.setattr(analysis._TrialChunkStream, "write", fail_after_first)
    with pytest.raises(RuntimeError, match="synthetic chunk failure"):
        analysis.main(["--config", str(config_path), "--stage", "search"])
    output = output_root / "search"
    assert not list(output.glob(".trial_chunks.*"))
    assert not (output / "random_search_trials_manifest.json").exists()


def test_corrupt_content_addressed_chunk_is_rehashed_and_repaired(synthetic_project):
    config_path, output_root, _ = synthetic_project
    analysis.main(["--config", str(config_path), "--stage", "screen"])
    analysis.main(["--config", str(config_path), "--stage", "three-seed"])
    analysis.main(["--config", str(config_path), "--stage", "search"])
    output = output_root / "search"
    manifest_path = output / "random_search_trials_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_path = output / manifest["chunks"][0]["path"]
    corrupted = bytearray(chunk_path.read_bytes())
    corrupted[0] ^= 0xFF
    chunk_path.write_bytes(corrupted)
    assert _sha(chunk_path) != manifest["chunks"][0]["sha256"]
    analysis.main(["--config", str(config_path), "--stage", "search"])
    repaired = json.loads(manifest_path.read_text(encoding="utf-8"))
    for chunk in repaired["chunks"]:
        path = output / chunk["path"]
        assert path.stat().st_size == chunk["size_bytes"]
        assert _sha(path) == chunk["sha256"]
    assert not (output / "random_search_trials_generation_invalid.json").exists()
    assert not list(output.glob(".*.quarantine.*"))


def test_failed_corrupt_replacement_preserves_manifest_and_marks_invalidation(
    synthetic_project, monkeypatch
):
    config_path, output_root, _ = synthetic_project
    analysis.main(["--config", str(config_path), "--stage", "screen"])
    analysis.main(["--config", str(config_path), "--stage", "three-seed"])
    analysis.main(["--config", str(config_path), "--stage", "search"])
    output = output_root / "search"
    manifest_path = output / "random_search_trials_manifest.json"
    previous_manifest = manifest_path.read_bytes()
    manifest = json.loads(previous_manifest)
    chunk_path = output / manifest["chunks"][0]["path"]
    corrupted = bytearray(chunk_path.read_bytes())
    corrupted[-1] ^= 0xFF
    chunk_path.write_bytes(corrupted)
    corrupted_bytes = chunk_path.read_bytes()
    original = analysis._TrialChunkStream._verify_directory
    calls = 0

    def fail_replacement(directory, expected):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(directory, expected)
        return False

    monkeypatch.setattr(
        analysis._TrialChunkStream,
        "_verify_directory",
        staticmethod(fail_replacement),
    )
    with pytest.raises(RuntimeError, match="replacement trial chunk"):
        analysis.main(["--config", str(config_path), "--stage", "search"])
    assert manifest_path.read_bytes() == previous_manifest
    assert chunk_path.read_bytes() == corrupted_bytes
    invalidation = json.loads(
        (output / "random_search_trials_generation_invalid.json").read_text(encoding="utf-8")
    )
    assert invalidation["status"] == "INVALIDATED"
    assert invalidation["previous_generation_restored"] is True
    assert not list(output.glob(".trial_chunks.*"))
    assert not list(output.glob(".*.quarantine.*"))


def test_role_constrained_pool_failure_is_explicit():
    mapping = {
        alias: {"role": "GLOBAL_BACKBONE", "roles": ["GLOBAL_BACKBONE"]}
        for alias in ALIASES
    }
    with pytest.raises(ValueError, match="complementary-role pools"):
        analysis._draw_model_trial(
            np.random.default_rng(1), list(ALIASES), mapping, max_models=3, constrained=True
        )
