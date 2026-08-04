"""E8A: E7 확률과 E8 LinearSVC decision score의 cross-fit 보정 앙상블 실행기."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from src.pipelines.pipeline_jh_v04 import (
    build_f0_matrix,
    build_profile_groups,
    make_class_burden_strata,
    parse_wide_mutations,
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_first(directory: Path, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        path = directory / name
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(
        f"{directory}에 필요한 파일이 없습니다: {', '.join(names)}"
    )


def temperature_softmax(decision: np.ndarray, temperature: float) -> np.ndarray:
    scaled = decision / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_score = np.exp(scaled)
    return exp_score / exp_score.sum(axis=1, keepdims=True)


def blend_probabilities(
    e7_probability: np.ndarray,
    e8_probability: np.ndarray,
    e8_weight: float,
) -> np.ndarray:
    blended = (1.0 - e8_weight) * e7_probability + e8_weight * e8_probability
    return blended / blended.sum(axis=1, keepdims=True)


def reconstruct_folds(
    features: pd.DataFrame,
    labels: pd.Series,
    seeds: list[int],
    n_splits: int,
) -> dict[int, np.ndarray]:
    events = parse_wide_mutations(features)
    f0 = build_f0_matrix(features)
    burden = np.asarray(f0.getnnz(axis=1)).ravel()
    groups = build_profile_groups(events, len(features))
    strata, _ = make_class_burden_strata(labels, burden, n_splits)
    result: dict[int, np.ndarray] = {}
    for seed in seeds:
        fold_ids = np.full(len(features), -1, dtype=np.int8)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed,
        )
        for fold, (_, valid_index) in enumerate(
            splitter.split(np.zeros(len(features)), strata, groups)
        ):
            fold_ids[valid_index] = fold
        if (fold_ids < 0).any():
            raise RuntimeError(f"seed={seed} fold 복원에 실패했습니다.")
        result[seed] = fold_ids
    return result


def select_crossfit_parameters(
    y: np.ndarray,
    e7_oof: np.ndarray,
    e8_oof: np.ndarray,
    folds: dict[int, np.ndarray],
    seeds: list[int],
    n_splits: int,
    temperatures: list[float],
    weights: list[float],
    n_classes: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    predictions = np.full((len(seeds), len(y)), -1, dtype=np.int16)
    rows: list[dict] = []
    labels = np.arange(n_classes)
    for seed_position, seed in enumerate(seeds):
        seed_folds = folds[seed]
        for fold in range(n_splits):
            evaluation_mask = seed_folds == fold
            tuning_mask = ~evaluation_mask
            best_score = -np.inf
            best_temperature = None
            best_weight = None
            best_penalty = np.inf
            for temperature in temperatures:
                tuning_e8 = temperature_softmax(
                    e8_oof[seed_position, tuning_mask], temperature,
                )
                for weight in weights:
                    tuning_prediction = blend_probabilities(
                        e7_oof[seed_position, tuning_mask], tuning_e8, weight,
                    ).argmax(axis=1)
                    score = f1_score(
                        y[tuning_mask], tuning_prediction, labels=labels,
                        average="macro", zero_division=0,
                    )
                    penalty = abs(np.log(temperature)) + 0.01 * abs(weight - 0.5)
                    if score > best_score or (
                        np.isclose(score, best_score) and penalty < best_penalty
                    ):
                        best_score = float(score)
                        best_temperature = float(temperature)
                        best_weight = float(weight)
                        best_penalty = float(penalty)
            evaluation_e8 = temperature_softmax(
                e8_oof[seed_position, evaluation_mask], best_temperature,
            )
            evaluation_prediction = blend_probabilities(
                e7_oof[seed_position, evaluation_mask],
                evaluation_e8,
                best_weight,
            ).argmax(axis=1)
            predictions[seed_position, evaluation_mask] = evaluation_prediction
            evaluation_score = f1_score(
                y[evaluation_mask], evaluation_prediction, labels=labels,
                average="macro", zero_division=0,
            )
            rows.append({
                "seed": seed,
                "fold": fold,
                "temperature": best_temperature,
                "e8_weight": best_weight,
                "tuning_macro_f1": best_score,
                "evaluation_macro_f1": float(evaluation_score),
                "evaluation_rows": int(evaluation_mask.sum()),
            })
    if (predictions < 0).any():
        raise RuntimeError("E8A OOF 예측이 완성되지 않았습니다.")
    return predictions, pd.DataFrame(rows)


def paired_bootstrap(
    y: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    repeats: int,
    random_state: int,
    n_classes: int,
) -> dict[str, float | bool]:
    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(y == index) for index in range(n_classes)]
    deltas = np.empty(repeats, dtype=float)
    labels = np.arange(n_classes)
    for repeat in range(repeats):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        seed_deltas = []
        for seed_position in range(candidate.shape[0]):
            candidate_score = f1_score(
                y[sampled], candidate[seed_position, sampled], labels=labels,
                average="macro", zero_division=0,
            )
            baseline_score = f1_score(
                y[sampled], baseline[seed_position, sampled], labels=labels,
                average="macro", zero_division=0,
            )
            seed_deltas.append(candidate_score - baseline_score)
        deltas[repeat] = np.mean(seed_deltas)
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return {
        "mean_delta": float(deltas.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_lower_ge_0": bool(lower >= 0),
        "probability_positive": float(np.mean(deltas > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/test_002_e8a.yaml")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    validation = config["validation"]
    ensemble = config["ensemble"]
    raw_dir = Path(data["raw_dir"])
    e7_dir = Path(data["e7_dir"])
    e8_dir = Path(data["e8_dir"])
    output_dir = Path(data["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(raw_dir / data["train_file"])
    target = data["target_column"]
    id_column = data["id_column"]
    gene_columns = [column for column in train.columns if column not in {id_column, target}]
    labels = train[target]
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    class_names = encoder.classes_
    n_classes = len(class_names)
    seeds = [int(value) for value in validation["seeds"]]
    n_splits = int(validation["n_splits"])

    e7_oof = load_first(
        e7_dir, ("oof_probability_by_seed.npy", "oof_probabilities_by_seed.npy")
    )
    e8_oof = load_first(e8_dir, ("oof_decision_by_seed.npy",))
    e7_test = load_first(e7_dir, ("test_probability_mean.npy",))
    e8_test = load_first(e8_dir, ("test_decision_mean.npy",))
    expected_oof_shape = (len(seeds), len(train), n_classes)
    if e7_oof.shape != expected_oof_shape or e8_oof.shape != expected_oof_shape:
        raise ValueError(
            f"OOF shape 불일치: E7={e7_oof.shape}, E8={e8_oof.shape}, "
            f"expected={expected_oof_shape}"
        )

    folds = reconstruct_folds(
        train[gene_columns], labels, seeds, n_splits,
    )
    e8a_prediction, selected = select_crossfit_parameters(
        y=y,
        e7_oof=e7_oof,
        e8_oof=e8_oof,
        folds=folds,
        seeds=seeds,
        n_splits=n_splits,
        temperatures=[float(value) for value in ensemble["temperatures"]],
        weights=[float(value) for value in ensemble["e8_weights"]],
        n_classes=n_classes,
    )
    e7_prediction = e7_oof.argmax(axis=2)
    e8_prediction = e8_oof.argmax(axis=2)
    seed_rows = []
    for seed_position, seed in enumerate(seeds):
        seed_rows.append({
            "seed": seed,
            "oof_macro_f1": float(f1_score(
                y, e8a_prediction[seed_position], labels=np.arange(n_classes),
                average="macro", zero_division=0,
            )),
        })
    seed_metrics = pd.DataFrame(seed_rows)

    test_probability_sum = np.zeros_like(e7_test, dtype=np.float64)
    for row in selected.itertuples(index=False):
        calibrated = temperature_softmax(e8_test, float(row.temperature))
        test_probability_sum += blend_probabilities(
            e7_test, calibrated, float(row.e8_weight),
        )
    test_probability = test_probability_sum / len(selected)
    test_probability /= test_probability.sum(axis=1, keepdims=True)
    submission = pd.read_csv(raw_dir / data["submission_file"])
    submission[target] = class_names[test_probability.argmax(axis=1)]

    bootstrap_e8 = paired_bootstrap(
        y, e8a_prediction, e8_prediction,
        int(ensemble["bootstrap_repeats"]), int(ensemble["bootstrap_seed"]), n_classes,
    )
    bootstrap_e7 = paired_bootstrap(
        y, e8a_prediction, e7_prediction,
        int(ensemble["bootstrap_repeats"]), int(ensemble["bootstrap_seed"]), n_classes,
    )
    mean_score = float(seed_metrics["oof_macro_f1"].mean())
    std_score = float(seed_metrics["oof_macro_f1"].std(ddof=1))
    summary = {
        "experiment": config["project"]["experiment_name"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "E7 probability + calibrated E8 LinearSVC decision blend",
        "oof_macro_f1_mean": mean_score,
        "oof_macro_f1_std": std_score,
        "bootstrap_e8a_minus_e8": bootstrap_e8,
        "bootstrap_e8a_minus_e7": bootstrap_e7,
    }
    selected.to_csv(output_dir / "selected_parameters.csv", index=False)
    seed_metrics.to_csv(output_dir / "seed_oof_metrics.csv", index=False)
    np.save(output_dir / "oof_predictions_by_seed.npy", e8a_prediction)
    np.save(output_dir / "test_probability_mean.npy", test_probability)
    submission.to_csv(output_dir / "submission_test_002_e8a.csv", index=False)
    pd.DataFrame({
        "class_index": np.arange(n_classes), target: class_names,
    }).to_csv(output_dir / "class_order.csv", index=False)
    (output_dir / "e8a_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
