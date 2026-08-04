"""E10B-1: E10과 ID 제외 Word·Char TF-IDF LinearSVC를 cross-fit 결합한다."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from src.ensembles.common import (
    blend_probabilities,
    load_config,
    load_first,
    load_fixed_folds,
    temperature_softmax,
)


def select_crossfit(
    y: np.ndarray,
    base_oof: np.ndarray,
    tfidf_decision: np.ndarray,
    folds: dict[int, np.ndarray],
    seeds: list[int],
    n_splits: int,
    temperatures: list[float],
    weights: list[float],
    n_classes: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    result = np.zeros_like(base_oof, dtype=np.float64)
    labels = np.arange(n_classes)
    rows: list[dict] = []
    for seed_position, seed in enumerate(seeds):
        for fold in range(n_splits):
            evaluation_mask = folds[seed] == fold
            tuning_mask = ~evaluation_mask
            best: tuple[float, float, float, float] | None = None
            for temperature in temperatures:
                tfidf_probability = temperature_softmax(
                    tfidf_decision[seed_position, tuning_mask], temperature,
                )
                for weight in weights:
                    probability = blend_probabilities(
                        base_oof[seed_position, tuning_mask],
                        tfidf_probability, weight,
                    )
                    score = f1_score(
                        y[tuning_mask], probability.argmax(axis=1), labels=labels,
                        average="macro", zero_division=0,
                    )
                    candidate = (
                        float(score), -float(weight), -float(temperature),
                        float(weight),
                    )
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate
            assert best is not None
            tuning_score, negative_weight, negative_temperature, weight = best
            temperature = -negative_temperature
            if not np.isclose(weight, -negative_weight):
                raise RuntimeError("TF-IDF weight 선택 결과가 잘못되었습니다.")
            evaluation_tfidf = temperature_softmax(
                tfidf_decision[seed_position, evaluation_mask], temperature,
            )
            evaluation_probability = blend_probabilities(
                base_oof[seed_position, evaluation_mask],
                evaluation_tfidf, weight,
            )
            result[seed_position, evaluation_mask] = evaluation_probability
            base_score = f1_score(
                y[evaluation_mask],
                base_oof[seed_position, evaluation_mask].argmax(axis=1),
                labels=labels, average="macro", zero_division=0,
            )
            candidate_score = f1_score(
                y[evaluation_mask], evaluation_probability.argmax(axis=1),
                labels=labels, average="macro", zero_division=0,
            )
            rows.append({
                "seed": seed,
                "fold": fold,
                "temperature": temperature,
                "tfidf_weight": weight,
                "tuning_macro_f1": tuning_score,
                "e10_evaluation_macro_f1": float(base_score),
                "candidate_evaluation_macro_f1": float(candidate_score),
                "evaluation_delta": float(candidate_score - base_score),
            })
    return result, pd.DataFrame(rows)


def paired_bootstrap_mean_probability(
    y: np.ndarray,
    candidate_oof: np.ndarray,
    baseline_oof: np.ndarray,
    repeats: int,
    random_state: int,
    n_classes: int,
) -> dict[str, float | bool]:
    candidate_prediction = candidate_oof.mean(axis=0).argmax(axis=1)
    baseline_prediction = baseline_oof.mean(axis=0).argmax(axis=1)
    labels = np.arange(n_classes)
    observed_candidate = f1_score(
        y, candidate_prediction, labels=labels, average="macro", zero_division=0,
    )
    observed_baseline = f1_score(
        y, baseline_prediction, labels=labels, average="macro", zero_division=0,
    )
    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(y == index) for index in range(n_classes)]
    deltas = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        baseline_score = f1_score(
            y[sampled], baseline_prediction[sampled], labels=labels,
            average="macro", zero_division=0,
        )
        candidate_score = f1_score(
            y[sampled], candidate_prediction[sampled], labels=labels,
            average="macro", zero_division=0,
        )
        deltas[repeat] = candidate_score - baseline_score
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return {
        "baseline_macro_f1": float(observed_baseline),
        "candidate_macro_f1": float(observed_candidate),
        "observed_delta": float(observed_candidate - observed_baseline),
        "mean_delta": float(deltas.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_lower_ge_0": bool(lower >= 0),
        "probability_positive": float(np.mean(deltas > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/test_006.yaml"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    project = config["project"]
    data = config["data"]
    validation = config["validation"]
    ensemble = config["ensemble"]
    raw_dir = Path(data["raw_dir"])
    output_dir = Path(data["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(raw_dir / data["train_file"])
    test = pd.read_csv(raw_dir / data["test_file"])
    submission = pd.read_csv(raw_dir / data["submission_file"])
    target = data["target_column"]
    id_column = data["id_column"]
    encoder = LabelEncoder().fit(train[target].astype(str))
    y = encoder.transform(train[target].astype(str))
    class_names = encoder.classes_
    n_classes = len(class_names)
    labels = np.arange(n_classes)
    seeds = [int(seed) for seed in validation["seeds"]]
    n_splits = int(validation["n_splits"])
    folds = load_fixed_folds(
        Path(data["fixed_split_file"]), train[id_column], seeds, n_splits, id_column,
    )

    e10_dir = Path(data["e10_dir"])
    tfidf_dir = Path(data["tfidf_dir"])
    base_oof = load_first(e10_dir, ["oof_probabilities_by_seed.npy"])
    base_test = load_first(e10_dir, ["test_probability_mean.npy"])
    tfidf_oof = np.load(tfidf_dir / data["tfidf_oof_file"])
    tfidf_test = np.load(tfidf_dir / data["tfidf_test_file"])
    expected_oof = (len(seeds), len(train), n_classes)
    expected_test = (len(seeds), n_splits, len(test), n_classes)
    if base_oof.shape != expected_oof or tfidf_oof.shape != expected_oof:
        raise ValueError(
            f"OOF shape 불일치: E10={base_oof.shape}, TF-IDF={tfidf_oof.shape}"
        )
    if base_test.shape != (len(test), n_classes) or tfidf_test.shape != expected_test:
        raise ValueError(
            f"Test shape 불일치: E10={base_test.shape}, TF-IDF={tfidf_test.shape}"
        )

    candidate_oof, selected = select_crossfit(
        y, base_oof, tfidf_oof, folds, seeds, n_splits,
        [float(value) for value in ensemble["tfidf_temperatures"]],
        [float(value) for value in ensemble["tfidf_weights"]], n_classes,
    )
    seed_rows = []
    for seed_position, seed in enumerate(seeds):
        base_score = f1_score(
            y, base_oof[seed_position].argmax(axis=1), labels=labels,
            average="macro", zero_division=0,
        )
        candidate_score = f1_score(
            y, candidate_oof[seed_position].argmax(axis=1), labels=labels,
            average="macro", zero_division=0,
        )
        seed_rows.append({
            "seed": seed,
            "e10_macro_f1": float(base_score),
            "candidate_macro_f1": float(candidate_score),
            "delta": float(candidate_score - base_score),
        })
    seed_metrics = pd.DataFrame(seed_rows)
    bootstrap = paired_bootstrap_mean_probability(
        y, candidate_oof, base_oof, int(ensemble["bootstrap_repeats"]),
        int(ensemble["bootstrap_seed"]), n_classes,
    )

    test_by_model = np.zeros(expected_test, dtype=np.float32)
    for row in selected.itertuples(index=False):
        seed_position = seeds.index(int(row.seed))
        fold = int(row.fold)
        tfidf_probability = temperature_softmax(
            tfidf_test[seed_position, fold], float(row.temperature),
        )
        test_by_model[seed_position, fold] = blend_probabilities(
            base_test, tfidf_probability, float(row.tfidf_weight),
        )
    test_probability = test_by_model.mean(axis=(0, 1))
    test_probability /= test_probability.sum(axis=1, keepdims=True)
    submission[target] = class_names[test_probability.argmax(axis=1)]

    seed_mean = float(seed_metrics["candidate_macro_f1"].mean())
    seed_std = float(seed_metrics["candidate_macro_f1"].std(ddof=1))
    summary = {
        "experiment": project["experiment_name"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "E10 + ID-excluded Word/Char TF-IDF Balanced LinearSVC",
        "oof_macro_f1_seed_mean": seed_mean,
        "oof_macro_f1_seed_std": seed_std,
        "seed_mean_delta": float(seed_metrics["delta"].mean()),
        "mean_probability_bootstrap": bootstrap,
        "adopted_by_internal_rule": bool(bootstrap["ci_lower_ge_0"]),
    }
    selected.to_csv(output_dir / "selected_parameters.csv", index=False)
    seed_metrics.to_csv(output_dir / "seed_oof_metrics.csv", index=False)
    np.save(output_dir / "oof_probabilities_by_seed.npy", candidate_oof)
    np.save(output_dir / "test_probability_by_seed_fold.npy", test_by_model)
    np.save(output_dir / "test_probability_mean.npy", test_probability)
    pd.DataFrame({"class_index": labels, target: class_names}).to_csv(
        output_dir / "class_order.csv", index=False,
    )
    submission.to_csv(
        output_dir / f"submission_{project['experiment_name']}.csv", index=False,
    )
    (output_dir / f"{project['experiment_name']}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
