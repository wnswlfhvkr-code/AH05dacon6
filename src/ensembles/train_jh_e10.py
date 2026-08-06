"""E10: E8 LinearSVC decision과 E9 LightGBM 확률의 cross-fit 앙상블."""

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
    load_config,
    load_first,
    load_fixed_folds,
    paired_bootstrap,
)


def probability_to_log_score(
    probability: np.ndarray,
    temperature: float,
) -> np.ndarray:
    return np.log(np.clip(probability, 1e-12, 1.0)) / float(temperature)


def combine_logit_scores(
    e8_decision: np.ndarray,
    e9_probability: np.ndarray,
    e8_temperature: float,
    e9_temperature: float,
    e9_weight: float,
) -> np.ndarray:
    return (
        (1.0 - float(e9_weight))
        * np.asarray(e8_decision, dtype=np.float64)
        / float(e8_temperature)
        + float(e9_weight)
        * probability_to_log_score(e9_probability, e9_temperature)
    )


def score_to_probability(score: np.ndarray) -> np.ndarray:
    stable = score - score.max(axis=-1, keepdims=True)
    probability = np.exp(stable)
    return probability / probability.sum(axis=-1, keepdims=True)


def select_crossfit(
    y: np.ndarray,
    e8_oof: np.ndarray,
    e9_oof: np.ndarray,
    folds: dict[int, np.ndarray],
    seeds: list[int],
    n_splits: int,
    e8_temperatures: list[float],
    e9_temperatures: list[float],
    e9_weights: list[float],
    n_classes: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    result = np.zeros_like(e9_oof, dtype=np.float64)
    rows: list[dict] = []
    labels = np.arange(n_classes)

    for seed_position, seed in enumerate(seeds):
        for fold in range(n_splits):
            evaluation_mask = folds[seed] == fold
            tuning_mask = ~evaluation_mask
            best: tuple[float, float, float, float, float] | None = None

            for e8_temperature in e8_temperatures:
                for e9_temperature in e9_temperatures:
                    for e9_weight in e9_weights:
                        tuning_score = combine_logit_scores(
                            e8_oof[seed_position, tuning_mask],
                            e9_oof[seed_position, tuning_mask],
                            e8_temperature,
                            e9_temperature,
                            e9_weight,
                        )
                        prediction = tuning_score.argmax(axis=1)
                        macro_f1 = f1_score(
                            y[tuning_mask], prediction, labels=labels,
                            average="macro", zero_division=0,
                        )
                        penalty = (
                            10.0 * float(e9_weight)
                            + abs(np.log(float(e8_temperature)))
                            + abs(np.log(float(e9_temperature)))
                        )
                        candidate = (
                            float(macro_f1), -float(penalty),
                            float(e8_temperature), float(e9_temperature),
                            float(e9_weight),
                        )
                        if best is None or candidate[:2] > best[:2]:
                            best = candidate

            assert best is not None
            tuning_macro_f1, _, e8_temperature, e9_temperature, e9_weight = best
            evaluation_score = combine_logit_scores(
                e8_oof[seed_position, evaluation_mask],
                e9_oof[seed_position, evaluation_mask],
                e8_temperature,
                e9_temperature,
                e9_weight,
            )
            evaluation_probability = score_to_probability(evaluation_score)
            result[seed_position, evaluation_mask] = evaluation_probability
            evaluation_macro_f1 = f1_score(
                y[evaluation_mask], evaluation_score.argmax(axis=1), labels=labels,
                average="macro", zero_division=0,
            )
            rows.append({
                "seed": seed,
                "fold": fold,
                "e8_temperature": e8_temperature,
                "e9_temperature": e9_temperature,
                "e9_weight": e9_weight,
                "tuning_macro_f1": tuning_macro_f1,
                "evaluation_macro_f1": float(evaluation_macro_f1),
                "evaluation_rows": int(evaluation_mask.sum()),
            })

    return result, pd.DataFrame(rows)


def _validate_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if actual != expected:
        raise ValueError(f"{name} shape={actual}, expected={expected}")


def load_e9_test_probability(
    directory: Path,
    n_seeds: int,
    n_splits: int,
) -> tuple[np.ndarray, str]:
    fold_names = [
        "test_probability_by_seed_fold.npy",
        "e9_test_probability_by_seed_fold.npy",
    ]
    for name in fold_names:
        path = directory / name
        if path.exists():
            return np.load(path), "fold_specific"

    mean_names = [
        "test_probability_mean.npy",
        "e9_test_probability_mean.npy",
    ]
    for name in mean_names:
        path = directory / name
        if path.exists():
            mean_probability = np.load(path)
            expanded = np.broadcast_to(
                mean_probability,
                (n_seeds, n_splits, *mean_probability.shape),
            )
            print(
                "경고: E9 Fold별 Test 확률이 없어 평균 Test 확률을 "
                "모든 seed/fold에 공통 적용합니다."
            )
            return expanded, "mean_probability_fallback"

    raise FileNotFoundError(
        f"{directory}에 E9 Test 확률 파일이 없습니다: "
        f"{', '.join(fold_names + mean_names)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/ensembles/test_008_v03.yaml"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    project = config["project"]
    data = config["data"]
    validation = config["validation"]
    ensemble = config["ensemble"]

    raw_dir = Path(data["raw_dir"])
    e8_dir = Path(data["e8_dir"])
    e9_dir = Path(data["e9_dir"])
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

    e8_oof = load_first(e8_dir, ["oof_decision_by_seed.npy"])
    e8_test = load_first(e8_dir, ["test_decision_mean.npy"])
    e9_oof = load_first(e9_dir, ["oof_probability_by_seed.npy"])
    e9_test_by_fold, e9_test_probability_source = load_e9_test_probability(
        e9_dir, len(seeds), n_splits,
    )

    _validate_shape(
        "E8 OOF", e8_oof.shape, (len(seeds), len(train), n_classes),
    )
    _validate_shape(
        "E9 OOF", e9_oof.shape, (len(seeds), len(train), n_classes),
    )
    _validate_shape("E8 Test", e8_test.shape, (len(test), n_classes))
    _validate_shape(
        "E9 Test", e9_test_by_fold.shape,
        (len(seeds), n_splits, len(test), n_classes),
    )

    candidate_oof, selected = select_crossfit(
        y, e8_oof, e9_oof, folds, seeds, n_splits,
        [float(value) for value in ensemble["e8_temperatures"]],
        [float(value) for value in ensemble["e9_temperatures"]],
        [float(value) for value in ensemble["e9_weights"]],
        n_classes,
    )

    seed_rows = []
    for seed_position, seed in enumerate(seeds):
        e8_score = f1_score(
            y, e8_oof[seed_position].argmax(axis=1), labels=labels,
            average="macro", zero_division=0,
        )
        e9_score = f1_score(
            y, e9_oof[seed_position].argmax(axis=1), labels=labels,
            average="macro", zero_division=0,
        )
        candidate_score = f1_score(
            y, candidate_oof[seed_position].argmax(axis=1), labels=labels,
            average="macro", zero_division=0,
        )
        seed_rows.append({
            "seed": seed,
            "e8_macro_f1": float(e8_score),
            "e9_macro_f1": float(e9_score),
            "candidate_macro_f1": float(candidate_score),
            "delta_vs_e8": float(candidate_score - e8_score),
        })
    seed_metrics = pd.DataFrame(seed_rows)

    bootstrap = paired_bootstrap(
        y, candidate_oof, score_to_probability(e8_oof),
        int(ensemble["bootstrap_repeats"]), int(ensemble["bootstrap_seed"]),
        n_classes,
    )

    test_sum = np.zeros((len(test), n_classes), dtype=np.float64)
    for row in selected.itertuples(index=False):
        seed_position = seeds.index(int(row.seed))
        combined = combine_logit_scores(
            e8_test,
            e9_test_by_fold[seed_position, int(row.fold)],
            float(row.e8_temperature),
            float(row.e9_temperature),
            float(row.e9_weight),
        )
        test_sum += score_to_probability(combined)
    test_probability = test_sum / len(selected)
    test_probability /= test_probability.sum(axis=1, keepdims=True)
    test_prediction = test_probability.argmax(axis=1)
    submission[target] = class_names[test_prediction]

    mean_score = float(seed_metrics["candidate_macro_f1"].mean())
    std_score = float(seed_metrics["candidate_macro_f1"].std(ddof=1))
    e8_mean = float(seed_metrics["e8_macro_f1"].mean())
    e9_mean = float(seed_metrics["e9_macro_f1"].mean())
    all_seeds_improved = bool((seed_metrics["delta_vs_e8"] > 0).all())
    adopted = bool(
        mean_score > e8_mean
        and bootstrap["ci_lower_ge_0"]
        and (
            all_seeds_improved
            or not bool(ensemble.get("require_all_seeds_improved", False))
        )
    )
    summary = {
        "experiment": project["experiment_name"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "E8 LinearSVC decision + E9 LightGBM log-probability",
        "e8_oof_macro_f1": e8_mean,
        "e9_oof_macro_f1": e9_mean,
        "oof_macro_f1_mean": mean_score,
        "oof_macro_f1_std": std_score,
        "delta_vs_e8": mean_score - e8_mean,
        "e9_test_probability_source": e9_test_probability_source,
        "all_seeds_improved": all_seeds_improved,
        "bootstrap_vs_e8": bootstrap,
        "adopted_by_internal_rule": adopted,
    }

    selected.to_csv(output_dir / "selected_parameters.csv", index=False)
    seed_metrics.to_csv(output_dir / "seed_oof_metrics.csv", index=False)
    np.save(output_dir / "oof_probabilities_by_seed.npy", candidate_oof)
    np.save(output_dir / "test_probability_mean.npy", test_probability)
    pd.DataFrame({
        "class_index": labels, target: class_names,
    }).to_csv(output_dir / "class_order.csv", index=False)
    submission.to_csv(
        output_dir / f"submission_{project['experiment_name']}.csv", index=False,
    )
    (output_dir / f"{project['experiment_name']}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
