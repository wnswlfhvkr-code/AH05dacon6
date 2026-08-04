"""E10A: E10에 EM16·EM24 pair expert를 cross-fit으로 적용한다."""

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
    load_fixed_folds,
    load_first,
    paired_bootstrap,
)


def mix_team_probability(
    em16_probability: np.ndarray,
    em24_probability: np.ndarray,
    em16_weight: float,
) -> np.ndarray:
    mixed = (
        float(em16_weight) * em16_probability
        + (1.0 - float(em16_weight)) * em24_probability
    )
    return mixed / mixed.sum(axis=1, keepdims=True)


def pair_right_probability(
    probability: np.ndarray,
    left_index: int,
    right_index: int,
    temperature: float,
    right_offset: float,
) -> np.ndarray:
    left = np.clip(probability[:, left_index], 1e-12, 1.0)
    right = np.clip(probability[:, right_index], 1e-12, 1.0)
    adjusted_log_odds = (
        (np.log(right) - np.log(left)) / float(temperature)
        - float(right_offset)
    )
    return 1.0 / (1.0 + np.exp(-adjusted_log_odds))


def apply_pair_experts(
    base_probability: np.ndarray,
    team_probability: np.ndarray,
    pair_indices: dict[str, tuple[int, int]],
    right_offsets: dict[str, float],
    temperature: float,
    kirc_pair_weight: float,
    lgg_pair_weight: float,
) -> np.ndarray:
    adjusted = np.asarray(base_probability, dtype=np.float64).copy()
    initial_prediction = base_probability.argmax(axis=1)
    settings = [
        ("KIRC_KIPAN", float(kirc_pair_weight)),
        ("LGG_GBMLGG", float(lgg_pair_weight)),
    ]
    for pair_name, pair_weight in settings:
        if pair_weight <= 0:
            continue
        left_index, right_index = pair_indices[pair_name]
        active = (initial_prediction == left_index) | (initial_prediction == right_index)
        if not active.any():
            continue
        pair_total = (
            base_probability[active, left_index]
            + base_probability[active, right_index]
        )
        base_right = (
            base_probability[active, right_index] / np.maximum(pair_total, 1e-12)
        )
        expert_right = pair_right_probability(
            team_probability[active], left_index, right_index,
            temperature, right_offsets[pair_name],
        )
        combined_right = (
            (1.0 - pair_weight) * base_right + pair_weight * expert_right
        )
        adjusted[active, left_index] = pair_total * (1.0 - combined_right)
        adjusted[active, right_index] = pair_total * combined_right
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def select_crossfit(
    y: np.ndarray,
    base_oof: np.ndarray,
    em16_oof: np.ndarray,
    em24_oof: np.ndarray,
    folds: dict[int, np.ndarray],
    seeds: list[int],
    n_splits: int,
    pair_indices: dict[str, tuple[int, int]],
    right_offsets: dict[str, float],
    ensemble: dict,
    n_classes: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    result = np.zeros_like(base_oof, dtype=np.float64)
    labels = np.arange(n_classes)
    rows: list[dict] = []
    for seed_position, seed in enumerate(seeds):
        for fold in range(n_splits):
            evaluation_mask = folds[seed] == fold
            tuning_mask = ~evaluation_mask
            best: tuple[float, float, float, float, float, float] | None = None
            for em16_weight in ensemble["em16_mix_weights"]:
                tuning_team = mix_team_probability(
                    em16_oof[seed_position, tuning_mask],
                    em24_oof[seed_position, tuning_mask],
                    em16_weight,
                )
                for temperature in ensemble["team_temperatures"]:
                    for kirc_weight in ensemble["kirc_pair_weights"]:
                        for lgg_weight in ensemble["lgg_pair_weights"]:
                            probability = apply_pair_experts(
                                base_oof[seed_position, tuning_mask], tuning_team,
                                pair_indices, right_offsets, temperature,
                                kirc_weight, lgg_weight,
                            )
                            score = f1_score(
                                y[tuning_mask], probability.argmax(axis=1),
                                labels=labels, average="macro", zero_division=0,
                            )
                            penalty = (
                                float(kirc_weight) + float(lgg_weight)
                                + 0.01 * abs(np.log(float(temperature)))
                                + 0.001 * (1.0 - float(em16_weight))
                            )
                            candidate = (
                                float(score), -float(penalty), float(em16_weight),
                                float(temperature), float(kirc_weight), float(lgg_weight),
                            )
                            if best is None or candidate[:2] > best[:2]:
                                best = candidate
            assert best is not None
            score, _, em16_weight, temperature, kirc_weight, lgg_weight = best
            evaluation_team = mix_team_probability(
                em16_oof[seed_position, evaluation_mask],
                em24_oof[seed_position, evaluation_mask], em16_weight,
            )
            evaluation_probability = apply_pair_experts(
                base_oof[seed_position, evaluation_mask], evaluation_team,
                pair_indices, right_offsets, temperature, kirc_weight, lgg_weight,
            )
            result[seed_position, evaluation_mask] = evaluation_probability
            evaluation_score = f1_score(
                y[evaluation_mask], evaluation_probability.argmax(axis=1),
                labels=labels, average="macro", zero_division=0,
            )
            rows.append({
                "seed": seed,
                "fold": fold,
                "em16_mix_weight": em16_weight,
                "temperature": temperature,
                "kirc_pair_weight": kirc_weight,
                "lgg_pair_weight": lgg_weight,
                "tuning_macro_f1": score,
                "evaluation_macro_f1": float(evaluation_score),
                "evaluation_rows": int(evaluation_mask.sum()),
            })
    return result, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/ensembles/test_006_v04.yaml"),
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
    class_index = {name: index for index, name in enumerate(class_names)}
    pair_indices = {
        "KIRC_KIPAN": (class_index["KIRC"], class_index["KIPAN"]),
        "LGG_GBMLGG": (class_index["LGG"], class_index["GBMLGG"]),
    }
    right_offsets = {
        name: float(value)
        for name, value in ensemble["pair_right_offsets"].items()
    }

    e10_dir = Path(data["e10_dir"])
    em16_dir = Path(data["em16_dir"])
    em24_dir = Path(data["em24_dir"])
    base_oof = load_first(e10_dir, ["oof_probabilities_by_seed.npy"])
    base_test = load_first(e10_dir, ["test_probability_mean.npy"])
    em16_oof = np.load(em16_dir / data["em16_oof_file"])
    em16_test = np.load(em16_dir / data["em16_test_file"])
    em24_oof = np.load(em24_dir / data["em24_oof_file"])
    em24_test = np.load(em24_dir / data["em24_test_file"])
    expected_oof = (len(seeds), len(train), n_classes)
    expected_test = (len(seeds), n_splits, len(test), n_classes)
    for name, value in {
        "E10 OOF": base_oof,
        "EM16 OOF": em16_oof,
        "EM24 OOF": em24_oof,
    }.items():
        if value.shape != expected_oof:
            raise ValueError(f"{name} shape={value.shape}, expected={expected_oof}")
    if base_test.shape != (len(test), n_classes):
        raise ValueError(f"E10 Test shape={base_test.shape}")
    for name, value in {"EM16 Test": em16_test, "EM24 Test": em24_test}.items():
        if value.shape != expected_test:
            raise ValueError(f"{name} shape={value.shape}, expected={expected_test}")

    candidate_oof, selected = select_crossfit(
        y, base_oof, em16_oof, em24_oof, folds, seeds, n_splits,
        pair_indices, right_offsets, ensemble, n_classes,
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
    bootstrap = paired_bootstrap(
        y, candidate_oof, base_oof, int(ensemble["bootstrap_repeats"]),
        int(ensemble["bootstrap_seed"]), n_classes,
    )

    test_sum = np.zeros_like(base_test, dtype=np.float64)
    for row in selected.itertuples(index=False):
        seed_position = seeds.index(int(row.seed))
        fold = int(row.fold)
        team_test = mix_team_probability(
            em16_test[seed_position, fold], em24_test[seed_position, fold],
            float(row.em16_mix_weight),
        )
        test_sum += apply_pair_experts(
            base_test, team_test, pair_indices, right_offsets,
            float(row.temperature), float(row.kirc_pair_weight),
            float(row.lgg_pair_weight),
        )
    test_probability = test_sum / len(selected)
    test_probability /= test_probability.sum(axis=1, keepdims=True)
    submission[target] = class_names[test_probability.argmax(axis=1)]

    mean_score = float(seed_metrics["candidate_macro_f1"].mean())
    std_score = float(seed_metrics["candidate_macro_f1"].std(ddof=1))
    base_mean = float(seed_metrics["e10_macro_f1"].mean())
    all_seeds_improved = bool((seed_metrics["delta"] > 0).all())
    adopted = bool(
        mean_score > base_mean and bootstrap["ci_lower_ge_0"]
        and (all_seeds_improved or not ensemble["require_all_seeds_improved"])
    )
    summary = {
        "experiment": project["experiment_name"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "E10 + EM16/EM24 pair experts",
        "e10_oof_macro_f1": base_mean,
        "oof_macro_f1_mean": mean_score,
        "oof_macro_f1_std": std_score,
        "delta": mean_score - base_mean,
        "all_seeds_improved": all_seeds_improved,
        "bootstrap": bootstrap,
        "adopted_by_internal_rule": adopted,
        "decision": "rejected_when_bootstrap_ci_includes_zero",
    }
    selected.to_csv(output_dir / "selected_parameters.csv", index=False)
    seed_metrics.to_csv(output_dir / "seed_oof_metrics.csv", index=False)
    np.save(output_dir / "oof_probabilities_by_seed.npy", candidate_oof)
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
