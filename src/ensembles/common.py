"""JH 앙상블 실행기의 공통 입출력·검증 함수."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_first(directory: Path, names: list[str] | tuple[str, ...]) -> np.ndarray:
    for name in names:
        path = directory / name
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(
        f"{directory}에 필요한 파일이 없습니다: {', '.join(names)}"
    )


def temperature_softmax(decision: np.ndarray, temperature: float) -> np.ndarray:
    scaled = decision.astype(np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_score = np.exp(scaled)
    return exp_score / exp_score.sum(axis=1, keepdims=True)


def calibrate_probability(probability: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.log(np.clip(probability, 1e-12, 1.0)) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    calibrated = np.exp(scaled)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def blend_probabilities(
    base_probability: np.ndarray,
    added_probability: np.ndarray,
    added_weight: float,
) -> np.ndarray:
    blended = (
        (1.0 - float(added_weight)) * base_probability
        + float(added_weight) * added_probability
    )
    return blended / blended.sum(axis=1, keepdims=True)


def load_fixed_folds(
    split_path: Path,
    train_ids: pd.Series,
    seeds: list[int],
    n_splits: int,
    id_column: str,
) -> dict[int, np.ndarray]:
    frame = pd.read_csv(split_path)
    if not np.array_equal(
        frame[id_column].astype(str).to_numpy(),
        train_ids.astype(str).to_numpy(),
    ):
        raise ValueError("Train과 고정 split의 ID 순서가 다릅니다.")
    result: dict[int, np.ndarray] = {}
    for seed in seeds:
        column = f"fold_seed_{seed}"
        if column not in frame:
            raise ValueError(f"고정 split에 {column} 열이 없습니다.")
        folds = frame[column].to_numpy(dtype=np.int8)
        if set(np.unique(folds)) != set(range(n_splits)):
            raise ValueError(f"seed={seed} fold 값이 잘못되었습니다.")
        result[seed] = folds
    return result


def reconstruct_e8a_probability(
    e7_oof: np.ndarray,
    e8_decision: np.ndarray,
    selected: pd.DataFrame,
    folds: dict[int, np.ndarray],
    seeds: list[int],
    n_splits: int,
) -> np.ndarray:
    result = np.zeros_like(e7_oof, dtype=np.float64)
    for seed_position, seed in enumerate(seeds):
        for fold in range(n_splits):
            row = selected.loc[
                (selected["seed"] == seed) & (selected["fold"] == fold)
            ]
            if len(row) != 1:
                raise ValueError(f"E8A seed={seed}, fold={fold} 파라미터 오류")
            row = row.iloc[0]
            mask = folds[seed] == fold
            e8_probability = temperature_softmax(
                e8_decision[seed_position, mask], float(row["temperature"])
            )
            result[seed_position, mask] = blend_probabilities(
                e7_oof[seed_position, mask], e8_probability, float(row["e8_weight"])
            )
    return result


def select_auxiliary_crossfit(
    y: np.ndarray,
    base_oof: np.ndarray,
    auxiliary_oof: np.ndarray,
    folds: dict[int, np.ndarray],
    seeds: list[int],
    n_splits: int,
    temperatures: list[float],
    weights: list[float],
    n_classes: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    result = np.zeros_like(base_oof, dtype=np.float64)
    rows: list[dict] = []
    labels = np.arange(n_classes)
    aligned = auxiliary_oof.ndim == 3
    for seed_position, seed in enumerate(seeds):
        seed_auxiliary = (
            auxiliary_oof[seed_position] if aligned else auxiliary_oof
        )
        for fold in range(n_splits):
            evaluation_mask = folds[seed] == fold
            tuning_mask = ~evaluation_mask
            best: tuple[float, float, float, float] | None = None
            for temperature in temperatures:
                tuning_auxiliary = calibrate_probability(
                    seed_auxiliary[tuning_mask], temperature
                )
                for weight in weights:
                    prediction = blend_probabilities(
                        base_oof[seed_position, tuning_mask],
                        tuning_auxiliary,
                        weight,
                    ).argmax(axis=1)
                    score = f1_score(
                        y[tuning_mask], prediction, labels=labels,
                        average="macro", zero_division=0,
                    )
                    penalty = abs(np.log(temperature)) + 0.01 * weight
                    candidate = (float(score), -float(penalty), temperature, weight)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
            assert best is not None
            best_score, _, best_temperature, best_weight = best
            evaluation_auxiliary = calibrate_probability(
                seed_auxiliary[evaluation_mask], best_temperature
            )
            evaluation_blend = blend_probabilities(
                base_oof[seed_position, evaluation_mask],
                evaluation_auxiliary,
                best_weight,
            )
            result[seed_position, evaluation_mask] = evaluation_blend
            evaluation_score = f1_score(
                y[evaluation_mask], evaluation_blend.argmax(axis=1), labels=labels,
                average="macro", zero_division=0,
            )
            rows.append({
                "seed": seed,
                "fold": fold,
                "em16_temperature": best_temperature,
                "em16_weight": best_weight,
                "tuning_macro_f1": best_score,
                "evaluation_macro_f1": float(evaluation_score),
                "evaluation_rows": int(evaluation_mask.sum()),
            })
    return result, pd.DataFrame(rows)


def paired_bootstrap(
    y: np.ndarray,
    candidate_probability: np.ndarray,
    baseline_probability: np.ndarray,
    repeats: int,
    random_state: int,
    n_classes: int,
) -> dict[str, float | bool]:
    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(y == index) for index in range(n_classes)]
    labels = np.arange(n_classes)
    deltas = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        seed_deltas = []
        for seed_position in range(candidate_probability.shape[0]):
            candidate_score = f1_score(
                y[sampled], candidate_probability[seed_position, sampled].argmax(axis=1),
                labels=labels, average="macro", zero_division=0,
            )
            baseline_score = f1_score(
                y[sampled], baseline_probability[seed_position, sampled].argmax(axis=1),
                labels=labels, average="macro", zero_division=0,
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


def run_em16_ensemble(config_path: Path) -> None:
    config = load_config(config_path)
    project = config["project"]
    data = config["data"]
    validation = config["validation"]
    ensemble = config["ensemble"]
    raw_dir = Path(data["raw_dir"])
    e7_dir = Path(data["e7_dir"])
    e8_dir = Path(data["e8_dir"])
    e8a_dir = Path(data["e8a_dir"])
    em16_dir = Path(data["em16_dir"])
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
    seeds = [int(seed) for seed in validation["seeds"]]
    n_splits = int(validation["n_splits"])
    folds = load_fixed_folds(
        Path(data["fixed_split_file"]), train[id_column], seeds, n_splits, id_column
    )

    e7_oof = load_first(e7_dir, ["oof_probabilities_by_seed.npy", "oof_probability_by_seed.npy"])
    e8_decision = load_first(e8_dir, ["oof_decision_by_seed.npy"])
    e8a_test = load_first(e8a_dir, ["test_probability_mean.npy"])
    e8a_selected = pd.read_csv(e8a_dir / "selected_parameters.csv")
    e8a_oof = reconstruct_e8a_probability(
        e7_oof, e8_decision, e8a_selected, folds, seeds, n_splits
    )
    auxiliary_oof = load_first(em16_dir, data["em16_oof_files"])
    auxiliary_test = load_first(em16_dir, data["em16_test_files"])

    expected = (len(seeds), len(train), n_classes)
    if e8a_oof.shape != expected:
        raise ValueError(f"E8A OOF shape: {e8a_oof.shape}, expected={expected}")
    aligned = bool(ensemble["aligned_by_seed_fold"])
    expected_aux_oof = expected if aligned else (len(train), n_classes)
    expected_aux_test = (
        (len(seeds), n_splits, len(test), n_classes)
        if aligned else (len(test), n_classes)
    )
    if auxiliary_oof.shape != expected_aux_oof:
        raise ValueError(f"EM16 OOF shape: {auxiliary_oof.shape}, expected={expected_aux_oof}")
    if auxiliary_test.shape != expected_aux_test:
        raise ValueError(f"EM16 Test shape: {auxiliary_test.shape}, expected={expected_aux_test}")

    candidate_oof, selected = select_auxiliary_crossfit(
        y, e8a_oof, auxiliary_oof, folds, seeds, n_splits,
        [float(value) for value in ensemble["temperatures"]],
        [float(value) for value in ensemble["weights"]],
        n_classes,
    )
    labels = np.arange(n_classes)
    seed_rows = []
    for seed_position, seed in enumerate(seeds):
        baseline_score = f1_score(
            y, e8a_oof[seed_position].argmax(axis=1), labels=labels,
            average="macro", zero_division=0,
        )
        candidate_score = f1_score(
            y, candidate_oof[seed_position].argmax(axis=1), labels=labels,
            average="macro", zero_division=0,
        )
        seed_rows.append({
            "seed": seed,
            "e8a_macro_f1": float(baseline_score),
            "candidate_macro_f1": float(candidate_score),
            "delta": float(candidate_score - baseline_score),
        })
    seed_metrics = pd.DataFrame(seed_rows)
    bootstrap = paired_bootstrap(
        y, candidate_oof, e8a_oof,
        int(ensemble["bootstrap_repeats"]), int(ensemble["bootstrap_seed"]), n_classes,
    )

    test_sum = np.zeros_like(e8a_test, dtype=np.float64)
    for row in selected.itertuples(index=False):
        seed_position = seeds.index(int(row.seed))
        auxiliary_fold_test = (
            auxiliary_test[seed_position, int(row.fold)] if aligned else auxiliary_test
        )
        calibrated = calibrate_probability(auxiliary_fold_test, row.em16_temperature)
        test_sum += blend_probabilities(e8a_test, calibrated, row.em16_weight)
    test_probability = test_sum / len(selected)
    test_probability /= test_probability.sum(axis=1, keepdims=True)
    submission[target] = class_names[test_probability.argmax(axis=1)]

    mean_score = float(seed_metrics["candidate_macro_f1"].mean())
    std_score = float(seed_metrics["candidate_macro_f1"].std(ddof=1))
    baseline_mean = float(seed_metrics["e8a_macro_f1"].mean())
    all_seeds_improved = bool((seed_metrics["delta"] > 0).all())
    adopted = bool(
        mean_score > baseline_mean
        and bootstrap["ci_lower_ge_0"]
        and (
            all_seeds_improved
            or not bool(ensemble.get("require_all_seeds_improved", False))
        )
    )
    summary = {
        "experiment": project["experiment_name"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_model": "E8A",
        "added_model": "EM v16 E4 XGBoost",
        "aligned_by_seed_fold": aligned,
        "e8a_oof_macro_f1": baseline_mean,
        "oof_macro_f1_mean": mean_score,
        "oof_macro_f1_std": std_score,
        "delta": mean_score - baseline_mean,
        "all_seeds_improved": all_seeds_improved,
        "bootstrap": bootstrap,
        "adopted_by_internal_rule": adopted,
    }
    selected.to_csv(output_dir / "selected_parameters.csv", index=False)
    seed_metrics.to_csv(output_dir / "seed_oof_metrics.csv", index=False)
    np.save(output_dir / "oof_probabilities_by_seed.npy", candidate_oof)
    np.save(output_dir / "test_probability_mean.npy", test_probability)
    submission.to_csv(output_dir / f"submission_{project['experiment_name']}.csv", index=False)
    pd.DataFrame({
        "class_index": np.arange(n_classes), target: class_names,
    }).to_csv(output_dir / "class_order.csv", index=False)
    (output_dir / f"{project['experiment_name']}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
