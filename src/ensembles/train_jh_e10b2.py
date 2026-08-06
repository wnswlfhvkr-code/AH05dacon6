"""E10B-2: E10B-1과 Numeric Structural LightGBM을 cross-fit 결합한다."""

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
)


def calibrate_probability(
    probability: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """확률을 log 공간에서 temperature 보정한다."""
    if temperature <= 0:
        raise ValueError(
            "temperature는 0보다 커야 합니다."
        )

    clipped = np.clip(
        np.asarray(
            probability,
            dtype=np.float64,
        ),
        1e-12,
        1.0,
    )

    scaled_log_probability = (
        np.log(clipped)
        / float(temperature)
    )

    scaled_log_probability -= (
        scaled_log_probability.max(
            axis=1,
            keepdims=True,
        )
    )

    calibrated = np.exp(
        scaled_log_probability
    )

    calibrated /= calibrated.sum(
        axis=1,
        keepdims=True,
    )

    return calibrated


def blend_numeric_probabilities(
    baseline_probability: np.ndarray,
    numeric_probability: np.ndarray,
    temperature: float,
    numeric_weight: float,
) -> np.ndarray:
    """E10B-1 확률과 보정된 Numeric LightGBM 확률을 결합한다."""
    if not 0.0 <= numeric_weight <= 1.0:
        raise ValueError(
            "numeric_weight는 0과 1 사이여야 합니다."
        )

    calibrated_numeric = calibrate_probability(
        numeric_probability,
        temperature,
    )

    blended = (
        (1.0 - numeric_weight)
        * np.asarray(
            baseline_probability,
            dtype=np.float64,
        )
        + numeric_weight
        * calibrated_numeric
    )

    blended /= blended.sum(
        axis=1,
        keepdims=True,
    )

    return blended


def select_crossfit_parameters(
    y: np.ndarray,
    baseline_oof: np.ndarray,
    numeric_oof: np.ndarray,
    folds: dict[int, np.ndarray],
    seeds: list[int],
    n_splits: int,
    temperatures: list[float],
    weights: list[float],
    n_classes: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fold별 tuning 영역에서 temperature와 weight를 선택한다."""
    candidate_oof = np.zeros_like(
        baseline_oof,
        dtype=np.float64,
    )

    labels = np.arange(
        n_classes
    )

    selected_rows: list[dict] = []

    for seed_position, seed in enumerate(
        seeds
    ):
        seed_folds = folds[seed]

        for evaluation_fold in range(
            n_splits
        ):
            tuning_mask = (
                seed_folds
                != evaluation_fold
            )

            evaluation_mask = (
                seed_folds
                == evaluation_fold
            )

            best_score = -np.inf
            best_penalty = np.inf
            best_temperature: float | None = None
            best_weight: float | None = None

            for temperature in temperatures:
                for numeric_weight in weights:
                    tuning_probability = (
                        blend_numeric_probabilities(
                            baseline_oof[
                                seed_position,
                                tuning_mask,
                            ],
                            numeric_oof[
                                seed_position,
                                tuning_mask,
                            ],
                            temperature,
                            numeric_weight,
                        )
                    )

                    tuning_prediction = (
                        tuning_probability.argmax(
                            axis=1
                        )
                    )

                    tuning_score = f1_score(
                        y[tuning_mask],
                        tuning_prediction,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )

                    # 동점이면 temperature=1에
                    # 가까운 후보를 우선한다.
                    # 그다음 가중치가 작은 후보를 우선한다.
                    penalty = (
                        abs(
                            np.log(
                                float(
                                    temperature
                                )
                            )
                        )
                        + 0.01
                        * float(
                            numeric_weight
                        )
                    )

                    is_better = (
                        tuning_score
                        > best_score
                        + 1e-12
                    )

                    is_tie_but_safer = (
                        np.isclose(
                            tuning_score,
                            best_score,
                        )
                        and penalty
                        < best_penalty
                    )

                    if (
                        is_better
                        or is_tie_but_safer
                    ):
                        best_score = float(
                            tuning_score
                        )
                        best_penalty = float(
                            penalty
                        )
                        best_temperature = float(
                            temperature
                        )
                        best_weight = float(
                            numeric_weight
                        )

            if (
                best_temperature is None
                or best_weight is None
            ):
                raise RuntimeError(
                    "결합 파라미터를 선택하지 "
                    "못했습니다."
                )

            evaluation_probability = (
                blend_numeric_probabilities(
                    baseline_oof[
                        seed_position,
                        evaluation_mask,
                    ],
                    numeric_oof[
                        seed_position,
                        evaluation_mask,
                    ],
                    best_temperature,
                    best_weight,
                )
            )

            candidate_oof[
                seed_position,
                evaluation_mask,
            ] = evaluation_probability

            baseline_prediction = (
                baseline_oof[
                    seed_position,
                    evaluation_mask,
                ].argmax(axis=1)
            )

            candidate_prediction = (
                evaluation_probability.argmax(
                    axis=1
                )
            )

            baseline_score = f1_score(
                y[evaluation_mask],
                baseline_prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )

            candidate_score = f1_score(
                y[evaluation_mask],
                candidate_prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )

            selected_rows.append(
                {
                    "seed": seed,
                    "fold": evaluation_fold,
                    "temperature": (
                        best_temperature
                    ),
                    "numeric_weight": (
                        best_weight
                    ),
                    "tuning_macro_f1": (
                        best_score
                    ),
                    "e10b1_evaluation_macro_f1": (
                        float(
                            baseline_score
                        )
                    ),
                    "e10b2_evaluation_macro_f1": (
                        float(
                            candidate_score
                        )
                    ),
                    "evaluation_delta": float(
                        candidate_score
                        - baseline_score
                    ),
                }
            )

            print(
                f"seed={seed}, "
                f"fold={evaluation_fold} | "
                f"T={best_temperature:.2f} | "
                f"weight={best_weight:.3f} | "
                f"E10B-1={baseline_score:.6f} | "
                f"E10B-2={candidate_score:.6f} | "
                f"delta="
                f"{candidate_score - baseline_score:+.6f}"
            )

    return (
        candidate_oof,
        pd.DataFrame(
            selected_rows
        ),
    )


def paired_stratified_bootstrap(
    y: np.ndarray,
    baseline_oof: np.ndarray,
    candidate_oof: np.ndarray,
    repeats: int,
    random_state: int,
    n_classes: int,
) -> dict[str, float | bool | int]:
    """Seed 평균 확률 예측의 Macro F1 차이를 bootstrap한다."""
    labels = np.arange(
        n_classes
    )

    baseline_prediction = (
        baseline_oof.mean(
            axis=0
        ).argmax(axis=1)
    )

    candidate_prediction = (
        candidate_oof.mean(
            axis=0
        ).argmax(axis=1)
    )

    baseline_score = f1_score(
        y,
        baseline_prediction,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    candidate_score = f1_score(
        y,
        candidate_prediction,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    class_indices = [
        np.flatnonzero(
            y == class_index
        )
        for class_index in labels
    ]

    rng = np.random.default_rng(
        random_state
    )

    deltas = np.empty(
        repeats,
        dtype=np.float64,
    )

    for repeat in range(
        repeats
    ):
        sampled_indices = np.concatenate(
            [
                rng.choice(
                    indices,
                    size=len(indices),
                    replace=True,
                )
                for indices
                in class_indices
            ]
        )

        sampled_baseline_score = f1_score(
            y[sampled_indices],
            baseline_prediction[
                sampled_indices
            ],
            labels=labels,
            average="macro",
            zero_division=0,
        )

        sampled_candidate_score = f1_score(
            y[sampled_indices],
            candidate_prediction[
                sampled_indices
            ],
            labels=labels,
            average="macro",
            zero_division=0,
        )

        deltas[repeat] = (
            sampled_candidate_score
            - sampled_baseline_score
        )

    ci_lower, ci_upper = np.quantile(
        deltas,
        [
            0.025,
            0.975,
        ],
    )

    return {
        "iterations": int(
            repeats
        ),
        "random_state": int(
            random_state
        ),
        "baseline_macro_f1": float(
            baseline_score
        ),
        "candidate_macro_f1": float(
            candidate_score
        ),
        "observed_delta": float(
            candidate_score
            - baseline_score
        ),
        "mean_delta": float(
            deltas.mean()
        ),
        "ci_lower": float(
            ci_lower
        ),
        "ci_upper": float(
            ci_upper
        ),
        "ci_lower_ge_0": bool(
            ci_lower >= 0
        ),
        "probability_positive": float(
            np.mean(
                deltas > 0
            )
        ),
    }
def write_experiment_report(
    path: Path,
    config_path: Path,
    config: dict,
    train_rows: int,
    seed_metrics: pd.DataFrame,
    selected_parameters: pd.DataFrame,
    bootstrap: dict,
    submission_path: Path,
    summary_path: Path,
) -> None:
    """앙상블 실행 결과를 Markdown 보고서 형식으로 저장한다."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    project = config["project"]
    validation = config["validation"]

    baseline_mean = float(
        seed_metrics[
            "e10b1_macro_f1"
        ].mean()
    )

    baseline_std = float(
        seed_metrics[
            "e10b1_macro_f1"
        ].std(ddof=0)
    )

    candidate_mean = float(
        seed_metrics[
            "e10b2_macro_f1"
        ].mean()
    )

    candidate_std = float(
        seed_metrics[
            "e10b2_macro_f1"
        ].std(ddof=0)
    )

    mean_delta = float(
        seed_metrics[
            "delta"
        ].mean()
    )

    improved_seed_count = int(
        (
            seed_metrics["delta"]
            > 0
        ).sum()
    )

    improved_fold_count = int(
        (
            selected_parameters[
                "evaluation_delta"
            ]
            > 0
        ).sum()
    )

    seed_table = dataframe_to_markdown(
        seed_metrics,
    )

    parameter_table = dataframe_to_markdown(
        selected_parameters,
    )

    report = f"""# {project["experiment_name"]}

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | {datetime.now().astimezone().isoformat(timespec="seconds")} |
| 실험 유형 | 앙상블 |
| 검증 | StratifiedGroupKFold {validation["n_splits"]}-Fold × {len(validation["seeds"])} seeds |
| 학습 데이터 행 수 | {train_rows} |
| 기준 모델 OOF Macro F1 | {baseline_mean:.6f} ± {baseline_std:.6f} |
| 후보 모델 OOF Macro F1 | {candidate_mean:.6f} ± {candidate_std:.6f} |
| Seed별 평균 변화 | {mean_delta:+.6f} |
| 최종 기준 모델 OOF | {bootstrap["baseline_macro_f1"]:.6f} |
| 최종 후보 모델 OOF | {bootstrap["candidate_macro_f1"]:.6f} |
| 최종 OOF 변화 | {bootstrap["observed_delta"]:+.6f} |
| 개선 Seed | {improved_seed_count}/{len(seed_metrics)} |
| 개선 Fold | {improved_fold_count}/{len(selected_parameters)} |
| Bootstrap 평균 변화 | {bootstrap["mean_delta"]:+.6f} |
| Bootstrap 95% CI | [{bootstrap["ci_lower"]:+.6f}, {bootstrap["ci_upper"]:+.6f}] |
| P(변화 > 0) | {bootstrap["probability_positive"]:.4f} |
| CI 하한 ≥ 0 | {bootstrap["ci_lower_ge_0"]} |
| 설정 파일 | `{config_path}` |
| 제출 파일 | `{submission_path}` |
| 요약 JSON | `{summary_path}` |

## Seed별 OOF

{seed_table}

## Fold별 선택 파라미터

{parameter_table}
"""

    path.write_text(
        report,
        encoding="utf-8",
    )

def normalize_test_baseline(
    baseline_test: np.ndarray,
    n_seeds: int,
    n_splits: int,
    n_test: int,
    n_classes: int,
) -> np.ndarray:
    """E10B-1 Test 배열을 최종 평균 확률로 변환한다."""
    expected_mean_shape = (
        n_test,
        n_classes,
    )

    expected_fold_shape = (
        n_seeds,
        n_splits,
        n_test,
        n_classes,
    )

    if (
        baseline_test.shape
        == expected_mean_shape
    ):
        result = np.asarray(
            baseline_test,
            dtype=np.float64,
        )

    elif (
        baseline_test.shape
        == expected_fold_shape
    ):
        result = baseline_test.mean(
            axis=(0, 1)
        )

    else:
        raise ValueError(
            "E10B-1 Test 배열 shape가 "
            "잘못되었습니다. "
            f"현재={baseline_test.shape}, "
            f"허용={expected_mean_shape} "
            f"또는 {expected_fold_shape}"
        )

    result /= result.sum(
        axis=1,
        keepdims=True,
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/ensembles/test_008_v06.yaml"
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    project = config["project"]
    data = config["data"]
    validation = config["validation"]
    ensemble = config["ensemble"]

    experiment_name = str(
        project["experiment_name"]
    )

    raw_dir = Path(
        data["raw_dir"]
    )

    output_dir = Path(
        data["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = pd.read_csv(
        raw_dir
        / data["train_file"]
    )

    test = pd.read_csv(
        raw_dir
        / data["test_file"]
    )

    submission = pd.read_csv(
        raw_dir
        / data["submission_file"]
    )

    target_column = str(
        data["target_column"]
    )

    id_column = str(
        data["id_column"]
    )

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        train[
            target_column
        ].astype(str)
    )

    class_names = (
        label_encoder.classes_
    )

    n_classes = len(
        class_names
    )

    labels = np.arange(
        n_classes
    )

    seeds = [
        int(seed)
        for seed
        in validation["seeds"]
    ]

    n_splits = int(
        validation["n_splits"]
    )

    folds = load_fixed_folds(
        Path(
            data[
                "fixed_split_file"
            ]
        ),
        train[
            id_column
        ],
        seeds,
        n_splits,
        id_column,
    )

    baseline_dir = Path(
        data["baseline_dir"]
    )

    numeric_dir = Path(
        data["numeric_dir"]
    )

    baseline_oof = np.load(
        baseline_dir
        / data[
            "baseline_oof_file"
        ]
    )

    baseline_test = np.load(
        baseline_dir
        / data[
            "baseline_test_file"
        ]
    )

    numeric_oof = np.load(
        numeric_dir
        / data[
            "numeric_oof_file"
        ]
    )

    numeric_test = np.load(
        numeric_dir
        / data[
            "numeric_test_file"
        ]
    )

    expected_oof_shape = (
        len(seeds),
        len(train),
        n_classes,
    )

    expected_test_fold_shape = (
        len(seeds),
        n_splits,
        len(test),
        n_classes,
    )

    if (
        baseline_oof.shape
        != expected_oof_shape
    ):
        raise ValueError(
            "E10B-1 OOF shape 불일치: "
            f"{baseline_oof.shape} != "
            f"{expected_oof_shape}"
        )

    if (
        numeric_oof.shape
        != expected_oof_shape
    ):
        raise ValueError(
            "Numeric OOF shape 불일치: "
            f"{numeric_oof.shape} != "
            f"{expected_oof_shape}"
        )

    if (
        numeric_test.shape
        != expected_test_fold_shape
    ):
        raise ValueError(
            "Numeric Test shape 불일치: "
            f"{numeric_test.shape} != "
            f"{expected_test_fold_shape}"
        )

    for array_name, array in [
        (
            "baseline_oof",
            baseline_oof,
        ),
        (
            "baseline_test",
            baseline_test,
        ),
        (
            "numeric_oof",
            numeric_oof,
        ),
        (
            "numeric_test",
            numeric_test,
        ),
    ]:
        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                f"{array_name}에 "
                "NaN 또는 무한대가 있습니다."
            )

    baseline_test_mean = (
        normalize_test_baseline(
            baseline_test,
            len(seeds),
            n_splits,
            len(test),
            n_classes,
        )
    )

    temperatures = [
        float(value)
        for value
        in ensemble[
            "numeric_temperatures"
        ]
    ]

    weights = [
        float(value)
        for value
        in ensemble[
            "numeric_weights"
        ]
    ]

    candidate_oof, selected_parameters = (
        select_crossfit_parameters(
            y=y,
            baseline_oof=baseline_oof,
            numeric_oof=numeric_oof,
            folds=folds,
            seeds=seeds,
            n_splits=n_splits,
            temperatures=temperatures,
            weights=weights,
            n_classes=n_classes,
        )
    )

    seed_rows: list[dict] = []

    for seed_position, seed in enumerate(
        seeds
    ):
        baseline_prediction = (
            baseline_oof[
                seed_position
            ].argmax(axis=1)
        )

        candidate_prediction = (
            candidate_oof[
                seed_position
            ].argmax(axis=1)
        )

        baseline_score = f1_score(
            y,
            baseline_prediction,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        candidate_score = f1_score(
            y,
            candidate_prediction,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        seed_rows.append(
            {
                "seed": seed,
                "e10b1_macro_f1": float(
                    baseline_score
                ),
                "e10b2_macro_f1": float(
                    candidate_score
                ),
                "delta": float(
                    candidate_score
                    - baseline_score
                ),
            }
        )

    seed_metrics = pd.DataFrame(
        seed_rows
    )

    bootstrap = (
        paired_stratified_bootstrap(
            y=y,
            baseline_oof=baseline_oof,
            candidate_oof=candidate_oof,
            repeats=int(
                ensemble[
                    "bootstrap_repeats"
                ]
            ),
            random_state=int(
                ensemble[
                    "bootstrap_seed"
                ]
            ),
            n_classes=n_classes,
        )
    )

    test_probability_by_seed_fold = (
        np.zeros(
            expected_test_fold_shape,
            dtype=np.float32,
        )
    )

    for row in selected_parameters.itertuples(
        index=False
    ):
        seed = int(
            row.seed
        )

        fold = int(
            row.fold
        )

        seed_position = seeds.index(
            seed
        )

        test_probability_by_seed_fold[
            seed_position,
            fold,
        ] = (
            blend_numeric_probabilities(
                baseline_test_mean,
                numeric_test[
                    seed_position,
                    fold,
                ],
                float(
                    row.temperature
                ),
                float(
                    row.numeric_weight
                ),
            ).astype(
                np.float32
            )
        )

    test_probability_mean = (
        test_probability_by_seed_fold.mean(
            axis=(0, 1)
        )
    )

    test_probability_mean /= (
        test_probability_mean.sum(
            axis=1,
            keepdims=True,
        )
    )

    test_prediction = (
        test_probability_mean.argmax(
            axis=1
        )
    )

    if len(
        np.unique(
            test_prediction
        )
    ) != n_classes:
        missing_classes = sorted(
            set(
                range(
                    n_classes
                )
            )
            - set(
                np.unique(
                    test_prediction
                )
            )
        )

        print(
            "경고: Test 예측이 0개인 클래스:",
            [
                class_names[index]
                for index
                in missing_classes
            ],
        )

    if id_column in submission.columns:
        if not np.array_equal(
            submission[
                id_column
            ].astype(str).to_numpy(),
            test[
                id_column
            ].astype(str).to_numpy(),
        ):
            raise ValueError(
                "submission과 Test의 ID "
                "순서가 다릅니다."
            )

    submission[
        target_column
    ] = class_names[
        test_prediction
    ]

    submission_path = (
        output_dir
        / (
            f"submission_"
            f"{experiment_name}.csv"
        )
    )

    submission.to_csv(
        submission_path,
        index=False,
    )

    selected_parameters.to_csv(
        output_dir
        / "selected_parameters.csv",
        index=False,
    )

    seed_metrics.to_csv(
        output_dir
        / "seed_oof_metrics.csv",
        index=False,
    )

    np.save(
        output_dir
        / "oof_probabilities_by_seed.npy",
        candidate_oof.astype(
            np.float32
        ),
    )

    np.save(
        output_dir
        / "test_probability_by_seed_fold.npy",
        test_probability_by_seed_fold,
    )

    np.save(
        output_dir
        / "test_probability_mean.npy",
        test_probability_mean.astype(
            np.float32
        ),
    )

    pd.DataFrame(
        {
            "class_index": labels,
            target_column: class_names,
        }
    ).to_csv(
        output_dir
        / "class_order.csv",
        index=False,
    )

    seed_mean = float(
        seed_metrics[
            "e10b2_macro_f1"
        ].mean()
    )

    seed_std = float(
        seed_metrics[
            "e10b2_macro_f1"
        ].std(
            ddof=0
        )
    )

    summary = {
        "experiment": experiment_name,
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),
        "model": (
            "E10B-1 + Numeric Structural "
            "LightGBM cross-fit ensemble"
        ),
        "cv_strategy": (
            "StratifiedGroupKFold"
        ),
        "n_splits": n_splits,
        "seeds": seeds,
        "model_count": (
            len(seeds)
            * n_splits
        ),
        "oof_macro_f1_seed_mean": (
            seed_mean
        ),
        "oof_macro_f1_seed_std": (
            seed_std
        ),
        "seed_mean_delta": float(
            seed_metrics[
                "delta"
            ].mean()
        ),
        "improved_seed_count": int(
            (
                seed_metrics[
                    "delta"
                ]
                > 0
            ).sum()
        ),
        "improved_fold_count": int(
            (
                selected_parameters[
                    "evaluation_delta"
                ]
                > 0
            ).sum()
        ),
        "total_fold_count": int(
            len(
                selected_parameters
            )
        ),
        "mean_probability_macro_f1": float(
            bootstrap[
                "candidate_macro_f1"
            ]
        ),
        "baseline_mean_probability_macro_f1": float(
            bootstrap[
                "baseline_macro_f1"
            ]
        ),
        "bootstrap": bootstrap,
        "adopted_by_internal_rule": bool(
            bootstrap[
                "ci_lower_ge_0"
            ]
        ),
        "submission": str(
            submission_path
        ),
    }

    summary_path = (
        output_dir
        / (
            f"{experiment_name}"
            "_summary.json"
        )
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = (
        Path("experiments")
        / "ensembles"
        / f"{experiment_name}.md"
    )
    write_experiment_report(
        path=report_path,
        config_path=args.config,
        config=config,
        train_rows=len(train),
        seed_metrics=seed_metrics,
        selected_parameters=selected_parameters,
        bootstrap=bootstrap,
        submission_path=submission_path,
        summary_path=summary_path,
    )

    print(
        "실험 보고서:",
        report_path,
    )

def dataframe_to_markdown(
    frame: pd.DataFrame,
    float_digits: int = 6,
) -> str:
    """추가 패키지 없이 DataFrame을 Markdown 표로 변환한다."""
    columns = [
        str(column)
        for column in frame.columns
    ]

    def format_value(
        value: object,
    ) -> str:
        if isinstance(
            value,
            (
                float,
                np.floating,
            ),
        ):
            return format(
                float(value),
                f".{float_digits}f",
            )
                

        return str(
            value
        ).replace(
            "|",
            "\\|",
        )

    rows = [
        (
            "| "
            + " | ".join(
                columns
            )
            + " |"
        ),
        (
            "| "
            + " | ".join(
                "---"
                for _ in columns
            )
            + " |"
        ),
    ]

    rows.extend(
        (
            "| "
            + " | ".join(
                format_value(
                    value
                )
                for value in row
            )
            + " |"
        )
        for row in frame.itertuples(
            index=False,
            name=None,
        )
    )

    return "\n".join(
        rows
    )


if __name__ == "__main__":
    main()
