"""E7 LR C=0.1과 TF-IDF LinearSVC C=0.2의 seed 42 cross-fit 결합."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    classification_report,
    f1_score,
)


def load_config(
    path: Path,
) -> dict:
    with path.open(
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def dataframe_to_markdown(
    frame: pd.DataFrame,
    float_digits: int = 6,
) -> str:
    columns = [
        str(column)
        for column in frame.columns
    ]

    def format_value(
        value: object,
    ) -> str:
        if isinstance(
            value,
            (float, np.floating),
        ):
            if np.isnan(value):
                return ""

            return format(
                float(value),
                f".{float_digits}f",
            )

        if isinstance(
            value,
            (bool, np.bool_),
        ):
            return str(bool(value))

        return (
            str(value)
            .replace("|", "\\|")
        )

    rows = [
        "| " + " | ".join(columns) + " |",
        (
            "| "
            + " | ".join(
                "---"
                for _ in columns
            )
            + " |"
        ),
    ]

    for row in frame.itertuples(
        index=False,
        name=None,
    ):
        rows.append(
            "| "
            + " | ".join(
                format_value(value)
                for value in row
            )
            + " |"
        )

    return "\n".join(rows)


def softmax_with_temperature(
    decision: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(
            "temperature는 0보다 커야 합니다."
        )

    scaled = (
        decision
        / float(temperature)
    )

    scaled = (
        scaled
        - scaled.max(
            axis=1,
            keepdims=True,
        )
    )

    exponential = np.exp(
        scaled
    )

    denominator = exponential.sum(
        axis=1,
        keepdims=True,
    )

    return (
        exponential
        / denominator
    )


def macro_f1(
    y_true: np.ndarray,
    prediction: np.ndarray,
    n_classes: int,
) -> float:
    return float(
        f1_score(
            y_true,
            prediction,
            labels=np.arange(
                n_classes
            ),
            average="macro",
            zero_division=0,
        )
    )


def load_seed_output(
    directory: Path,
    array_filename: str,
    seed_metrics_filename: str,
    seed: int,
) -> np.ndarray:
    array_path = (
        directory
        / array_filename
    )

    seed_metrics_path = (
        directory
        / seed_metrics_filename
    )

    if not array_path.exists():
        raise FileNotFoundError(
            f"OOF 파일이 없습니다: "
            f"{array_path}"
        )

    if not seed_metrics_path.exists():
        raise FileNotFoundError(
            f"Seed 지표 파일이 없습니다: "
            f"{seed_metrics_path}"
        )

    values = np.load(
        array_path
    )

    seed_metrics = pd.read_csv(
        seed_metrics_path
    )

    if "seed" not in seed_metrics:
        raise ValueError(
            f"{seed_metrics_path}에 seed 열이 "
            "없습니다."
        )

    if values.ndim != 3:
        raise ValueError(
            f"{array_path}는 "
            "(seed, sample, class) 형태여야 합니다. "
            f"현재 shape={values.shape}"
        )

    if values.shape[0] != len(
        seed_metrics
    ):
        raise ValueError(
            f"{array_path}의 seed 축과 "
            f"{seed_metrics_path} 행 수가 다릅니다."
        )

    matched = np.flatnonzero(
        seed_metrics[
            "seed"
        ].to_numpy(
            dtype=int
        )
        == int(seed)
    )

    if len(matched) != 1:
        raise ValueError(
            f"seed={seed} 결과가 정확히 "
            f"1개가 아닙니다."
        )

    return np.asarray(
        values[
            matched[0]
        ],
        dtype=np.float64,
    )


def load_class_names(
    path: Path,
    target_column: str,
) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"클래스 순서 파일이 없습니다: "
            f"{path}"
        )

    frame = pd.read_csv(
        path
    )

    if target_column not in frame:
        raise ValueError(
            f"{path}에 {target_column} 열이 "
            "없습니다."
        )

    return (
        frame[
            target_column
        ]
        .astype(str)
        .tolist()
    )


def encode_labels(
    labels: pd.Series,
    class_names: list[str],
) -> np.ndarray:
    mapping = {
        class_name: index
        for index, class_name
        in enumerate(
            class_names
        )
    }

    encoded = (
        labels.astype(str)
        .map(mapping)
    )

    if encoded.isna().any():
        unknown = sorted(
            labels.loc[
                encoded.isna()
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"클래스 순서에 없는 라벨이 있습니다: "
            f"{unknown}"
        )

    return encoded.to_numpy(
        dtype=np.int64
    )


def select_parameters(
    y: np.ndarray,
    tuning_mask: np.ndarray,
    logistic_probability: np.ndarray,
    linear_svc_decision: np.ndarray,
    temperatures: list[float],
    linear_svc_weights: list[float],
    n_classes: int,
) -> dict:
    best_result: dict | None = None

    for temperature in temperatures:
        svc_probability = (
            softmax_with_temperature(
                linear_svc_decision,
                temperature,
            )
        )

        for weight in linear_svc_weights:
            weight = float(
                weight
            )

            blended = (
                (
                    1.0
                    - weight
                )
                * logistic_probability
                + weight
                * svc_probability
            )

            prediction = (
                blended[
                    tuning_mask
                ]
                .argmax(
                    axis=1
                )
            )

            score = macro_f1(
                y_true=y[
                    tuning_mask
                ],
                prediction=prediction,
                n_classes=n_classes,
            )

            result = {
                "temperature": float(
                    temperature
                ),
                "linear_svc_weight": weight,
                "tuning_macro_f1": float(
                    score
                ),
            }

            if best_result is None:
                best_result = result
                continue

            if (
                result[
                    "tuning_macro_f1"
                ]
                > best_result[
                    "tuning_macro_f1"
                ]
                + 1e-12
            ):
                best_result = result
                continue

            if np.isclose(
                result[
                    "tuning_macro_f1"
                ],
                best_result[
                    "tuning_macro_f1"
                ],
            ):
                if (
                    result[
                        "linear_svc_weight"
                    ]
                    < best_result[
                        "linear_svc_weight"
                    ]
                ):
                    best_result = result

    if best_result is None:
        raise RuntimeError(
            "선택된 결합 파라미터가 없습니다."
        )

    return best_result


def stratified_paired_bootstrap(
    y: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    n_classes: int,
    iterations: int,
    random_state: int,
) -> dict:
    rng = np.random.default_rng(
        random_state
    )

    class_indices = [
        np.flatnonzero(
            y == class_index
        )
        for class_index in range(
            n_classes
        )
    ]

    if any(
        len(indices) == 0
        for indices in class_indices
    ):
        raise ValueError(
            "표본이 0개인 클래스가 있습니다."
        )

    deltas = np.empty(
        iterations,
        dtype=np.float64,
    )

    for iteration in range(
        iterations
    ):
        sampled = np.concatenate(
            [
                rng.choice(
                    indices,
                    size=len(indices),
                    replace=True,
                )
                for indices in class_indices
            ]
        )

        baseline_score = macro_f1(
            y_true=y[sampled],
            prediction=(
                baseline_prediction[
                    sampled
                ]
            ),
            n_classes=n_classes,
        )

        candidate_score = macro_f1(
            y_true=y[sampled],
            prediction=(
                candidate_prediction[
                    sampled
                ]
            ),
            n_classes=n_classes,
        )

        deltas[
            iteration
        ] = (
            candidate_score
            - baseline_score
        )

    ci_lower, ci_upper = (
        np.quantile(
            deltas,
            [
                0.025,
                0.975,
            ],
        )
    )

    return {
        "iterations": int(
            iterations
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


def build_class_comparison(
    y: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    n_classes = len(
        class_names
    )

    baseline_report = pd.DataFrame(
        classification_report(
            y,
            baseline_prediction,
            labels=np.arange(
                n_classes
            ),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).T.loc[
        class_names
    ]

    candidate_report = pd.DataFrame(
        classification_report(
            y,
            candidate_prediction,
            labels=np.arange(
                n_classes
            ),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).T.loc[
        class_names
    ]

    comparison = pd.DataFrame(
        {
            "SUBCLASS": class_names,
            "support": (
                baseline_report[
                    "support"
                ].to_numpy()
            ),
            "baseline_f1": (
                baseline_report[
                    "f1-score"
                ].to_numpy()
            ),
            "candidate_f1": (
                candidate_report[
                    "f1-score"
                ].to_numpy()
            ),
            "baseline_precision": (
                baseline_report[
                    "precision"
                ].to_numpy()
            ),
            "candidate_precision": (
                candidate_report[
                    "precision"
                ].to_numpy()
            ),
            "baseline_recall": (
                baseline_report[
                    "recall"
                ].to_numpy()
            ),
            "candidate_recall": (
                candidate_report[
                    "recall"
                ].to_numpy()
            ),
        }
    )

    comparison[
        "f1_delta"
    ] = (
        comparison[
            "candidate_f1"
        ]
        - comparison[
            "baseline_f1"
        ]
    )

    comparison[
        "precision_delta"
    ] = (
        comparison[
            "candidate_precision"
        ]
        - comparison[
            "baseline_precision"
        ]
    )

    comparison[
        "recall_delta"
    ] = (
        comparison[
            "candidate_recall"
        ]
        - comparison[
            "baseline_recall"
        ]
    )

    return comparison.sort_values(
        "f1_delta"
    ).reset_index(
        drop=True
    )


def write_report(
    path: Path,
    config_path: Path,
    config: dict,
    started_at: datetime,
    finished_at: datetime,
    summary: dict,
    selected_parameters: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    class_comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameter_table = (
        dataframe_to_markdown(
            selected_parameters
        )
    )

    fold_table = (
        dataframe_to_markdown(
            fold_metrics
        )
    )

    class_table = (
        dataframe_to_markdown(
            class_comparison
        )
    )

    elapsed = (
        finished_at
        - started_at
    )

    experiment_name = str(
        config["project"][
            "experiment_name"
        ]
    )

    bootstrap = summary[
        "bootstrap"
    ]

    report = f"""# {experiment_name}

| 항목 | 결과 |
| --- | --- |
| 설명 | {config["project"].get("description", "")} |
| 실행 시작 | {started_at.isoformat(timespec="seconds")} |
| 실행 종료 | {finished_at.isoformat(timespec="seconds")} |
| 총 실행 시간 | {elapsed} |
| Seed | {int(config["validation"]["seed"])} |
| 기준 OOF Macro F1 | {float(summary["baseline_oof_macro_f1"]):.6f} |
| LinearSVC OOF Macro F1 | {float(summary["linear_svc_oof_macro_f1"]):.6f} |
| 결합 OOF Macro F1 | {float(summary["blend_oof_macro_f1"]):.6f} |
| 기준 대비 변화 | {float(summary["observed_delta"]):+.6f} |
| 개선 Fold 수 | {int(summary["improved_fold_count"])}/{int(config["validation"]["n_splits"])} |
| OOF 예측 변화율 | {float(summary["prediction_change_rate"]):.6f} |
| 오답→정답 | {int(summary["fixed_count"])} |
| 정답→오답 | {int(summary["broken_count"])} |
| 순교정 | {int(summary["net_correction"]):+d} |
| Bootstrap 평균 변화 | {float(bootstrap["mean_delta"]):+.6f} |
| Bootstrap 95% CI | [{float(bootstrap["ci_lower"]):+.6f}, {float(bootstrap["ci_upper"]):+.6f}] |
| P(변화 > 0) | {float(bootstrap["probability_positive"]):.4f} |
| 내부 판정 | {bool(summary["adopted_by_internal_rule"])} |
| 설정 파일 | `{config_path}` |
| 결과 폴더 | `{output_dir}` |

## 선택 파라미터

{parameter_table}

## Fold별 결과

{fold_table}

## 암종별 결과

{class_table}

## 설정

```yaml
{yaml.safe_dump(config, allow_unicode=True, sort_keys=False).rstrip()}
```
"""
    
    path.write_text(
        report,
        encoding="utf-8",
    )

    json_path = (
        path.parent
        / f"{path.stem}.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "experiment_name": experiment_name,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "elapsed_seconds": elapsed.total_seconds(),
                "summary": summary,
                "selected_parameters": selected_parameters.to_dict(
                    orient="records"
                ),
                "fold_metrics": fold_metrics.to_dict(
                    orient="records"
                ),
                "class_comparison": class_comparison.to_dict(
                    orient="records"
                ),
            },
            file,
            ensure_ascii=False,
            indent=4,
        )
def main() -> None:
    started_at = (
        datetime.now()
        .astimezone()
    )
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/"
            "test_002_e11_blend_smoke.yaml"
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    data_config = config["data"]
    input_config = config["inputs"]
    validation_config = (
        config["validation"]
    )

    raw_dir = Path(
        data_config["raw_dir"]
    )

    output_dir = Path(
        data_config["processed_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = pd.read_csv(
        raw_dir
        / data_config["train_file"]
    )

    target_column = str(
        data_config["target_column"]
    )

    id_column = str(
        data_config["id_column"]
    )

    seed = int(
        validation_config["seed"]
    )

    n_splits = int(
        validation_config[
            "n_splits"
        ]
    )

    fold_column = str(
        validation_config[
            "fold_column"
        ]
    )

    logistic_dir = Path(
        input_config[
            "logistic_dir"
        ]
    )

    linear_svc_dir = Path(
        input_config[
            "linear_svc_dir"
        ]
    )

    logistic_probability = (
        load_seed_output(
            directory=logistic_dir,
            array_filename=(
                input_config[
                    "logistic_oof_file"
                ]
            ),
            seed_metrics_filename=(
                input_config[
                    "logistic_seed_metrics_file"
                ]
            ),
            seed=seed,
        )
    )

    linear_svc_decision = (
        load_seed_output(
            directory=linear_svc_dir,
            array_filename=(
                input_config[
                    "linear_svc_oof_file"
                ]
            ),
            seed_metrics_filename=(
                input_config[
                    "linear_svc_seed_metrics_file"
                ]
            ),
            seed=seed,
        )
    )

    logistic_class_names = (
        load_class_names(
            path=(
                logistic_dir
                / input_config[
                    "logistic_class_order_file"
                ]
            ),
            target_column=target_column,
        )
    )

    linear_svc_class_names = (
        load_class_names(
            path=(
                linear_svc_dir
                / input_config[
                    "linear_svc_class_order_file"
                ]
            ),
            target_column=target_column,
        )
    )

    if (
        logistic_class_names
        != linear_svc_class_names
    ):
        raise ValueError(
            "Logistic Regression과 LinearSVC의 "
            "클래스 순서가 다릅니다."
        )

    class_names = (
        logistic_class_names
    )

    n_classes = len(
        class_names
    )

    if logistic_probability.shape != (
        len(train),
        n_classes,
    ):
        raise ValueError(
            "Logistic Regression OOF shape가 "
            f"잘못되었습니다: "
            f"{logistic_probability.shape}"
        )

    if linear_svc_decision.shape != (
        len(train),
        n_classes,
    ):
        raise ValueError(
            "LinearSVC OOF shape가 "
            f"잘못되었습니다: "
            f"{linear_svc_decision.shape}"
        )

    y = encode_labels(
        labels=train[
            target_column
        ],
        class_names=class_names,
    )

    fixed_split_path = Path(
        data_config[
            "fixed_split_file"
        ]
    )

    fixed_frame = pd.read_csv(
        fixed_split_path
    )

    if id_column not in fixed_frame:
        raise ValueError(
            f"고정 split에 {id_column} 열이 "
            "없습니다."
        )

    if not np.array_equal(
        fixed_frame[
            id_column
        ]
        .astype(str)
        .to_numpy(),
        train[
            id_column
        ]
        .astype(str)
        .to_numpy(),
    ):
        raise ValueError(
            "Train과 고정 split의 ID 순서가 "
            "다릅니다."
        )

    if fold_column not in fixed_frame:
        raise ValueError(
            f"고정 split에 {fold_column} 열이 "
            "없습니다."
        )

    fold_ids = fixed_frame[
        fold_column
    ].to_numpy(
        dtype=np.int8
    )

    if set(
        np.unique(
            fold_ids
        )
    ) != set(
        range(
            n_splits
        )
    ):
        raise ValueError(
            "고정 split의 Fold 값이 "
            "잘못되었습니다."
        )

    temperatures = [
        float(value)
        for value
        in config["grid"][
            "temperatures"
        ]
    ]

    linear_svc_weights = [
        float(value)
        for value
        in config["grid"][
            "linear_svc_weights"
        ]
    ]

    crossfit_probability = np.zeros_like(
        logistic_probability,
        dtype=np.float64,
    )

    selected_rows = []
    fold_rows = []

    for evaluation_fold in range(
        n_splits
    ):
        tuning_mask = (
            fold_ids
            != evaluation_fold
        )

        evaluation_mask = (
            fold_ids
            == evaluation_fold
        )

        selected = select_parameters(
            y=y,
            tuning_mask=tuning_mask,
            logistic_probability=(
                logistic_probability
            ),
            linear_svc_decision=(
                linear_svc_decision
            ),
            temperatures=temperatures,
            linear_svc_weights=(
                linear_svc_weights
            ),
            n_classes=n_classes,
        )

        temperature = float(
            selected["temperature"]
        )

        linear_svc_weight = float(
            selected[
                "linear_svc_weight"
            ]
        )

        svc_probability = (
            softmax_with_temperature(
                linear_svc_decision,
                temperature,
            )
        )

        blended = (
            (
                1.0
                - linear_svc_weight
            )
            * logistic_probability
            + linear_svc_weight
            * svc_probability
        )

        crossfit_probability[
            evaluation_mask
        ] = blended[
            evaluation_mask
        ]

        logistic_fold_prediction = (
            logistic_probability[
                evaluation_mask
            ]
            .argmax(
                axis=1
            )
        )

        svc_fold_prediction = (
            linear_svc_decision[
                evaluation_mask
            ]
            .argmax(
                axis=1
            )
        )

        blend_fold_prediction = (
            blended[
                evaluation_mask
            ]
            .argmax(
                axis=1
            )
        )

        logistic_fold_score = macro_f1(
            y_true=y[
                evaluation_mask
            ],
            prediction=(
                logistic_fold_prediction
            ),
            n_classes=n_classes,
        )

        svc_fold_score = macro_f1(
            y_true=y[
                evaluation_mask
            ],
            prediction=(
                svc_fold_prediction
            ),
            n_classes=n_classes,
        )

        blend_fold_score = macro_f1(
            y_true=y[
                evaluation_mask
            ],
            prediction=(
                blend_fold_prediction
            ),
            n_classes=n_classes,
        )

        selected_rows.append(
            {
                "seed": seed,
                "evaluation_fold": (
                    evaluation_fold
                ),
                "temperature": (
                    temperature
                ),
                "linear_svc_weight": (
                    linear_svc_weight
                ),
                "tuning_macro_f1": float(
                    selected[
                        "tuning_macro_f1"
                    ]
                ),
            }
        )

        fold_rows.append(
            {
                "seed": seed,
                "fold": evaluation_fold,
                "sample_count": int(
                    evaluation_mask.sum()
                ),
                "baseline_macro_f1": (
                    logistic_fold_score
                ),
                "linear_svc_macro_f1": (
                    svc_fold_score
                ),
                "blend_macro_f1": (
                    blend_fold_score
                ),
                "delta_vs_baseline": float(
                    blend_fold_score
                    - logistic_fold_score
                ),
            }
        )

        print(
            f"fold={evaluation_fold} "
            f"T={temperature:.2f} "
            f"SVC weight="
            f"{linear_svc_weight:.2f} "
            f"LR={logistic_fold_score:.6f} "
            f"SVC={svc_fold_score:.6f} "
            f"Blend={blend_fold_score:.6f} "
            f"Delta="
            f"{blend_fold_score - logistic_fold_score:+.6f}"
        )

    baseline_prediction = (
        logistic_probability.argmax(
            axis=1
        )
    )

    linear_svc_prediction = (
        linear_svc_decision.argmax(
            axis=1
        )
    )

    blend_prediction = (
        crossfit_probability.argmax(
            axis=1
        )
    )

    baseline_score = macro_f1(
        y_true=y,
        prediction=baseline_prediction,
        n_classes=n_classes,
    )

    linear_svc_score = macro_f1(
        y_true=y,
        prediction=linear_svc_prediction,
        n_classes=n_classes,
    )

    blend_score = macro_f1(
        y_true=y,
        prediction=blend_prediction,
        n_classes=n_classes,
    )

    observed_delta = (
        blend_score
        - baseline_score
    )

    changed = (
        baseline_prediction
        != blend_prediction
    )

    baseline_correct = (
        baseline_prediction
        == y
    )

    blend_correct = (
        blend_prediction
        == y
    )

    fixed_count = int(
        (
            changed
            & ~baseline_correct
            & blend_correct
        ).sum()
    )

    broken_count = int(
        (
            changed
            & baseline_correct
            & ~blend_correct
        ).sum()
    )

    net_correction = (
        fixed_count
        - broken_count
    )

    prediction_change_rate = float(
        changed.mean()
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    selected_parameters = pd.DataFrame(
        selected_rows
    )

    improved_fold_count = int(
        (
            fold_metrics[
                "delta_vs_baseline"
            ]
            > 0
        ).sum()
    )

    bootstrap = (
        stratified_paired_bootstrap(
            y=y,
            baseline_prediction=(
                baseline_prediction
            ),
            candidate_prediction=(
                blend_prediction
            ),
            n_classes=n_classes,
            iterations=int(
                config["bootstrap"][
                    "iterations"
                ]
            ),
            random_state=int(
                config["bootstrap"][
                    "random_state"
                ]
            ),
        )
    )

    selection_config = (
        config["selection"]
    )

    adopted_by_internal_rule = bool(
        (
            observed_delta
            >= float(
                selection_config[
                    "minimum_oof_delta"
                ]
            )
        )
        and (
            improved_fold_count
            >= int(
                selection_config[
                    "minimum_improved_fold_count"
                ]
            )
        )
        and (
            (
                not bool(
                    selection_config[
                        "require_bootstrap_ci_lower_ge_0"
                    ]
                )
            )
            or bootstrap[
                "ci_lower_ge_0"
            ]
        )
    )

    class_comparison = (
        build_class_comparison(
            y=y,
            baseline_prediction=(
                baseline_prediction
            ),
            candidate_prediction=(
                blend_prediction
            ),
            class_names=class_names,
        )
    )

    prediction_frame = pd.DataFrame(
        {
            id_column: train[
                id_column
            ],
            target_column: train[
                target_column
            ],
            "fold": fold_ids,
            "baseline_prediction": [
                class_names[index]
                for index
                in baseline_prediction
            ],
            "linear_svc_prediction": [
                class_names[index]
                for index
                in linear_svc_prediction
            ],
            "blend_prediction": [
                class_names[index]
                for index
                in blend_prediction
            ],
            "changed": changed,
            "baseline_correct": (
                baseline_correct
            ),
            "blend_correct": (
                blend_correct
            ),
        }
    )

    np.save(
        output_dir
        / "crossfit_blend_probability.npy",
        crossfit_probability,
    )

    selected_parameters.to_csv(
        output_dir
        / "selected_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_metrics.to_csv(
        output_dir
        / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    class_comparison.to_csv(
        output_dir
        / "class_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_frame.to_csv(
        output_dir
        / "oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    finished_at = (
        datetime.now()
        .astimezone()
    )

    summary = {
        "experiment": config[
            "project"
        ][
            "experiment_name"
        ],
        "created_at": (
            finished_at.isoformat(
                timespec="seconds"
            )
        ),
        "seed": seed,
        "baseline_oof_macro_f1": (
            baseline_score
        ),
        "linear_svc_oof_macro_f1": (
            linear_svc_score
        ),
        "blend_oof_macro_f1": (
            blend_score
        ),
        "observed_delta": float(
            observed_delta
        ),
        "improved_fold_count": (
            improved_fold_count
        ),
        "prediction_change_rate": (
            prediction_change_rate
        ),
        "fixed_count": fixed_count,
        "broken_count": broken_count,
        "net_correction": (
            net_correction
        ),
        "bootstrap": bootstrap,
        "adopted_by_internal_rule": (
            adopted_by_internal_rule
        ),
    }

    summary_path = (
        output_dir
        / "summary.json"
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
        / (
            f"{config['project']['experiment_name']}"
            ".md"
        )
    )

    write_report(
        path=report_path,
        config_path=args.config,
        config=config,
        started_at=started_at,
        finished_at=finished_at,
        summary=summary,
        selected_parameters=(
            selected_parameters
        ),
        fold_metrics=fold_metrics,
        class_comparison=(
            class_comparison
        ),
        output_dir=output_dir,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "E11 seed 42 cross-fit 결합 완료"
    )
    print(
        "=" * 70
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "결과:",
        output_dir,
    )

    print(
        "보고서:",
        report_path,
    )

if __name__ == "__main__":
    main()