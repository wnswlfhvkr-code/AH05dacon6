"""E7 Raw/TF-IDF 피처의 XGBoost seed 42 예비 비교."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    classification_report,
    f1_score,
)
from sklearn.preprocessing import (
    LabelEncoder,
)

from sklearn.utils.class_weight import (
    compute_class_weight,
)
from xgboost import XGBClassifier

from src.pipelines.preprocessing_registry import (
    create_preprocessing_pipeline,
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def dataframe_to_markdown(
    frame: pd.DataFrame,
    float_digits: int = 6,
) -> str:
    columns = [
        str(column)
        for column in frame.columns
    ]

    def format_value(value: object) -> str:
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
        "| "
        + " | ".join(
            "---"
            for _ in columns
        )
        + " |",
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


def make_fold_weights(
    y_train: np.ndarray,
    y_valid: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold-Train 빈도로 Train/Validation 가중치를 만든다."""
    observed_classes = np.unique(
        y_train
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=observed_classes,
        y=y_train,
    )

    weight_lookup = np.ones(
        n_classes,
        dtype=np.float64,
    )

    weight_lookup[
        observed_classes
    ] = class_weights

    train_weights = weight_lookup[
        y_train
    ]

    valid_weights = weight_lookup[
        y_valid
    ]

    return (
        train_weights,
        valid_weights,
    )


def align_probability(
    probability: np.ndarray,
    model_classes: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    aligned = np.zeros(
        (
            len(probability),
            n_classes,
        ),
        dtype=np.float64,
    )

    aligned[
        :,
        model_classes.astype(int),
    ] = probability

    return aligned


def get_best_iteration(
    model: XGBClassifier,
) -> int:
    try:
        return int(
            model.best_iteration
        )
    except (
        AttributeError,
        TypeError,
    ):
        return -1


def create_xgb_model(
    model_config: dict,
    seed: int,
    n_classes: int,
) -> XGBClassifier:
    parameters = {
        key: value
        for key, value
        in model_config.items()
        if key != "name"
    }

    parameters[
        "random_state"
    ] = seed

    parameters[
        "num_class"
    ] = n_classes

    return XGBClassifier(
        **parameters
    )


def write_report(
    path: Path,
    config_path: Path,
    config: dict,
    started_at: datetime,
    finished_at: datetime,
    candidate_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_columns = [
        "candidate",
        "pipeline",
        "oof_macro_f1",
        "reference_oof_macro_f1",
        "delta_vs_reference",
        "train_macro_f1_mean",
        "validation_macro_f1_mean",
        "generalization_gap_mean",
        "generalization_gap_max",
        "feature_count_mean",
        "all_folds_converged",
    ]

    fold_columns = [
        "candidate",
        "pipeline",
        "seed",
        "fold",
        "train_size",
        "validation_size",
        "feature_count",
        "best_iteration",
        "train_macro_f1",
        "validation_macro_f1",
        "generalization_gap",
        "elapsed_seconds",
    ]

    candidate_table = (
        dataframe_to_markdown(
            candidate_metrics[
                candidate_columns
            ]
        )
    )

    fold_table = (
        dataframe_to_markdown(
            fold_metrics[
                fold_columns
            ]
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

    report = f"""# {experiment_name}

| 항목 | 결과 |
| --- | --- |
| 설명 | {config["project"].get("description", "")} |
| 실행 시작 | {started_at.isoformat(timespec="seconds")} |
| 실행 종료 | {finished_at.isoformat(timespec="seconds")} |
| 총 실행 시간 | {elapsed} |
| 기준 모델 | {config["reference"]["name"]} |
| 기준 OOF Macro F1 | {float(config["reference"]["oof_macro_f1"]):.6f} |
| Seed | {config["validation"]["seeds"]} |
| Fold 수 | {int(config["validation"]["n_splits"])} |
| 설정 파일 | `{config_path}` |
| 결과 폴더 | `{output_dir}` |

## 후보별 결과

{candidate_table}

## Fold별 결과

{fold_table}

## 설정

```yaml
{yaml.safe_dump(config, allow_unicode=True, sort_keys=False).rstrip()}
```
"""
    path.write_text(
        report,
        encoding="utf-8",
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
            "test_002_e7_xgb_smoke.yaml"
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    data_config = config["data"]
    validation_config = (
        config["validation"]
    )
    model_config = config["model"]

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

    gene_columns = [
        column
        for column in train.columns
        if column not in {
            id_column,
            target_column,
        }
    ]

    train_features = train[
        gene_columns
    ]

    labels = train[
        target_column
    ]

    label_encoder = (
        LabelEncoder()
        .fit(labels)
    )

    y = label_encoder.transform(
        labels
    )

    class_names = (
        label_encoder.classes_
    )

    n_classes = len(
        class_names
    )

    n_splits = int(
        validation_config[
            "n_splits"
        ]
    )

    seeds = [
        int(seed)
        for seed
        in validation_config["seeds"]
    ]

    if seeds != [42]:
        raise ValueError(
            "예비 실험은 seeds: [42]만 "
            "사용하도록 설정해야 합니다."
        )

    fixed_split_path = Path(
        validation_config[
            "fixed_split_file"
        ]
    )

    fixed_frame = pd.read_csv(
        fixed_split_path
    )

    if not np.array_equal(
        fixed_frame[id_column]
        .astype(str)
        .to_numpy(),
        train[id_column]
        .astype(str)
        .to_numpy(),
    ):
        raise ValueError(
            "Train과 고정 split의 "
            "ID 순서가 다릅니다."
        )

    fold_column = (
        "fold_seed_42"
    )

    if fold_column not in fixed_frame:
        raise ValueError(
            f"고정 split에 {fold_column} "
            "열이 없습니다."
        )

    fold_ids = fixed_frame[
        fold_column
    ].to_numpy(
        dtype=np.int8
    )

    if set(
        np.unique(fold_ids)
    ) != set(
        range(n_splits)
    ):
        raise ValueError(
            "고정 split의 Fold 값이 "
            "잘못되었습니다."
        )

    fold_rows: list[dict] = []
    candidate_rows: list[dict] = []
    class_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    oof_probabilities: dict[
        str,
        np.ndarray,
    ] = {}

    reference_score = float(
        config["reference"][
            "oof_macro_f1"
        ]
    )

    for candidate in config[
        "candidates"
    ]:
        candidate_name = str(
            candidate["name"]
        )

        preprocessing_config = (
            candidate[
                "preprocessing"
            ]
        )

        pipeline_name = str(
            preprocessing_config[
                "name"
            ]
        )

        print()
        print(
            "=" * 70
        )
        print(
            f"후보: {candidate_name} "
            f"| pipeline={pipeline_name}"
        )
        print(
            "=" * 70
        )

        oof_probability = np.zeros(
            (
                len(train),
                n_classes,
            ),
            dtype=np.float64,
        )

        oof_seen = np.zeros(
            len(train),
            dtype=bool,
        )

        candidate_fold_rows = []

        for fold in range(
            n_splits
        ):
            fold_started = time.time()

            train_index = np.flatnonzero(
                fold_ids != fold
            )

            valid_index = np.flatnonzero(
                fold_ids == fold
            )

            preprocessor = (
                create_preprocessing_pipeline(
                    preprocessing_config
                )
            )

            train_matrix = (
                preprocessor.fit_transform(
                    train_features.iloc[
                        train_index
                    ],
                    labels.iloc[
                        train_index
                    ],
                )
            )

            valid_matrix = (
                preprocessor.transform(
                    train_features.iloc[
                        valid_index
                    ]
                )
            )

            feature_summary = (
                preprocessor.summary()
            )

            train_weights, valid_weights = (
                make_fold_weights(
                    y_train=y[train_index],
                    y_valid=y[valid_index],
                    n_classes=n_classes,
                )
            )

            model_seed = (
                42
                + fold
            )

            model = create_xgb_model(
                model_config=model_config,
                seed=model_seed,
                n_classes=n_classes,
            )

            model.fit(
                train_matrix,
                y[train_index],
                sample_weight=train_weights,
                eval_set=[
                    (
                        valid_matrix,
                        y[valid_index],
                    )
                ],
                sample_weight_eval_set=[
                    valid_weights
                ],
                verbose=False,
            )

            train_probability = (
                align_probability(
                    probability=(
                        model.predict_proba(
                            train_matrix
                        )
                    ),
                    model_classes=model.classes_,
                    n_classes=n_classes,
                )
            )

            valid_probability = (
                align_probability(
                    probability=(
                        model.predict_proba(
                            valid_matrix
                        )
                    ),
                    model_classes=model.classes_,
                    n_classes=n_classes,
                )
            )

            train_prediction = (
                train_probability.argmax(
                    axis=1
                )
            )

            valid_prediction = (
                valid_probability.argmax(
                    axis=1
                )
            )

            train_score = f1_score(
                y[train_index],
                train_prediction,
                labels=np.arange(
                    n_classes
                ),
                average="macro",
                zero_division=0,
            )

            validation_score = f1_score(
                y[valid_index],
                valid_prediction,
                labels=np.arange(
                    n_classes
                ),
                average="macro",
                zero_division=0,
            )

            gap = (
                train_score
                - validation_score
            )

            oof_probability[
                valid_index
            ] = valid_probability

            oof_seen[
                valid_index
            ] = True

            elapsed_seconds = (
                time.time()
                - fold_started
            )

            row = {
                "candidate": (
                    candidate_name
                ),
                "pipeline": pipeline_name,
                "seed": 42,
                "fold": fold,
                "train_size": int(
                    len(train_index)
                ),
                "validation_size": int(
                    len(valid_index)
                ),
                "feature_count": int(
                    train_matrix.shape[1]
                ),
                "best_iteration": (
                    get_best_iteration(
                        model
                    )
                ),
                "train_macro_f1": float(
                    train_score
                ),
                "validation_macro_f1": float(
                    validation_score
                ),
                "generalization_gap": float(
                    gap
                ),
                "elapsed_seconds": float(
                    elapsed_seconds
                ),
                "converged": True,
            }

            for key, value in (
                feature_summary.items()
            ):
                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                        np.integer,
                        np.floating,
                    ),
                ):
                    row[
                        f"feature_{key}"
                    ] = value

            fold_rows.append(row)
            candidate_fold_rows.append(
                row
            )

            print(
                f"fold={fold} "
                f"Train={train_score:.6f} "
                f"Validation="
                f"{validation_score:.6f} "
                f"Gap={gap:+.6f} "
                f"features="
                f"{train_matrix.shape[1]} "
                f"best_iteration="
                f"{row['best_iteration']} "
                f"time="
                f"{elapsed_seconds:.1f}s"
            )

        if not oof_seen.all():
            raise RuntimeError(
                f"{candidate_name}의 OOF가 "
                "완성되지 않았습니다."
            )

        oof_prediction = (
            oof_probability.argmax(
                axis=1
            )
        )

        oof_score = f1_score(
            y,
            oof_prediction,
            labels=np.arange(
                n_classes
            ),
            average="macro",
            zero_division=0,
        )

        candidate_fold_frame = (
            pd.DataFrame(
                candidate_fold_rows
            )
        )

        candidate_summary = {
            "candidate": candidate_name,
            "pipeline": pipeline_name,
            "oof_macro_f1": float(
                oof_score
            ),
            "reference_oof_macro_f1": (
                reference_score
            ),
            "delta_vs_reference": float(
                oof_score
                - reference_score
            ),
            "train_macro_f1_mean": float(
                candidate_fold_frame[
                    "train_macro_f1"
                ].mean()
            ),
            "validation_macro_f1_mean": float(
                candidate_fold_frame[
                    "validation_macro_f1"
                ].mean()
            ),
            "generalization_gap_mean": float(
                candidate_fold_frame[
                    "generalization_gap"
                ].mean()
            ),
            "generalization_gap_max": float(
                candidate_fold_frame[
                    "generalization_gap"
                ].max()
            ),
            "feature_count_mean": float(
                candidate_fold_frame[
                    "feature_count"
                ].mean()
            ),
            "all_folds_converged": bool(
                candidate_fold_frame[
                    "converged"
                ].all()
            ),
        }

        candidate_rows.append(
            candidate_summary
        )

        report = pd.DataFrame(
            classification_report(
                y,
                oof_prediction,
                labels=np.arange(
                    n_classes
                ),
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
        ).T.loc[
            class_names
        ].reset_index(
            names=target_column
        )

        report.insert(
            0,
            "candidate",
            candidate_name,
        )

        class_frames.append(
            report
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    id_column: train[
                        id_column
                    ],
                    target_column: labels,
                    "candidate": (
                        candidate_name
                    ),
                    "prediction": (
                        label_encoder
                        .inverse_transform(
                            oof_prediction
                        )
                    ),
                }
            )
        )

        oof_probabilities[
            candidate_name
        ] = oof_probability

        print(
            f"{candidate_name} "
            f"OOF Macro F1="
            f"{oof_score:.6f} "
            f"| 기준 대비 "
            f"{oof_score - reference_score:+.6f}"
        )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    candidate_metrics = pd.DataFrame(
        candidate_rows
    )

    class_metrics = pd.concat(
        class_frames,
        ignore_index=True,
    )

    oof_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    fold_metrics.to_csv(
        output_dir
        / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidate_metrics.to_csv(
        output_dir
        / "candidate_oof_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    class_metrics.to_csv(
        output_dir
        / "class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof_predictions.to_csv(
        output_dir
        / "oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidate_order = [
        str(candidate["name"])
        for candidate
        in config["candidates"]
    ]

    probability_stack = np.stack(
        [
            oof_probabilities[
                candidate_name
            ]
            for candidate_name
            in candidate_order
        ]
    )

    np.save(
        output_dir
        / "oof_probabilities_by_candidate.npy",
        probability_stack,
    )

    pd.DataFrame(
        {
            "candidate_index": np.arange(
                len(candidate_order)
            ),
            "candidate": candidate_order,
        }
    ).to_csv(
        output_dir
        / "candidate_order.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "class_index": np.arange(
                n_classes
            ),
            target_column: class_names,
        }
    ).to_csv(
        output_dir
        / "class_order.csv",
        index=False,
        encoding="utf-8-sig",
    )

    finished_at = (
        datetime.now()
        .astimezone()
    )

    experiment_name = str(
        config["project"][
            "experiment_name"
        ]
    )

    summary = {
        "experiment": experiment_name,
        "created_at": (
            finished_at.isoformat(
                timespec="seconds"
            )
        ),
        "reference": config[
            "reference"
        ],
        "candidate_count": int(
            len(candidate_metrics)
        ),
        "candidates": (
            candidate_metrics.to_dict(
                orient="records"
            )
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
        / f"{experiment_name}.md"
    )

    write_report(
        path=report_path,
        config_path=args.config,
        config=config,
        started_at=started_at,
        finished_at=finished_at,
        candidate_metrics=candidate_metrics,
        fold_metrics=fold_metrics,
        output_dir=output_dir,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "E7 XGBoost 예비 비교 완료"
    )
    print(
        "=" * 70
    )

    print(
        candidate_metrics.to_string(
            index=False
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