from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.pipelines.pipeline_jsj_v3 import (
    add_e1_features,
    make_tree_features,
)


def load_config(path: Path) -> dict:
    with path.open(
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def make_model_config(
    config: dict,
    n_classes: int,
) -> tuple[dict, int]:
    model_config = dict(config)

    early_stopping_rounds = int(
        model_config.pop(
            "early_stopping_rounds"
        )
    )

    model_config.update(
        {
            "num_class": n_classes,
            "metric": "multi_logloss",
            "n_jobs": -1,
            "verbosity": -1,
        }
    )

    return (
        model_config,
        early_stopping_rounds,
    )


def dataframe_to_markdown(
    frame: pd.DataFrame,
) -> str:
    columns = [
        str(column)
        for column in frame.columns
    ]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(
            "---"
            for _ in columns
        ) + " |",
    ]

    for row in frame.itertuples(
        index=False,
        name=None,
    ):
        values = []

        for value in row:
            if isinstance(
                value,
                (
                    float,
                    np.floating,
                ),
            ):
                values.append(
                    f"{float(value):.6f}"
                )
            else:
                values.append(
                    str(value).replace(
                        "|",
                        "\\|",
                    )
                )

        lines.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(lines)


def evaluate_model(
    *,
    model_name: str,
    model_config: dict,
    early_stopping_rounds: int,
    numeric_features: pd.DataFrame,
    y_label: np.ndarray,
    fold_by_seed: dict[int, np.ndarray],
    seeds: list[int],
    n_splits: int,
    n_classes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []

    oof_probability_by_seed = np.zeros(
        (
            len(seeds),
            len(y_label),
            n_classes,
        ),
        dtype=np.float32,
    )

    for seed_position, seed in enumerate(
        seeds
    ):
        fold_assignment = fold_by_seed[
            seed
        ]

        for fold in range(n_splits):
            started_at = time.time()

            train_index = np.flatnonzero(
                fold_assignment != fold
            )

            validation_index = np.flatnonzero(
                fold_assignment == fold
            )

            fold_train_frame = (
                numeric_features.iloc[
                    train_index
                ]
            )

            # 피처 선택은 Fold-Train에서만 수행한다.
            active_mask = (
                fold_train_frame.nunique(
                    dropna=False
                )
                > 1
            )

            active_columns = (
                numeric_features.columns[
                    active_mask.to_numpy()
                ]
            )

            fold_train_x = (
                fold_train_frame[
                    active_columns
                ]
            )

            fold_validation_x = (
                numeric_features.iloc[
                    validation_index
                ][active_columns]
            )

            fold_train_y = y_label[
                train_index
            ]

            fold_validation_y = y_label[
                validation_index
            ]

            sample_weight = (
                compute_sample_weight(
                    class_weight="balanced",
                    y=fold_train_y,
                )
            )

            model = lgb.LGBMClassifier(
                **model_config,
                random_state=seed + fold,
            )

            model.fit(
                fold_train_x,
                fold_train_y,
                sample_weight=sample_weight,
                eval_set=[
                    (
                        fold_validation_x,
                        fold_validation_y,
                    )
                ],
                eval_metric="multi_logloss",
                callbacks=[
                    lgb.early_stopping(
                        early_stopping_rounds,
                        verbose=False,
                    ),
                    lgb.log_evaluation(0),
                ],
            )

            train_probability = (
                model.predict_proba(
                    fold_train_x
                )
                .astype(np.float32)
            )

            validation_probability = (
                model.predict_proba(
                    fold_validation_x
                )
                .astype(np.float32)
            )

            oof_probability_by_seed[
                seed_position,
                validation_index,
            ] = validation_probability

            train_prediction = (
                train_probability.argmax(
                    axis=1
                )
            )

            validation_prediction = (
                validation_probability.argmax(
                    axis=1
                )
            )

            train_macro_f1 = f1_score(
                fold_train_y,
                train_prediction,
                average="macro",
                zero_division=0,
            )

            validation_macro_f1 = f1_score(
                fold_validation_y,
                validation_prediction,
                average="macro",
                zero_division=0,
            )

            generalization_gap = (
                train_macro_f1
                - validation_macro_f1
            )

            fold_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "fold": fold,
                    "train_size": len(
                        train_index
                    ),
                    "validation_size": len(
                        validation_index
                    ),
                    "feature_count": len(
                        active_columns
                    ),
                    "best_iteration": int(
                        model.best_iteration_
                    ),
                    "train_macro_f1": float(
                        train_macro_f1
                    ),
                    "validation_macro_f1": float(
                        validation_macro_f1
                    ),
                    "generalization_gap": float(
                        generalization_gap
                    ),
                    "elapsed_seconds": float(
                        time.time()
                        - started_at
                    ),
                }
            )

            print(
                f"{model_name} | "
                f"seed={seed}, fold={fold} | "
                f"Train={train_macro_f1:.6f} | "
                f"Validation={validation_macro_f1:.6f} | "
                f"Gap={generalization_gap:+.6f}"
            )

    seed_rows = []

    for seed_position, seed in enumerate(
        seeds
    ):
        prediction = (
            oof_probability_by_seed[
                seed_position
            ].argmax(axis=1)
        )

        score = f1_score(
            y_label,
            prediction,
            average="macro",
            zero_division=0,
        )

        seed_rows.append(
            {
                "model": model_name,
                "seed": seed,
                "oof_macro_f1": float(
                    score
                ),
            }
        )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(seed_rows),
    )


def write_report(
    *,
    path: Path,
    config_path: Path,
    config: dict,
    fold_metrics: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    summary_path: Path,
) -> None:
    project = config["project"]

    report_sections = [
        f"# {project['experiment_name']}",
        "",
        "## 실험 정보",
        "",
        f"- 설명: {project.get('description', '')}",
        f"- 부모 실험: {project.get('parent_experiment', '')}",
        f"- 설정 파일: `{config_path}`",
        "",
        "## 모델 비교",
        "",
        dataframe_to_markdown(
            comparison
        ),
        "",
        "## Seed별 OOF Macro F1",
        "",
        dataframe_to_markdown(
            seed_metrics
        ),
        "",
        "## Fold별 Train–Validation 결과",
        "",
        dataframe_to_markdown(
            fold_metrics
        ),
        "",
        "## 산출물",
        "",
        f"- 요약 JSON: `{summary_path}`",
        "- Fold 지표: `train_validation_gap.csv`",
        "- Seed별 OOF: `seed_oof_metrics.csv`",
        "- 모델 비교: `model_comparison.csv`",
        "",
        "## 결과 해석",
        "",
        "<!-- 실행 결과를 확인한 후 작성 -->",
        "",
        "## 채택 여부",
        "",
        "<!-- 채택 / 보류 / 폐기 및 근거 작성 -->",
        "",
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(
            report_sections
        ),
        encoding="utf-8",
    )

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/ensembles/"
            "test_008_v06_overfit.yaml"
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    project = config["project"]
    data = config["data"]
    validation = config["validation"]

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

    fixed_split = pd.read_csv(
        data["fixed_split_file"]
    )

    id_column = data["id_column"]
    target_column = data["target_column"]

    assert np.array_equal(
        train[id_column].astype(str).to_numpy(),
        fixed_split[id_column].astype(str).to_numpy(),
    )

    seeds = [
        int(seed)
        for seed in validation["seeds"]
    ]

    n_splits = int(
        validation["n_splits"]
    )

    label_encoder = LabelEncoder()

    y_label = label_encoder.fit_transform(
        train[target_column]
    )

    n_classes = len(
        label_encoder.classes_
    )

    fold_by_seed = {
        seed: fixed_split[
            f"fold_seed_{seed}"
        ].to_numpy(
            dtype=np.int8
        )
        for seed in seeds
    }

    gene_columns = [
        column
        for column in test.columns
        if column != id_column
        and column in train.columns
    ]

    print("Numeric Structural 피처 생성 중...")

    train_numeric_base = (
        make_tree_features(
            train[gene_columns]
        )
    )

    train_numeric = add_e1_features(
        train_numeric_base
    )

    print(
        "Numeric feature shape:",
        train_numeric.shape,
    )

    original_config, original_es = (
        make_model_config(
            config["original_model"],
            n_classes,
        )
    )

    regularized_config, regularized_es = (
        make_model_config(
            config["regularized_model"],
            n_classes,
        )
    )

    original_fold, original_seed = (
        evaluate_model(
            model_name="original",
            model_config=original_config,
            early_stopping_rounds=(
                original_es
            ),
            numeric_features=train_numeric,
            y_label=y_label,
            fold_by_seed=fold_by_seed,
            seeds=seeds,
            n_splits=n_splits,
            n_classes=n_classes,
        )
    )

    regularized_fold, regularized_seed = (
        evaluate_model(
            model_name="regularized",
            model_config=regularized_config,
            early_stopping_rounds=(
                regularized_es
            ),
            numeric_features=train_numeric,
            y_label=y_label,
            fold_by_seed=fold_by_seed,
            seeds=seeds,
            n_splits=n_splits,
            n_classes=n_classes,
        )
    )

    fold_metrics = pd.concat(
        [
            original_fold,
            regularized_fold,
        ],
        ignore_index=True,
    )

    seed_metrics = pd.concat(
        [
            original_seed,
            regularized_seed,
        ],
        ignore_index=True,
    )

    fold_summary = (
        fold_metrics.groupby(
            "model",
            as_index=False,
        )
        .agg(
            train_macro_f1_mean=(
                "train_macro_f1",
                "mean",
            ),
            validation_macro_f1_mean=(
                "validation_macro_f1",
                "mean",
            ),
            generalization_gap_mean=(
                "generalization_gap",
                "mean",
            ),
            gap_min=(
                "generalization_gap",
                "min",
            ),
            gap_max=(
                "generalization_gap",
                "max",
            ),
        )
    )

    seed_summary = (
        seed_metrics.groupby(
            "model",
            as_index=False,
        )
        .agg(
            oof_macro_f1_mean=(
                "oof_macro_f1",
                "mean",
            ),
            oof_macro_f1_std=(
                "oof_macro_f1",
                lambda values: values.std(
                    ddof=0
                ),
            ),
        )
    )

    comparison = fold_summary.merge(
        seed_summary,
        on="model",
    )

    fold_path = (
        output_dir
        / "train_validation_gap.csv"
    )

    seed_path = (
        output_dir
        / "seed_oof_metrics.csv"
    )

    comparison_path = (
        output_dir
        / "model_comparison.csv"
    )

    fold_metrics.to_csv(
        fold_path,
        index=False,
    )

    seed_metrics.to_csv(
        seed_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    summary = {
        "experiment": (
            project["experiment_name"]
        ),
        "parent_experiment": (
            project["parent_experiment"]
        ),
        "created_at": datetime.now(
            ZoneInfo("Asia/Seoul")
        ).isoformat(
            timespec="seconds"
        ),
        "public_result": (
            config["public_result"]
        ),
        "comparison": (
            comparison.to_dict(
                orient="records"
            )
        ),
        "decision_status": (
        "pending_manual_review"
    ),
    }

    summary_path = (
        output_dir
        / (
            project["experiment_name"]
            + "_summary.json"
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
        / (
            project["experiment_name"]
            + ".md"
        )
    )

    write_report(
        path=report_path,
        config_path=args.config,
        config=config,
        fold_metrics=fold_metrics,
        seed_metrics=seed_metrics,
        comparison=comparison,
        summary_path=summary_path,
    )

    print()
    print(comparison.to_string(index=False))
    print()
    print("요약 JSON:", summary_path)
    print("실험 보고서:", report_path)


if __name__ == "__main__":
    main()