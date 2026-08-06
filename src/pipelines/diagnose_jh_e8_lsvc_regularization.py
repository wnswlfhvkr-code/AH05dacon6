"""E7 TF-IDF Balanced LinearSVC의 C 규제 강도 비교 실행기."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path) -> dict:
    with path.open(
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def save_yaml(
    path: Path,
    content: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            content,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


def c_slug(value: float) -> str:
    return (
        f"{value:g}"
        .replace(".", "p")
        .replace("-", "m")
    )


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
        (
            "| "
            + " | ".join(columns)
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


def normalize_converged(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    if normalized.isna().any():
        raise ValueError(
            "converged 열에 True/False가 아닌 "
            "값이 있습니다."
        )

    return normalized.astype(bool)


def run_candidate(
    candidate_config_path: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "src.train_sgkf",
        "--config",
        str(candidate_config_path),
    ]

    print()
    print(
        "=" * 70
    )
    print(
        "실행:",
        " ".join(command),
    )
    print(
        "=" * 70
    )

    subprocess.run(
        command,
        check=True,
    )


def load_candidate_result(
    result_dir: Path,
    candidate_name: str,
    C: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    fold_path = (
        result_dir
        / "fold_metrics.csv"
    )

    seed_path = (
        result_dir
        / "seed_oof_metrics.csv"
    )

    if not fold_path.exists():
        raise FileNotFoundError(
            f"Fold 결과가 없습니다: "
            f"{fold_path}"
        )

    if not seed_path.exists():
        raise FileNotFoundError(
            f"Seed 결과가 없습니다: "
            f"{seed_path}"
        )

    fold_metrics = pd.read_csv(
        fold_path
    )

    seed_metrics = pd.read_csv(
        seed_path
    )

    required_fold_columns = {
        "seed",
        "fold",
        "feature_count",
        "train_macro_f1",
        "validation_macro_f1",
        "generalization_gap",
        "converged",
        "elapsed_seconds",
    }

    missing_fold_columns = (
        required_fold_columns
        - set(fold_metrics.columns)
    )

    if missing_fold_columns:
        raise ValueError(
            f"{fold_path}에 필요한 열이 "
            f"없습니다: "
            f"{sorted(missing_fold_columns)}"
        )

    required_seed_columns = {
        "seed",
        "oof_macro_f1",
    }

    missing_seed_columns = (
        required_seed_columns
        - set(seed_metrics.columns)
    )

    if missing_seed_columns:
        raise ValueError(
            f"{seed_path}에 필요한 열이 "
            f"없습니다: "
            f"{sorted(missing_seed_columns)}"
        )

    fold_metrics.insert(
        0,
        "candidate",
        candidate_name,
    )

    fold_metrics.insert(
        1,
        "C",
        float(C),
    )

    seed_metrics.insert(
        0,
        "candidate",
        candidate_name,
    )

    seed_metrics.insert(
        1,
        "C",
        float(C),
    )

    return (
        fold_metrics,
        seed_metrics,
    )


def summarize_candidate(
    candidate_name: str,
    C: float,
    fold_metrics: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    reference_score: float,
) -> dict:
    converged = normalize_converged(
        fold_metrics["converged"]
    )

    oof_score = float(
        seed_metrics[
            "oof_macro_f1"
        ].mean()
    )

    return {
        "candidate": candidate_name,
        "C": float(C),
        "oof_macro_f1": oof_score,
        "reference_oof_macro_f1": float(
            reference_score
        ),
        "delta_vs_reference": float(
            oof_score
            - reference_score
        ),
        "train_macro_f1_mean": float(
            fold_metrics[
                "train_macro_f1"
            ].mean()
        ),
        "validation_macro_f1_mean": float(
            fold_metrics[
                "validation_macro_f1"
            ].mean()
        ),
        "generalization_gap_mean": float(
            fold_metrics[
                "generalization_gap"
            ].mean()
        ),
        "generalization_gap_min": float(
            fold_metrics[
                "generalization_gap"
            ].min()
        ),
        "generalization_gap_max": float(
            fold_metrics[
                "generalization_gap"
            ].max()
        ),
        "feature_count_mean": float(
            fold_metrics[
                "feature_count"
            ].mean()
        ),
        "converged_fold_count": int(
            converged.sum()
        ),
        "total_fold_count": int(
            len(fold_metrics)
        ),
        "all_folds_converged": bool(
            converged.all()
        ),
        "elapsed_seconds_sum": float(
            fold_metrics[
                "elapsed_seconds"
            ].sum()
        ),
    }


def write_report(
    path: Path,
    config_path: Path,
    config: dict,
    started_at: datetime,
    finished_at: datetime,
    summary_frame: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    best_candidate: dict,
    output_dir: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_columns = [
        "rank",
        "candidate",
        "C",
        "oof_macro_f1",
        "reference_oof_macro_f1",
        "delta_vs_reference",
        "train_macro_f1_mean",
        "validation_macro_f1_mean",
        "generalization_gap_mean",
        "generalization_gap_min",
        "generalization_gap_max",
        "feature_count_mean",
        "all_folds_converged",
        "passes_reference",
    ]

    fold_columns = [
        "candidate",
        "C",
        "seed",
        "fold",
        "feature_count",
        "train_macro_f1",
        "validation_macro_f1",
        "generalization_gap",
        "converged",
        "elapsed_seconds",
    ]

    summary_table = dataframe_to_markdown(
        summary_frame[
            summary_columns
        ]
    )

    fold_table = dataframe_to_markdown(
        fold_metrics[
            fold_columns
        ]
    )

    experiment_name = str(
        config["project"][
            "experiment_name"
        ]
    )

    elapsed = (
        finished_at
        - started_at
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
| 비교 C | {config["grid"]["C_values"]} |
| Seed | {config["validation"]["seeds"]} |
| Fold 수 | {int(config["validation"]["n_splits"])} |
| 최상위 후보 | {best_candidate["candidate"]} |
| 최상위 C | {float(best_candidate["C"]):g} |
| 최상위 OOF Macro F1 | {float(best_candidate["oof_macro_f1"]):.6f} |
| 설정 파일 | `{config_path}` |
| 결과 폴더 | `{output_dir}` |

## 후보별 결과

{summary_table}

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
            "test_002_e8_lsvc_grid.yaml"
        ),
    )

    args = parser.parse_args()

    config = load_yaml(
        args.config
    )

    validation_config = (
        config["validation"]
    )

    seeds = [
        int(seed)
        for seed
        in validation_config["seeds"]
    ]

    if seeds != [42]:
        raise ValueError(
            "예비 C 그리드는 seeds: [42]만 "
            "사용해야 합니다."
        )

    n_splits = int(
        validation_config[
            "n_splits"
        ]
    )

    if n_splits != 3:
        raise ValueError(
            "예비 C 그리드는 3-Fold를 "
            "사용해야 합니다."
        )

    if str(
        config["preprocessing"]["name"]
    ) != "jh_v09":
        raise ValueError(
            "이 실행기는 jh_v09 전처리를 "
            "사용해야 합니다."
        )

    if str(
        config["model"]["name"]
    ) != "linear_svc":
        raise ValueError(
            "이 실행기는 linear_svc 모델을 "
            "사용해야 합니다."
        )

    experiment_name = str(
        config["project"][
            "experiment_name"
        ]
    )

    output_dir = Path(
        config["data"][
            "processed_dir"
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_config_dir = (
        output_dir
        / "generated_configs"
    )

    generated_config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_score = float(
        config["reference"][
            "oof_macro_f1"
        ]
    )

    reuse_completed = bool(
        config["grid"].get(
            "reuse_completed",
            True,
        )
    )

    all_fold_frames = []
    all_seed_frames = []
    summary_rows = []

    for C_value in config[
        "grid"
    ][
        "C_values"
    ]:
        C = float(
            C_value
        )

        slug = c_slug(
            C
        )

        candidate_name = (
            f"C_{slug}"
        )

        candidate_dir = (
            output_dir
            / candidate_name
        )

        candidate_config_path = (
            generated_config_dir
            / f"{candidate_name}.yaml"
        )

        fold_path = (
            candidate_dir
            / "fold_metrics.csv"
        )

        seed_path = (
            candidate_dir
            / "seed_oof_metrics.csv"
        )

        completed = (
            fold_path.exists()
            and seed_path.exists()
        )

        if (
            completed
            and reuse_completed
        ):
            print(
                f"{candidate_name}: "
                "기존 완료 결과 재사용"
            )
        else:
            candidate_config = deepcopy(
                config
            )

            candidate_config.pop(
                "grid",
                None,
            )

            candidate_config[
                "project"
            ][
                "experiment_name"
            ] = (
                f"{experiment_name}_"
                f"{candidate_name}"
            )

            candidate_config[
                "project"
            ][
                "description"
            ] = (
                f"Balanced LinearSVC C={C:g}"
            )

            candidate_config[
                "project"
            ][
                "task_type"
            ] = (
                "linear_svc_regularization_candidate"
            )

            candidate_config[
                "data"
            ][
                "processed_dir"
            ] = str(
                candidate_dir
            )

            candidate_config[
                "model"
            ][
                "C"
            ] = C

            save_yaml(
                candidate_config_path,
                candidate_config,
            )

            run_candidate(
                candidate_config_path
            )

        (
            candidate_fold_metrics,
            candidate_seed_metrics,
        ) = load_candidate_result(
            result_dir=candidate_dir,
            candidate_name=candidate_name,
            C=C,
        )

        all_fold_frames.append(
            candidate_fold_metrics
        )

        all_seed_frames.append(
            candidate_seed_metrics
        )

        summary_rows.append(
            summarize_candidate(
                candidate_name=(
                    candidate_name
                ),
                C=C,
                fold_metrics=(
                    candidate_fold_metrics
                ),
                seed_metrics=(
                    candidate_seed_metrics
                ),
                reference_score=(
                    reference_score
                ),
            )
        )

    fold_metrics_all = pd.concat(
        all_fold_frames,
        ignore_index=True,
    )

    seed_metrics_all = pd.concat(
        all_seed_frames,
        ignore_index=True,
    )

    summary_frame = pd.DataFrame(
        summary_rows
    )

    summary_frame[
        "passes_reference"
    ] = (
        summary_frame[
            "oof_macro_f1"
        ]
        >= reference_score
    )

    summary_frame = (
        summary_frame.sort_values(
            [
                "oof_macro_f1",
                "generalization_gap_mean",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    summary_frame.insert(
        0,
        "rank",
        np.arange(
            1,
            len(summary_frame) + 1,
        ),
    )

    best_candidate = (
        summary_frame.iloc[0]
        .to_dict()
    )

    fold_output_path = (
        output_dir
        / "fold_metrics_by_C.csv"
    )

    seed_output_path = (
        output_dir
        / "seed_metrics_by_C.csv"
    )

    summary_output_path = (
        output_dir
        / "linear_svc_grid_summary.csv"
    )

    json_output_path = (
        output_dir
        / "linear_svc_grid_summary.json"
    )

    fold_metrics_all.to_csv(
        fold_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    seed_metrics_all.to_csv(
        seed_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_frame.to_csv(
        summary_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    finished_at = (
        datetime.now()
        .astimezone()
    )

    json_summary = {
        "experiment": (
            experiment_name
        ),
        "created_at": (
            finished_at.isoformat(
                timespec="seconds"
            )
        ),
        "reference": config[
            "reference"
        ],
        "best_candidate": (
            best_candidate
        ),
        "candidate_count": int(
            len(summary_frame)
        ),
        "all_candidates": (
            summary_frame.to_dict(
                orient="records"
            )
        ),
        "summary_csv": str(
            summary_output_path
        ),
    }

    json_output_path.write_text(
        json.dumps(
            json_summary,
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
        summary_frame=summary_frame,
        fold_metrics=fold_metrics_all,
        best_candidate=best_candidate,
        output_dir=output_dir,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "Balanced LinearSVC C 그리드 완료"
    )
    print(
        "=" * 70
    )

    print(
        summary_frame[
            [
                "rank",
                "candidate",
                "C",
                "oof_macro_f1",
                "delta_vs_reference",
                "train_macro_f1_mean",
                "validation_macro_f1_mean",
                "generalization_gap_mean",
                "all_folds_converged",
                "passes_reference",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "결과:",
        summary_output_path,
    )

    print(
        "보고서:",
        report_path,
    )

if __name__ == "__main__":
    main()