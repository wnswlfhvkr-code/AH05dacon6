"""E7 Logistic Regression의 L2 C 규제 강도 비교 실행기.

기존 E7 과적합 진단 결과를 C=1.0 기준으로 재사용하고,
지정한 C 후보만 공용 SGKF 실행기로 학습한다.
"""

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
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(path: Path, content: dict) -> None:
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
    """C 값을 파일명에 사용할 수 있는 문자열로 변환한다."""
    return (
        f"{value:g}"
        .replace(".", "p")
        .replace("-", "m")
    )


def dataframe_to_markdown(
    frame: pd.DataFrame,
    float_digits: int = 6,
) -> str:
    """tabulate 없이 DataFrame을 Markdown 표로 변환한다."""
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


def load_result(
    result_dir: Path,
    C: float,
    candidate_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            f"Fold 결과가 없습니다: {fold_path}"
        )

    if not seed_path.exists():
        raise FileNotFoundError(
            f"Seed 결과가 없습니다: {seed_path}"
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
        "train_macro_f1",
        "validation_macro_f1",
        "generalization_gap",
        "converged",
    }

    missing_fold_columns = (
        required_fold_columns
        - set(fold_metrics.columns)
    )

    if missing_fold_columns:
        raise ValueError(
            f"{fold_path}에 필요한 열이 없습니다: "
            f"{sorted(missing_fold_columns)}"
        )

    if not {
        "seed",
        "oof_macro_f1",
    }.issubset(seed_metrics.columns):
        raise ValueError(
            f"{seed_path}에 seed 또는 "
            "oof_macro_f1 열이 없습니다."
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
) -> dict:
    converged_series = (
        fold_metrics["converged"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    if converged_series.isna().any():
        converged_series = (
            fold_metrics["converged"]
            .astype(bool)
        )

    return {
        "candidate": candidate_name,
        "C": float(C),
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
        "oof_macro_f1_mean": float(
            seed_metrics[
                "oof_macro_f1"
            ].mean()
        ),
        "oof_macro_f1_std": float(
            seed_metrics[
                "oof_macro_f1"
            ].std(ddof=1)
        ),
        "converged_fold_count": int(
            converged_series.sum()
        ),
        "total_fold_count": int(
            len(fold_metrics)
        ),
    }


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

def write_grid_report(
    path: Path,
    config_path: Path,
    config: dict,
    started_at: datetime,
    finished_at: datetime,
    summary_frame: pd.DataFrame,
    seed_comparison: pd.DataFrame,
    recommended: dict | None,
    output_dir: Path,
) -> None:
    """규제 그리드 실행 결과를 공통 Markdown 형식으로 저장한다."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    project_config = config["project"]
    grid_config = config["grid"]
    selection_config = grid_config["selection"]

    experiment_name = str(
        project_config["experiment_name"]
    )

    description = str(
        project_config.get(
            "description",
            "",
        )
    )

    elapsed = (
        finished_at
        - started_at
    )

    display_columns = [
        "candidate",
        "C",
        "train_macro_f1_mean",
        "validation_macro_f1_mean",
        "generalization_gap_mean",
        "generalization_gap_min",
        "generalization_gap_max",
        "oof_macro_f1_mean",
        "oof_macro_f1_std",
        "oof_delta",
        "gap_reduction",
        "improved_seed_count",
        "converged_fold_count",
        "total_fold_count",
        "eligible",
    ]

    seed_columns = [
        "seed",
        "baseline_oof_macro_f1",
        "candidate",
        "C",
        "candidate_oof_macro_f1",
        "delta",
    ]

    summary_table = dataframe_to_markdown(
        summary_frame[display_columns]
    )

    seed_table = dataframe_to_markdown(
        seed_comparison[seed_columns]
    )

    if recommended is None:
        recommended_name = ""
        recommended_C = ""
        recommended_oof = ""
        recommended_gap = ""
    else:
        recommended_name = str(
            recommended["candidate"]
        )

        recommended_C = (
            f"{float(recommended['C']):g}"
        )

        recommended_oof = (
            f"{float(recommended['oof_macro_f1_mean']):.6f}"
        )

        recommended_gap = (
            f"{float(recommended['generalization_gap_mean']):.6f}"
        )

    report = f"""# {experiment_name}

| 항목 | 결과 |
| --- | --- |
| 설명 | {description} |
| 실행 시작 | {started_at.isoformat(timespec="seconds")} |
| 실행 종료 | {finished_at.isoformat(timespec="seconds")} |
| 총 실행 시간 | {elapsed} |
| 기준 C | {float(grid_config["baseline"]["C"]):g} |
| 비교 C | {grid_config["C_values"]} |
| OOF 하락 허용값 | {float(selection_config["oof_drop_tolerance"]):.6f} |
| 최소 gap 감소값 | {float(selection_config["min_gap_reduction"]):.6f} |
| 설정 파일 | `{config_path}` |
| 결과 폴더 | `{output_dir}` |

## 후보별 결과

{summary_table}

## Seed별 결과

{seed_table}

## 선택 결과

| 항목 | 결과 |
| --- | --- |
| 추천 후보 | {recommended_name} |
| 추천 C | {recommended_C} |
| 추천 OOF Macro F1 | {recommended_oof} |
| 추천 평균 generalization gap | {recommended_gap} |

## 설정

```yaml
{yaml.safe_dump(config, allow_unicode=True, sort_keys=False).rstrip()}
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
            "test_002_v08_overfit_grid.yaml"
        ),
    )

    args = parser.parse_args()

    grid_config = load_yaml(
        args.config
    )

    project_config = (
        grid_config["project"]
    )

    grid_section = (
        grid_config["grid"]
    )

    output_config = (
        grid_config["output"]
    )

    experiment_name = str(
        project_config[
            "experiment_name"
        ]
    )

    base_config_path = Path(
        grid_section[
            "base_config"
        ]
    )

    base_config = load_yaml(
        base_config_path
    )

    output_dir = Path(
        output_config[
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

    baseline_config = (
        grid_section["baseline"]
    )

    baseline_C = float(
        baseline_config["C"]
    )

    baseline_dir = Path(
        baseline_config[
            "result_dir"
        ]
    )

    reuse_completed = bool(
        grid_section.get(
            "reuse_completed",
            True,
        )
    )

    selection_config = (
        grid_section["selection"]
    )

    all_fold_frames = []
    all_seed_frames = []
    summary_rows = []

    # 기존 완료 결과를 C=1.0 기준값으로 재사용한다.
    baseline_name = "baseline_C_1"

    (
        baseline_fold_metrics,
        baseline_seed_metrics,
    ) = load_result(
        result_dir=baseline_dir,
        C=baseline_C,
        candidate_name=baseline_name,
    )

    all_fold_frames.append(
        baseline_fold_metrics
    )

    all_seed_frames.append(
        baseline_seed_metrics
    )

    summary_rows.append(
        summarize_candidate(
            candidate_name=baseline_name,
            C=baseline_C,
            fold_metrics=baseline_fold_metrics,
            seed_metrics=baseline_seed_metrics,
        )
    )

    # C 후보를 순서대로 실행한다.
    for C in grid_section["C_values"]:
        C = float(C)

        slug = c_slug(C)

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
                base_config
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
                "E7 Logistic Regression "
                f"L2 규제 실험 C={C:g}"
            )

            candidate_config[
                "project"
            ][
                "task_type"
            ] = (
                "overfit_regularization_candidate"
            )

            candidate_config[
                "project"
            ][
                "parent_experiment"
            ] = (
                "test_002_v08_overfit"
            )

            candidate_config[
                "data"
            ][
                "processed_dir"
            ] = str(candidate_dir)

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
        ) = load_result(
            result_dir=candidate_dir,
            C=C,
            candidate_name=candidate_name,
        )

        all_fold_frames.append(
            candidate_fold_metrics
        )

        all_seed_frames.append(
            candidate_seed_metrics
        )

        summary_rows.append(
            summarize_candidate(
                candidate_name=candidate_name,
                C=C,
                fold_metrics=candidate_fold_metrics,
                seed_metrics=candidate_seed_metrics,
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

    baseline_summary = (
        summary_frame.loc[
            np.isclose(
                summary_frame["C"],
                baseline_C,
            )
        ]
        .iloc[0]
    )

    baseline_oof = float(
        baseline_summary[
            "oof_macro_f1_mean"
        ]
    )

    baseline_gap = float(
        baseline_summary[
            "generalization_gap_mean"
        ]
    )

    summary_frame[
        "oof_delta"
    ] = (
        summary_frame[
            "oof_macro_f1_mean"
        ]
        - baseline_oof
    )

    summary_frame[
        "gap_reduction"
    ] = (
        baseline_gap
        - summary_frame[
            "generalization_gap_mean"
        ]
    )

    baseline_seed = (
        baseline_seed_metrics[
            [
                "seed",
                "oof_macro_f1",
            ]
        ]
        .rename(
            columns={
                "oof_macro_f1":
                    "baseline_oof_macro_f1",
            }
        )
    )

    candidate_seed_metrics = (
        seed_metrics_all.loc[
            ~np.isclose(
                seed_metrics_all["C"],
                baseline_C,
            )
        ]
        .copy()
    )

    seed_comparison = (
        candidate_seed_metrics.merge(
            baseline_seed,
            on="seed",
            how="left",
            validate="many_to_one",
        )
    )

    seed_comparison = (
        seed_comparison.rename(
            columns={
                "oof_macro_f1":
                    "candidate_oof_macro_f1",
            }
        )
    )

    seed_comparison[
        "delta"
    ] = (
        seed_comparison[
            "candidate_oof_macro_f1"
        ]
        - seed_comparison[
            "baseline_oof_macro_f1"
        ]
    )

    improved_seed_counts = (
        seed_comparison.assign(
            improved=(
                seed_comparison[
                    "delta"
                ]
                > 0
            )
        )
        .groupby(
            "candidate"
        )["improved"]
        .sum()
        .astype(int)
    )

    summary_frame[
        "improved_seed_count"
    ] = (
        summary_frame[
            "candidate"
        ]
        .map(
            improved_seed_counts
        )
        .fillna(0)
        .astype(int)
    )

    oof_drop_tolerance = float(
        selection_config[
            "oof_drop_tolerance"
        ]
    )

    min_gap_reduction = float(
        selection_config[
            "min_gap_reduction"
        ]
    )

    summary_frame[
        "eligible"
    ] = (
        ~np.isclose(
            summary_frame["C"],
            baseline_C,
        )
        & (
            summary_frame[
                "oof_delta"
            ]
            >= -oof_drop_tolerance
        )
        & (
            summary_frame[
                "gap_reduction"
            ]
            >= min_gap_reduction
        )
        & (
            summary_frame[
                "converged_fold_count"
            ]
            == summary_frame[
                "total_fold_count"
            ]
        )
    )

    eligible_frame = (
        summary_frame.loc[
            summary_frame[
                "eligible"
            ]
        ]
        .sort_values(
            [
                "oof_macro_f1_mean",
                "generalization_gap_mean",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    recommended = (
        None
        if eligible_frame.empty
        else eligible_frame.iloc[0].to_dict()
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
        / "regularization_grid_summary.csv"
    )

    json_output_path = (
        output_dir
        / "regularization_grid_summary.json"
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
        "experiment": experiment_name,
        "created_at": (
            finished_at.isoformat(
                timespec="seconds"
            )
        ),
        "baseline_C": baseline_C,
        "baseline_oof_macro_f1": (
            baseline_oof
        ),
        "baseline_generalization_gap": (
            baseline_gap
        ),
        "selection": selection_config,
        "recommended_candidate": (
            recommended
        ),
        "candidate_count": int(
            len(summary_frame) - 1
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

    report_path = Path(
        output_config[
            "report_file"
        ]
    )

    write_grid_report(
        path=report_path,
        config_path=args.config,
        config=grid_config,
        started_at=started_at,
        finished_at=finished_at,
        summary_frame=summary_frame,
        seed_comparison=seed_comparison,
        recommended=recommended,
        output_dir=output_dir,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "E7 C 규제 그리드 완료"
    )
    print(
        "=" * 70
    )

    print(
        summary_frame[
            [
                "candidate",
                "C",
                "train_macro_f1_mean",
                "validation_macro_f1_mean",
                "generalization_gap_mean",
                "oof_macro_f1_mean",
                "oof_delta",
                "gap_reduction",
                "eligible",
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

    if recommended is None:
        print(
            "자동 판정: 채택 조건을 "
            "만족하는 후보 없음"
        )
    else:
        print(
            "자동 추천:",
            recommended[
                "candidate"
            ],
            f"C={recommended['C']:g}",
        )


if __name__ == "__main__":
    main()