"""데이터 특성과 전처리 전 품질을 Markdown 보고서로 기록합니다."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.pipelines.base import infer_numeric_columns


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_report(features: pd.DataFrame) -> str:
    missing = features.isna().sum()
    constant_columns = features.columns[features.nunique(dropna=False) <= 1].tolist()
    without_constants = features.drop(columns=constant_columns)
    numeric_columns = infer_numeric_columns(without_constants)
    categorical_columns = [column for column in without_constants.columns if column not in numeric_columns]
    missing_top = missing[missing > 0].sort_values(ascending=False).head(20)
    numeric_data = without_constants[numeric_columns].apply(pd.to_numeric, errors="coerce") if numeric_columns else pd.DataFrame()
    numeric_summary = numeric_data.describe().T if not numeric_data.empty else pd.DataFrame()

    lines = [
        "# Data quality profile", "", "| 항목 | 값 |", "| --- | ---: |",
        f"| 행 수 | {len(features):,} |", f"| 전체 피처 수 | {features.shape[1]:,} |",
        f"| 수치형 피처 수 | {len(numeric_columns):,} |", f"| 범주형 피처 수 | {len(categorical_columns):,} |",
        f"| 결측치를 가진 피처 수 | {(missing > 0).sum():,} |", f"| 상수 피처 수 | {len(constant_columns):,} |",
        "", "## 결측치 상위 20개 피처", "",
    ]
    if missing_top.empty:
        lines.append("결측치가 없습니다.")
    else:
        lines.extend(["| 피처 | 결측치 수 |", "| --- | ---: |"])
        lines.extend(f"| {column} | {count:,} |" for column, count in missing_top.items())
    lines.extend(["", "## 상수 피처", ""])
    lines.append(f"- 상수 피처: {', '.join(constant_columns) if constant_columns else '없음'}")
    lines.extend(["", "## 수치형 분포 요약", ""])
    if numeric_summary.empty:
        lines.append("수치형 피처가 없습니다.")
    else:
        selected = numeric_summary[["mean", "std", "min", "max"]].head(20).round(4)
        lines.extend(["| 피처 | 평균 | 표준편차 | 최솟값 | 최댓값 |", "| --- | ---: | ---: | ---: | ---: |"])
        lines.extend(f"| {index} | {row['mean']} | {row['std']} | {row['min']} | {row['max']} |" for index, row in selected.iterrows())
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--output", type=Path, default=Path("experiments/data_profile.md"))
    args = parser.parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    train = pd.read_csv(Path(data_config["raw_dir"]) / data_config["train_file"])
    features = train.drop(columns=[data_config["id_column"], data_config["target_column"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(features), encoding="utf-8")
    print(f"데이터 품질 보고서 생성: {args.output}")


if __name__ == "__main__":
    main()
