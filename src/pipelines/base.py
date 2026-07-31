"""전처리 파이프라인의 공통 구현입니다."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def infer_numeric_columns(features: pd.DataFrame, minimum_valid_ratio: float = 0.99) -> list[str]:
    """문자열로 저장됐어도 대부분 숫자로 변환되는 피처를 수치형으로 판별합니다."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    for column in features.columns.difference(numeric):
        values = features[column]
        non_missing = values.notna().sum()
        if non_missing == 0:
            continue
        converted = pd.to_numeric(values, errors="coerce")
        if converted.notna().sum() / non_missing >= minimum_valid_ratio:
            numeric.append(column)
    return numeric


class PreprocessingPipeline:
    """상수 피처 제거, 범주형 순서 인코딩, 타깃 레이블 인코딩을 적용합니다."""
    print(f"[pipeline_base]:{'>'*50}")
    def __init__(self, **_: object) -> None:
        self.steps = ("상수 열 제거", "범주형 순서 인코딩")
        self.feature_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.dropped_constant_columns: list[str] = []
        self.ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.label_encoder = LabelEncoder()

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "PreprocessingPipeline":
        self.dropped_constant_columns = features.columns[features.nunique(dropna=False) <= 1].tolist()
        cleaned = features.drop(columns=self.dropped_constant_columns)
        self.feature_columns = cleaned.columns.tolist()
        self.categorical_columns = cleaned.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        if self.categorical_columns:
            self.ordinal_encoder.fit(cleaned[self.categorical_columns])
        self.label_encoder.fit(labels)
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        return self.fit(features, labels).transform(features)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"누락된 피처가 있습니다: {sorted(missing)}")
        transformed = features[self.feature_columns].copy()
        if self.categorical_columns:
            transformed[self.categorical_columns] = self.ordinal_encoder.transform(
                transformed[self.categorical_columns]
            )
        return transformed

    def summary(self) -> dict[str, int]:
        return {
            "remaining_features": len(self.feature_columns),
            "dropped_constant_features": len(self.dropped_constant_columns),
        }

    def encode_labels(self, labels: pd.Series):
        return self.label_encoder.transform(labels)

    def decode_labels(self, labels):
        return self.label_encoder.inverse_transform(labels.astype(int))

