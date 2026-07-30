"""표형 데이터의 범주형 피처와 라벨을 인코딩합니다."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


class TabularPreprocessor:
    """학습·추론에서 동일한 범주형 인코딩을 적용합니다."""

    def __init__(self) -> None:
        self.categorical_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.feature_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        self.label_encoder = LabelEncoder()

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "TabularPreprocessor":
        self.feature_columns = features.columns.tolist()
        self.categorical_columns = features.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.categorical_columns:
            self.feature_encoder.fit(features[self.categorical_columns])
        self.label_encoder.fit(labels)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"누락된 피처가 있습니다: {sorted(missing)}")
        encoded = features[self.feature_columns].copy()
        if self.categorical_columns:
            encoded[self.categorical_columns] = self.feature_encoder.transform(encoded[self.categorical_columns])
        return encoded

    def encode_labels(self, labels: pd.Series):
        return self.label_encoder.transform(labels)

    def decode_labels(self, labels):
        return self.label_encoder.inverse_transform(labels.astype(int))
