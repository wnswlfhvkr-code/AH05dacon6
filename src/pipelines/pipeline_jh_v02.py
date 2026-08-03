"""JH v02: F0 유전자 변이 유무 + F1 환자별 변이 요약."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

from src.pipelines.pipeline_jh_v04 import (
    build_f0_matrix,
    build_f1_matrix,
    parse_wide_mutations,
)
from src.pipelines.base import PreprocessingPipeline


class JHV02PreprocessingPipeline(PreprocessingPipeline):
    """F0에 Fold-local scaling을 적용한 F1을 추가합니다."""

    name = "jh_v02"

    def __init__(self, **_: object) -> None:
        self.label_encoder = LabelEncoder()
        self.f1_scaler = StandardScaler()

        self.gene_columns: list[str] = []
        self.f1_feature_names: list[str] = []

        self.f0_active_mask: (
            np.ndarray | None
        ) = None

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "JHV02PreprocessingPipeline":
        self.gene_columns = (
            features.columns.tolist()
        )

        events = parse_wide_mutations(
            features
        )

        f0 = build_f0_matrix(features)

        f1, self.f1_feature_names = (
            build_f1_matrix(
                events,
                len(features),
            )
        )

        self.f0_active_mask = (
            np.asarray(
                f0.getnnz(axis=0)
            ).ravel()
            > 0
        )

        self.f1_scaler.fit(f1)
        self.label_encoder.fit(labels)

        return self

    def transform(
        self,
        features: pd.DataFrame,
    ) -> sparse.csr_matrix:
        if (
            features.columns.tolist()
            != self.gene_columns
        ):
            raise ValueError(
                "학습 때와 유전자 열 또는 순서가 다릅니다."
            )

        if self.f0_active_mask is None:
            raise RuntimeError(
                "fit을 먼저 실행해야 합니다."
            )

        events = parse_wide_mutations(
            features
        )

        f0 = build_f0_matrix(features)[
            :,
            self.f0_active_mask,
        ]

        f1, _ = build_f1_matrix(
            events,
            len(features),
        )

        f1_scaled = sparse.csr_matrix(
            self.f1_scaler
            .transform(f1)
            .astype(np.float32)
        )

        return sparse.hstack(
            [
                f0,
                f1_scaled,
            ],
            format="csr",
            dtype=np.float32,
        )

    def fit_transform(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> sparse.csr_matrix:
        return (
            self.fit(features, labels)
            .transform(features)
        )

    def encode_labels(
        self,
        labels: pd.Series,
    ) -> np.ndarray:
        return self.label_encoder.transform(
            labels
        )

    def decode_labels(
        self,
        labels: np.ndarray,
    ) -> np.ndarray:
        return (
            self.label_encoder
            .inverse_transform(
                np.asarray(labels).astype(int)
            )
        )

    def summary(self) -> dict[str, int]:
        f0_count = (
            int(self.f0_active_mask.sum())
            if self.f0_active_mask
            is not None
            else 0
        )

        f1_count = len(
            self.f1_feature_names
        )

        return {
            "f0_features": f0_count,
            "f1_features": f1_count,
            "remaining_features": (
                f0_count + f1_count
            ),
        }