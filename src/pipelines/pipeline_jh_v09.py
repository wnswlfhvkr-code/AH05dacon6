"""JH v09: v08 토큰 TF-IDF + F1 요약 피처."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

from src.pipelines.pipeline_jh_v08 import JHV08PreprocessingPipeline
from src.pipelines.base import PreprocessingPipeline


class JHV09PreprocessingPipeline(PreprocessingPipeline):
    """F0·global F3·F4에 TF-IDF를 적용하고 F1을 결합합니다."""

    name = "jh_v09"

    def __init__(
        self,
        tfidf_sublinear_tf: bool = True,
        tfidf_norm: str = "l2",
        **kwargs: object,
    ) -> None:
        self.base_pipeline = JHV08PreprocessingPipeline(**kwargs)
        self.tfidf_sublinear_tf = bool(tfidf_sublinear_tf)
        self.tfidf_norm = str(tfidf_norm)
        self.tfidf_transformer = TfidfTransformer(
            sublinear_tf=self.tfidf_sublinear_tf,
            norm=self.tfidf_norm,
        )
        self.token_column_indices: np.ndarray | None = None
        self.f1_column_indices: np.ndarray | None = None

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "JHV09PreprocessingPipeline":
        self.base_pipeline.fit(features, labels)
        raw = self.base_pipeline.transform(features)
        base_summary = self.base_pipeline.summary()

        f0_count = int(base_summary["f0_features"])
        f1_count = int(base_summary["f1_features"])
        f3_count = int(base_summary["f3_features"])
        f4_count = int(base_summary["f4_features"])

        f1_start = f0_count
        f1_stop = f1_start + f1_count
        token_indices = np.concatenate([
            np.arange(0, f0_count, dtype=int),
            np.arange(f1_stop, f1_stop + f3_count + f4_count, dtype=int),
        ])
        self.token_column_indices = token_indices
        self.f1_column_indices = np.arange(f1_start, f1_stop, dtype=int)
        self.tfidf_transformer.fit(raw[:, self.token_column_indices])
        return self

    def transform(self, features: pd.DataFrame) -> sparse.csr_matrix:
        if self.token_column_indices is None or self.f1_column_indices is None:
            raise RuntimeError("v09 TF-IDF가 학습되지 않았습니다.")
        raw = self.base_pipeline.transform(features)
        token_tfidf = self.tfidf_transformer.transform(
            raw[:, self.token_column_indices]
        ).astype(np.float32, copy=False)
        f1 = raw[:, self.f1_column_indices].astype(np.float32, copy=False)
        return sparse.hstack([token_tfidf, f1], format="csr", dtype=np.float32)

    def fit_transform(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> sparse.csr_matrix:
        return self.fit(features, labels).transform(features)

    def summary(self) -> dict[str, int | str | bool]:
        result = self.base_pipeline.summary()
        result.update({
            "tfidf_token_features": int(
                0 if self.token_column_indices is None else len(self.token_column_indices)
            ),
            "tfidf_sublinear_tf": self.tfidf_sublinear_tf,
            "tfidf_norm": self.tfidf_norm,
        })
        return result

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        return self.base_pipeline.encode_labels(labels)

    def decode_labels(self, labels: np.ndarray) -> np.ndarray:
        return self.base_pipeline.decode_labels(labels)
