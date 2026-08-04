"""JH v10: v09 TF-IDF 토큰 SVD 압축 + F1 수치 피처."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_jh_v09 import JHV09PreprocessingPipeline


class JHV10PreprocessingPipeline(PreprocessingPipeline):
    """F0·global F3·F4 TF-IDF를 SVD로 압축하고 F1을 결합합니다."""

    name = "jh_v10"

    def __init__(
        self,
        svd_components: int = 128,
        svd_n_iter: int = 7,
        svd_random_state: int = 42,
        **kwargs: object,
    ) -> None:
        self.base_pipeline = JHV09PreprocessingPipeline(**kwargs)
        self.svd_components = int(svd_components)
        self.svd_n_iter = int(svd_n_iter)
        self.svd_random_state = int(svd_random_state)
        self.svd = TruncatedSVD(
            n_components=self.svd_components,
            algorithm="randomized",
            n_iter=self.svd_n_iter,
            random_state=self.svd_random_state,
        )
        self.token_feature_count = 0
        self.f1_feature_count = 0

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "JHV10PreprocessingPipeline":
        self.base_pipeline.fit(features, labels)
        raw = self.base_pipeline.transform(features)
        summary = self.base_pipeline.summary()
        self.token_feature_count = int(summary["tfidf_token_features"])
        self.f1_feature_count = int(summary["f1_features"])
        if raw.shape[1] != self.token_feature_count + self.f1_feature_count:
            raise RuntimeError("v10 TF-IDF·F1 피처 경계가 잘못되었습니다.")
        if self.token_feature_count <= self.svd_components:
            raise ValueError(
                f"SVD 토큰 피처 {self.token_feature_count}개가 "
                f"n_components={self.svd_components}보다 많아야 합니다."
            )
        self.svd.fit(raw[:, : self.token_feature_count])
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if self.token_feature_count == 0:
            raise RuntimeError("v10 SVD가 학습되지 않았습니다.")
        raw = self.base_pipeline.transform(features)
        token = raw[:, : self.token_feature_count]
        f1 = raw[:, self.token_feature_count :].toarray().astype(
            np.float32, copy=False
        )
        reduced = self.svd.transform(token).astype(np.float32, copy=False)
        return np.hstack([reduced, f1]).astype(np.float32, copy=False)

    def fit_transform(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> np.ndarray:
        return self.fit(features, labels).transform(features)

    def summary(self) -> dict[str, int | float | str | bool]:
        result = self.base_pipeline.summary()
        result.update({
            "svd_components": self.svd_components,
            "svd_n_iter": self.svd_n_iter,
            "svd_random_state": self.svd_random_state,
            "svd_explained_variance": float(
                self.svd.explained_variance_ratio_.sum()
            ),
            "final_features": self.svd_components + self.f1_feature_count,
        })
        return result

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        return self.base_pipeline.encode_labels(labels)

    def decode_labels(self, labels: np.ndarray) -> np.ndarray:
        return self.base_pipeline.decode_labels(labels)
