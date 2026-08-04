"""EMV45와 JSJ9 전체 피처를 XGBoost용 단일 희소행렬로 결합합니다."""

from __future__ import annotations

import pandas as pd
from scipy.sparse import csr_matrix, hstack

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_em_v45 import EMV45PreprocessingPipeline
from src.pipelines.pipeline_jsj_v1 import TextTreeFeatureBundle
from src.pipelines.pipeline_jsj_v9 import JSJV9PreprocessingPipeline


class PipeCombEMV2PreprocessingPipeline(PreprocessingPipeline):
    """EMV45 + JSJ9를 XGBoost가 받는 하나의 CSR 행렬로 반환합니다.

    EMV45는 consequence·burden·hotspot·OOF signature를 제공하고, JSJ9는
    Word/Char TF-IDF와 유전자 변이 여부·구조 요약을 제공합니다. 학습 행의
    EMV45 label 기반 피처는 기존 ``fit_transform``의 inner-fold OOF 값을
    그대로 사용합니다.
    """

    name = "pipeComb_em_v2"
    evaluation_folds = 5

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.emv45_pipeline = EMV45PreprocessingPipeline(**parameters)
        self.jsj9_pipeline = JSJV9PreprocessingPipeline(**parameters)
        self.emv45_feature_count_ = 0
        self.jsj9_text_feature_count_ = 0
        self.jsj9_tree_feature_count_ = 0
        self.combined_feature_count_ = 0
        self.steps = (
            "EMV45 consequence·burden·hotspot·OOF signature",
            "JSJ9 Word·Char TF-IDF",
            "JSJ9 유전자 변이 여부·구조 요약",
            "XGBoost용 CSR 희소행렬 결합",
            "원본 SUBCLASS 유지",
        )

    @staticmethod
    def _require_jsj9_bundle(features) -> TextTreeFeatureBundle:
        if not isinstance(features, TextTreeFeatureBundle):
            raise TypeError("JSJ9 전처리가 TextTreeFeatureBundle을 반환하지 않았습니다.")
        return features

    def _combine(
        self,
        emv45_features: pd.DataFrame,
        jsj9_features: TextTreeFeatureBundle,
    ) -> csr_matrix:
        combined = hstack(
            (
                csr_matrix(emv45_features.to_numpy(dtype="float32", copy=False)),
                jsj9_features.text,
                jsj9_features.tree,
            ),
            format="csr",
            dtype="float32",
        )
        self.emv45_feature_count_ = emv45_features.shape[1]
        self.jsj9_text_feature_count_ = jsj9_features.text.shape[1]
        self.jsj9_tree_feature_count_ = jsj9_features.tree.shape[1]
        self.combined_feature_count_ = combined.shape[1]
        return combined

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "PipeCombEMV2PreprocessingPipeline":
        self.feature_columns = features.columns.tolist()
        self.emv45_pipeline.fit(features, labels)
        self.jsj9_pipeline.fit(features, labels)
        self._combine(
            self.emv45_pipeline.transform(features),
            self._require_jsj9_bundle(self.jsj9_pipeline.transform(features)),
        )
        self.label_encoder.fit(labels)
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> csr_matrix:
        self.feature_columns = features.columns.tolist()
        emv45 = self.emv45_pipeline.fit_transform(features, labels)
        jsj9 = self._require_jsj9_bundle(
            self.jsj9_pipeline.fit_transform(features, labels)
        )
        self.label_encoder.fit(labels)
        return self._combine(emv45, jsj9)

    def transform(self, features: pd.DataFrame) -> csr_matrix:
        return self._combine(
            self.emv45_pipeline.transform(features),
            self._require_jsj9_bundle(self.jsj9_pipeline.transform(features)),
        )

    def summary(self) -> dict[str, int]:
        return {
            "remaining_features": self.combined_feature_count_,
            "emv45_features": self.emv45_feature_count_,
            "jsj9_text_features": self.jsj9_text_feature_count_,
            "jsj9_tree_features": self.jsj9_tree_feature_count_,
            "dropped_constant_features": int(
                self.emv45_pipeline.summary().get("dropped_constant_features", 0)
            ),
        }

