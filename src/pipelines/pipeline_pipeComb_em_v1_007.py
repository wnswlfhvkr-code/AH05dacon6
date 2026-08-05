"""pipeComb_em_v1_007: 강화된 L2 meta learner용 OOF stacking 피처."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_em_v45 import EMV45PreprocessingPipeline
from src.pipelines.pipeline_jsj_v9 import JSJV9PreprocessingPipeline


@dataclass
class OOFStackingV007FeatureBundle:
    """텍스트 모델과 트리 모델이 각각 사용하는 피처 뷰입니다."""

    text: csr_matrix
    tree: csr_matrix


class PipeCombEMV1007PreprocessingPipeline(PreprocessingPipeline):
    """JSJ9 텍스트와 EMV45 트리 피처를 모델별 뷰로 분리합니다.

    v006과 같은 피처 및 base model 입력을 재현하되 다른 버전 파이프라인을
    상속하지 않고 ``PreprocessingPipeline``만 직접 상속합니다.
    """

    name = "pipeComb_em_v1_007"
    evaluation_folds = 5

    def __init__(
        self,
        text_parameters: dict[str, object] | None = None,
        emv45_parameters: dict[str, object] | None = None,
        **parameters: object,
    ) -> None:
        super().__init__()
        text_config = dict(parameters)
        text_config.update(text_parameters or {})
        tree_config = dict(parameters)
        tree_config.update(emv45_parameters or {})

        self.text_pipeline = JSJV9PreprocessingPipeline(**text_config)
        self.tree_pipeline = EMV45PreprocessingPipeline(**tree_config)
        self.text_feature_count_ = 0
        self.tree_feature_count_ = 0
        self.steps = (
            "JSJ9 word/char TF-IDF 텍스트 뷰",
            "EMV45 mutation·consequence·hotspot·OOF signature 트리 뷰",
            "base learner별 피처 분리",
            "outer fold-train에서만 전처리 fit",
            "강화된 L2 meta learner 입력 제공",
            "GBMLGG·KIPAN·STES 원본 SUBCLASS 유지",
        )

    @staticmethod
    def _to_sparse(features: pd.DataFrame) -> csr_matrix:
        return csr_matrix(features.to_numpy(dtype="float32", copy=False))

    def _bundle(
        self, text_bundle, tree: pd.DataFrame
    ) -> OOFStackingV007FeatureBundle:
        self.text_feature_count_ = int(text_bundle.text.shape[1])
        self.tree_feature_count_ = int(tree.shape[1])
        return OOFStackingV007FeatureBundle(
            text=text_bundle.text.tocsr(),
            tree=self._to_sparse(tree),
        )

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "PipeCombEMV1007PreprocessingPipeline":
        self.feature_columns = features.columns.tolist()
        self.text_pipeline.fit(features, labels)
        self.tree_pipeline.fit(features, labels)
        self.label_encoder.fit(labels)
        self.text_feature_count_ = int(self.text_pipeline.summary()["remaining_features"])
        self.tree_feature_count_ = int(self.tree_pipeline.summary()["remaining_features"])
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> OOFStackingV007FeatureBundle:
        self.feature_columns = features.columns.tolist()
        text = self.text_pipeline.fit_transform(features, labels)
        tree = self.tree_pipeline.fit_transform(features, labels)
        self.label_encoder.fit(labels)
        return self._bundle(text, tree)

    def transform(self, features: pd.DataFrame) -> OOFStackingV007FeatureBundle:
        return self._bundle(
            self.text_pipeline.transform(features),
            self.tree_pipeline.transform(features),
        )

    def summary(self) -> dict[str, int]:
        return {
            "remaining_features": self.text_feature_count_ + self.tree_feature_count_,
            "text_features": self.text_feature_count_,
            "tree_features": self.tree_feature_count_,
            "model_specific_views": 2,
            "duplicated_columns_in_same_view": 0,
            "dropped_constant_features": int(
                self.tree_pipeline.summary().get("dropped_constant_features", 0)
            ),
        }
