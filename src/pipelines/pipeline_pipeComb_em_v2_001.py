"""pipeComb_em_v1_007의 EMV45 tree를 EMV46으로 교체한 v2_001."""

from __future__ import annotations

import pandas as pd
from scipy.sparse import csr_matrix

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_jsj_v1 import TextTreeFeatureBundle
from src.pipelines.pipeline_jsj_v9 import JSJV9PreprocessingPipeline
from src.pipelines.pipeline_em_v46 import EMV46PreprocessingPipeline


class PipeCombEMV2001PreprocessingPipeline(PreprocessingPipeline):
    """JSJ9 text와 EMV46 tree를 stacking용 독립 뷰로 반환합니다.

    EMV46은 EMV45와 F01~F19 고유 피처를 이미 중복 없이 포함합니다.
    따라서 EMV45를 다시 붙이지 않고, JSJ9의 mutation·structure tree도
    제외하여 Word/Char TF-IDF text와 EMV46 tree만 사용합니다.
    """

    name = "pipeComb_em_v2_001"
    evaluation_folds = 5

    def __init__(
        self,
        text_parameters: dict[str, object] | None = None,
        emv46_parameters: dict[str, object] | None = None,
        **parameters: object,
    ) -> None:
        super().__init__()
        text_config = dict(parameters)
        text_config.update(text_parameters or {})
        tree_config = dict(parameters)
        tree_config.update(emv46_parameters or {})
        self.text_pipeline = JSJV9PreprocessingPipeline(**text_config)
        self.tree_pipeline = EMV46PreprocessingPipeline(**tree_config)
        self.text_feature_count_ = 0
        self.tree_feature_count_ = 0
        self.removed_jsj9_tree_feature_count_ = 0
        self.steps = (
            "JSJ9 Word·Char TF-IDF text 뷰",
            "EMV46 mutation·파생 피처 tree 뷰",
            "EMV46 내부 EMV45 베이스 1회만 유지",
            "중복 JSJ9 mutation·structure tree 제외",
            "Outer 5-fold·Inner 5-fold OOF stacking 입력",
            "fold-train에서만 모든 전처리 fit",
            "GBMLGG·KIPAN·STES 원본 SUBCLASS 유지",
        )

    @staticmethod
    def _require_bundle(features) -> TextTreeFeatureBundle:
        if not isinstance(features, TextTreeFeatureBundle):
            raise TypeError("JSJ9 전처리가 TextTreeFeatureBundle을 반환하지 않았습니다.")
        return features

    @staticmethod
    def _to_sparse(features: pd.DataFrame) -> csr_matrix:
        return csr_matrix(features.to_numpy(dtype="float32", copy=False))

    def _bundle(
        self,
        text_features: TextTreeFeatureBundle,
        tree_features: pd.DataFrame,
    ) -> TextTreeFeatureBundle:
        self.text_feature_count_ = int(text_features.text.shape[1])
        self.tree_feature_count_ = int(tree_features.shape[1])
        self.removed_jsj9_tree_feature_count_ = int(text_features.tree.shape[1])
        return TextTreeFeatureBundle(
            text=text_features.text.tocsr(),
            tree=self._to_sparse(tree_features),
        )

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "PipeCombEMV2001PreprocessingPipeline":
        self.feature_columns = features.columns.tolist()
        self.text_pipeline.fit(features, labels)
        self.tree_pipeline.fit(features, labels)
        text = self._require_bundle(self.text_pipeline.transform(features))
        tree = self.tree_pipeline.transform(features)
        self._bundle(text, tree)
        self.label_encoder.fit(labels)
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> TextTreeFeatureBundle:
        self.feature_columns = features.columns.tolist()
        text = self._require_bundle(self.text_pipeline.fit_transform(features, labels))
        tree = self.tree_pipeline.fit_transform(features, labels)
        self.label_encoder.fit(labels)
        return self._bundle(text, tree)

    def transform(self, features: pd.DataFrame) -> TextTreeFeatureBundle:
        return self._bundle(
            self._require_bundle(self.text_pipeline.transform(features)),
            self.tree_pipeline.transform(features),
        )

    def summary(self) -> dict[str, int]:
        return {
            "remaining_features": self.text_feature_count_ + self.tree_feature_count_,
            "jsj9_text_features": self.text_feature_count_,
            "emv46_tree_features": self.tree_feature_count_,
            "removed_duplicate_jsj9_tree_features": (
                self.removed_jsj9_tree_feature_count_
            ),
            "retained_emv46_instances": 1,
            "duplicate_emv45_tree_added": 0,
            "dropped_constant_features": int(
                self.tree_pipeline.summary().get("dropped_constant_features", 0)
            ),
        }
