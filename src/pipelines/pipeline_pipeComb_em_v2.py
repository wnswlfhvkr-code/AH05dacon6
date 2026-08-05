"""EMV46과 JSJ9 텍스트를 stacking 모델별 뷰로 분리한 pipeComb EM v2."""

from __future__ import annotations

import pandas as pd
from scipy.sparse import csr_matrix

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_em_v46 import EMV46PreprocessingPipeline
from src.pipelines.pipeline_jsj_v1 import TextTreeFeatureBundle
from src.pipelines.pipeline_jsj_v9 import JSJV9PreprocessingPipeline


class PipeCombEMV2PreprocessingPipeline(PreprocessingPipeline):
    """EMV46 수치 뷰와 JSJ9 TF-IDF를 겹치지 않는 두 뷰로 반환합니다.

    JSJ9의 tree 뷰는 유전자 변이 여부와 구조 요약으로 구성되어 EMV46의
    mutation·burden·consequence·multi-hit 계열과 중복됩니다. 따라서 JSJ9의
    Word/Char TF-IDF만 유지하고 tree 뷰는 결합하지 않습니다.
    """

    name = "pipeComb_em_v2"
    # Stacking 자체가 train 내부 OOF를 수행하므로 train.py의 outer OOF는
    # 비활성화합니다. 평가용 holdout과 최종 전체 학습에서 각각 한 번만
    # single-level stacking을 수행합니다.
    evaluation_folds = 1

    def __init__(
        self,
        emv46_parameters: dict[str, object] | None = None,
        text_parameters: dict[str, object] | None = None,
        **parameters: object,
    ) -> None:
        super().__init__()
        em_config = dict(parameters)
        em_config.update(emv46_parameters or {})
        jsj_config = dict(parameters)
        jsj_config.update(text_parameters or {})
        self.emv46_pipeline = EMV46PreprocessingPipeline(**em_config)
        self.jsj9_pipeline = JSJV9PreprocessingPipeline(**jsj_config)
        self.emv46_feature_count_ = 0
        self.jsj9_text_feature_count_ = 0
        self.removed_jsj9_tree_feature_count_ = 0
        self.combined_feature_count_ = 0
        self.steps = (
            "EMV46 수치 피처 1회 생성",
            "JSJ9 Word·Char TF-IDF 텍스트 피처",
            "중복 JSJ9 유전자 변이 여부·구조 tree 뷰 제외",
            "LinearSVC용 text·XGBoost/LightGBM용 tree 뷰 분리",
            "단일 레벨 5-fold OOF stacking 입력 제공",
            "중첩 방지를 위한 outer OOF 비활성화",
            "fold-train에서만 EMV46·JSJ9 fit",
            "원본 SUBCLASS 유지",
        )

    @staticmethod
    def _require_jsj9_bundle(features) -> TextTreeFeatureBundle:
        if not isinstance(features, TextTreeFeatureBundle):
            raise TypeError("JSJ9 전처리가 TextTreeFeatureBundle을 반환하지 않았습니다.")
        return features

    def _bundle(
        self,
        emv46_features: pd.DataFrame,
        jsj9_features: TextTreeFeatureBundle,
    ) -> TextTreeFeatureBundle:
        tree = csr_matrix(emv46_features.to_numpy(dtype="float32", copy=False))
        self.emv46_feature_count_ = int(emv46_features.shape[1])
        self.jsj9_text_feature_count_ = int(jsj9_features.text.shape[1])
        self.removed_jsj9_tree_feature_count_ = int(jsj9_features.tree.shape[1])
        self.combined_feature_count_ = int(
            self.emv46_feature_count_ + self.jsj9_text_feature_count_
        )
        return TextTreeFeatureBundle(text=jsj9_features.text.tocsr(), tree=tree)

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "PipeCombEMV2PreprocessingPipeline":
        self.feature_columns = features.columns.tolist()
        self.emv46_pipeline.fit(features, labels)
        self.jsj9_pipeline.fit(features, labels)
        self._bundle(
            self.emv46_pipeline.transform(features),
            self._require_jsj9_bundle(self.jsj9_pipeline.transform(features)),
        )
        self.label_encoder.fit(labels)
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> TextTreeFeatureBundle:
        self.feature_columns = features.columns.tolist()
        emv46 = self.emv46_pipeline.fit_transform(features, labels)
        jsj9 = self._require_jsj9_bundle(
            self.jsj9_pipeline.fit_transform(features, labels)
        )
        self.label_encoder.fit(labels)
        return self._bundle(emv46, jsj9)

    def transform(self, features: pd.DataFrame) -> TextTreeFeatureBundle:
        return self._bundle(
            self.emv46_pipeline.transform(features),
            self._require_jsj9_bundle(self.jsj9_pipeline.transform(features)),
        )

    def summary(self) -> dict[str, int]:
        return {
            "remaining_features": self.combined_feature_count_,
            "emv46_features": self.emv46_feature_count_,
            "jsj9_text_features": self.jsj9_text_feature_count_,
            "removed_duplicate_jsj9_tree_features": (
                self.removed_jsj9_tree_feature_count_
            ),
            "dropped_constant_features": int(
                self.emv46_pipeline.summary().get("dropped_constant_features", 0)
            ),
        }
