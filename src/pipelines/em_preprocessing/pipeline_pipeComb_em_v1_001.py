"""pipeComb_em_v1에 규제형 E14 전용 뷰를 추가한 Weighted Voting 입력."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E14 import (
    EME14PreprocessingPipeline,
)
from src.pipelines.em_preprocessing.pipeline_em_F05 import (
    EMF05PreprocessingPipeline,
)
from src.pipelines.em_preprocessing.pipeline_em_F10 import (
    EMF10PreprocessingPipeline,
)
from src.pipelines.pipeline_jsj_v1 import TextTreeFeatureBundle
from src.pipelines.pipeline_pipeComb_em_v1 import (
    PipeCombEMV1PreprocessingPipeline,
)


@dataclass
class WeightedSoftVotingFeatureBundle:
    """broad·E14·F05·F10 모델에 전달할 분리 피처입니다."""

    broad: TextTreeFeatureBundle
    regularized_e14: csr_matrix
    em_f05: csr_matrix
    em_f10: csr_matrix


class PipeCombEMV1001PreprocessingPipeline(PreprocessingPipeline):
    """broad·E14·F05·F10 피처를 모델별 뷰로 분리합니다.

    같은 열을 하나의 행렬에 두 번 붙이지 않습니다. 기존 결합 모델은
    ``broad``만, 강한 규제 모델은 ``regularized_e14``만 사용합니다.
    """

    name = "pipeComb_em_v1_001"
    evaluation_folds = 5

    def __init__(
        self,
        e14_parameters: dict[str, object] | None = None,
        f05_parameters: dict[str, object] | None = None,
        f10_parameters: dict[str, object] | None = None,
        include_high_performance_views: bool = False,
        **parameters: object,
    ) -> None:
        super().__init__()
        self.broad_pipeline = PipeCombEMV1PreprocessingPipeline(**parameters)
        regularized_parameters = dict(parameters)
        regularized_parameters.update(e14_parameters or {})
        self.e14_pipeline = EME14PreprocessingPipeline(**regularized_parameters)
        self.include_high_performance_views = bool(include_high_performance_views)
        self.f05_pipeline = None
        self.f10_pipeline = None
        if self.include_high_performance_views:
            f05_combined_parameters = dict(parameters)
            f05_combined_parameters.update(f05_parameters or {})
            self.f05_pipeline = EMF05PreprocessingPipeline(**f05_combined_parameters)
            f10_combined_parameters = dict(parameters)
            f10_combined_parameters.update(f10_parameters or {})
            self.f10_pipeline = EMF10PreprocessingPipeline(**f10_combined_parameters)
        self.regularized_feature_count_ = 0
        self.f05_feature_count_ = 0
        self.f10_feature_count_ = 0
        self.steps = (
            "pipeComb_em_v1 텍스트·EMV45 broad 뷰",
            "E14 전용 규제 뷰",
            "EMV45+F05 XGBoost 전용 뷰",
            "EMV45+F10 XGBoost 전용 뷰",
            "모델별 피처 분리로 열 중복 결합 방지",
            "fold-train 내부에서만 네 전처리 fit",
            "원본 SUBCLASS 유지",
        )

    @staticmethod
    def _to_sparse(features: pd.DataFrame) -> csr_matrix:
        return csr_matrix(features.to_numpy(dtype="float32", copy=False))

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "PipeCombEMV1001PreprocessingPipeline":
        self.feature_columns = features.columns.tolist()
        self.broad_pipeline.fit(features, labels)
        self.e14_pipeline.fit(features, labels)
        if self.f05_pipeline is not None and self.f10_pipeline is not None:
            self.f05_pipeline.fit(features, labels)
            self.f10_pipeline.fit(features, labels)
        self.regularized_feature_count_ = int(
            self.e14_pipeline.summary()["remaining_features"]
        )
        if self.f05_pipeline is not None and self.f10_pipeline is not None:
            self.f05_feature_count_ = int(
                self.f05_pipeline.summary()["remaining_features"]
            )
            self.f10_feature_count_ = int(
                self.f10_pipeline.summary()["remaining_features"]
            )
        self.label_encoder.fit(labels)
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> WeightedSoftVotingFeatureBundle:
        self.feature_columns = features.columns.tolist()
        broad = self.broad_pipeline.fit_transform(features, labels)
        regularized = self.e14_pipeline.fit_transform(features, labels)
        if self.f05_pipeline is not None and self.f10_pipeline is not None:
            f05 = self.f05_pipeline.fit_transform(features, labels)
            f10 = self.f10_pipeline.fit_transform(features, labels)
        else:
            f05 = pd.DataFrame(index=features.index)
            f10 = pd.DataFrame(index=features.index)
        self.regularized_feature_count_ = regularized.shape[1]
        self.f05_feature_count_ = f05.shape[1]
        self.f10_feature_count_ = f10.shape[1]
        self.label_encoder.fit(labels)
        return WeightedSoftVotingFeatureBundle(
            broad=broad,
            regularized_e14=self._to_sparse(regularized),
            em_f05=self._to_sparse(f05),
            em_f10=self._to_sparse(f10),
        )

    def transform(self, features: pd.DataFrame) -> WeightedSoftVotingFeatureBundle:
        if self.f05_pipeline is not None and self.f10_pipeline is not None:
            f05 = self.f05_pipeline.transform(features)
            f10 = self.f10_pipeline.transform(features)
        else:
            f05 = pd.DataFrame(index=features.index)
            f10 = pd.DataFrame(index=features.index)
        return WeightedSoftVotingFeatureBundle(
            broad=self.broad_pipeline.transform(features),
            regularized_e14=self._to_sparse(self.e14_pipeline.transform(features)),
            em_f05=self._to_sparse(f05),
            em_f10=self._to_sparse(f10),
        )

    def summary(self) -> dict[str, int]:
        broad_summary = self.broad_pipeline.summary()
        e14_summary = self.e14_pipeline.summary()
        f05_summary = {} if self.f05_pipeline is None else self.f05_pipeline.summary()
        f10_summary = {} if self.f10_pipeline is None else self.f10_pipeline.summary()
        return {
            "remaining_features": (
                int(broad_summary["remaining_features"])
                + self.regularized_feature_count_
                + self.f05_feature_count_
                + self.f10_feature_count_
            ),
            "broad_view_features": int(broad_summary["remaining_features"]),
            "regularized_e14_features": self.regularized_feature_count_,
            "em_f05_features": self.f05_feature_count_,
            "em_f10_features": self.f10_feature_count_,
            "dropped_constant_features": (
                int(broad_summary.get("dropped_constant_features", 0))
                + int(e14_summary.get("dropped_constant_features", 0))
                + int(f05_summary.get("dropped_constant_features", 0))
                + int(f10_summary.get("dropped_constant_features", 0))
            ),
            "duplicated_columns_in_same_view": 0,
            "model_specific_views": 4,
        }
