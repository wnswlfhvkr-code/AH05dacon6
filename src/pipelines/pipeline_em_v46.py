"""EMV45와 F22 실험(F20)의 고유 피처를 중복 없이 결합한 독립 EMV46."""

from __future__ import annotations

import pandas as pd

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_em_v45 import EMV45PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F01 import (_F01DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F02 import (_F02DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F03 import (_F03DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F04 import (_F04DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F05 import (_F05DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F06 import (_F06DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F07 import (_F07DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F08 import (_F08DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F09 import (_F09DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F10 import (_F10DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F11 import (_F11DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F12 import (_F12DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F13 import (_F13DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F14 import (_F14DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F15 import (_F15DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F16 import (_F16DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F17 import (_F17DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F18 import (_F18DerivedFeaturePipeline)
from src.pipelines.em_preprocessing.pipeline_em_F19 import (_F19DerivedFeaturePipeline)


V46_DERIVED_PIPELINE_CLASSES = {
    "F01": _F01DerivedFeaturePipeline,
    "F02": _F02DerivedFeaturePipeline,
    "F03": _F03DerivedFeaturePipeline,
    "F04": _F04DerivedFeaturePipeline,
    "F05": _F05DerivedFeaturePipeline,
    "F06": _F06DerivedFeaturePipeline,
    "F07": _F07DerivedFeaturePipeline,
    "F08": _F08DerivedFeaturePipeline,
    "F09": _F09DerivedFeaturePipeline,
    "F10": _F10DerivedFeaturePipeline,
    "F11": _F11DerivedFeaturePipeline,
    "F12": _F12DerivedFeaturePipeline,
    "F13": _F13DerivedFeaturePipeline,
    "F14": _F14DerivedFeaturePipeline,
    "F15": _F15DerivedFeaturePipeline,
    "F16": _F16DerivedFeaturePipeline,
    "F17": _F17DerivedFeaturePipeline,
    "F18": _F18DerivedFeaturePipeline,
    "F19": _F19DerivedFeaturePipeline,
}


class EMV46PreprocessingPipeline(PreprocessingPipeline):
    """EMV45를 한 번 유지하고 F22가 사용한 F20 고유 피처를 추가합니다.

    ``test_006_em_F22``는 별도 F22 파이프라인이 아니라 EMF20을 사용하므로,
    EMF20 전체를 결합하지 않고 그 안의 F01~F19 파생 부분만 한 번 추가합니다.
    """

    name = "em_v46"
    evaluation_folds = 5

    def __init__(
        self,
        feature_parameters: dict[str, dict[str, object]] | None = None,
        **parameters: object,
    ) -> None:
        super().__init__()
        overrides = {
            str(name).upper(): dict(values)
            for name, values in (feature_parameters or {}).items()
        }
        unknown = set(overrides) - set(V46_DERIVED_PIPELINE_CLASSES)
        if unknown:
            raise ValueError(f"지원하지 않는 EMV46 구성 피처입니다: {sorted(unknown)}")
        self.v45_pipeline = EMV45PreprocessingPipeline(**parameters)
        self.derived_pipelines: dict[str, PreprocessingPipeline] = {}
        for feature_name, pipeline_class in V46_DERIVED_PIPELINE_CLASSES.items():
            feature_config = dict(parameters)
            feature_config.update(overrides.get(feature_name, {}))
            self.derived_pipelines[feature_name] = pipeline_class(**feature_config)
        self.v45_feature_count_ = 0
        self.derived_feature_counts_: dict[str, int] = {
            name: 0 for name in V46_DERIVED_PIPELINE_CLASSES
        }
        self.steps = (
            "EMV45 베이스 피처 1회 생성",
            "F22 실험의 실제 전처리 EMF20에서 중복 EMV45 제외",
            "F01~F15 비레이블 파생 피처",
            "F16~F19 inner-fold OOF signature 피처",
            "F번호 접두사로 컬럼 충돌 제거",
            "원본 SUBCLASS 유지",
        )

    @staticmethod
    def _combine(
        v45: pd.DataFrame,
        derived: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        parts = [v45]
        for feature_name in V46_DERIVED_PIPELINE_CLASSES:
            frame = derived[feature_name].copy()
            frame.columns = [
                f"{feature_name}__{column}" for column in frame.columns
            ]
            parts.append(frame)
        return pd.concat(parts, axis=1)

    def _transformed_derived(
        self, features: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        return {
            name: pipeline.transform(features)
            for name, pipeline in self.derived_pipelines.items()
        }

    def _update_counts(
        self,
        v45: pd.DataFrame,
        derived: dict[str, pd.DataFrame],
    ) -> None:
        self.v45_feature_count_ = v45.shape[1]
        self.derived_feature_counts_ = {
            name: frame.shape[1] for name, frame in derived.items()
        }

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "EMV46PreprocessingPipeline":
        self.feature_columns = features.columns.tolist()
        self.v45_pipeline.fit(features, labels)
        for pipeline in self.derived_pipelines.values():
            pipeline.fit(features, labels)
        v45 = self.v45_pipeline.transform(features)
        derived = self._transformed_derived(features)
        self._update_counts(v45, derived)
        PreprocessingPipeline.fit(self, self._combine(v45, derived), labels)
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> pd.DataFrame:
        self.feature_columns = features.columns.tolist()
        v45 = self.v45_pipeline.fit_transform(features, labels)
        derived = {
            name: pipeline.fit_transform(features, labels)
            for name, pipeline in self.derived_pipelines.items()
        }
        self._update_counts(v45, derived)
        combined = self._combine(v45, derived)
        PreprocessingPipeline.fit(self, combined, labels)
        return PreprocessingPipeline.transform(self, combined).astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        combined = self._combine(
            self.v45_pipeline.transform(features),
            self._transformed_derived(features),
        )
        return PreprocessingPipeline.transform(self, combined).astype("float32")

    def summary(self) -> dict[str, int]:
        result = PreprocessingPipeline.summary(self)
        result.update({
            "v45_base_features": self.v45_feature_count_,
            "f01_f19_derived_features": sum(self.derived_feature_counts_.values()),
            "included_feature_groups": len(self.derived_feature_counts_),
            "combined_before_constant_filter": (
                self.v45_feature_count_ + sum(self.derived_feature_counts_.values())
            ),
        })
        return result
