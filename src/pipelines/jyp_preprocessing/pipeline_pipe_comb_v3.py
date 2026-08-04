"""JYP F9과 EM24 dual signature를 결합한 pipeComb_v3 파이프라인입니다."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_jyp_f9 import (
    F9GlobalAAPairNoRawPreprocessingPipeline,
)
from src.pipelines.pipeline_em_v24 import EMV24PreprocessingPipeline


class PipeCombV3PreprocessingPipeline(PreprocessingPipeline):
    """F9 뒤에 EM24 dual signature만 추가합니다(원본 26클래스는 104개).

    학습 행에는 두 구성 파이프라인의 ``fit_transform``을 사용하므로 F9의
    pair contrast와 EM24의 암종 signature가 모두 내부 OOF 통계로 생성됩니다.
    검증·테스트 행에는 동일 학습 데이터에서 만든 full-fit 통계만 적용합니다.
    """

    name = "pipeComb_v3"
    evaluation_folds = 5
    artifact_schema_version = 1

    def __init__(
        self,
        *,
        burden_clip_quantile: float = 0.99,
        f3_position_min_support: int = 3,
        f3_aa_min_support: int = 3,
        f4_min_support: int = 5,
        f7_pairs: Sequence[Sequence[str]] = (
            ("KIRC", "KIPAN"),
            ("LGG", "GBMLGG"),
        ),
        f7_top_k_per_direction: int = 3,
        f7_min_gene_support: int = 10,
        f7_laplace_alpha: float = 4.0,
        f7_burden_quantiles: int = 5,
        f7_stability_folds: int = 5,
        f7_min_direction_consistency: int = 4,
        f7_min_selection_frequency: int = 3,
        f7_random_state: int = 42,
        em24_min_mutation_count: int = 5,
        em24_min_functional_mutation_count: int = 5,
        em24_top_genes_per_class: int = 20,
        em24_smoothing: float = 0.5,
        em24_max_log2_odds: float = 8.0,
        em24_shrinkage: float = 10.0,
        em24_min_hotspot_count: int = 5,
        em24_max_hotspots: int = 384,
        em24_inner_signature_folds: int = 5,
        em24_signature_random_state: int = 42,
        show_progress: bool = True,
        progress_interval: int = 25000,
    ) -> None:
        super().__init__()
        self.pipeline_name = self.name
        self.feature_blocks = (
            "jyp_f9",
            "em24_signature_all",
            "em24_signature_functional",
        )
        self.steps = (
            "JYP F9 피처 생성",
            "EM24 전체 변이 암종 signature 생성",
            "EM24 기능 변이 암종 signature 생성",
            "F9 CSR 뒤에 EM24 dual signature 결합",
        )
        self.f9_parameters = {
            "burden_clip_quantile": burden_clip_quantile,
            "f3_position_min_support": f3_position_min_support,
            "f3_aa_min_support": f3_aa_min_support,
            "f4_min_support": f4_min_support,
            "f7_pairs": f7_pairs,
            "f7_top_k_per_direction": f7_top_k_per_direction,
            "f7_min_gene_support": f7_min_gene_support,
            "f7_laplace_alpha": f7_laplace_alpha,
            "f7_burden_quantiles": f7_burden_quantiles,
            "f7_stability_folds": f7_stability_folds,
            "f7_min_direction_consistency": f7_min_direction_consistency,
            "f7_min_selection_frequency": f7_min_selection_frequency,
            "f7_random_state": f7_random_state,
            "show_progress": show_progress,
            "progress_interval": progress_interval,
        }
        self.em24_parameters = {
            "min_mutation_count": em24_min_mutation_count,
            "min_functional_mutation_count": em24_min_functional_mutation_count,
            "top_genes_per_class": em24_top_genes_per_class,
            "smoothing": em24_smoothing,
            "max_log2_odds": em24_max_log2_odds,
            "shrinkage": em24_shrinkage,
            "min_hotspot_count": em24_min_hotspot_count,
            "max_hotspots": em24_max_hotspots,
            "inner_signature_folds": em24_inner_signature_folds,
            "signature_random_state": em24_signature_random_state,
        }
        self._reset_components()

    def _reset_components(self) -> None:
        self.f9_pipeline_ = F9GlobalAAPairNoRawPreprocessingPipeline(
            **self.f9_parameters
        )
        self.em24_pipeline_ = EMV24PreprocessingPipeline(**self.em24_parameters)
        self.feature_names_out_: np.ndarray | None = None
        self.em24_signature_columns_: tuple[str, ...] = ()

    @staticmethod
    def _normalize_labels(
        features: pd.DataFrame,
        labels: pd.Series | Sequence[str],
    ) -> pd.Series:
        if isinstance(labels, pd.Series):
            return labels
        if isinstance(labels, (str, bytes)):
            raise TypeError("labels는 문자열 하나가 아니라 1차원 시퀀스여야 합니다.")
        return pd.Series(labels, index=features.index)

    def _finalize_schema(self, em24_frame: pd.DataFrame) -> None:
        f9_classes = np.asarray(self.f9_pipeline_.label_encoder.classes_, dtype=str)
        em24_classes = np.asarray(self.em24_pipeline_.label_encoder.classes_, dtype=str)
        if not np.array_equal(f9_classes, em24_classes):
            raise RuntimeError("F9과 EM24의 원본 SUBCLASS 순서가 다릅니다.")

        columns = em24_frame.columns.astype(str).tolist()
        signature_all = [
            name for name in columns if name.startswith("signature_all_")
        ]
        signature_functional = [
            name for name in columns if name.startswith("signature_functional_")
        ]
        expected_per_channel = 2 * len(f9_classes)
        if (
            len(signature_all) != expected_per_channel
            or len(signature_functional) != expected_per_channel
        ):
            raise RuntimeError(
                "EM24 dual signature 스키마가 클래스당 weighted·match_count "
                "2개 구성과 일치하지 않습니다."
            )

        self.em24_signature_columns_ = tuple(
            [*signature_all, *signature_functional]
        )
        f9_names = self.f9_pipeline_.get_feature_names_out().astype(str).tolist()
        em24_names = [
            f"EM24__{name}" for name in self.em24_signature_columns_
        ]
        self.feature_names_out_ = np.asarray([*f9_names, *em24_names], dtype=object)
        self.label_encoder = self.f9_pipeline_.label_encoder

    @staticmethod
    def _validate_output_index(
        features: pd.DataFrame,
        em24_frame: pd.DataFrame,
    ) -> None:
        if not em24_frame.index.equals(features.index):
            raise RuntimeError("EM24 출력 행의 인덱스 또는 순서가 입력과 다릅니다.")

    def _combine(
        self,
        f9_matrix: object,
        em24_frame: pd.DataFrame,
    ) -> sparse.csr_matrix:
        self._require_fitted()
        if not isinstance(em24_frame, pd.DataFrame):
            raise TypeError("EM24 전처리 결과는 pandas DataFrame이어야 합니다.")
        missing = [
            name
            for name in self.em24_signature_columns_
            if name not in em24_frame.columns
        ]
        if missing:
            raise RuntimeError(f"EM24 dual signature 열이 누락되었습니다: {missing}")

        f9_csr = sparse.csr_matrix(f9_matrix, dtype=np.float32)
        if f9_csr.shape[0] != len(em24_frame):
            raise RuntimeError("F9과 EM24 행 수가 다릅니다.")
        signature_values = em24_frame.loc[
            :, list(self.em24_signature_columns_)
        ].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(signature_values).all():
            raise RuntimeError("EM24 dual signature에 NaN 또는 무한대가 있습니다.")

        output = sparse.hstack(
            [f9_csr, sparse.csr_matrix(signature_values)],
            format="csr",
            dtype=np.float32,
        )
        if output.shape[1] != len(self.feature_names_out_):
            raise RuntimeError("생성된 피처 수와 pipeComb_v3 스키마가 다릅니다.")
        return output

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series | Sequence[str],
    ) -> "PipeCombV3PreprocessingPipeline":
        normalized_labels = self._normalize_labels(features, labels)
        self._reset_components()
        self.f9_pipeline_.fit(features, normalized_labels)
        self.em24_pipeline_.fit(features, normalized_labels)
        em24_frame = self.em24_pipeline_.transform(features)
        self._validate_output_index(features, em24_frame)
        self._finalize_schema(em24_frame)
        return self

    def fit_transform(
        self,
        features: pd.DataFrame,
        labels: pd.Series | Sequence[str],
    ) -> sparse.csr_matrix:
        normalized_labels = self._normalize_labels(features, labels)
        self._reset_components()
        f9_matrix = self.f9_pipeline_.fit_transform(features, normalized_labels)
        em24_frame = self.em24_pipeline_.fit_transform(features, normalized_labels)
        self._validate_output_index(features, em24_frame)
        self._finalize_schema(em24_frame)
        return self._combine(f9_matrix, em24_frame)

    def transform(self, features: pd.DataFrame) -> sparse.csr_matrix:
        self._require_fitted()
        f9_matrix = self.f9_pipeline_.transform(features)
        em24_frame = self.em24_pipeline_.transform(features)
        self._validate_output_index(features, em24_frame)
        return self._combine(f9_matrix, em24_frame)

    def get_feature_names_out(self) -> np.ndarray:
        self._require_fitted()
        return self.feature_names_out_.copy()

    def summary(self) -> dict[str, object]:
        self._require_fitted()
        f9_summary = self.f9_pipeline_.summary()
        em24_summary = self.em24_pipeline_.summary()
        signature_all_count = sum(
            name.startswith("signature_all_")
            for name in self.em24_signature_columns_
        )
        signature_functional_count = (
            len(self.em24_signature_columns_) - signature_all_count
        )
        return {
            "pipeline_name": self.pipeline_name,
            "feature_blocks": list(self.feature_blocks),
            "remaining_features": len(self.feature_names_out_),
            "raw_features": 0,
            "includes_raw_ordinal": False,
            "f9_features": int(f9_summary["remaining_features"]),
            "em24_dual_signature_features": len(self.em24_signature_columns_),
            "em24_signature_all_features": signature_all_count,
            "em24_signature_functional_features": signature_functional_count,
            "class_count": len(self.label_encoder.classes_),
            "f9_summary": f9_summary,
            "em24_summary": em24_summary,
        }

    def get_diagnostics(self) -> dict[str, object]:
        self._require_fitted()
        return {
            "pipeline_name": self.pipeline_name,
            "summary": self.summary(),
            "f9": self.f9_pipeline_.get_diagnostics(),
            "em24": self.em24_pipeline_.summary(),
            "em24_signature_columns": list(self.em24_signature_columns_),
        }

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(self.feature_blocks)

    def encode_labels(
        self,
        labels: pd.Series | Sequence[str],
    ) -> np.ndarray:
        self._require_fitted()
        return self.label_encoder.transform(np.asarray(labels))

    def decode_labels(self, labels: Sequence[int]) -> np.ndarray:
        self._require_fitted()
        return self.label_encoder.inverse_transform(np.asarray(labels, dtype=int))

    def _require_fitted(self) -> None:
        if self.feature_names_out_ is None:
            raise RuntimeError("fit을 먼저 실행해야 합니다.")
