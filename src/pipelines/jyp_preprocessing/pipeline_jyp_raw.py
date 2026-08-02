"""jyp_raw 고정 JYP 전처리 파이프라인입니다.

AGENTS.md의 버전 독립 규칙에 따라 이 파일이 사용하는 파싱, 피처 생성,
학습 및 변환 로직을 파일 안에 직접 보관합니다.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from src.pipelines.base import PreprocessingPipeline
RAW_MISSING_TOKEN = '__JYP_RAW_MISSING__'

class RawPreprocessingPipeline(PreprocessingPipeline):
    """RawPreprocessingPipeline의 단계 고정 전처리 구현입니다."""
    name = 'jyp_raw'
    artifact_schema_version = 4

    def __init__(self, *, show_progress: bool=True, progress_interval: int=25000) -> None:
        super().__init__()
        self.pipeline_name = self.name
        self.feature_blocks = ('raw',)
        self.steps = self.feature_blocks
        if progress_interval < 1:
            raise ValueError('progress_interval은 1 이상이어야 합니다.')
        self.show_progress = bool(show_progress)
        self.progress_interval = int(progress_interval)
        self.label_encoder = LabelEncoder()
        self.raw_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32)

    def fit(self, features: pd.DataFrame, labels: pd.Series | Sequence[str]) -> 'RawPreprocessingPipeline':
        frame = self._validate_frame(features)
        label_values = self._validate_labels(frame, labels)
        self._fit_frame_reference_ = frame
        self.gene_columns_ = frame.columns.tolist()
        self.label_encoder.fit(label_values)
        self.raw_encoder.fit(self._prepare_raw(frame))
        self.raw_wt_codes_ = np.asarray([float(np.flatnonzero(categories == 'WT')[0]) if np.any(categories == 'WT') else -2.0 for categories in self.raw_encoder.categories_], dtype=np.float32)
        self.active_gene_indices_ = np.asarray([], dtype=np.int64)
        self.active_gene_columns_: list[str] = []
        self.dropped_constant_columns: list[str] = []
        self.active_f2_indices_ = np.asarray([], dtype=np.int64)
        self.f2_feature_names_: list[str] = []
        self.active_f3_position_indices_ = np.asarray([], dtype=np.int64)
        self.f3_position_feature_names_: list[str] = []
        self.active_f3_aa_indices_ = np.asarray([], dtype=np.int64)
        self.f3_aa_feature_names_: list[str] = []
        self.f4_hotspot_vocabulary_: tuple[tuple[int, str], ...] = ()
        self.f4_feature_names_: list[str] = []
        self.f5_signature_model_: object | None = None
        self._fit_f5_oof_matrix_: sparse.csr_matrix | None = None
        self.f5_feature_names_: list[str] = []
        self.f5_output_class_labels_: tuple[str, ...] = ()
        self.f5_output_class_indices_ = np.asarray([], dtype=np.int64)
        self.f5_output_score_names_: tuple[str, ...] = ()
        self.f5_output_column_indices_ = np.asarray([], dtype=np.int64)
        self.f5_selected_gene_indices_ = np.asarray([], dtype=np.int64)
        self.f7_pair_contrast_model_: object | None = None
        self._fit_f7_oof_matrix_: sparse.csr_matrix | None = None
        self.f7_feature_names_: list[str] = []
        self.f7_selected_gene_indices_ = np.asarray([], dtype=np.int64)
        self.f7_selected_gene_catalog_: list[dict[str, object]] = []
        self.burden_clip_value_ = 0.0
        self.feature_names_out_ = np.asarray([f'RAW__gene__{gene}' for gene in self.gene_columns_], dtype=object)
        self._log(f'fit 완료 | RAW={len(self.gene_columns_):,}')
        return self

    def transform(self, features: pd.DataFrame):
        self._require_fitted()
        frame = self._validate_transform_frame(features)
        use_fit_oof = getattr(self, '_fit_frame_reference_', None) is features
        scan = None
        return self._build_matrix(frame, scan=scan, use_fit_oof=use_fit_oof)

    def _build_matrix(self, frame: pd.DataFrame, *, scan: object | None, use_fit_oof: bool) -> sparse.csr_matrix:
        """고정 스키마로 행렬을 만들고 target-aware OOF 사용 여부를 제어합니다."""
        raw = self._build_raw_matrix(frame)
        return raw

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series | Sequence[str]):
        self.fit(features, labels)
        frame = self._validate_transform_frame(features)
        scan = None
        self._log('학습 행렬의 target-aware 블록은 내부 OOF 통계만 사용합니다.')
        return self._build_matrix(frame, scan=scan, use_fit_oof=True)

    def get_feature_names_out(self) -> np.ndarray:
        self._require_fitted()
        return self.feature_names_out_.copy()

    def summary(self) -> dict[str, object]:
        self._require_fitted()
        result: dict[str, object] = {'pipeline_name': self.pipeline_name, 'feature_blocks': list(self.feature_blocks), 'remaining_features': len(self.feature_names_out_), 'raw_features': len(self.gene_columns_), 'includes_raw_ordinal': True, 'f0_features': len(self.active_gene_columns_), 'f1_features': 0, 'f2_features': len(self.f2_feature_names_), 'f3_position_features': len(self.f3_position_feature_names_), 'f3_aa_features': len(self.f3_aa_feature_names_), 'f4_exact_hotspot_features': len(self.f4_feature_names_), 'f5_signature_features': len(self.f5_feature_names_), 'f5_selected_gene_union': len(self.f5_selected_gene_indices_), 'f7_pair_count': 0, 'f7_pair_contrast_features': len(self.f7_feature_names_), 'f7_selected_gene_union': len(self.f7_selected_gene_indices_), 'f5_missing_policy': 'legacy_treat_as_wt', 'dropped_constant_features': len(self.dropped_constant_columns), 'burden_clip_value': self.burden_clip_value_}
        f6_model = getattr(self, 'f6_compression_model_', None)
        if f6_model is not None:
            result.update({'f6_min_document_frequency': f6_model.minimum_document_frequency, 'f6_selected_token_features': len(f6_model.selected_token_indices), 'f6_requested_components': f6_model.requested_components, 'f6_actual_components': f6_model.actual_components, 'f6_explained_variance_ratio_sum': f6_model.explained_variance_ratio_sum})
        return result

    def get_diagnostics(self) -> dict[str, object]:
        """검증 코드가 private 속성에 의존하지 않도록 학습 상태를 복사해 반환합니다."""
        self._require_fitted()
        f5_model = getattr(self, 'f5_signature_model_', None)
        return {'pipeline_name': self.pipeline_name, 'capabilities': sorted(self.capabilities), 'summary': self.summary(), 'f5_signature_model': f5_model, 'f5_selected_gene_indices': self.f5_selected_gene_indices_.copy(), 'f5_output_column_indices': self.f5_output_column_indices_.copy(), 'f5_fit_oof_matrix': self._fit_f5_oof_matrix_, 'f7_pair_contrast_model': getattr(self, 'f7_pair_contrast_model_', None), 'f7_selected_gene_indices': self.f7_selected_gene_indices_.copy(), 'f7_selected_gene_catalog': [dict(item) for item in self.f7_selected_gene_catalog_], 'f7_fit_oof_matrix': self._fit_f7_oof_matrix_}

    @property
    def capabilities(self) -> frozenset[str]:
        """이 파이프라인이 제공하는 고정 피처 블록 집합입니다."""
        return frozenset(self.feature_blocks)

    def encode_labels(self, labels: pd.Series | Sequence[str]) -> np.ndarray:
        self._require_fitted()
        return self.label_encoder.transform(np.asarray(labels))

    def decode_labels(self, labels: Sequence[int]) -> np.ndarray:
        self._require_fitted()
        return self.label_encoder.inverse_transform(np.asarray(labels, dtype=int))

    @staticmethod
    def _prepare_raw(features: pd.DataFrame) -> pd.DataFrame:
        """도메인 가공 없이 결측만 명시한 exact-cell 범주를 보존합니다."""
        return features.fillna(RAW_MISSING_TOKEN).astype(str)

    def _build_raw_matrix(self, features: pd.DataFrame) -> sparse.csr_matrix:
        """exact 원문 범주를 유지하되 WT가 0이 되도록 이동해 희소 행렬로 만듭니다."""
        encoded = self.raw_encoder.transform(self._prepare_raw(features)).astype(np.float32, copy=False)
        encoded -= self.raw_wt_codes_[None, :]
        return sparse.csr_matrix(encoded, dtype=np.float32)

    @staticmethod
    def _validate_frame(features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError('features는 pandas DataFrame이어야 합니다.')
        if features.columns.has_duplicates:
            raise ValueError('중복된 유전자 열 이름이 있습니다.')
        if not features.index.is_unique:
            raise ValueError('features 인덱스는 고유해야 합니다.')
        return features

    @staticmethod
    def _validate_labels(frame: pd.DataFrame, labels: pd.Series | Sequence[str]) -> np.ndarray:
        if isinstance(labels, pd.Series):
            if not labels.index.is_unique:
                raise ValueError('labels 인덱스는 고유해야 합니다.')
            if not frame.index.equals(labels.index):
                raise ValueError('features와 labels 인덱스 및 순서가 일치해야 합니다.')
            values = labels.to_numpy()
        else:
            if isinstance(labels, (str, bytes)):
                raise TypeError('labels는 문자열 하나가 아니라 1차원 시퀀스여야 합니다.')
            values = np.asarray(labels)
        if values.ndim != 1:
            raise ValueError('labels는 1차원이어야 합니다.')
        if len(values) != len(frame):
            raise ValueError('features와 labels의 행 수가 일치하지 않습니다.')
        if pd.isna(values).any():
            raise ValueError('labels에 결측값이 있습니다.')
        return values

    def _validate_transform_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        frame = self._validate_frame(features)
        expected = list(self.gene_columns_)
        actual = list(frame.columns)
        expected_set = set(expected)
        actual_set = set(actual)
        missing = [column for column in expected if column not in actual_set]
        unexpected = [column for column in actual if column not in expected_set]
        if missing or unexpected:
            details = []
            if missing:
                details.append(f'누락={missing}')
            if unexpected:
                details.append(f'예상 밖={unexpected}')
            raise ValueError('유전자 피처 스키마가 다릅니다: ' + ', '.join(details))
        if actual != expected:
            raise ValueError('유전자 피처 열 순서가 학습 시점과 다릅니다.')
        return frame

    def _require_fitted(self) -> None:
        if not hasattr(self, 'feature_names_out_'):
            raise RuntimeError('fit을 먼저 실행해야 합니다.')

    def __getstate__(self) -> dict[str, object]:
        """학습 DataFrame 참조를 artifact에 중복 저장하지 않습니다."""
        state = self.__dict__.copy()
        state['_fit_frame_reference_'] = None
        return state

    def _log(self, message: str) -> None:
        if self.show_progress:
            print(f'[{self.pipeline_name}] {message}')
