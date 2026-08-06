"""JYP F9과 자체 dual-signature 블록을 결합한 pipeComb_v3 파이프라인입니다."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_jyp_f9 import (
    F9GlobalAAPairNoRawPreprocessingPipeline,
)


_SYNONYMOUS = 1
_MISSENSE = 2
_INFRAME = 3
_NONSENSE = 4
_FRAMESHIFT = 5
_SYNONYMOUS_PATTERN = re.compile(r"^([A-Z])(\d+)\1$")
_MISSENSE_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")
_FRAMESHIFT_PATTERN = re.compile(r"FS", re.IGNORECASE)
_STOP_PATTERN = re.compile(r"(?:\*|TER$|X$)", re.IGNORECASE)
_INFRAME_PATTERN = re.compile(r"(?:DEL|INS|DUP|>|_)", re.IGNORECASE)
_SIGNATURE_COLUMN_CHUNK_SIZE = 256


@lru_cache(maxsize=None)
def _classify_mutation_token(token: str) -> int:
    normalized = token.strip().upper()
    if not normalized or normalized in {"WT", "<NA>"}:
        return 0
    if _SYNONYMOUS_PATTERN.fullmatch(normalized):
        return _SYNONYMOUS
    if _FRAMESHIFT_PATTERN.search(normalized):
        return _FRAMESHIFT
    if _STOP_PATTERN.search(normalized):
        return _NONSENSE
    if _INFRAME_PATTERN.search(normalized):
        return _INFRAME
    match = _MISSENSE_PATTERN.fullmatch(normalized)
    if match and match.group(1) != match.group(3):
        return _MISSENSE
    return _MISSENSE


@lru_cache(maxsize=None)
def _split_unique_mutations(value: str) -> tuple[tuple[str, int], ...]:
    normalized = value.strip().upper()
    if not normalized or normalized in {"WT", "<NA>"}:
        return ()
    return tuple(
        (token, _classify_mutation_token(token))
        for token in sorted(set(normalized.split()))
        if _classify_mutation_token(token) != 0
    )


def _build_signature_matrices(
    features: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"변환에 필요한 유전자 컬럼이 없습니다: {sorted(missing)}")
    mutation_values = np.zeros((len(features), len(columns)), dtype="int8")
    functional_values = np.zeros_like(mutation_values)
    for start in range(0, len(columns), _SIGNATURE_COLUMN_CHUNK_SIZE):
        stop = min(start + _SIGNATURE_COLUMN_CHUNK_SIZE, len(columns))
        chunk = features.loc[:, columns[start:stop]].to_numpy(
            dtype=object,
            copy=False,
        )
        missing = pd.isna(chunk)
        wild_type = np.zeros(chunk.shape, dtype=bool)
        present = ~missing
        wild_type[present] = chunk[present] == "WT"
        candidate_rows, candidate_columns = np.nonzero(~(missing | wild_type))
        for row_position, chunk_column_position in zip(
            candidate_rows,
            candidate_columns,
            strict=True,
        ):
            normalized = str(chunk[row_position, chunk_column_position]).strip().upper()
            mutations = _split_unique_mutations(normalized)
            if not mutations:
                continue
            column_position = start + chunk_column_position
            mutation_values[row_position, column_position] = 1
            functional_values[row_position, column_position] = int(
                any(consequence >= _MISSENSE for _, consequence in mutations)
            )
    frame_arguments = {"index": features.index, "columns": columns, "dtype": "int8"}
    return (
        pd.DataFrame(mutation_values, **frame_arguments),
        pd.DataFrame(functional_values, **frame_arguments),
    )


def _learn_class_weights(
    matrix: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
    signal_name: str,
) -> dict[str, dict[str, float]]:
    aligned = labels.reindex(matrix.index).astype("string")
    if aligned.isna().any():
        raise ValueError("labels가 모든 signature 입력 행을 포함해야 합니다.")
    matrix_values = matrix.to_numpy(dtype=np.int8, copy=False)
    gene_names = matrix.columns.astype(str).to_numpy(dtype=object)
    total = matrix_values.sum(axis=0, dtype=np.float64)
    weights_by_class: dict[str, dict[str, float]] = {}
    for raw_class_name in sorted(aligned.unique()):
        class_name = str(raw_class_name)
        in_class = aligned.eq(class_name).fillna(False).to_numpy(dtype=bool)
        class_count = int(np.count_nonzero(in_class))
        other_count = len(aligned) - class_count
        positive = matrix_values[in_class].sum(axis=0, dtype=np.float64)
        negative = total - positive
        class_odds = (positive + smoothing) / (class_count - positive + smoothing)
        other_odds = (negative + smoothing) / (other_count - negative + smoothing)
        score = np.clip(np.log2(class_odds / other_odds), 0.0, max_log2_odds)
        support = positive + negative
        score *= np.sqrt(support / (support + shrinkage))
        selected_positions = np.argsort(-score, kind="stable")[:top_genes_per_class]
        selected_positions = selected_positions[score[selected_positions] > 0]
        if not len(selected_positions):
            raise ValueError(f"{class_name}의 {signal_name} signature를 만들 수 없습니다.")
        weights_by_class[class_name] = {
            str(gene_names[position]): float(score[position])
            for position in selected_positions
        }
    return weights_by_class


class PipeCombV3PreprocessingPipeline(PreprocessingPipeline):
    """F9 뒤에 독립 dual signature만 추가합니다(원본 26클래스는 104개).

    EMV24 파이프라인을 하위 파이프라인으로 실행하지 않습니다. 학습 행에는
    F9 pair contrast와 이 파일이 직접 만드는 암종 signature를 각각 내부 OOF
    통계로 생성하고, 검증·테스트 행에는 학습 데이터의 full-fit 통계를 적용합니다.
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
        self._validate_signature_parameters()
        self._reset_components()

    def _validate_signature_parameters(self) -> None:
        integer_values = {
            "min_mutation_count": (self.em24_parameters["min_mutation_count"], 1),
            "min_functional_mutation_count": (
                self.em24_parameters["min_functional_mutation_count"],
                1,
            ),
            "top_genes_per_class": (self.em24_parameters["top_genes_per_class"], 1),
            "min_hotspot_count": (self.em24_parameters["min_hotspot_count"], 1),
            "max_hotspots": (self.em24_parameters["max_hotspots"], 1),
            "inner_signature_folds": (
                self.em24_parameters["inner_signature_folds"],
                2,
            ),
        }
        for name, (value, minimum) in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"em24_{name}는 {minimum} 이상의 정수여야 합니다.")
        for name in ("smoothing", "max_log2_odds", "shrinkage"):
            if float(self.em24_parameters[name]) <= 0:
                raise ValueError(f"em24_{name}는 0보다 커야 합니다.")

    def _reset_components(self) -> None:
        self.f9_pipeline_ = F9GlobalAAPairNoRawPreprocessingPipeline(
            **self.f9_parameters
        )
        self.feature_names_out_: np.ndarray | None = None
        self.em24_signature_columns_: tuple[str, ...] = ()
        self.selected_gene_columns_: list[str] = []
        self.dropped_gene_columns_: list[str] = []
        self.mutation_signature_genes_: list[str] = []
        self.functional_signature_genes_: list[str] = []
        self.all_variant_weights_: dict[str, dict[str, float]] = {}
        self.functional_weights_: dict[str, dict[str, float]] = {}
        self.signature_oof_fold_count_ = 0

    @staticmethod
    def _normalize_labels(
        features: pd.DataFrame,
        labels: pd.Series | Sequence[str],
    ) -> pd.Series:
        if isinstance(labels, pd.Series):
            normalized = labels.reindex(features.index)
            if normalized.isna().any():
                raise ValueError("labels가 모든 입력 행을 포함해야 합니다.")
            return normalized
        if isinstance(labels, (str, bytes)):
            raise TypeError("labels는 문자열 하나가 아니라 1차원 시퀀스여야 합니다.")
        return pd.Series(labels, index=features.index)

    @staticmethod
    def _signature_columns(classes: Sequence[object]) -> tuple[str, ...]:
        class_names = tuple(str(class_name) for class_name in classes)
        return tuple(
            f"signature_{channel}_{class_name}_{suffix}"
            for channel in ("all", "functional")
            for class_name in class_names
            for suffix in ("weighted", "match_count")
        )

    @staticmethod
    def _validate_weight_classes(
        weights: dict[str, dict[str, float]],
        classes: tuple[str, ...],
        *,
        context: str,
    ) -> None:
        if tuple(weights) != classes:
            raise RuntimeError(
                f"{context}의 암종 순서가 전체 학습 클래스와 다릅니다: "
                f"expected={classes}, actual={tuple(weights)}"
            )

    @staticmethod
    def _channel_values(
        matrix: pd.DataFrame,
        weights_by_class: dict[str, dict[str, float]],
    ) -> list[np.ndarray]:
        columns: list[np.ndarray] = []
        matrix_values = matrix.to_numpy(dtype=np.float32, copy=False)
        column_positions = {
            str(name): position for position, name in enumerate(matrix.columns)
        }
        for weights in weights_by_class.values():
            genes = list(weights)
            positions = np.fromiter(
                (column_positions[gene] for gene in genes),
                dtype=np.intp,
                count=len(genes),
            )
            values = matrix_values[:, positions]
            vector = np.asarray(
                [weights[gene] for gene in genes],
                dtype=np.float32,
            )
            columns.extend((values @ vector, values.sum(axis=1)))
        return columns

    def _build_signature_values(
        self,
        mutation: pd.DataFrame,
        functional: pd.DataFrame,
        all_weights: dict[str, dict[str, float]],
        functional_weights: dict[str, dict[str, float]],
    ) -> np.ndarray:
        classes = tuple(all_weights)
        self._validate_weight_classes(
            functional_weights,
            classes,
            context="functional signature",
        )
        columns = self._signature_columns(classes)
        values = [
            *self._channel_values(mutation, all_weights),
            *self._channel_values(functional, functional_weights),
        ]
        if len(values) != len(columns):
            raise RuntimeError("dual signature 값과 스키마의 열 수가 다릅니다.")
        return np.column_stack(values).astype(np.float32, copy=False)

    def _build_signature_frame(
        self,
        mutation: pd.DataFrame,
        functional: pd.DataFrame,
        all_weights: dict[str, dict[str, float]],
        functional_weights: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        classes = tuple(all_weights)
        values = self._build_signature_values(
            mutation,
            functional,
            all_weights,
            functional_weights,
        )
        return pd.DataFrame(
            values,
            index=mutation.index,
            columns=self._signature_columns(classes),
            dtype="float32",
        )

    def _fit_signature_state(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        columns = features.columns.tolist()
        mutation, functional = _build_signature_matrices(features, columns)
        mutation_counts = mutation.to_numpy(dtype=np.int8, copy=False).sum(axis=0)
        functional_counts = functional.to_numpy(dtype=np.int8, copy=False).sum(axis=0)
        mutation_support = dict(zip(columns, mutation_counts, strict=True))
        functional_support = dict(zip(columns, functional_counts, strict=True))
        self.mutation_signature_genes_ = [
            gene
            for gene in columns
            if mutation_support[gene] >= self.em24_parameters["min_mutation_count"]
        ]
        self.functional_signature_genes_ = [
            gene
            for gene in columns
            if functional_support[gene]
            >= self.em24_parameters["min_functional_mutation_count"]
        ]
        selected = set(self.mutation_signature_genes_) | set(
            self.functional_signature_genes_
        )
        self.selected_gene_columns_ = [gene for gene in columns if gene in selected]
        self.dropped_gene_columns_ = [gene for gene in columns if gene not in selected]
        if not self.mutation_signature_genes_:
            raise ValueError("최소 전체 변이 빈도를 만족하는 유전자가 없습니다.")
        if not self.functional_signature_genes_:
            raise ValueError("최소 기능 변이 빈도를 만족하는 유전자가 없습니다.")

        mutation_channel = mutation.loc[:, self.mutation_signature_genes_]
        functional_channel = functional.loc[:, self.functional_signature_genes_]
        classes = tuple(str(name) for name in sorted(labels.unique()))
        parameters = self.em24_parameters
        self.all_variant_weights_ = _learn_class_weights(
            mutation_channel,
            labels,
            parameters["top_genes_per_class"],
            parameters["smoothing"],
            parameters["max_log2_odds"],
            parameters["shrinkage"],
            "전체 변이",
        )
        self.functional_weights_ = _learn_class_weights(
            functional_channel,
            labels,
            parameters["top_genes_per_class"],
            parameters["smoothing"],
            parameters["max_log2_odds"],
            parameters["shrinkage"],
            "기능 변이",
        )
        self._validate_weight_classes(
            self.all_variant_weights_,
            classes,
            context="full-fit all signature",
        )
        self._validate_weight_classes(
            self.functional_weights_,
            classes,
            context="full-fit functional signature",
        )
        return mutation_channel, functional_channel

    def _create_oof_signatures(
        self,
        mutation: pd.DataFrame,
        functional: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        classes = tuple(str(name) for name in sorted(labels.unique()))
        parameters = self.em24_parameters
        n_splits = min(
            parameters["inner_signature_folds"],
            int(labels.value_counts().min()),
        )
        if n_splits < 2:
            raise ValueError("OOF signature에는 클래스별 표본이 최소 2개 필요합니다.")
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=parameters["signature_random_state"],
        )
        output_columns = self._signature_columns(classes)
        output_values = np.zeros(
            (len(mutation), len(output_columns)),
            dtype=np.float32,
        )
        row_coverage = np.zeros(len(mutation), dtype=np.int8)
        for train_positions, valid_positions in splitter.split(mutation, labels):
            fold_labels = labels.iloc[train_positions]
            all_weights = _learn_class_weights(
                mutation.iloc[train_positions],
                fold_labels,
                parameters["top_genes_per_class"],
                parameters["smoothing"],
                parameters["max_log2_odds"],
                parameters["shrinkage"],
                "all",
            )
            functional_weights = _learn_class_weights(
                functional.iloc[train_positions],
                fold_labels,
                parameters["top_genes_per_class"],
                parameters["smoothing"],
                parameters["max_log2_odds"],
                parameters["shrinkage"],
                "functional",
            )
            self._validate_weight_classes(
                all_weights,
                classes,
                context="OOF all signature",
            )
            self._validate_weight_classes(
                functional_weights,
                classes,
                context="OOF functional signature",
            )
            fold_values = self._build_signature_values(
                mutation.iloc[valid_positions],
                functional.iloc[valid_positions],
                all_weights,
                functional_weights,
            )
            output_values[valid_positions] = fold_values
            row_coverage[valid_positions] += 1
        if not np.all(row_coverage == 1):
            raise RuntimeError("OOF signature의 각 행은 정확히 한 번 생성되어야 합니다.")
        self.signature_oof_fold_count_ = n_splits
        return pd.DataFrame(
            output_values,
            index=mutation.index,
            columns=output_columns,
            dtype="float32",
        )

    def _transform_signatures(self, features: pd.DataFrame) -> pd.DataFrame:
        mutation, functional = _build_signature_matrices(
            features,
            self.selected_gene_columns_,
        )
        return self._build_signature_frame(
            mutation.loc[:, self.mutation_signature_genes_],
            functional.loc[:, self.functional_signature_genes_],
            self.all_variant_weights_,
            self.functional_weights_,
        )

    def _finalize_schema(self, em24_frame: pd.DataFrame) -> None:
        f9_classes = np.asarray(self.f9_pipeline_.label_encoder.classes_, dtype=str)
        signature_classes = np.asarray(tuple(self.all_variant_weights_), dtype=str)
        if not np.array_equal(f9_classes, signature_classes):
            raise RuntimeError("F9과 dual signature의 원본 SUBCLASS 순서가 다릅니다.")

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
        self._fit_signature_state(features, normalized_labels)
        em24_frame = self._transform_signatures(features)
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
        mutation, functional = self._fit_signature_state(features, normalized_labels)
        em24_frame = self._create_oof_signatures(
            mutation,
            functional,
            normalized_labels,
        )
        self._validate_output_index(features, em24_frame)
        self._finalize_schema(em24_frame)
        return self._combine(f9_matrix, em24_frame)

    def transform(self, features: pd.DataFrame) -> sparse.csr_matrix:
        self._require_fitted()
        f9_matrix = self.f9_pipeline_.transform(features)
        em24_frame = self._transform_signatures(features)
        self._validate_output_index(features, em24_frame)
        return self._combine(f9_matrix, em24_frame)

    def get_feature_names_out(self) -> np.ndarray:
        self._require_fitted()
        return self.feature_names_out_.copy()

    def summary(self) -> dict[str, object]:
        self._require_fitted()
        f9_summary = self.f9_pipeline_.summary()
        em24_summary = {
            "selected_gene_features": len(self.selected_gene_columns_),
            "dropped_gene_features": len(self.dropped_gene_columns_),
            "all_variant_signature_genes": len(self.mutation_signature_genes_),
            "functional_signature_genes": len(self.functional_signature_genes_),
            "all_variant_signature_features": 2 * len(self.all_variant_weights_),
            "functional_signature_features": 2 * len(self.functional_weights_),
            "signature_oof_fold_count": self.signature_oof_fold_count_,
            "standalone_signature_block": True,
        }
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
            "em24": {
                "standalone_signature_block": True,
                "signature_oof_fold_count": self.signature_oof_fold_count_,
                "selected_gene_columns": list(self.selected_gene_columns_),
                "mutation_signature_genes": list(self.mutation_signature_genes_),
                "functional_signature_genes": list(
                    self.functional_signature_genes_
                ),
            },
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
