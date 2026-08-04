"""Exact group-safe F9 plus the weighted-only EM24 W52 block."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_jyp_f9 import (
    F9GlobalAAPairNoRawPreprocessingPipeline,
    parse_cell,
)
from src.pipelines.pipeline_em_v24 import (
    EMV24PreprocessingPipeline,
    split_unique_mutations,
)


EM24_SIGNATURE_PREFIXES = ("signature_all_", "signature_functional_")
EM24_SIGNATURE_CHANNELS = ("all", "functional")
EM24_SIGNATURE_SUFFIXES = ("weighted", "match_count")
EXPECTED_CLASS_COUNT = 26
EXPECTED_EM24_SIGNATURE_FEATURES = 104
EXPECTED_EM24_WEIGHTED_FEATURES = 52
PROFILE_GROUP_COLUMN_CHUNK_SIZE = 256


def build_exact_profile_groups(features: pd.DataFrame) -> np.ndarray:
    """Build groups only when both F9 and EM24 see the same mutation profile."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if features.columns.has_duplicates:
        raise ValueError("Exact-profile groups require unique feature columns.")

    profile_tokens: list[list[str]] = [[] for _ in range(len(features))]
    columns = features.columns
    for start in range(0, len(columns), PROFILE_GROUP_COLUMN_CHUNK_SIZE):
        chunk_columns = columns[start : start + PROFILE_GROUP_COLUMN_CHUNK_SIZE]
        values = features.loc[:, chunk_columns].to_numpy(dtype=object, copy=False)
        missing_mask = pd.isna(values)
        wt_mask = np.zeros(values.shape, dtype=bool)
        nonmissing_mask = ~missing_mask
        wt_mask[nonmissing_mask] = values[nonmissing_mask] == "WT"
        candidate_rows, candidate_columns = np.nonzero(~(missing_mask | wt_mask))
        gene_names = chunk_columns.astype(str).to_numpy(dtype=object)

        for row_index, column_index in zip(candidate_rows, candidate_columns):
            value = str(values[row_index, column_index]).strip().upper()
            f9_events = parse_cell(value).events
            em24_events = tuple(
                token for token, _ in split_unique_mutations(value)
            )
            if f9_events or em24_events:
                profile_tokens[row_index].append(
                    repr((gene_names[column_index], f9_events, em24_events))
                )

    return np.asarray(
        [
            hashlib.sha256(
                ("|".join(sorted(tokens)) if tokens else "ALL_WT").encode("utf-8")
            ).hexdigest()
            for tokens in profile_tokens
        ],
        dtype=object,
    )


def build_em24_signature_columns(classes: Sequence[object]) -> tuple[str, ...]:
    """Return EM24's canonical channel-major dual-signature schema."""
    class_names = tuple(str(class_name) for class_name in classes)
    if len(class_names) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"pipeComb_v4 requires exactly {EXPECTED_CLASS_COUNT} fitted classes."
        )
    if any(not class_name for class_name in class_names) or len(set(class_names)) != len(
        class_names
    ):
        raise ValueError("Fitted label classes must be non-empty and unique.")

    columns = tuple(
        f"signature_{channel}_{class_name}_{suffix}"
        for channel in EM24_SIGNATURE_CHANNELS
        for class_name in class_names
        for suffix in EM24_SIGNATURE_SUFFIXES
    )
    if len(columns) != EXPECTED_EM24_SIGNATURE_FEATURES:
        raise RuntimeError(
            "Canonical EM24 dual-signature schema must contain exactly "
            f"{EXPECTED_EM24_SIGNATURE_FEATURES} features."
        )
    return columns


def build_em24_weighted_signature_columns(
    classes: Sequence[object],
) -> tuple[str, ...]:
    """Return the exact channel-major EM24 weighted-only W52 schema."""
    weighted_columns = tuple(
        name
        for name in build_em24_signature_columns(classes)
        if name.endswith("_weighted")
    )
    if len(weighted_columns) != EXPECTED_EM24_WEIGHTED_FEATURES:
        raise RuntimeError(
            "Canonical EM24 weighted-only schema must contain exactly "
            f"{EXPECTED_EM24_WEIGHTED_FEATURES} features."
        )
    return weighted_columns


class PipeCombV4PreprocessingPipeline(PreprocessingPipeline):
    """Combine exact group-safe F9 with EM24's weighted-only W52 signatures."""

    name = "pipeComb_v4"
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
        em24_group_weight_power: float = 0.0,
        auto_profile_groups: bool = True,
        show_progress: bool = True,
        progress_interval: int = 25000,
    ) -> None:
        super().__init__()
        self.pipeline_name = self.name
        self.feature_blocks = (
            "jyp_f9_exact",
            "em24_signature_all_weighted",
            "em24_signature_functional_weighted",
        )
        self.steps = (
            "build exact group-safe F9 matrix",
            "build and validate group-safe EM24 dual signatures",
            "append only the channel-major weighted W52 signatures",
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
            "group_weight_power": em24_group_weight_power,
        }
        self.auto_profile_groups = bool(auto_profile_groups)
        self._reset_components()

    def _reset_components(self) -> None:
        self.f9_pipeline_ = F9GlobalAAPairNoRawPreprocessingPipeline(
            **self.f9_parameters
        )
        self.em24_pipeline_ = EMV24PreprocessingPipeline(**self.em24_parameters)
        self.f9_feature_names_: tuple[str, ...] = ()
        self.em24_signature_columns_: tuple[str, ...] = ()
        self.em24_weighted_signature_columns_: tuple[str, ...] = ()
        self.output_column_indices_: np.ndarray | None = None
        self.feature_names_out_: np.ndarray | None = None
        self.profile_group_source_ = "none"

    @staticmethod
    def _normalize_labels(
        features: pd.DataFrame,
        labels: pd.Series | Sequence[str],
    ) -> pd.Series:
        if isinstance(labels, pd.Series):
            return labels
        if isinstance(labels, (str, bytes)):
            raise TypeError("labels must be a one-dimensional sequence, not a string.")
        return pd.Series(labels, index=features.index)

    def _resolve_groups(
        self,
        features: pd.DataFrame,
        groups: Sequence[object] | np.ndarray | pd.Series | None,
    ) -> Sequence[object] | np.ndarray | pd.Series | None:
        if groups is not None:
            if isinstance(groups, pd.Series):
                if not features.index.is_unique or not groups.index.is_unique:
                    raise ValueError(
                        "features and groups indexes must be unique for alignment."
                    )
                if not groups.index.equals(features.index):
                    raise ValueError(
                        "groups Series index and order must exactly match features."
                    )
            self.profile_group_source_ = "provided"
            return groups
        if not self.auto_profile_groups:
            self.profile_group_source_ = "disabled"
            return None
        self.profile_group_source_ = "derived"
        return build_exact_profile_groups(features)

    @staticmethod
    def _validate_em24_index(
        features: pd.DataFrame,
        em24_frame: pd.DataFrame,
    ) -> None:
        if not isinstance(em24_frame, pd.DataFrame):
            raise TypeError("EM24 preprocessing output must be a pandas DataFrame.")
        if not em24_frame.index.equals(features.index):
            raise RuntimeError("EM24 output index or row order differs from the input.")

    def _finalize_schema(self, em24_frame: pd.DataFrame) -> None:
        f9_classes = np.asarray(self.f9_pipeline_.label_encoder.classes_, dtype=str)
        em24_classes = np.asarray(self.em24_pipeline_.label_encoder.classes_, dtype=str)
        if not np.array_equal(f9_classes, em24_classes):
            raise RuntimeError("F9 and EM24 label-class order differs.")
        if em24_frame.columns.has_duplicates:
            raise RuntimeError("EM24 output schema contains duplicate feature names.")

        emitted_columns = em24_frame.columns.astype(str).tolist()
        signature_columns = tuple(
            name for name in emitted_columns if name.startswith(EM24_SIGNATURE_PREFIXES)
        )
        expected_signature_columns = build_em24_signature_columns(f9_classes)
        if signature_columns != expected_signature_columns:
            raise RuntimeError(
                "EM24 dual-signature columns differ from the canonical fitted-class "
                "schema or order."
            )

        f9_names_array = np.asarray(
            self.f9_pipeline_.get_feature_names_out(), dtype=object
        )
        if f9_names_array.ndim != 1 or not len(f9_names_array):
            raise RuntimeError("F9 feature schema must be a non-empty vector.")
        if any(not isinstance(name, str) or not name for name in f9_names_array):
            raise RuntimeError("F9 feature names must be non-empty strings.")

        f9_names = tuple(str(name) for name in f9_names_array)
        weighted_columns = build_em24_weighted_signature_columns(f9_classes)
        full_output_names = (
            *f9_names,
            *(f"EM24__{name}" for name in signature_columns),
        )
        output_names = (
            *f9_names,
            *(f"EM24__{name}" for name in weighted_columns),
        )
        if len(output_names) != len(f9_names) + EXPECTED_EM24_WEIGHTED_FEATURES:
            raise RuntimeError("pipeComb_v4 must append exactly 52 EM24 weighted features.")
        if any(not name for name in output_names) or len(set(output_names)) != len(
            output_names
        ):
            raise RuntimeError("pipeComb_v4 feature schema must be non-empty and unique.")

        full_name_to_index = {
            name: index for index, name in enumerate(full_output_names)
        }
        output_indices = np.asarray(
            [full_name_to_index[name] for name in output_names], dtype=np.int64
        )

        self.f9_feature_names_ = f9_names
        self.em24_signature_columns_ = signature_columns
        self.em24_weighted_signature_columns_ = weighted_columns
        self.output_column_indices_ = output_indices
        self.feature_names_out_ = np.asarray(output_names, dtype=object)
        self.label_encoder = self.f9_pipeline_.label_encoder

    def _combine(
        self,
        f9_matrix: object,
        em24_frame: pd.DataFrame,
    ) -> sparse.csr_matrix:
        self._require_fitted()
        missing = [
            name
            for name in self.em24_weighted_signature_columns_
            if name not in em24_frame
        ]
        if missing:
            raise RuntimeError(f"EM24 weighted signature columns are missing: {missing}")

        f9_csr = sparse.csr_matrix(f9_matrix, dtype=np.float32)
        if f9_csr.shape[1] != len(self.f9_feature_names_):
            raise RuntimeError("F9 matrix width differs from the fitted F9 schema.")
        if f9_csr.shape[0] != len(em24_frame):
            raise RuntimeError("F9 and EM24 row counts differ.")
        if not np.isfinite(f9_csr.data).all():
            raise RuntimeError("F9 output contains NaN or infinity.")

        signature_values = em24_frame.loc[
            :, list(self.em24_weighted_signature_columns_)
        ].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(signature_values).all():
            raise RuntimeError("EM24 weighted signatures contain NaN or infinity.")

        output = sparse.hstack(
            (f9_csr, sparse.csr_matrix(signature_values)),
            format="csr",
            dtype=np.float32,
        )
        output.sum_duplicates()
        output.eliminate_zeros()
        output.sort_indices()
        if output.shape[1] != len(self.feature_names_out_):
            raise RuntimeError("pipeComb_v4 matrix width differs from its fitted schema.")
        if not output.has_canonical_format or not np.isfinite(output.data).all():
            raise RuntimeError("pipeComb_v4 output must be canonical and finite.")
        return output

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series | Sequence[str],
        *,
        groups: Sequence[object] | np.ndarray | pd.Series | None = None,
    ) -> "PipeCombV4PreprocessingPipeline":
        normalized_labels = self._normalize_labels(features, labels)
        self._reset_components()
        resolved_groups = self._resolve_groups(features, groups)
        self.f9_pipeline_.fit(features, normalized_labels, groups=resolved_groups)
        self.em24_pipeline_.fit(
            features, normalized_labels, groups=resolved_groups
        )
        em24_frame = self.em24_pipeline_.transform(features)
        self._validate_em24_index(features, em24_frame)
        self._finalize_schema(em24_frame)
        return self

    def fit_transform(
        self,
        features: pd.DataFrame,
        labels: pd.Series | Sequence[str],
        *,
        groups: Sequence[object] | np.ndarray | pd.Series | None = None,
    ) -> sparse.csr_matrix:
        normalized_labels = self._normalize_labels(features, labels)
        self._reset_components()
        resolved_groups = self._resolve_groups(features, groups)
        f9_matrix = self.f9_pipeline_.fit_transform(
            features, normalized_labels, groups=resolved_groups
        )
        em24_frame = self.em24_pipeline_.fit_transform(
            features, normalized_labels, groups=resolved_groups
        )
        self._validate_em24_index(features, em24_frame)
        self._finalize_schema(em24_frame)
        return self._combine(f9_matrix, em24_frame)

    def transform(self, features: pd.DataFrame) -> sparse.csr_matrix:
        self._require_fitted()
        f9_matrix = self.f9_pipeline_.transform(features)
        em24_frame = self.em24_pipeline_.transform(features)
        self._validate_em24_index(features, em24_frame)
        return self._combine(f9_matrix, em24_frame)

    def get_feature_names_out(self) -> np.ndarray:
        self._require_fitted()
        return self.feature_names_out_.copy()

    @staticmethod
    def _diagnostics(component: object) -> dict[str, object]:
        getter = getattr(component, "get_diagnostics", None)
        if callable(getter):
            return getter()
        return component.summary()

    def summary(self) -> dict[str, object]:
        self._require_fitted()
        f9_summary = self.f9_pipeline_.summary()
        em24_summary = self.em24_pipeline_.summary()
        f9_group_safe = bool(f9_summary.get("f7_oof_group_safe", False))
        em24_group_safe = bool(em24_summary.get("signature_oof_group_safe", False))
        return {
            "pipeline_name": self.pipeline_name,
            "feature_blocks": list(self.feature_blocks),
            "remaining_features": len(self.feature_names_out_),
            "raw_features": 0,
            "includes_raw_ordinal": False,
            "f9_features": len(self.f9_feature_names_),
            "f9_prefix_features": len(self.f9_feature_names_),
            "em24_weighted_signature_features": len(
                self.em24_weighted_signature_columns_
            ),
            "em24_signature_all_weighted_features": EXPECTED_CLASS_COUNT,
            "em24_signature_functional_weighted_features": EXPECTED_CLASS_COUNT,
            "em24_match_count_features": 0,
            "class_count": len(self.label_encoder.classes_),
            "auto_profile_groups": self.auto_profile_groups,
            "profile_group_source": self.profile_group_source_,
            "f9_oof_group_safe": f9_group_safe,
            "f9_oof_group_count": f9_summary.get("f7_oof_group_count"),
            "f9_oof_fold_count": f9_summary.get("f7_oof_fold_count", 0),
            "em24_oof_group_safe": em24_group_safe,
            "em24_oof_group_count": em24_summary.get("signature_oof_group_count"),
            "em24_oof_fold_count": em24_summary.get("signature_oof_fold_count", 0),
            "em24_group_weight_power": em24_summary.get("group_weight_power", 0.0),
            "em24_group_count": em24_summary.get("group_count"),
            "em24_group_size_stats": em24_summary.get("group_size_stats"),
            "oof_group_safe": f9_group_safe and em24_group_safe,
            "f9_summary": f9_summary,
            "em24_summary": em24_summary,
        }

    def get_diagnostics(self) -> dict[str, object]:
        self._require_fitted()
        f9_diagnostics = self._diagnostics(self.f9_pipeline_)
        em24_diagnostics = self._diagnostics(self.em24_pipeline_)
        return {
            "pipeline_name": self.pipeline_name,
            "summary": self.summary(),
            "f9": f9_diagnostics,
            "em24": em24_diagnostics,
            "f9_oof_fold_audits": f9_diagnostics.get("f7_oof_fold_audits", []),
            "em24_oof_fold_audits": em24_diagnostics.get(
                "signature_oof_fold_audits", []
            ),
            "f9_feature_names": list(self.f9_feature_names_),
            "em24_weighted_signature_columns": list(
                self.em24_weighted_signature_columns_
            ),
            "output_column_indices": self.output_column_indices_.copy(),
        }

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(self.feature_blocks)

    def encode_labels(self, labels: pd.Series | Sequence[str]) -> np.ndarray:
        self._require_fitted()
        return self.label_encoder.transform(np.asarray(labels))

    def decode_labels(self, labels: Sequence[int]) -> np.ndarray:
        self._require_fitted()
        return self.label_encoder.inverse_transform(np.asarray(labels, dtype=int))

    def _require_fitted(self) -> None:
        if self.feature_names_out_ is None:
            raise RuntimeError("fit must be called first.")
