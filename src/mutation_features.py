"""암종 분류용 변이 문자열 전처리와 구조적 피처 생성을 담당합니다.

라벨을 사용하는 marker 통계는 ``fit``에 전달된 학습 데이터에서만
계산됩니다. 교차검증에서는 반드시 각 학습 fold마다 새 인스턴스를
생성하여 ``fit_transform``하고, 검증 fold에는 ``transform``만 적용합니다.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.mutation_preprocessing import (
    MUTATION_TYPES,
    NO_MUTATION_VALUES,
    POSITION_BINS,
    PROTEIN_PREFIX,
    mutation_position,
    mutation_type,
    normalize_variant,
    split_mutation_events,
)


def _mutated_mask(series: pd.Series) -> np.ndarray:
    normalized = series.fillna("WT").astype(str).str.strip().str.upper()
    return (~normalized.isin(NO_MUTATION_VALUES)).to_numpy(dtype=bool)


def _safe_feature_token(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")


class MutationFeatureEngineer(BaseEstimator, TransformerMixin):
    """변이 통계, marker 점수와 유전자×변이 유형 피처를 생성합니다.

    기본 선별 조건은 팀 분석 기준을 반영합니다.

    - 해당 암종 내부 변이율 10% 이상
    - 해당 암종 내부 변이 건수 5건 이상
    - 나머지 암종 대비 변이율 차이 5%p 이상
    - Lift 1.5 이상
    """

    def __init__(
        self,
        min_class_rate: float = 0.10,
        min_class_count: int = 5,
        min_rate_difference: float = 0.05,
        min_lift: float = 1.5,
        max_markers_per_class: int = 5,
        smoothing: float = 0.5,
        log2_or_clip: float = 8.0,
        include_gene_type_features: bool = True,
    ) -> None:
        self.min_class_rate = min_class_rate
        self.min_class_count = min_class_count
        self.min_rate_difference = min_rate_difference
        self.min_lift = min_lift
        self.max_markers_per_class = max_markers_per_class
        self.smoothing = smoothing
        self.log2_or_clip = log2_or_clip
        self.include_gene_type_features = include_gene_type_features

    def fit(self, features: pd.DataFrame, labels: Iterable[object]):
        frame = self._validate_frame(features)
        y = pd.Series(labels, index=frame.index, dtype="object").astype(str)
        if len(frame) != len(y):
            raise ValueError("features와 labels의 행 수가 다릅니다.")
        if y.isna().any():
            raise ValueError("labels에 결측값이 있습니다.")

        self.feature_columns_ = frame.columns.tolist()
        self.classes_ = np.asarray(sorted(y.unique()), dtype=object)
        class_to_index = {name: index for index, name in enumerate(self.classes_)}
        y_codes = y.map(class_to_index).to_numpy(dtype=np.int32)
        class_sizes = np.bincount(y_codes, minlength=len(self.classes_))
        total_rows = len(frame)
        candidates: list[dict[str, object]] = []

        for gene in self.feature_columns_:
            mutated = _mutated_mask(frame[gene])
            counts = np.bincount(y_codes[mutated], minlength=len(self.classes_))
            total_mutated = int(mutated.sum())

            for class_index, class_name in enumerate(self.classes_):
                count_in = int(counts[class_index])
                class_size = int(class_sizes[class_index])
                out_size = total_rows - class_size
                count_out = total_mutated - count_in
                rate_in = count_in / class_size if class_size else 0.0
                rate_out = count_out / out_size if out_size else 0.0
                rate_difference = rate_in - rate_out
                lift = (rate_in + 1e-12) / (rate_out + 1e-12)

                a = count_in + self.smoothing
                b = class_size - count_in + self.smoothing
                c = count_out + self.smoothing
                d = out_size - count_out + self.smoothing
                log2_or = float(np.log2((a * d) / (b * c)))
                clipped_log2_or = float(
                    np.clip(log2_or, -self.log2_or_clip, self.log2_or_clip)
                )
                rank_score = clipped_log2_or * float(np.log2(1 + count_in))

                if (
                    count_in >= self.min_class_count
                    and rate_in >= self.min_class_rate
                    and rate_difference >= self.min_rate_difference
                    and lift >= self.min_lift
                    and log2_or > 0
                ):
                    candidates.append(
                        {
                            "SUBCLASS": class_name,
                            "gene": gene,
                            "class_count": class_size,
                            "mutation_count": count_in,
                            "class_rate": rate_in,
                            "other_rate": rate_out,
                            "rate_difference": rate_difference,
                            "lift": lift,
                            "log2_or": log2_or,
                            "rank_score": rank_score,
                        }
                    )

        marker_table = pd.DataFrame(candidates)
        if marker_table.empty:
            marker_table = pd.DataFrame(
                columns=[
                    "SUBCLASS",
                    "gene",
                    "class_count",
                    "mutation_count",
                    "class_rate",
                    "other_rate",
                    "rate_difference",
                    "lift",
                    "log2_or",
                    "rank_score",
                ]
            )
        else:
            marker_table = (
                marker_table.sort_values(
                    ["SUBCLASS", "rank_score", "mutation_count", "gene"],
                    ascending=[True, False, False, True],
                )
                .groupby("SUBCLASS", sort=False, as_index=False)
                .head(self.max_markers_per_class)
                .reset_index(drop=True)
            )

        self.marker_table_ = marker_table
        self.selected_genes_ = sorted(marker_table["gene"].unique().tolist())
        selected_gene_types: list[tuple[str, str]] = []
        if self.include_gene_type_features:
            for gene in self.selected_genes_:
                observed_types = set()
                for value in frame.loc[_mutated_mask(frame[gene]), gene]:
                    observed_types.update(
                        mutation_type(event)
                        for event in split_mutation_events(value)
                    )
                selected_gene_types.extend(
                    (gene, kind)
                    for kind in MUTATION_TYPES
                    if kind in observed_types
                )
        self.selected_gene_types_ = selected_gene_types
        marker_map: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for row in marker_table.itertuples(index=False):
            marker_map[row.gene].append(
                (
                    str(row.SUBCLASS),
                    float(np.clip(row.log2_or, 0.0, self.log2_or_clip)),
                )
            )
        self.marker_map_ = dict(marker_map)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=[
                "feature_columns_",
                "classes_",
                "marker_table_",
                "selected_genes_",
                "selected_gene_types_",
                "marker_map_",
            ],
        )
        frame = self._validate_frame(features)
        missing = sorted(set(self.feature_columns_) - set(frame.columns))
        if missing:
            raise ValueError(f"필요한 유전자 컬럼이 없습니다: {missing[:10]}")
        frame = frame[self.feature_columns_]

        row_count = len(frame)
        gene_count = len(self.feature_columns_)
        mutation_count = np.zeros(row_count, dtype=np.int32)
        mutation_event_count = np.zeros(row_count, dtype=np.int32)
        missing_count = np.zeros(row_count, dtype=np.int32)
        protein_prefix_count = np.zeros(row_count, dtype=np.int32)
        parsed_position_count = np.zeros(row_count, dtype=np.int32)
        position_sum = np.zeros(row_count, dtype=np.float64)
        position_max = np.zeros(row_count, dtype=np.int32)
        type_counts = {
            name: np.zeros(row_count, dtype=np.int32) for name in MUTATION_TYPES
        }
        position_bin_counts = {
            name: np.zeros(row_count, dtype=np.int32)
            for name, _, _ in POSITION_BINS
        }
        marker_scores = {
            str(name): np.zeros(row_count, dtype=np.float64)
            for name in self.classes_
        }
        marker_hits = {
            str(name): np.zeros(row_count, dtype=np.int32)
            for name in self.classes_
        }
        selected_gene_set = set(self.selected_genes_)
        gene_mutation_features: dict[str, np.ndarray] = {}
        gene_type_features = {
            key: np.zeros(row_count, dtype=np.int8)
            for key in self.selected_gene_types_
        }

        for gene in self.feature_columns_:
            raw_values = frame[gene]
            missing_count += raw_values.isna().to_numpy(dtype=np.int32)
            mutated = _mutated_mask(raw_values)
            indices = np.flatnonzero(mutated)
            if not len(indices):
                if gene in selected_gene_set:
                    gene_mutation_features[gene] = mutated.astype(np.int8)
                continue

            mutation_count[indices] += 1
            values = raw_values.iloc[indices].tolist()
            if gene in selected_gene_set:
                gene_mutation_features[gene] = mutated.astype(np.int8)

            for class_name, weight in self.marker_map_.get(gene, ()):
                marker_scores[class_name][indices] += weight
                marker_hits[class_name][indices] += 1

            for row_index, raw_value in zip(indices, values):
                for event in split_mutation_events(raw_value):
                    mutation_event_count[row_index] += 1
                    if PROTEIN_PREFIX.match(event):
                        protein_prefix_count[row_index] += 1
                    kind = mutation_type(event)
                    type_counts[kind][row_index] += 1

                    if self.include_gene_type_features and gene in selected_gene_set:
                        key = (gene, kind)
                        if key in gene_type_features:
                            gene_type_features[key][row_index] = 1

                    position = mutation_position(event)
                    if position is None:
                        continue
                    parsed_position_count[row_index] += 1
                    position_sum[row_index] += position
                    position_max[row_index] = max(position_max[row_index], position)
                    for bin_name, lower, upper in POSITION_BINS:
                        if lower <= position <= upper:
                            position_bin_counts[bin_name][row_index] += 1
                            break

        safe_event_count = np.maximum(mutation_event_count, 1)
        output: dict[str, np.ndarray] = {
            "sample_mutation_count": mutation_count,
            "sample_mutation_event_count": mutation_event_count,
            "sample_missing_count": missing_count,
            "sample_wt_count": gene_count - mutation_count - missing_count,
            "sample_mutation_ratio": mutation_count / max(gene_count, 1),
            "protein_prefix_count": protein_prefix_count,
            "parsed_position_count": parsed_position_count,
            "unparsed_mutation_count": mutation_event_count - parsed_position_count,
            "position_mean": np.divide(
                position_sum,
                np.maximum(parsed_position_count, 1),
            ),
            "position_max": position_max,
        }
        for kind in MUTATION_TYPES:
            output[f"type_{kind}_count"] = type_counts[kind]
            output[f"type_{kind}_ratio"] = type_counts[kind] / safe_event_count
        for bin_name, _, _ in POSITION_BINS:
            output[f"position_{bin_name}_count"] = position_bin_counts[bin_name]
            output[f"position_{bin_name}_ratio"] = (
                position_bin_counts[bin_name] / safe_event_count
            )

        for class_name in map(str, self.classes_):
            token = _safe_feature_token(class_name)
            output[f"marker_{token}_score"] = marker_scores[class_name]
            output[f"marker_{token}_hits"] = marker_hits[class_name]

        if len(self.classes_):
            score_matrix = np.column_stack(
                [marker_scores[str(name)] for name in self.classes_]
            )
            sorted_scores = np.sort(score_matrix, axis=1)
            output["marker_score_max"] = sorted_scores[:, -1]
            output["marker_score_margin"] = (
                sorted_scores[:, -1] - sorted_scores[:, -2]
                if len(self.classes_) > 1
                else sorted_scores[:, -1]
            )
        output["marker_gene_count"] = (
            np.column_stack(list(gene_mutation_features.values())).sum(axis=1)
            if gene_mutation_features
            else np.zeros(row_count, dtype=np.int32)
        )

        for gene in self.selected_genes_:
            values = gene_mutation_features.get(
                gene, np.zeros(row_count, dtype=np.int8)
            )
            output[f"gene_{_safe_feature_token(gene)}_mutated"] = values
        for gene, kind in sorted(gene_type_features):
            output[
                f"gene_type_{_safe_feature_token(gene)}_{kind}"
            ] = gene_type_features[(gene, kind)]

        transformed = pd.DataFrame(output, index=frame.index)
        self.output_features_ = transformed.columns.to_numpy(dtype=object)
        return transformed

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, attributes=["output_features_"])
        return self.output_features_.copy()

    def get_marker_table(self) -> pd.DataFrame:
        """학습 fold에서 선택된 marker와 근거 지표를 반환합니다."""
        check_is_fitted(self, attributes=["marker_table_"])
        return self.marker_table_.copy()

    @staticmethod
    def _validate_frame(features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features는 pandas DataFrame이어야 합니다.")
        if features.columns.duplicated().any():
            duplicated = features.columns[features.columns.duplicated()].tolist()
            raise ValueError(f"중복 컬럼이 있습니다: {duplicated[:10]}")
        return features
