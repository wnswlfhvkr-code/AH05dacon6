"""셀 단위 변이 토큰을 유전자별 count/hotspot/multi-hit으로 만드는 EM v17."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re

import numpy as np
import pandas as pd

from src.pipelines.base import PreprocessingPipeline


SYNONYMOUS_PATTERN = re.compile(r"([A-Z])\d+\1")
TRUNCATING_PATTERN = re.compile(r"FS|\*|TER")
INFRAME_PATTERN = re.compile(r"DEL|INS|DUP|>")
SAFE_NAME_PATTERN = re.compile(r"[^A-Z0-9]+")

CONSEQUENCE_SYNONYMOUS = 1
CONSEQUENCE_MISSENSE = 2
CONSEQUENCE_INFRAME = 3
CONSEQUENCE_TRUNCATING = 4


@lru_cache(maxsize=None)
def split_unique_mutations(value: str) -> tuple[tuple[str, int], ...]:
    """셀 문자열을 분리하고 중복 제거 후 각 토큰의 consequence를 반환합니다."""
    normalized = value.strip().upper()
    if not normalized or normalized == "WT" or normalized == "<NA>":
        return ()

    classified: list[tuple[str, int]] = []
    for token in sorted(set(normalized.split())):
        if SYNONYMOUS_PATTERN.fullmatch(token):
            consequence = CONSEQUENCE_SYNONYMOUS
        elif TRUNCATING_PATTERN.search(token):
            consequence = CONSEQUENCE_TRUNCATING
        elif INFRAME_PATTERN.search(token):
            consequence = CONSEQUENCE_INFRAME
        else:
            consequence = CONSEQUENCE_MISSENSE
        classified.append((token, consequence))
    return tuple(classified)


def absolute_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """상수 배열을 안전하게 처리하는 절대 Pearson 상관계수입니다."""
    left_centered = left.astype("float32", copy=True) - float(left.mean())
    right_centered = right.astype("float32", copy=True) - float(right.mean())
    denominator = float(
        np.sqrt(np.square(left_centered).sum() * np.square(right_centered).sum())
    )
    if denominator == 0.0:
        return 0.0
    return abs(float(left_centered @ right_centered) / denominator)


class EMV17PreprocessingPipeline(PreprocessingPipeline):
    """유전자별 unique mutation count, hotspot, multi-hit을 생성합니다."""

    name = "em_v17"
    evaluation_folds = 5

    def __init__(
        self,
        min_gene_mutation_count: int = 5,
        max_gene_features: int = 3000,
        min_hotspot_count: int = 5,
        max_hotspots: int = 384,
        correlation_threshold: float = 0.9,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        integer_parameters = {
            "min_gene_mutation_count": min_gene_mutation_count,
            "max_gene_features": max_gene_features,
            "min_hotspot_count": min_hotspot_count,
            "max_hotspots": max_hotspots,
        }
        for parameter, value in integer_parameters.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{parameter}는 1 이상의 정수여야 합니다.")
        if not 0 < correlation_threshold < 1:
            raise ValueError("correlation_threshold는 0과 1 사이여야 합니다.")

        self.min_gene_mutation_count = min_gene_mutation_count
        self.max_gene_features = max_gene_features
        self.min_hotspot_count = min_hotspot_count
        self.max_hotspots = max_hotspots
        self.correlation_threshold = float(correlation_threshold)
        self.input_gene_columns: list[str] = []
        self.selected_gene_columns: list[str] = []
        self.gene_mutation_support_: dict[str, int] = {}
        self.hotspot_definitions_: dict[str, tuple[str, str, int]] = {}
        self.dropped_rare_gene_columns: list[str] = []
        self.dropped_correlated_columns: list[str] = []
        self.max_checked_absolute_correlation_ = 0.0
        self.steps = (
            "셀 문자열 토큰 분리",
            "표본·유전자 내 중복 토큰 제거",
            "변이별 consequence 판정",
            "유전자별 mutation_count·hotspot·multi-hit 생성",
            "학습 fold 빈도 선택 및 고상관 중복 제거",
        )

    @staticmethod
    def _normalized_column(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("WT").str.strip().str.upper()

    def _learn_gene_and_hotspot_support(self, features: pd.DataFrame) -> None:
        gene_support: dict[str, int] = {}
        token_support: Counter[tuple[str, str]] = Counter()

        for gene in self.input_gene_columns:
            support = 0
            normalized = self._normalized_column(features[gene])
            positions = np.flatnonzero(normalized.ne("WT").to_numpy())
            for row_position in positions:
                mutations = split_unique_mutations(str(normalized.iloc[row_position]))
                if not mutations:
                    continue
                support += 1
                token_support.update((gene, token) for token, _ in mutations)
            gene_support[gene] = support

        self.gene_mutation_support_ = gene_support
        eligible_genes = [
            gene for gene in self.input_gene_columns
            if gene_support[gene] >= self.min_gene_mutation_count
        ]
        self.selected_gene_columns = sorted(
            eligible_genes,
            key=lambda gene: (-gene_support[gene], gene),
        )[:self.max_gene_features]
        selected = set(self.selected_gene_columns)
        self.dropped_rare_gene_columns = [
            gene for gene in self.input_gene_columns if gene not in selected
        ]

        recurrent = [
            (gene, token, count)
            for (gene, token), count in token_support.items()
            if gene in selected and count >= self.min_hotspot_count
        ]
        recurrent.sort(key=lambda item: (-item[2], item[0], item[1]))
        definitions: dict[str, tuple[str, str, int]] = {}
        for rank, (gene, token, count) in enumerate(
            recurrent[:self.max_hotspots], start=1
        ):
            safe_token = SAFE_NAME_PATTERN.sub("_", token).strip("_") or "VARIANT"
            feature_name = f"hotspot_{rank:03d}_{gene}_{safe_token}"
            definitions[feature_name] = (gene, token, count)
        self.hotspot_definitions_ = definitions

    def _candidate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.input_gene_columns) - set(features.columns)
        if missing:
            raise ValueError(f"변환에 필요한 유전자 컬럼이 없습니다: {sorted(missing)}")

        rows = len(features)
        genes = self.selected_gene_columns
        gene_positions = {gene: position for position, gene in enumerate(genes)}
        mutation_counts = np.zeros((rows, len(genes)), dtype="int16")
        consequence_max = np.zeros((rows, len(genes)), dtype="int8")
        multi_hit = np.zeros((rows, len(genes)), dtype="int8")
        consequence_counts = np.zeros((rows, 4), dtype="int16")
        total_unique = np.zeros(rows, dtype="int16")
        multihit_gene_count = np.zeros(rows, dtype="int16")
        hotspot_lookup: dict[tuple[str, str], int] = {
            (gene, token): position
            for position, (_, (gene, token, _)) in enumerate(
                self.hotspot_definitions_.items()
            )
        }
        hotspot_values = np.zeros(
            (rows, len(self.hotspot_definitions_)), dtype="int8"
        )

        for gene in self.input_gene_columns:
            normalized = self._normalized_column(features[gene])
            positions = np.flatnonzero(normalized.ne("WT").to_numpy())
            selected_position = gene_positions.get(gene)
            for row_position in positions:
                mutations = split_unique_mutations(str(normalized.iloc[row_position]))
                if not mutations:
                    continue
                total_unique[row_position] += len(mutations)
                for token, consequence in mutations:
                    consequence_counts[row_position, consequence - 1] += 1
                    hotspot_position = hotspot_lookup.get((gene, token))
                    if hotspot_position is not None:
                        hotspot_values[row_position, hotspot_position] = 1
                if selected_position is not None:
                    mutation_counts[row_position, selected_position] = len(mutations)
                    consequence_max[row_position, selected_position] = max(
                        consequence for _, consequence in mutations
                    )
                    if len(mutations) >= 2:
                        multi_hit[row_position, selected_position] = 1
                        multihit_gene_count[row_position] += 1

        count_frame = pd.DataFrame(
            mutation_counts,
            index=features.index,
            columns=[f"gene_{gene}_mutation_count" for gene in genes],
        )
        consequence_frame = pd.DataFrame(
            consequence_max,
            index=features.index,
            columns=[f"gene_{gene}_consequence_max" for gene in genes],
        )
        multihit_frame = pd.DataFrame(
            multi_hit,
            index=features.index,
            columns=[f"gene_{gene}_multi_hit" for gene in genes],
        )
        hotspot_frame = pd.DataFrame(
            hotspot_values,
            index=features.index,
            columns=list(self.hotspot_definitions_),
        )
        summaries = pd.DataFrame({
            "sample_unique_mutation_count_log1p": np.log1p(total_unique),
            "sample_multihit_gene_count_log1p": np.log1p(multihit_gene_count),
            "consequence_synonymous_count_log1p": np.log1p(consequence_counts[:, 0]),
            "consequence_missense_count_log1p": np.log1p(consequence_counts[:, 1]),
            "consequence_inframe_count_log1p": np.log1p(consequence_counts[:, 2]),
            "consequence_truncating_count_log1p": np.log1p(consequence_counts[:, 3]),
        }, index=features.index, dtype="float32")
        return pd.concat(
            [hotspot_frame, count_frame, multihit_frame, consequence_frame, summaries],
            axis=1,
        ).astype("float32")

    def _find_correlated_columns(self, candidate: pd.DataFrame) -> tuple[list[str], float]:
        """동일 유전자 파생값과 같은 유전자 hotspot의 구조적 중복을 제거합니다."""
        hotspots_by_gene: dict[str, list[str]] = {}
        for feature_name, (gene, _, _) in self.hotspot_definitions_.items():
            hotspots_by_gene.setdefault(gene, []).append(feature_name)

        dropped: list[str] = []
        maximum = 0.0
        for gene in self.selected_gene_columns:
            ordered = hotspots_by_gene.get(gene, []) + [
                f"gene_{gene}_mutation_count",
                f"gene_{gene}_multi_hit",
                f"gene_{gene}_consequence_max",
            ]
            kept: list[str] = []
            for column in ordered:
                values = candidate[column].to_numpy()
                redundant = False
                for kept_column in kept:
                    correlation = absolute_correlation(
                        values, candidate[kept_column].to_numpy()
                    )
                    maximum = max(maximum, correlation)
                    if correlation >= self.correlation_threshold:
                        redundant = True
                        break
                if redundant:
                    dropped.append(column)
                else:
                    kept.append(column)

        summary_columns = [
            column for column in candidate.columns
            if column.startswith("sample_") or column.startswith("consequence_")
        ]
        kept_summaries: list[str] = []
        for column in summary_columns:
            values = candidate[column].to_numpy()
            if any(
                absolute_correlation(values, candidate[kept].to_numpy())
                >= self.correlation_threshold
                for kept in kept_summaries
            ):
                dropped.append(column)
            else:
                kept_summaries.append(column)
        return dropped, maximum

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EMV17PreprocessingPipeline":
        self.input_gene_columns = features.columns.tolist()
        self._learn_gene_and_hotspot_support(features)
        if not self.selected_gene_columns:
            raise ValueError("최소 변이 빈도를 만족하는 유전자가 없습니다.")
        candidate = self._candidate_features(features)
        (
            self.dropped_correlated_columns,
            self.max_checked_absolute_correlation_,
        ) = self._find_correlated_columns(candidate)
        retained = candidate.drop(columns=self.dropped_correlated_columns)
        super().fit(retained, labels)
        print(
            f"[{self.name}] 선택 유전자 {len(self.selected_gene_columns)}개, "
            f"hotspot {len(self.hotspot_definitions_)}개, "
            f"multi-hit 유전자 피처 {len(self.selected_gene_columns)}개, "
            f"고상관 제거 {len(self.dropped_correlated_columns)}개"
        )
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        candidate = self._candidate_features(features).drop(
            columns=self.dropped_correlated_columns,
            errors="ignore",
        )
        return super().transform(candidate).astype("float32")

    def summary(self) -> dict[str, int | float]:
        summary: dict[str, int | float] = super().summary()
        summary.update({
            "selected_gene_groups": len(self.selected_gene_columns),
            "gene_mutation_count_features_created": len(self.selected_gene_columns),
            "gene_consequence_features_created": len(self.selected_gene_columns),
            "gene_multihit_features_created": len(self.selected_gene_columns),
            "hotspot_features_created": len(self.hotspot_definitions_),
            "dropped_gene_features": len(self.dropped_rare_gene_columns),
            "dropped_correlated_features": len(self.dropped_correlated_columns),
            "max_checked_absolute_correlation": self.max_checked_absolute_correlation_,
        })
        return summary
