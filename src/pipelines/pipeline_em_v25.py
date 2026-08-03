"""em_v16과 em_v17을 중복 없이 결합한 EM v25."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.pipelines.base import PreprocessingPipeline


SYNONYMOUS = 1
MISSENSE = 2
INFRAME = 3
TRUNCATING = 4
CONSEQUENCE_NAMES = {
    SYNONYMOUS: "synonymous",
    MISSENSE: "missense",
    INFRAME: "inframe",
    TRUNCATING: "truncating",
}
SYNONYMOUS_PATTERN = re.compile(r"^([A-Z])\d+\1$")
TRUNCATING_PATTERN = re.compile(r"FS|\*|TER", re.IGNORECASE)
INFRAME_PATTERN = re.compile(r"DEL|INS|DUP|>", re.IGNORECASE)
SAFE_NAME_PATTERN = re.compile(r"[^A-Z0-9]+")


@lru_cache(maxsize=None)
def split_unique_mutations(value: str) -> tuple[tuple[str, int], ...]:
    """셀 문자열을 분리하고 중복 토큰을 제거한 뒤 consequence를 판정합니다."""
    normalized = value.strip().upper()
    if not normalized or normalized in {"WT", "<NA>"}:
        return ()
    classified: list[tuple[str, int]] = []
    for token in sorted(set(normalized.split())):
        if SYNONYMOUS_PATTERN.fullmatch(token):
            consequence = SYNONYMOUS
        elif TRUNCATING_PATTERN.search(token):
            consequence = TRUNCATING
        elif INFRAME_PATTERN.search(token):
            consequence = INFRAME
        else:
            consequence = MISSENSE
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


def parse_feature_matrices(
    features: pd.DataFrame,
    input_columns: list[str],
    selected_columns: list[str],
    hotspots: dict[str, tuple[str, str, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """한 번의 토큰 순회로 v16/v17의 공통·파생 피처를 모두 계산합니다."""
    missing = set(input_columns) - set(features.columns)
    if missing:
        raise ValueError(f"변환에 필요한 유전자 컬럼이 없습니다: {sorted(missing)}")
    rows = len(features)
    gene_positions = {gene: position for position, gene in enumerate(selected_columns)}
    mutation_counts = np.zeros((rows, len(selected_columns)), dtype="int16")
    consequence_max = np.zeros((rows, len(selected_columns)), dtype="int8")
    multi_hit = np.zeros((rows, len(selected_columns)), dtype="int8")
    consequence_counts = np.zeros((rows, len(CONSEQUENCE_NAMES)), dtype="int16")
    unique_token_count = np.zeros(rows, dtype="int32")
    mutated_gene_count = np.zeros(rows, dtype="int16")
    multihit_gene_count = np.zeros(rows, dtype="int16")
    hotspot_lookup = {
        (gene, token): position
        for position, (_, (gene, token, _)) in enumerate(hotspots.items())
    }
    hotspot_values = np.zeros((rows, len(hotspots)), dtype="int8")

    for gene in input_columns:
        normalized = features[gene].astype("string").fillna("WT").str.strip().str.upper()
        selected_position = gene_positions.get(gene)
        for row_position in np.flatnonzero(normalized.ne("WT").to_numpy()):
            mutations = split_unique_mutations(str(normalized.iloc[row_position]))
            if not mutations:
                continue
            unique_token_count[row_position] += len(mutations)
            if selected_position is not None:
                mutated_gene_count[row_position] += 1
                mutation_counts[row_position, selected_position] = len(mutations)
                consequence_max[row_position, selected_position] = max(
                    consequence for _, consequence in mutations
                )
                if len(mutations) >= 2:
                    multi_hit[row_position, selected_position] = 1
                    multihit_gene_count[row_position] += 1
            for token, consequence in mutations:
                consequence_counts[row_position, consequence - 1] += 1
                hotspot_position = hotspot_lookup.get((gene, token))
                if hotspot_position is not None:
                    hotspot_values[row_position, hotspot_position] = 1

    count_frame = pd.DataFrame(
        mutation_counts,
        index=features.index,
        columns=[f"gene_{gene}_mutation_count" for gene in selected_columns],
    )
    consequence_frame = pd.DataFrame(
        consequence_max,
        index=features.index,
        columns=[f"gene_{gene}_consequence_max" for gene in selected_columns],
    )
    multihit_frame = pd.DataFrame(
        multi_hit,
        index=features.index,
        columns=[f"gene_{gene}_multi_hit" for gene in selected_columns],
    )
    hotspot_frame = pd.DataFrame(
        hotspot_values, index=features.index, columns=list(hotspots)
    )
    total_tokens = consequence_counts.sum(axis=1).astype("float32")
    token_denominator = np.maximum(total_tokens, 1.0)
    gene_denominator = np.maximum(mutated_gene_count.astype("float32"), 1.0)
    summaries: dict[str, np.ndarray] = {
        "mutation_burden_log1p": np.log1p(mutated_gene_count),
        "sample_unique_mutation_count_log1p": np.log1p(unique_token_count),
        "sample_multihit_gene_count_log1p": np.log1p(multihit_gene_count),
    }
    for position, name in enumerate(CONSEQUENCE_NAMES.values()):
        count = consequence_counts[:, position].astype("float32")
        summaries[f"consequence_{name}_count_log1p"] = np.log1p(count)
        summaries[f"consequence_{name}_token_ratio"] = count / token_denominator
        gene_count = (consequence_max == position + 1).sum(axis=1).astype("float32")
        summaries[f"consequence_{name}_gene_ratio"] = gene_count / gene_denominator
    summary_frame = pd.DataFrame(summaries, index=features.index, dtype="float32")
    candidate = pd.concat(
        [count_frame, consequence_frame, multihit_frame, hotspot_frame, summary_frame],
        axis=1,
    ).astype("float32")
    mutation_presence = count_frame.gt(0).astype("int8")
    mutation_presence.columns = selected_columns
    return candidate, mutation_presence


def learn_support(
    features: pd.DataFrame,
    columns: list[str],
    minimum_count: int,
    maximum_genes: int,
    minimum_hotspot_count: int,
    maximum_hotspots: int,
) -> tuple[
    list[str],
    dict[str, int],
    dict[str, tuple[str, str, int]],
    list[str],
]:
    """공통 유전자 지지도와 recurrent hotspot을 한 번만 학습합니다."""
    gene_support: dict[str, int] = {}
    token_support: Counter[tuple[str, str]] = Counter()
    for gene in columns:
        support = 0
        normalized = features[gene].astype("string").fillna("WT").str.strip().str.upper()
        for value in normalized[normalized.ne("WT")]:
            mutations = split_unique_mutations(str(value))
            if not mutations:
                continue
            support += 1
            token_support.update((gene, token) for token, _ in mutations)
        gene_support[gene] = support
    eligible = [gene for gene in columns if gene_support[gene] >= minimum_count]
    selected = sorted(eligible, key=lambda gene: (-gene_support[gene], gene))[:maximum_genes]
    selected_set = set(selected)
    dropped = [gene for gene in columns if gene not in selected_set]
    recurrent = sorted(
        (
            (gene, token, count)
            for (gene, token), count in token_support.items()
            if gene in selected_set and count >= minimum_hotspot_count
        ),
        key=lambda item: (-item[2], item[0], item[1]),
    )[:maximum_hotspots]
    definitions: dict[str, tuple[str, str, int]] = {}
    for rank, (gene, token, count) in enumerate(recurrent, start=1):
        safe_token = SAFE_NAME_PATTERN.sub("_", token).strip("_") or "VARIANT"
        definitions[f"hotspot_{rank:03d}_{gene}_{safe_token}"] = (gene, token, count)
    return selected, gene_support, definitions, dropped


def find_correlated_columns(
    candidate: pd.DataFrame,
    selected_genes: list[str],
    hotspots: dict[str, tuple[str, str, int]],
    threshold: float,
) -> tuple[list[str], float]:
    """같은 유전자 파생 피처와 summary 안의 구조적 중복만 제거합니다."""
    hotspots_by_gene: dict[str, list[str]] = {}
    for name, (gene, _, _) in hotspots.items():
        hotspots_by_gene.setdefault(gene, []).append(name)
    dropped: list[str] = []
    maximum = 0.0
    for gene in selected_genes:
        # 넓은 집계 피처를 우선 보존하고 동일 신호인 세부 hotspot을 후순위로 둡니다.
        ordered = [
            f"gene_{gene}_mutation_count",
            f"gene_{gene}_consequence_max",
            f"gene_{gene}_multi_hit",
            *hotspots_by_gene.get(gene, []),
        ]
        kept: list[str] = []
        for column in ordered:
            values = candidate[column].to_numpy()
            redundant = False
            for kept_column in kept:
                correlation = absolute_correlation(values, candidate[kept_column].to_numpy())
                maximum = max(maximum, correlation)
                if correlation >= threshold:
                    redundant = True
                    break
            if redundant:
                dropped.append(column)
            else:
                kept.append(column)

    summary_columns = [
        column for column in candidate
        if column.startswith("sample_")
        or column.startswith("mutation_burden_")
        or column.startswith("consequence_")
    ]
    kept_summaries: list[str] = []
    for column in summary_columns:
        values = candidate[column].to_numpy()
        redundant = False
        for kept_column in kept_summaries:
            correlation = absolute_correlation(values, candidate[kept_column].to_numpy())
            maximum = max(maximum, correlation)
            if correlation >= threshold:
                redundant = True
                break
        if redundant:
            dropped.append(column)
        else:
            kept_summaries.append(column)
    return list(dict.fromkeys(dropped)), maximum


def learn_class_weights(
    mutation: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
) -> dict[str, dict[str, float]]:
    aligned = labels.reindex(mutation.index).astype("string")
    weights_by_class: dict[str, dict[str, float]] = {}
    for raw_class_name in sorted(aligned.dropna().unique()):
        class_name = str(raw_class_name)
        in_class = aligned.eq(class_name).fillna(False)
        class_count = int(in_class.sum())
        other_count = len(aligned) - class_count
        positive = mutation.loc[in_class].sum(axis=0).astype("float64")
        negative = mutation.loc[~in_class].sum(axis=0).astype("float64")
        class_odds = (positive + smoothing) / (class_count - positive + smoothing)
        other_odds = (negative + smoothing) / (other_count - negative + smoothing)
        score = np.log2(class_odds / other_odds).clip(0.0, max_log2_odds)
        support = positive + negative
        score *= np.sqrt(support / (support + shrinkage))
        selected = score.nlargest(top_genes_per_class)
        selected = selected[selected.gt(0)]
        if selected.empty:
            raise ValueError(f"{class_name}의 변이 signature를 만들 수 없습니다.")
        weights_by_class[class_name] = selected.astype(float).to_dict()
    return weights_by_class


def add_signatures(
    output: pd.DataFrame,
    mutation: pd.DataFrame,
    weights_by_class: dict[str, dict[str, float]],
) -> None:
    for class_name, weights in weights_by_class.items():
        genes = list(weights)
        matrix = mutation[genes].to_numpy(dtype="float32")
        vector = np.array([weights[gene] for gene in genes], dtype="float32")
        output[f"signature_{class_name}_weighted"] = matrix @ vector
        output[f"signature_{class_name}_match_count"] = matrix.sum(axis=1)


def create_oof_signatures(
    mutation: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
    folds: int,
    random_state: int,
) -> pd.DataFrame:
    aligned = labels.reindex(mutation.index)
    n_splits = min(folds, int(aligned.value_counts().min()))
    if n_splits < 2:
        raise ValueError("OOF signature에는 클래스별 표본이 최소 2개 필요합니다.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    output = pd.DataFrame(index=mutation.index)
    for train_positions, valid_positions in splitter.split(mutation, aligned):
        train, valid = mutation.iloc[train_positions], mutation.iloc[valid_positions]
        weights = learn_class_weights(
            train, aligned.iloc[train_positions], top_genes_per_class,
            smoothing, max_log2_odds, shrinkage,
        )
        fold_output = pd.DataFrame(index=valid.index)
        add_signatures(fold_output, valid, weights)
        for column in fold_output:
            output.loc[valid.index, column] = fold_output[column]
    return output.astype("float32")


class EMV25PreprocessingPipeline(PreprocessingPipeline):
    """v16 OOF signature와 v17 token/count/multi-hit을 단일 흐름으로 결합합니다."""

    name = "em_v25"
    evaluation_folds = 5

    def __init__(
        self,
        min_mutation_count: int = 5,
        min_gene_mutation_count: int | None = None,
        max_gene_features: int = 3000,
        top_genes_per_class: int = 20,
        smoothing: float = 0.5,
        max_log2_odds: float = 8.0,
        shrinkage: float = 10.0,
        min_hotspot_count: int = 5,
        max_hotspots: int = 384,
        correlation_threshold: float = 0.9,
        inner_signature_folds: int = 5,
        signature_random_state: int = 42,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        minimum_count = (
            min_mutation_count
            if min_gene_mutation_count is None
            else min_gene_mutation_count
        )
        integer_values = {
            "min_gene_mutation_count": (minimum_count, 1),
            "max_gene_features": (max_gene_features, 1),
            "top_genes_per_class": (top_genes_per_class, 1),
            "min_hotspot_count": (min_hotspot_count, 1),
            "max_hotspots": (max_hotspots, 1),
            "inner_signature_folds": (inner_signature_folds, 2),
        }
        for name, (value, minimum) in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name}는 {minimum} 이상의 정수여야 합니다.")
        if not 0 < correlation_threshold < 1:
            raise ValueError("correlation_threshold는 0과 1 사이여야 합니다.")
        if smoothing <= 0 or max_log2_odds <= 0 or shrinkage <= 0:
            raise ValueError("signature 안정화 파라미터는 0보다 커야 합니다.")
        self.min_gene_mutation_count = minimum_count
        self.max_gene_features = max_gene_features
        self.top_genes_per_class = top_genes_per_class
        self.smoothing = float(smoothing)
        self.max_log2_odds = float(max_log2_odds)
        self.shrinkage = float(shrinkage)
        self.min_hotspot_count = min_hotspot_count
        self.max_hotspots = max_hotspots
        self.correlation_threshold = float(correlation_threshold)
        self.inner_signature_folds = inner_signature_folds
        self.signature_random_state = signature_random_state
        self.input_gene_columns: list[str] = []
        self.selected_gene_columns: list[str] = []
        self.dropped_gene_columns: list[str] = []
        self.gene_mutation_support_: dict[str, int] = {}
        self.hotspot_definitions_: dict[str, tuple[str, str, int]] = {}
        self.class_gene_weights_: dict[str, dict[str, float]] = {}
        self.dropped_correlated_columns: list[str] = []
        self.max_checked_absolute_correlation_ = 0.0
        self.steps = (
            "변이 토큰 단일 분리 및 중복 제거",
            "공통 변이 빈도 선택과 유전자 상한",
            "mutation_count·consequence_max·multi-hit·hotspot",
            "구조적 고상관 파생 피처 제거",
            "inner-fold OOF 암종 signature",
        )

    def _base_features(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        candidate, mutation = parse_feature_matrices(
            features, self.input_gene_columns, self.selected_gene_columns,
            self.hotspot_definitions_,
        )
        retained = candidate.drop(columns=self.dropped_correlated_columns, errors="ignore")
        return retained, mutation

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        retained, mutation = self._base_features(features)
        add_signatures(retained, mutation, self.class_gene_weights_)
        return retained

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMV25PreprocessingPipeline":
        self.input_gene_columns = features.columns.tolist()
        (
            self.selected_gene_columns,
            self.gene_mutation_support_,
            self.hotspot_definitions_,
            self.dropped_gene_columns,
        ) = learn_support(
            features, self.input_gene_columns, self.min_gene_mutation_count,
            self.max_gene_features, self.min_hotspot_count, self.max_hotspots,
        )
        if not self.selected_gene_columns:
            raise ValueError("최소 변이 빈도를 만족하는 유전자가 없습니다.")
        candidate, mutation = parse_feature_matrices(
            features, self.input_gene_columns, self.selected_gene_columns,
            self.hotspot_definitions_,
        )
        (
            self.dropped_correlated_columns,
            self.max_checked_absolute_correlation_,
        ) = find_correlated_columns(
            candidate, self.selected_gene_columns, self.hotspot_definitions_,
            self.correlation_threshold,
        )
        self.class_gene_weights_ = learn_class_weights(
            mutation, labels, self.top_genes_per_class, self.smoothing,
            self.max_log2_odds, self.shrinkage,
        )
        engineered = candidate.drop(columns=self.dropped_correlated_columns, errors="ignore")
        add_signatures(engineered, mutation, self.class_gene_weights_)
        super().fit(engineered, labels)
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        self.fit(features, labels)
        transformed = self.transform(features)
        _, mutation = self._base_features(features)
        oof = create_oof_signatures(
            mutation, labels, self.top_genes_per_class, self.smoothing,
            self.max_log2_odds, self.shrinkage, self.inner_signature_folds,
            self.signature_random_state,
        )
        for column in oof:
            if column in transformed:
                transformed.loc[:, column] = oof[column].to_numpy()
        return transformed.astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return super().transform(self._build_features(features)).astype("float32")

    def summary(self) -> dict[str, int | float]:
        result: dict[str, int | float] = super().summary()
        result.update({
            "selected_gene_groups": len(self.selected_gene_columns),
            "dropped_gene_features": len(self.dropped_gene_columns),
            "gene_mutation_count_features_created": len(self.selected_gene_columns),
            "gene_consequence_features_created": len(self.selected_gene_columns),
            "gene_multihit_features_created": len(self.selected_gene_columns),
            "hotspot_features_created": len(self.hotspot_definitions_),
            "signature_features": 2 * len(self.class_gene_weights_),
            "dropped_correlated_features": len(self.dropped_correlated_columns),
            "max_checked_absolute_correlation": self.max_checked_absolute_correlation_,
        })
        return result
