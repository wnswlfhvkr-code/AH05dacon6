"""em_v24와 em_v9을 중복 없이 결합한 EM v40."""

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
NONSENSE = 4
FRAMESHIFT = 5
CONSEQUENCE_NAMES = {
    SYNONYMOUS: "synonymous",
    MISSENSE: "missense",
    INFRAME: "inframe",
    NONSENSE: "nonsense",
    FRAMESHIFT: "frameshift",
}
SYNONYMOUS_PATTERN = re.compile(r"^([A-Z])(\d+)\1$")
MISSENSE_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")
FRAMESHIFT_PATTERN = re.compile(r"FS", re.IGNORECASE)
STOP_PATTERN = re.compile(r"(?:\*|TER$|X$)", re.IGNORECASE)
INFRAME_PATTERN = re.compile(r"(?:DEL|INS|DUP|>|_)", re.IGNORECASE)
SAFE_NAME_PATTERN = re.compile(r"[^A-Z0-9]+")


@lru_cache(maxsize=None)
def classify_mutation_token(token: str) -> int:
    """단백질 변이를 WT=0부터 frameshift=5까지 상호 배타적으로 분류합니다."""
    normalized = token.strip().upper()
    if not normalized or normalized in {"WT", "<NA>"}:
        return 0
    if SYNONYMOUS_PATTERN.fullmatch(normalized):
        return SYNONYMOUS
    if FRAMESHIFT_PATTERN.search(normalized):
        return FRAMESHIFT
    if STOP_PATTERN.search(normalized):
        return NONSENSE
    if INFRAME_PATTERN.search(normalized):
        return INFRAME
    match = MISSENSE_PATTERN.fullmatch(normalized)
    if match and match.group(1) != match.group(3):
        return MISSENSE
    return MISSENSE


@lru_cache(maxsize=None)
def split_unique_mutations(value: str) -> tuple[tuple[str, int], ...]:
    """셀 문자열을 분리하고 동일 토큰을 표본 안에서 한 번만 유지합니다."""
    normalized = value.strip().upper()
    if not normalized or normalized in {"WT", "<NA>"}:
        return ()
    return tuple(
        (token, classify_mutation_token(token))
        for token in sorted(set(normalized.split()))
        if classify_mutation_token(token) != 0
    )


def build_mutation_matrices(
    features: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """한 번의 토큰 순회로 severity·전체/기능 변이·multi-hit·결과 수를 만듭니다."""
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"변환에 필요한 유전자 컬럼이 없습니다: {sorted(missing)}")
    rows, width = len(features), len(columns)
    severity_values = np.zeros((rows, width), dtype="int8")
    mutation_values = np.zeros((rows, width), dtype="int8")
    functional_values = np.zeros((rows, width), dtype="int8")
    multi_hit_values = np.zeros((rows, width), dtype="int8")
    category_counts = np.zeros((rows, len(CONSEQUENCE_NAMES)), dtype="int16")

    for column_position, column in enumerate(columns):
        normalized = features[column].astype("string").fillna("WT").str.strip().str.upper()
        for row_position in np.flatnonzero(normalized.ne("WT").to_numpy()):
            mutations = split_unique_mutations(str(normalized.iloc[row_position]))
            if not mutations:
                continue
            consequences = [consequence for _, consequence in mutations]
            severity_values[row_position, column_position] = max(consequences)
            mutation_values[row_position, column_position] = 1
            functional_values[row_position, column_position] = int(
                any(consequence >= MISSENSE for consequence in consequences)
            )
            multi_hit_values[row_position, column_position] = int(len(mutations) >= 2)
            for consequence in consequences:
                category_counts[row_position, consequence - 1] += 1

    frame_arguments = {"index": features.index, "columns": columns, "dtype": "int8"}
    return (
        pd.DataFrame(severity_values, **frame_arguments),
        pd.DataFrame(mutation_values, **frame_arguments),
        pd.DataFrame(functional_values, **frame_arguments),
        pd.DataFrame(multi_hit_values, **frame_arguments),
        category_counts,
    )


def add_summary_features(
    output: pd.DataFrame,
    mutation: pd.DataFrame,
    functional: pd.DataFrame,
    multi_hit: pd.DataFrame,
    category_counts: np.ndarray,
) -> None:
    """v16/v18의 중복 burden을 상세한 단일 summary 집합으로 통합합니다."""
    token_count = category_counts.sum(axis=1).astype("float32")
    denominator = np.maximum(token_count, 1.0)
    functional_token_count = category_counts[:, 1:].sum(axis=1).astype("float32")
    output["mutated_gene_count_log1p"] = np.log1p(
        mutation.sum(axis=1).to_numpy(dtype="float32")
    ).astype("float32")
    output["mutation_token_count_log1p"] = np.log1p(token_count).astype("float32")
    output["functional_mutation_count_log1p"] = np.log1p(
        functional_token_count
    ).astype("float32")
    output["functional_gene_count_log1p"] = np.log1p(
        functional.sum(axis=1).to_numpy(dtype="float32")
    ).astype("float32")
    output["multi_variant_gene_count_log1p"] = np.log1p(
        multi_hit.sum(axis=1).to_numpy(dtype="float32")
    ).astype("float32")
    output["functional_mutation_ratio"] = (
        functional_token_count / denominator
    ).astype("float32")
    for position, name in enumerate(CONSEQUENCE_NAMES.values()):
        count = category_counts[:, position].astype("float32")
        output[f"consequence_{name}_count_log1p"] = np.log1p(count).astype("float32")
        output[f"consequence_{name}_ratio"] = (count / denominator).astype("float32")


def learn_functional_hotspots(
    features: pd.DataFrame,
    columns: list[str],
    minimum_count: int,
    maximum_hotspots: int,
) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str], int]]:
    """동의 변이를 제외한 recurrent hotspot만 선택합니다."""
    support: Counter[tuple[str, str]] = Counter()
    for gene in columns:
        normalized = features[gene].astype("string").fillna("WT").str.strip().str.upper()
        for value in normalized[normalized.ne("WT")]:
            support.update(
                (gene, token)
                for token, consequence in split_unique_mutations(str(value))
                if consequence >= MISSENSE
            )
    pairs = sorted(
        (pair for pair, count in support.items() if count >= minimum_count),
        key=lambda pair: (-support[pair], pair[0], pair[1]),
    )[:maximum_hotspots]
    hotspots = []
    for rank, (gene, token) in enumerate(pairs, start=1):
        safe_token = SAFE_NAME_PATTERN.sub("_", token).strip("_") or "VARIANT"
        hotspots.append((f"hotspot_{rank:03d}_{gene}_{safe_token}", gene, token))
    return hotspots, dict(support)


def create_hotspot_features(
    features: pd.DataFrame,
    hotspots: list[tuple[str, str, str]],
) -> pd.DataFrame:
    by_gene: dict[str, list[tuple[str, str]]] = {}
    for name, gene, token in hotspots:
        by_gene.setdefault(gene, []).append((name, token))
    output: dict[str, pd.Series] = {}
    for gene, definitions in by_gene.items():
        normalized = features[gene].astype("string").fillna("WT").str.strip().str.upper()
        token_sets = normalized.map(
            lambda value: frozenset(token for token, _ in split_unique_mutations(str(value)))
        )
        for name, token in definitions:
            output[name] = token_sets.map(lambda tokens: token in tokens).astype("int8")
    return pd.DataFrame(output, index=features.index)


def learn_class_weights(
    matrix: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
    signal_name: str,
) -> dict[str, dict[str, float]]:
    """전체 또는 기능 변이 채널의 암종별 안정화 log2 odds를 학습합니다."""
    aligned = labels.reindex(matrix.index).astype("string")
    weights_by_class: dict[str, dict[str, float]] = {}
    for raw_class_name in sorted(aligned.dropna().unique()):
        class_name = str(raw_class_name)
        in_class = aligned.eq(class_name).fillna(False)
        class_count = int(in_class.sum())
        other_count = len(aligned) - class_count
        positive = matrix.loc[in_class].sum(axis=0).astype("float64")
        negative = matrix.loc[~in_class].sum(axis=0).astype("float64")
        class_odds = (positive + smoothing) / (class_count - positive + smoothing)
        other_odds = (negative + smoothing) / (other_count - negative + smoothing)
        score = np.log2(class_odds / other_odds).clip(0.0, max_log2_odds)
        support = positive + negative
        score *= np.sqrt(support / (support + shrinkage))
        selected = score.nlargest(top_genes_per_class)
        selected = selected[selected.gt(0)]
        if selected.empty:
            raise ValueError(f"{class_name}의 {signal_name} signature를 만들 수 없습니다.")
        weights_by_class[class_name] = selected.astype(float).to_dict()
    return weights_by_class


def add_signature_channel(
    output: pd.DataFrame,
    matrix: pd.DataFrame,
    weights_by_class: dict[str, dict[str, float]],
    channel: str,
) -> None:
    """두 채널이 충돌하지 않도록 이름을 구분해 weighted/match 피처를 만듭니다."""
    for class_name, weights in weights_by_class.items():
        genes = list(weights)
        values = matrix[genes].to_numpy(dtype="float32")
        vector = np.array([weights[gene] for gene in genes], dtype="float32")
        output[f"signature_{channel}_{class_name}_weighted"] = values @ vector
        output[f"signature_{channel}_{class_name}_match_count"] = values.sum(axis=1)


def create_oof_dual_signatures(
    mutation: pd.DataFrame,
    functional: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
    folds: int,
    random_state: int,
) -> pd.DataFrame:
    """동일 inner-fold에서 전체 변이와 기능 변이 signature를 함께 교차 적합합니다."""
    aligned = labels.reindex(mutation.index)
    n_splits = min(folds, int(aligned.value_counts().min()))
    if n_splits < 2:
        raise ValueError("OOF signature에는 클래스별 표본이 최소 2개 필요합니다.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    output = pd.DataFrame(index=mutation.index)
    for train_positions, valid_positions in splitter.split(mutation, aligned):
        fold_labels = aligned.iloc[train_positions]
        fold_output = pd.DataFrame(index=mutation.iloc[valid_positions].index)
        for channel, source in (("all", mutation), ("functional", functional)):
            weights = learn_class_weights(
                source.iloc[train_positions], fold_labels, top_genes_per_class,
                smoothing, max_log2_odds, shrinkage, channel,
            )
            add_signature_channel(
                fold_output, source.iloc[valid_positions], weights, channel
            )
        for column in fold_output:
            output.loc[fold_output.index, column] = fold_output[column]
    return output.astype("float32")


class EMV40PreprocessingPipeline(PreprocessingPipeline):
    """v9 signature를 중복 생성하지 않고 v24의 OOF all/functional signature를 유지합니다."""

    name = "em_v40"
    evaluation_folds = 5

    def __init__(
        self,
        min_mutation_count: int = 5,
        min_functional_mutation_count: int = 5,
        top_genes_per_class: int = 20,
        smoothing: float = 0.5,
        max_log2_odds: float = 8.0,
        shrinkage: float = 10.0,
        min_hotspot_count: int = 5,
        max_hotspots: int = 384,
        inner_signature_folds: int = 5,
        signature_random_state: int = 42,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        integer_values = {
            "min_mutation_count": (min_mutation_count, 1),
            "min_functional_mutation_count": (min_functional_mutation_count, 1),
            "top_genes_per_class": (top_genes_per_class, 1),
            "min_hotspot_count": (min_hotspot_count, 1),
            "max_hotspots": (max_hotspots, 1),
            "inner_signature_folds": (inner_signature_folds, 2),
        }
        for name, (value, minimum) in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name}는 {minimum} 이상의 정수여야 합니다.")
        if smoothing <= 0 or max_log2_odds <= 0 or shrinkage <= 0:
            raise ValueError("signature 안정화 파라미터는 0보다 커야 합니다.")
        self.min_mutation_count = min_mutation_count
        self.min_functional_mutation_count = min_functional_mutation_count
        self.top_genes_per_class = top_genes_per_class
        self.smoothing = float(smoothing)
        self.max_log2_odds = float(max_log2_odds)
        self.shrinkage = float(shrinkage)
        self.min_hotspot_count = min_hotspot_count
        self.max_hotspots = max_hotspots
        self.inner_signature_folds = inner_signature_folds
        self.signature_random_state = signature_random_state
        self.selected_gene_columns: list[str] = []
        self.mutation_signature_genes_: list[str] = []
        self.functional_signature_genes_: list[str] = []
        self.dropped_gene_columns: list[str] = []
        self.mutation_counts_: dict[str, int] = {}
        self.functional_mutation_counts_: dict[str, int] = {}
        self.all_variant_weights_: dict[str, dict[str, float]] = {}
        self.functional_weights_: dict[str, dict[str, float]] = {}
        self.hotspots_: list[tuple[str, str, str]] = []
        self.hotspot_support_: dict[tuple[str, str], int] = {}
        self.steps = (
            "변이 토큰 단일 분리 및 중복 제거",
            "전체 변이·기능 변이 채널 분리",
            "채널별 최소 빈도와 유전자 합집합",
            "단일 consequence summary와 기능 hotspot",
            "공통 inner-fold OOF dual signature",
        )

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        severity, mutation, functional, multi_hit, category_counts = build_mutation_matrices(
            features, self.selected_gene_columns
        )
        output = severity.astype("float32")
        add_summary_features(output, mutation, functional, multi_hit, category_counts)
        add_signature_channel(
            output, mutation[self.mutation_signature_genes_],
            self.all_variant_weights_, "all",
        )
        add_signature_channel(
            output, functional[self.functional_signature_genes_],
            self.functional_weights_, "functional",
        )
        return pd.concat(
            [output, create_hotspot_features(features, self.hotspots_)], axis=1
        )

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMV40PreprocessingPipeline":
        columns = features.columns.tolist()
        _, mutation, functional, _, _ = build_mutation_matrices(features, columns)
        mutation_counts = mutation.sum(axis=0).astype(int)
        functional_counts = functional.sum(axis=0).astype(int)
        self.mutation_counts_ = mutation_counts.to_dict()
        self.functional_mutation_counts_ = functional_counts.to_dict()
        self.mutation_signature_genes_ = [
            gene for gene in columns if mutation_counts[gene] >= self.min_mutation_count
        ]
        self.functional_signature_genes_ = [
            gene for gene in columns
            if functional_counts[gene] >= self.min_functional_mutation_count
        ]
        selected = set(self.mutation_signature_genes_) | set(self.functional_signature_genes_)
        self.selected_gene_columns = [gene for gene in columns if gene in selected]
        self.dropped_gene_columns = [gene for gene in columns if gene not in selected]
        if not self.mutation_signature_genes_:
            raise ValueError("최소 전체 변이 빈도를 만족하는 유전자가 없습니다.")
        if not self.functional_signature_genes_:
            raise ValueError("최소 기능 변이 빈도를 만족하는 유전자가 없습니다.")

        _, selected_mutation, selected_functional, _, _ = build_mutation_matrices(
            features, self.selected_gene_columns
        )
        mutation_channel = selected_mutation[self.mutation_signature_genes_]
        functional_channel = selected_functional[self.functional_signature_genes_]
        self.all_variant_weights_ = learn_class_weights(
            mutation_channel, labels, self.top_genes_per_class, self.smoothing,
            self.max_log2_odds, self.shrinkage, "전체 변이",
        )
        self.functional_weights_ = learn_class_weights(
            functional_channel, labels, self.top_genes_per_class, self.smoothing,
            self.max_log2_odds, self.shrinkage, "기능 변이",
        )
        self.hotspots_, self.hotspot_support_ = learn_functional_hotspots(
            features, self.functional_signature_genes_, self.min_hotspot_count,
            self.max_hotspots,
        )
        super().fit(self._build_features(features), labels)
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        self.fit(features, labels)
        transformed = self.transform(features)
        _, mutation, functional, _, _ = build_mutation_matrices(
            features, self.selected_gene_columns
        )
        oof = create_oof_dual_signatures(
            mutation[self.mutation_signature_genes_],
            functional[self.functional_signature_genes_],
            labels, self.top_genes_per_class, self.smoothing, self.max_log2_odds,
            self.shrinkage, self.inner_signature_folds, self.signature_random_state,
        )
        for column in oof:
            if column in transformed:
                transformed.loc[:, column] = oof[column].to_numpy()
        return transformed.astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return super().transform(self._build_features(features)).astype("float32")

    def summary(self) -> dict[str, int]:
        result = super().summary()
        result.update({
            "selected_gene_features": len(self.selected_gene_columns),
            "dropped_gene_features": len(self.dropped_gene_columns),
            "all_variant_signature_genes": len(self.mutation_signature_genes_),
            "functional_signature_genes": len(self.functional_signature_genes_),
            "consequence_summary_features": 16,
            "all_variant_signature_features": 2 * len(self.all_variant_weights_),
            "functional_signature_features": 2 * len(self.functional_weights_),
            "functional_hotspot_features": len(self.hotspots_),
        })
        return result

