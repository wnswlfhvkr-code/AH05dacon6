"""EMV46·JSJ9 통합 로직을 Outer 5-fold 실험에 적용한 pipeComb EM v2_001."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
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


def create_pattern_keys(features: pd.DataFrame) -> pd.Series:
    """결측과 WT를 동일하게 정규화한 전체 유전자 패턴 키를 만듭니다."""
    normalized = features.astype("string").fillna("WT").apply(
        lambda column: column.str.strip().str.upper()
    )
    hashes = pd.util.hash_pandas_object(normalized, index=False, categorize=True)
    return hashes.map(lambda value: f"{int(value):016x}")


def learn_pattern_uncertainty(
    features: pd.DataFrame,
    labels: pd.Series,
) -> dict[str, tuple[float, float, float, float]]:
    """학습 데이터의 동일 패턴 label support·entropy·충돌도를 학습합니다."""
    keys = create_pattern_keys(features).astype(str)
    frame = pd.DataFrame({
        "key": keys.to_numpy(),
        "label": labels.astype(str).to_numpy(),
    })
    class_count = max(int(labels.nunique()), 2)
    lookup: dict[str, tuple[float, float, float, float]] = {}
    for key, group in frame.groupby("key", sort=False):
        counts = group["label"].value_counts().to_numpy(dtype="float64")
        probabilities = counts / counts.sum()
        entropy = float(
            -(probabilities * np.log(probabilities)).sum() / np.log(class_count)
        )
        lookup[str(key)] = (
            float(np.log1p(len(group))),
            entropy,
            float(probabilities.max()),
            float(len(counts) > 1),
        )
    return lookup


def create_pattern_uncertainty_features(
    features: pd.DataFrame,
    lookup: dict[str, tuple[float, float, float, float]],
) -> pd.DataFrame:
    """학습 lookup에 없는 패턴은 0으로 두고 불확실성 4개를 생성합니다."""
    keys = create_pattern_keys(features).astype(str)
    values = keys.map(
        lambda key: lookup.get(str(key), (0.0, 0.0, 0.0, 0.0))
    )
    matrix = np.asarray(values.tolist(), dtype="float32")
    return pd.DataFrame({
        "pattern_label_support_log1p": matrix[:, 0],
        "pattern_label_entropy": matrix[:, 1],
        "pattern_label_max_probability": matrix[:, 2],
        "pattern_label_conflict": matrix[:, 3],
    }, index=features.index)


def create_oof_pattern_uncertainty(
    features: pd.DataFrame,
    labels: pd.Series,
    folds: int,
    random_state: int,
) -> pd.DataFrame:
    """각 학습 행의 정답을 제외한 inner-fold 패턴 불확실성을 만듭니다."""
    n_splits = min(folds, int(labels.value_counts().min()))
    if n_splits < 2:
        raise ValueError("OOF 패턴 불확실성에는 클래스별 표본이 최소 2개 필요합니다.")
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    fold_outputs: list[pd.DataFrame] = []
    for train_positions, valid_positions in splitter.split(features, labels):
        lookup = learn_pattern_uncertainty(
            features.iloc[train_positions], labels.iloc[train_positions]
        )
        fold_outputs.append(
            create_pattern_uncertainty_features(
                features.iloc[valid_positions], lookup
            )
        )
    return pd.concat(fold_outputs).reindex(features.index).astype("float32")


def build_mutation_matrices(
    features: pd.DataFrame,
    columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """severity·전체/기능 변이·multi-hit·결과 수·최대 token 수를 만듭니다."""
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"변환에 필요한 유전자 컬럼이 없습니다: {sorted(missing)}")
    rows, width = len(features), len(columns)
    severity_values = np.zeros((rows, width), dtype="int8")
    mutation_values = np.zeros((rows, width), dtype="int8")
    functional_values = np.zeros((rows, width), dtype="int8")
    multi_hit_values = np.zeros((rows, width), dtype="int8")
    category_counts = np.zeros((rows, len(CONSEQUENCE_NAMES)), dtype="int16")
    max_gene_token_counts = np.zeros(rows, dtype="int16")

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
            max_gene_token_counts[row_position] = max(
                max_gene_token_counts[row_position], len(mutations)
            )
            for consequence in consequences:
                category_counts[row_position, consequence - 1] += 1

    frame_arguments = {"index": features.index, "columns": columns, "dtype": "int8"}
    return (
        pd.DataFrame(severity_values, **frame_arguments),
        pd.DataFrame(mutation_values, **frame_arguments),
        pd.DataFrame(functional_values, **frame_arguments),
        pd.DataFrame(multi_hit_values, **frame_arguments),
        category_counts,
        max_gene_token_counts,
    )


def create_summary_features(
    mutation: pd.DataFrame,
    functional: pd.DataFrame,
    multi_hit: pd.DataFrame,
    category_counts: np.ndarray,
    max_gene_token_counts: np.ndarray,
) -> pd.DataFrame:
    """v16/v18의 중복 burden을 상세한 단일 summary 집합으로 통합합니다."""
    token_count = category_counts.sum(axis=1).astype("float32")
    denominator = np.maximum(token_count, 1.0)
    functional_token_count = category_counts[:, 1:].sum(axis=1).astype("float32")
    mutated_gene_count = mutation.sum(axis=1).to_numpy(dtype="float32")
    multi_hit_gene_count = multi_hit.sum(axis=1).to_numpy(dtype="float32")
    values: dict[str, np.ndarray] = {
        "mutated_gene_count_log1p": np.log1p(mutated_gene_count),
        "mutation_token_count_log1p": np.log1p(token_count),
        "functional_mutation_count_log1p": np.log1p(functional_token_count),
        "functional_gene_count_log1p": np.log1p(
            functional.sum(axis=1).to_numpy(dtype="float32")
        ),
        "multi_variant_gene_count_log1p": np.log1p(multi_hit_gene_count),
        "token_excess_count_log1p": np.log1p(
            np.maximum(token_count - mutated_gene_count, 0.0)
        ),
        "max_gene_token_count": max_gene_token_counts.astype("float32"),
        "multi_hit_gene_ratio": (
            multi_hit_gene_count / np.maximum(mutated_gene_count, 1.0)
        ),
        "functional_mutation_ratio": functional_token_count / denominator,
    }
    for position, name in enumerate(CONSEQUENCE_NAMES.values()):
        count = category_counts[:, position].astype("float32")
        values[f"consequence_{name}_count_log1p"] = np.log1p(count)
        values[f"consequence_{name}_ratio"] = count / denominator
    return pd.DataFrame(values, index=mutation.index, dtype="float32")


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


def create_signature_channel(
    matrix: pd.DataFrame,
    weights_by_class: dict[str, dict[str, float]],
    channel: str,
) -> pd.DataFrame:
    """두 채널이 충돌하지 않도록 이름을 구분해 weighted/match 피처를 만듭니다."""
    columns: dict[str, np.ndarray] = {}
    for class_name, weights in weights_by_class.items():
        genes = list(weights)
        values = matrix[genes].to_numpy(dtype="float32")
        vector = np.array([weights[gene] for gene in genes], dtype="float32")
        columns[f"signature_{channel}_{class_name}_weighted"] = values @ vector
        columns[f"signature_{channel}_{class_name}_match_count"] = values.sum(axis=1)
    return pd.DataFrame(columns, index=matrix.index, dtype="float32")


def create_dual_pair_contrasts(output: pd.DataFrame) -> pd.DataFrame:
    """V24 dual signature를 복제하지 않고 중첩 암종 쌍의 차이만 추가합니다."""
    columns: dict[str, pd.Series] = {}
    for channel in ("all", "functional"):
        for left, right in (("KIRC", "KIPAN"), ("LGG", "GBMLGG")):
            left_weighted = f"signature_{channel}_{left}_weighted"
            right_weighted = f"signature_{channel}_{right}_weighted"
            left_count = f"signature_{channel}_{left}_match_count"
            right_count = f"signature_{channel}_{right}_match_count"
            if left_weighted in output and right_weighted in output:
                columns[f"pair_{channel}_{left}_{right}_weighted_diff"] = (
                    output[left_weighted] - output[right_weighted]
                )
                columns[f"pair_{channel}_{left}_{right}_count_diff"] = (
                    output[left_count] - output[right_count]
                )
    return pd.DataFrame(columns, index=output.index, dtype="float32")


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
    fold_outputs: list[pd.DataFrame] = []
    for train_positions, valid_positions in splitter.split(mutation, aligned):
        fold_labels = aligned.iloc[train_positions]
        channel_outputs: list[pd.DataFrame] = []
        for channel, source in (("all", mutation), ("functional", functional)):
            weights = learn_class_weights(
                source.iloc[train_positions], fold_labels, top_genes_per_class,
                smoothing, max_log2_odds, shrinkage, channel,
            )
            channel_outputs.append(
                create_signature_channel(
                    source.iloc[valid_positions], weights, channel
                )
            )
        fold_outputs.append(pd.concat(channel_outputs, axis=1))
    return pd.concat(fold_outputs).reindex(mutation.index).astype("float32")


class _IntegratedEMV45PreprocessingPipeline(PreprocessingPipeline):
    """V24 dual channel에 OOF 패턴 불확실성·고유 multi-hit·pair contrast를 결합합니다."""

    name = "em_v45"
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
        self.pattern_uncertainty_: dict[
            str, tuple[float, float, float, float]
        ] = {}
        self.steps = (
            "변이 토큰 단일 분리 및 중복 제거",
            "전체 변이·기능 변이 채널 분리",
            "채널별 최소 빈도와 유전자 합집합",
            "단일 consequence summary와 기능 hotspot",
            "공통 inner-fold OOF dual signature",
            "inner-fold OOF 동일 패턴 label 불확실성",
            "중복 제거된 multi-hit token complexity",
            "dual signature 기반 중첩 암종 pair contrast",
        )

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        (
            severity,
            mutation,
            functional,
            multi_hit,
            category_counts,
            max_gene_token_counts,
        ) = build_mutation_matrices(features, self.selected_gene_columns)
        output = pd.concat(
            [
                severity.astype("float32"),
                create_summary_features(
                    mutation,
                    functional,
                    multi_hit,
                    category_counts,
                    max_gene_token_counts,
                ),
                create_signature_channel(
                    mutation[self.mutation_signature_genes_],
                    self.all_variant_weights_,
                    "all",
                ),
                create_signature_channel(
                    functional[self.functional_signature_genes_],
                    self.functional_weights_,
                    "functional",
                ),
            ],
            axis=1,
        )
        return pd.concat(
            [
                output,
                create_dual_pair_contrasts(output),
                create_hotspot_features(features, self.hotspots_),
                create_pattern_uncertainty_features(
                    features, self.pattern_uncertainty_
                ),
            ],
            axis=1,
        )

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "_IntegratedEMV45PreprocessingPipeline":
        columns = features.columns.tolist()
        _, mutation, functional, _, _, _ = build_mutation_matrices(features, columns)
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

        _, selected_mutation, selected_functional, _, _, _ = build_mutation_matrices(
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
        self.pattern_uncertainty_ = learn_pattern_uncertainty(features, labels)
        super().fit(self._build_features(features), labels)
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        self.fit(features, labels)
        transformed = self.transform(features)
        _, mutation, functional, _, _, _ = build_mutation_matrices(
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
        contrasts = create_dual_pair_contrasts(transformed)
        transformed.loc[:, contrasts.columns] = contrasts.to_numpy()
        oof_pattern = create_oof_pattern_uncertainty(
            features,
            labels,
            self.inner_signature_folds,
            self.signature_random_state,
        )
        for column in oof_pattern:
            if column in transformed:
                transformed.loc[:, column] = oof_pattern[column].to_numpy()
        return transformed.astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return super().transform(self._build_features(features)).astype("float32")

    def summary(self) -> dict[str, int]:
        pair_contrast_count = 0
        for weights in (self.all_variant_weights_, self.functional_weights_):
            for left, right in (("KIRC", "KIPAN"), ("LGG", "GBMLGG")):
                if left in weights and right in weights:
                    pair_contrast_count += 2
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
            "pattern_uncertainty_features": 4,
            "unique_multi_hit_features": 3,
            "pair_contrast_features": pair_contrast_count,
        })
        return result



FEATURE_NAMES = tuple(f"F{index:02d}" for index in range(1, 20))
SIGNATURE_FEATURE_NAMES = ("F16", "F17", "F18", "F19")
_SPLICE = re.compile(r"SPLICE|SPL_|IVS|(?:[+-]\d+)", re.IGNORECASE)
_FRAMESHIFT = FRAMESHIFT_PATTERN
_NONSENSE = re.compile(r"\*|TER$|STOP", re.IGNORECASE)
_MISSENSE = re.compile(r"^([A-Z*])\d+([A-Z*])$", re.IGNORECASE)


@dataclass
class _SharedMatrices:
    event_counts: pd.DataFrame
    consequence_event_summary: pd.DataFrame
    consequence_presence: dict[str, pd.DataFrame]


@dataclass
class _SignatureState:
    selected_genes: list[str]
    class_names: list[str]
    weights: np.ndarray
    profiles: np.ndarray
    top_genes_per_class: int
    smoothing: float
    shrinkage: float
    max_log2_odds: float
    folds: int
    temperature: float
    random_state: int


def _tokens(value: object) -> tuple[str, ...]:
    """내장 EMV45의 캐시된 토큰 정규화 결과를 공통으로 재사용합니다."""
    if pd.isna(value):
        return ()
    return tuple(token for token, _ in split_unique_mutations(str(value)))


def _consequence_values(
    tokens: tuple[str, ...],
) -> tuple[int, int, int, int, bool, bool, bool]:
    """F04 이벤트 수와 F13~F15 유전자 여부를 한 번에 판정합니다."""
    event_counts = [0, 0, 0, 0]
    truncating = False
    gene_missense = False
    gene_splice = False
    for token in tokens:
        is_splice = bool(_SPLICE.search(token))
        is_frameshift = bool(_FRAMESHIFT.search(token))
        is_nonsense = bool(_NONSENSE.search(token))
        match = _MISSENSE.fullmatch(token)
        is_missense = bool(
            match and match.group(1) != match.group(2) and not is_nonsense
        )
        truncating = truncating or is_frameshift or is_nonsense
        gene_missense = gene_missense or is_missense
        gene_splice = gene_splice or is_splice
        if is_splice:
            event_counts[3] += 1
        elif is_frameshift:
            event_counts[2] += 1
        elif is_nonsense:
            event_counts[1] += 1
        else:
            if is_missense:
                event_counts[0] += 1
    return (*event_counts, truncating, gene_missense, gene_splice)


def _shared_matrices(
    features: pd.DataFrame,
    gene_columns: list[str],
) -> _SharedMatrices:
    missing = set(gene_columns) - set(features.columns)
    if missing:
        raise ValueError(f"누락된 유전자 컬럼이 있습니다: {sorted(missing)}")
    event_data: dict[str, pd.Series] = {}
    consequence_event_summary = pd.DataFrame(
        0,
        index=features.index,
        columns=("missense", "nonsense", "frameshift", "splice"),
        dtype="int32",
    )
    consequence_presence_data: dict[str, dict[str, pd.Series]] = {
        name: {} for name in ("truncating", "missense", "splice")
    }
    for gene in gene_columns:
        parsed = features[gene].map(_tokens)
        event_data[gene] = parsed.map(len).astype("int16")
        classified = parsed.map(_consequence_values)
        for position, name in enumerate(consequence_event_summary.columns):
            consequence_event_summary[name] += classified.map(
                lambda values, index=position: values[index]
            ).astype("int32")
        for position, name in enumerate(consequence_presence_data, start=4):
            consequence_presence_data[name][gene] = classified.map(
                lambda values, index=position: values[index]
            ).astype("int8")
    return _SharedMatrices(
        event_counts=pd.DataFrame(event_data, index=features.index),
        consequence_event_summary=consequence_event_summary,
        consequence_presence={
            name: pd.DataFrame(values, index=features.index)
            for name, values in consequence_presence_data.items()
        },
    )


def _learn_hotspots(
    features: pd.DataFrame,
    gene_columns: list[str],
    minimum_count: int,
    maximum_hotspots: int,
) -> list[tuple[str, str]]:
    support: Counter[tuple[str, str]] = Counter()
    for gene in gene_columns:
        for value in features[gene]:
            for token in _tokens(value):
                support[(gene, token)] += 1
    eligible = [pair for pair, count in support.items() if count >= minimum_count]
    return sorted(eligible, key=lambda pair: (-support[pair], pair))[
        :maximum_hotspots
    ]


def _hotspot_outputs(
    features: pd.DataFrame,
    hotspots: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = list(dict.fromkeys(gene for gene, _ in hotspots))
    token_cache = {gene: features[gene].map(_tokens) for gene in genes}
    total = pd.Series(0, index=features.index, dtype="int32")
    grouped: dict[str, set[str]] = {}
    for gene, token in hotspots:
        total += token_cache[gene].map(lambda values: token in values).astype("int32")
        grouped.setdefault(gene, set()).add(token)
    per_gene = pd.DataFrame(
        {
            f"gene_hotspot__{gene}": token_cache[gene].map(
                lambda values, selected=frozenset(tokens): not selected.isdisjoint(values)
            ).astype("int8")
            for gene, tokens in grouped.items()
        },
        index=features.index,
    )
    return (
        pd.DataFrame({"hotspot_mutation_count": total}, index=features.index),
        per_gene,
    )


def _learn_signature_arrays(
    mutation: pd.DataFrame,
    labels: pd.Series,
    class_names: list[str],
    top_genes_per_class: int,
    smoothing: float,
    shrinkage: float,
    max_log2_odds: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = mutation.to_numpy(dtype="float64")
    label_values = labels.astype(str).to_numpy()
    weights = np.zeros((len(class_names), matrix.shape[1]), dtype="float64")
    profiles = np.zeros_like(weights)
    for class_index, class_name in enumerate(class_names):
        inside = label_values == class_name
        outside = ~inside
        inside_count = matrix[inside].sum(axis=0)
        outside_count = matrix[outside].sum(axis=0)
        inside_rate = (inside_count + smoothing) / (
            inside.sum() + 2.0 * smoothing
        )
        outside_rate = (outside_count + smoothing) / (
            outside.sum() + 2.0 * smoothing
        )
        log_odds = np.log2(
            (inside_rate / np.maximum(1.0 - inside_rate, 1e-12))
            / (outside_rate / np.maximum(1.0 - outside_rate, 1e-12))
        )
        stabilized = np.clip(log_odds, -max_log2_odds, max_log2_odds)
        stabilized *= inside_count / (inside_count + shrinkage)
        ranked = np.argsort(stabilized)[::-1][:top_genes_per_class]
        positive = ranked[stabilized[ranked] > 0]
        weights[class_index, positive] = stabilized[positive]
        profiles[class_index] = inside_rate
    return weights, profiles


def _signature_outputs(
    mutation: pd.DataFrame,
    class_names: list[str],
    weights: np.ndarray,
    profiles: np.ndarray,
    temperature: float,
) -> dict[str, pd.DataFrame]:
    matrix = mutation.to_numpy(dtype="float64")
    weighted = matrix @ weights.T
    sample_norm = np.sqrt((matrix * matrix).sum(axis=1, keepdims=True))
    profile_norm = np.sqrt((profiles * profiles).sum(axis=1))[None, :]
    similarity = (matrix @ profiles.T) / np.maximum(
        sample_norm * profile_norm, 1e-12
    )
    if similarity.shape[1] >= 2:
        top_two = np.sort(
            np.partition(similarity, -2, axis=1)[:, -2:], axis=1
        )
        margin = top_two[:, 1] - top_two[:, 0]
    else:
        margin = similarity[:, 0]
    logits = similarity / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
    entropy /= np.log(max(similarity.shape[1], 2))
    return {
        "F16": pd.DataFrame(
            weighted,
            index=mutation.index,
            columns=[f"signature_weighted__{name}" for name in class_names],
            dtype="float32",
        ),
        "F17": pd.DataFrame(
            similarity,
            index=mutation.index,
            columns=[f"signature_similarity__{name}" for name in class_names],
            dtype="float32",
        ),
        "F18": pd.DataFrame(
            {"signature_top2_margin": margin.astype("float32")},
            index=mutation.index,
        ),
        "F19": pd.DataFrame(
            {"signature_entropy": entropy.astype("float32")},
            index=mutation.index,
        ),
    }


class _IntegratedEMV46PreprocessingPipeline(PreprocessingPipeline):
    """내장 EMV45 한 번과 통합 F01~F19 계산 한 번을 결합합니다."""

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
        unknown = set(overrides) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"지원하지 않는 F20 구성 피처입니다: {sorted(unknown)}")
        self.base_parameters = dict(parameters)
        self.feature_configs = {
            name: {**parameters, **overrides.get(name, {})}
            for name in FEATURE_NAMES
        }
        self.v45_pipeline = _IntegratedEMV45PreprocessingPipeline(**parameters)
        self.gene_columns_: list[str] = []
        self.selected_genes_: dict[str, list[str]] = {}
        self.hotspot_states_: dict[tuple[int, int], list[tuple[str, str]]] = {}
        self.feature_hotspot_keys_: dict[str, tuple[int, int]] = {}
        self.signature_states_: dict[tuple[object, ...], _SignatureState] = {}
        self.feature_signature_keys_: dict[str, tuple[object, ...]] = {}
        self.derived_columns_: dict[str, list[str]] = {
            name: [] for name in FEATURE_NAMES
        }
        self.v45_feature_count_ = 0
        self.derived_feature_counts_: dict[str, int] = {
            name: 0 for name in FEATURE_NAMES
        }
        self.steps = (
            "EMV45 베이스 피처 1회 생성",
            "공통 토큰 수·consequence 행렬 1회 계산",
            "F05·F12 공통 hotspot 상태 재사용",
            "F16~F19 공통 inner-fold OOF signature 계산 재사용",
            "F01~F19 접두사로 컬럼 충돌 제거",
            "원본 SUBCLASS 유지",
        )

    def _int_config(self, feature: str, key: str, default: int) -> int:
        value = int(self.feature_configs[feature].get(key, default))
        if value < 1:
            raise ValueError(f"{feature}.{key}는 1 이상이어야 합니다.")
        return value

    def _learn_feature_states(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        shared: _SharedMatrices,
    ) -> None:
        mutated_support = shared.event_counts.gt(0).sum(axis=0)
        multi_support = shared.event_counts.gt(1).sum(axis=0)
        self.selected_genes_ = {}
        for feature in ("F09", "F10"):
            minimum = self._int_config(feature, "min_feature_support", 2)
            self.selected_genes_[feature] = mutated_support[
                mutated_support >= minimum
            ].index.tolist()
        minimum = self._int_config("F11", "min_feature_support", 2)
        self.selected_genes_["F11"] = multi_support[
            multi_support >= minimum
        ].index.tolist()
        consequence_map = {
            "F13": shared.consequence_presence["truncating"].gt(0),
            "F14": shared.consequence_presence["missense"].gt(0),
            "F15": shared.consequence_presence["splice"].gt(0),
        }
        for feature, matrix in consequence_map.items():
            minimum = self._int_config(feature, "min_feature_support", 2)
            support = matrix.sum(axis=0)
            self.selected_genes_[feature] = support[
                support >= minimum
            ].index.tolist()
            if not self.selected_genes_[feature]:
                raise ValueError(f"{feature} 지지도를 만족하는 유전자가 없습니다.")

        self.hotspot_states_ = {}
        self.feature_hotspot_keys_ = {}
        for feature in ("F05", "F12"):
            minimum = self._int_config(feature, "min_hotspot_count", 5)
            maximum = self._int_config(feature, "max_hotspots", 384)
            if minimum < 2:
                raise ValueError(f"{feature}.min_hotspot_count는 2 이상이어야 합니다.")
            key = (minimum, maximum)
            self.feature_hotspot_keys_[feature] = key
            if key not in self.hotspot_states_:
                self.hotspot_states_[key] = _learn_hotspots(
                    features, self.gene_columns_, minimum, maximum
                )
        if not self.hotspot_states_[self.feature_hotspot_keys_["F12"]]:
            raise ValueError("F12 최소 지지도를 만족하는 hotspot이 없습니다.")

        presence = shared.event_counts.gt(0).astype("int8")
        aligned_labels = pd.Series(
            labels.astype(str).to_numpy(), index=features.index
        )
        class_names = sorted(aligned_labels.unique().tolist())
        self.signature_states_ = {}
        self.feature_signature_keys_ = {}
        for feature in SIGNATURE_FEATURE_NAMES:
            config = self.feature_configs[feature]
            key = (
                int(config.get("min_mutation_count", 5)),
                int(config.get("top_genes_per_class", 20)),
                float(config.get("smoothing", 0.5)),
                float(config.get("shrinkage", 10.0)),
                float(config.get("max_log2_odds", 8.0)),
                int(config.get("inner_signature_folds", 5)),
                float(config.get("signature_temperature", 1.0)),
                int(config.get("random_state", 42)),
            )
            self.feature_signature_keys_[feature] = key
            if key in self.signature_states_:
                continue
            (
                minimum,
                top_genes,
                smoothing,
                shrinkage,
                max_odds,
                folds,
                temperature,
                random_state,
            ) = key
            if minimum < 1 or top_genes < 1 or folds < 2:
                raise ValueError("signature 빈도·유전자 수·fold 설정이 잘못되었습니다.")
            if min(smoothing, shrinkage, max_odds, temperature) <= 0:
                raise ValueError("signature 안정화 파라미터는 0보다 커야 합니다.")
            support = presence.sum(axis=0)
            selected = support[support >= minimum].index.tolist()
            if not selected:
                raise ValueError("signature 최소 변이 빈도를 만족하는 유전자가 없습니다.")
            mutation = presence[selected]
            weights, profiles = _learn_signature_arrays(
                mutation,
                aligned_labels,
                class_names,
                top_genes,
                smoothing,
                shrinkage,
                max_odds,
            )
            self.signature_states_[key] = _SignatureState(
                selected_genes=selected,
                class_names=class_names,
                weights=weights,
                profiles=profiles,
                top_genes_per_class=top_genes,
                smoothing=smoothing,
                shrinkage=shrinkage,
                max_log2_odds=max_odds,
                folds=folds,
                temperature=temperature,
                random_state=random_state,
            )

    def _non_signature_features(
        self,
        features: pd.DataFrame,
        shared: _SharedMatrices,
    ) -> dict[str, pd.DataFrame]:
        counts = shared.event_counts
        total = counts.sum(axis=1).astype("float32")
        active = counts.gt(0).sum(axis=1).astype("float32")
        matrix = counts.to_numpy(dtype="float64")
        probabilities = matrix / np.maximum(total.to_numpy(), 1.0)[:, None]
        entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
        normalized_entropy = entropy / np.maximum(
            np.log(np.maximum(active.to_numpy(), 2)), 1e-12
        )
        consequence = shared.consequence_presence
        hotspot_cache = {
            key: _hotspot_outputs(features, hotspots)
            for key, hotspots in self.hotspot_states_.items()
        }
        selected09 = self.selected_genes_["F09"]
        selected10 = self.selected_genes_["F10"]
        selected11 = self.selected_genes_["F11"]
        selected13 = self.selected_genes_["F13"]
        selected14 = self.selected_genes_["F14"]
        selected15 = self.selected_genes_["F15"]
        return {
            "F01": pd.DataFrame(
                {"mutated_gene_count": active}, index=features.index
            ),
            "F02": pd.DataFrame(
                {"mutation_event_count": total}, index=features.index
            ),
            "F03": pd.DataFrame(
                {"mutation_burden_log1p": np.log1p(total).astype("float32")},
                index=features.index,
            ),
            "F04": pd.DataFrame(
                {
                    f"{name}_event_count": shared.consequence_event_summary[name]
                    for name in ("missense", "nonsense", "frameshift", "splice")
                },
                index=features.index,
            ),
            "F05": hotspot_cache[self.feature_hotspot_keys_["F05"]][0],
            "F06": pd.DataFrame(
                {"multi_hit_gene_count": counts.gt(1).sum(axis=1)},
                index=features.index,
            ),
            "F07": pd.DataFrame(
                {"no_mutation_sample": total.eq(0).astype("int8")},
                index=features.index,
            ),
            "F08": pd.DataFrame(
                {
                    "mutation_event_concentration": matrix.max(axis=1)
                    / np.maximum(total.to_numpy(), 1.0),
                    "mutation_gene_entropy": normalized_entropy,
                    "mutated_gene_diversity_ratio": active
                    / max(len(self.gene_columns_), 1),
                },
                index=features.index,
                dtype="float32",
            ),
            "F09": pd.DataFrame(
                {
                    f"gene_mutated__{gene}": counts[gene].gt(0).astype("int8")
                    for gene in selected09
                },
                index=features.index,
            ),
            "F10": pd.DataFrame(
                {
                    f"gene_mutation_count__{gene}": counts[gene].astype("int16")
                    for gene in selected10
                },
                index=features.index,
            ),
            "F11": pd.DataFrame(
                {
                    f"gene_multi_hit__{gene}": counts[gene].gt(1).astype("int8")
                    for gene in selected11
                },
                index=features.index,
            ),
            "F12": hotspot_cache[self.feature_hotspot_keys_["F12"]][1],
            "F13": consequence["truncating"][selected13].rename(
                columns=lambda gene: f"gene_truncating__{gene}"
            ).astype("int8"),
            "F14": consequence["missense"][selected14].gt(0).rename(
                columns=lambda gene: f"gene_missense__{gene}"
            ).astype("int8"),
            "F15": consequence["splice"][selected15].gt(0).rename(
                columns=lambda gene: f"gene_splice__{gene}"
            ).astype("int8"),
        }

    def _full_signature_features(
        self,
        shared: _SharedMatrices,
    ) -> dict[str, pd.DataFrame]:
        presence = shared.event_counts.gt(0).astype("int8")
        cache: dict[tuple[object, ...], dict[str, pd.DataFrame]] = {}
        for key, state in self.signature_states_.items():
            cache[key] = _signature_outputs(
                presence[state.selected_genes],
                state.class_names,
                state.weights,
                state.profiles,
                state.temperature,
            )
        return {
            feature: cache[self.feature_signature_keys_[feature]][feature]
            for feature in SIGNATURE_FEATURE_NAMES
        }

    def _oof_signature_features(
        self,
        shared: _SharedMatrices,
        labels: pd.Series,
    ) -> dict[str, pd.DataFrame]:
        presence = shared.event_counts.gt(0).astype("int8")
        aligned_labels = pd.Series(
            labels.astype(str).to_numpy(), index=presence.index
        )
        cache: dict[tuple[object, ...], dict[str, pd.DataFrame]] = {}
        for key, state in self.signature_states_.items():
            minimum_class = int(aligned_labels.value_counts().min())
            folds = min(state.folds, minimum_class)
            if folds < 2:
                raise ValueError("OOF signature에는 클래스별 표본이 최소 2개 필요합니다.")
            mutation = presence[state.selected_genes]
            fold_outputs: dict[str, list[pd.DataFrame]] = {
                feature: []
                for feature in SIGNATURE_FEATURE_NAMES
            }
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=state.random_state,
            )
            for train_positions, valid_positions in splitter.split(
                mutation, aligned_labels
            ):
                weights, profiles = _learn_signature_arrays(
                    mutation.iloc[train_positions],
                    aligned_labels.iloc[train_positions],
                    state.class_names,
                    state.top_genes_per_class,
                    state.smoothing,
                    state.shrinkage,
                    state.max_log2_odds,
                )
                fold_output = _signature_outputs(
                    mutation.iloc[valid_positions],
                    state.class_names,
                    weights,
                    profiles,
                    state.temperature,
                )
                for feature, frame in fold_output.items():
                    fold_outputs[feature].append(frame)
            cache[key] = {
                feature: pd.concat(frames)
                .reindex(presence.index)
                .astype("float32")
                for feature, frames in fold_outputs.items()
            }
        return {
            feature: cache[self.feature_signature_keys_[feature]][feature]
            for feature in SIGNATURE_FEATURE_NAMES
        }

    def _derived_features(
        self,
        features: pd.DataFrame,
        shared: _SharedMatrices,
        labels: pd.Series | None = None,
        use_oof: bool = False,
    ) -> dict[str, pd.DataFrame]:
        derived = self._non_signature_features(features, shared)
        if use_oof:
            if labels is None:
                raise ValueError("OOF signature 생성에는 labels가 필요합니다.")
            derived.update(self._oof_signature_features(shared, labels))
        else:
            derived.update(self._full_signature_features(shared))
        return derived

    def _learn_derived_columns(
        self,
        derived: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        filtered: dict[str, pd.DataFrame] = {}
        for feature in FEATURE_NAMES:
            frame = derived[feature]
            columns = frame.columns[frame.nunique(dropna=False) > 1].tolist()
            self.derived_columns_[feature] = columns
            filtered[feature] = frame[columns]
        return filtered

    def _select_derived_columns(
        self,
        derived: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        return {
            feature: derived[feature][self.derived_columns_[feature]]
            for feature in FEATURE_NAMES
        }

    @staticmethod
    def _combine(
        v45: pd.DataFrame,
        derived: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        parts = [v45]
        for feature in FEATURE_NAMES:
            frame = derived[feature].copy()
            frame.columns = [f"{feature}__{column}" for column in frame.columns]
            parts.append(frame)
        return pd.concat(parts, axis=1)

    def _update_counts(
        self,
        v45: pd.DataFrame,
        derived: dict[str, pd.DataFrame],
    ) -> None:
        self.v45_feature_count_ = v45.shape[1]
        self.derived_feature_counts_ = {
            feature: derived[feature].shape[1] for feature in FEATURE_NAMES
        }

    def _fit_integrated_state(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> _SharedMatrices:
        self.gene_columns_ = features.columns.tolist()
        shared = _shared_matrices(features, self.gene_columns_)
        self._learn_feature_states(features, labels, shared)
        return shared

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "_IntegratedEMV46PreprocessingPipeline":
        self.v45_pipeline.fit(features, labels)
        shared = self._fit_integrated_state(features, labels)
        v45 = self.v45_pipeline.transform(features)
        derived = self._learn_derived_columns(
            self._derived_features(features, shared)
        )
        self._update_counts(v45, derived)
        PreprocessingPipeline.fit(self, self._combine(v45, derived), labels)
        return self

    def fit_transform(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        v45 = self.v45_pipeline.fit_transform(features, labels)
        shared = self._fit_integrated_state(features, labels)
        derived = self._learn_derived_columns(
            self._derived_features(features, shared, labels, use_oof=True)
        )
        self._update_counts(v45, derived)
        combined = self._combine(v45, derived)
        PreprocessingPipeline.fit(self, combined, labels)
        return PreprocessingPipeline.transform(self, combined).astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        shared = _shared_matrices(features, self.gene_columns_)
        derived = self._select_derived_columns(
            self._derived_features(features, shared)
        )
        combined = self._combine(
            self.v45_pipeline.transform(features),
            derived,
        )
        return PreprocessingPipeline.transform(self, combined).astype("float32")

    def summary(self) -> dict[str, int]:
        result = PreprocessingPipeline.summary(self)
        result.update({
            "v45_base_features": self.v45_feature_count_,
            "f01_f19_derived_features": sum(self.derived_feature_counts_.values()),
            "included_feature_groups": len(self.derived_feature_counts_),
            "shared_hotspot_states": len(self.hotspot_states_),
            "shared_signature_states": len(self.signature_states_),
            "combined_before_constant_filter": (
                self.v45_feature_count_ + sum(self.derived_feature_counts_.values())
            ),
        })
        return result

@dataclass
class TextTreeFeatureBundle:
    """LinearSVC 텍스트 뷰와 트리 모델 수치 뷰를 함께 전달합니다."""

    text: csr_matrix
    tree: csr_matrix


def _jsj_mutation_type(value: str) -> str:
    """JSJ9 텍스트 vocabulary와 호환되는 변이 유형을 반환합니다."""
    lowered = value.lower()
    if "delins" in lowered:
        return "DELINS"
    if "fs" in lowered:
        return "FS"
    if "del" in lowered:
        return "DEL"
    if "ins" in lowered:
        return "INS"
    if "dup" in lowered:
        return "DUP"
    if "*" in value or "stop" in lowered:
        return "STOP"
    if ">" in value:
        return "SUB"
    return "OTHER"


def _jsj_position_bin(position: int) -> str:
    if position <= 50:
        return "001_050"
    if position <= 100:
        return "051_100"
    if position <= 250:
        return "101_250"
    if position <= 500:
        return "251_500"
    return "501_PLUS"


class _JSJ9TextOnlyEngine:
    """JSJ9에서 실제 앙상블에 쓰이는 Word·Char TF-IDF만 생성합니다.

    기존 JSJ9 delegate는 이후 폐기되는 gene-mutated/structural tree 행렬도
    계산했습니다. 이 엔진은 같은 텍스트 token 규칙과 vocabulary를 유지하면서
    문서 생성과 두 vectorizer 계산만 수행합니다.
    """

    _event_pattern = re.compile(r"^([A-Za-z*]+)(\d+)([A-Za-z*]+)$")

    def __init__(
        self,
        word_ngram_range: list[int] | tuple[int, int] = (1, 2),
        word_min_df: int = 2,
        word_max_features: int = 250_000,
        char_ngram_range: list[int] | tuple[int, int] = (3, 5),
        char_min_df: int = 3,
        char_max_features: int = 180_000,
        char_weight: float = 0.5,
        sublinear_tf: bool = True,
        split_multi_event: bool = False,
        **_: object,
    ) -> None:
        self.char_weight = float(char_weight)
        self.split_multi_event = bool(split_multi_event)
        self.feature_columns: list[str] = []
        self.word_vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            ngram_range=tuple(word_ngram_range),
            min_df=int(word_min_df),
            max_features=int(word_max_features),
            sublinear_tf=bool(sublinear_tf),
            dtype=np.float32,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tuple(char_ngram_range),
            min_df=int(char_min_df),
            max_features=int(char_max_features),
            sublinear_tf=bool(sublinear_tf),
            dtype=np.float32,
        )

    def _validate_columns(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"누락된 피처가 있습니다: {sorted(missing)}")
        return features[self.feature_columns]

    def _documents(self, features: pd.DataFrame) -> np.ndarray:
        values = features.fillna("WT").astype(str).to_numpy()
        documents: list[str] = []
        columns = self.feature_columns
        split_multi_event = self.split_multi_event
        event_pattern = self._event_pattern

        for row in values:
            tokens: list[str] = []
            for gene_index, raw_value in enumerate(row):
                value = raw_value.strip()
                if not value or value.upper() == "WT":
                    continue
                events = value.split() if split_multi_event else [value.replace(" ", "")]
                gene = columns[gene_index]
                for event in events:
                    if event.startswith("p."):
                        event = event[2:]
                    event_type = _jsj_mutation_type(event)
                    tokens.extend((
                        f"G={gene}",
                        f"GT={gene}:{event_type}",
                        f"E={gene}:{event}",
                        f"T={event_type}",
                    ))
                    match = event_pattern.match(event)
                    if match:
                        reference, position, alternate = match.groups()
                        bin_name = _jsj_position_bin(int(position))
                        tokens.extend((
                            f"AA={gene}:{reference}>{alternate}",
                            f"AAG={reference}>{alternate}",
                            f"POS={gene}:{bin_name}",
                            f"POSG={bin_name}",
                        ))
            documents.append(" ".join(tokens) if tokens else "NO_MUTATION")
        return np.asarray(documents, dtype=object)

    def _combine_text(self, documents: np.ndarray, *, fit: bool) -> csr_matrix:
        if fit:
            word = self.word_vectorizer.fit_transform(documents)
            char = self.char_vectorizer.fit_transform(documents)
        else:
            word = self.word_vectorizer.transform(documents)
            char = self.char_vectorizer.transform(documents)
        return hstack((word, char * self.char_weight), format="csr")

    def fit(self, features: pd.DataFrame) -> "_JSJ9TextOnlyEngine":
        self.feature_columns = features.columns.tolist()
        documents = self._documents(features)
        self.word_vectorizer.fit(documents)
        self.char_vectorizer.fit(documents)
        return self

    def fit_transform(self, features: pd.DataFrame) -> csr_matrix:
        self.feature_columns = features.columns.tolist()
        return self._combine_text(self._documents(features), fit=True)

    def transform(self, features: pd.DataFrame) -> csr_matrix:
        selected = self._validate_columns(features)
        return self._combine_text(self._documents(selected), fit=False)

    def feature_count(self) -> int:
        return len(self.word_vectorizer.vocabulary_) + len(
            self.char_vectorizer.vocabulary_
        )


class PipeCombEMV2001PreprocessingPipeline(PreprocessingPipeline):
    """통합 EMV46 수치 뷰와 JSJ9 TF-IDF 텍스트 뷰를 반환합니다."""

    name = "pipeComb_em_v2_001"
    # v2와 달리 Outer 5-fold 평가를 유지하는 비교 실험입니다.
    evaluation_folds = 5

    def __init__(
        self,
        emv46_parameters: dict[str, object] | None = None,
        text_parameters: dict[str, object] | None = None,
        **parameters: object,
    ) -> None:
        super().__init__()
        em_config = dict(parameters)
        em_config.update(emv46_parameters or {})
        text_config = dict(parameters)
        text_config.update(text_parameters or {})
        self.emv46_pipeline = _IntegratedEMV46PreprocessingPipeline(**em_config)
        self.jsj9_text_engine = _JSJ9TextOnlyEngine(**text_config)
        self.emv46_feature_count_ = 0
        self.jsj9_text_feature_count_ = 0
        self.combined_feature_count_ = 0
        self.steps = (
            "내장 EMV46 수치 피처 1회 생성",
            "JSJ9 Word·Char TF-IDF 문서·행렬 1회 생성",
            "중복 JSJ9 tree 생성 경로 제거",
            "LinearSVC text·XGBoost/LightGBM tree 뷰 분리",
            "Outer 5-fold·Inner 5-fold 중첩 OOF stacking 입력 제공",
            "fold-train에서만 모든 상태 fit",
            "원본 SUBCLASS 유지",
        )

    def _update_counts(
        self,
        emv46_count: int | None = None,
        text_count: int | None = None,
    ) -> None:
        if emv46_count is not None:
            self.emv46_feature_count_ = int(emv46_count)
        if text_count is not None:
            self.jsj9_text_feature_count_ = int(text_count)
        self.combined_feature_count_ = (
            self.emv46_feature_count_ + self.jsj9_text_feature_count_
        )

    def _bundle(
        self,
        emv46_features: pd.DataFrame,
        text_features: csr_matrix,
    ) -> TextTreeFeatureBundle:
        tree = csr_matrix(emv46_features.to_numpy(dtype="float32", copy=False))
        text = text_features.tocsr()
        self._update_counts(tree.shape[1], text.shape[1])
        return TextTreeFeatureBundle(text=text, tree=tree)

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "PipeCombEMV2001PreprocessingPipeline":
        self.feature_columns = features.columns.tolist()
        self.emv46_pipeline.fit(features, labels)
        self.jsj9_text_engine.fit(features)
        self._update_counts(
            self.emv46_pipeline.summary()["remaining_features"],
            self.jsj9_text_engine.feature_count(),
        )
        self.label_encoder.fit(labels)
        return self

    def fit_transform(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> TextTreeFeatureBundle:
        self.feature_columns = features.columns.tolist()
        emv46 = self.emv46_pipeline.fit_transform(features, labels)
        text = self.jsj9_text_engine.fit_transform(features)
        self.label_encoder.fit(labels)
        return self._bundle(emv46, text)

    def transform(self, features: pd.DataFrame) -> TextTreeFeatureBundle:
        return self._bundle(
            self.emv46_pipeline.transform(features),
            self.jsj9_text_engine.transform(features),
        )

    def summary(self) -> dict[str, int]:
        return {
            "remaining_features": self.combined_feature_count_,
            "emv46_features": self.emv46_feature_count_,
            "jsj9_text_features": self.jsj9_text_feature_count_,
            "removed_duplicate_jsj9_tree_features": 0,
            "jsj9_tree_generation_skipped": 1,
            "dropped_constant_features": int(
                self.emv46_pipeline.summary().get("dropped_constant_features", 0)
            ),
        }
