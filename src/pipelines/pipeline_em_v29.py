"""em_v16과 em_v19를 중복 없이 결합한 EM v29."""

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
def classify_token(token: str) -> int:
    """단백질 변이를 WT=0부터 frameshift=5까지 분류합니다."""
    token = token.strip().upper()
    if not token or token in {"WT", "<NA>"}:
        return 0
    if SYNONYMOUS_PATTERN.fullmatch(token):
        return SYNONYMOUS
    if FRAMESHIFT_PATTERN.search(token):
        return FRAMESHIFT
    if STOP_PATTERN.search(token):
        return NONSENSE
    if INFRAME_PATTERN.search(token):
        return INFRAME
    match = MISSENSE_PATTERN.fullmatch(token)
    if match and match.group(1) != match.group(3):
        return MISSENSE
    return MISSENSE


@lru_cache(maxsize=None)
def split_mutations(value: str) -> tuple[tuple[str, int], ...]:
    """셀 안의 변이 토큰을 분리하고 중복을 제거합니다."""
    value = value.strip().upper()
    if not value or value in {"WT", "<NA>"}:
        return ()
    return tuple(
        (token, classify_token(token))
        for token in sorted(set(value.split()))
        if classify_token(token) != 0
    )


def build_matrices(
    features: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """유전자 severity, 기능 변이, multi-hit과 결과별 토큰 수를 만듭니다."""
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"변환에 필요한 유전자 컬럼이 없습니다: {sorted(missing)}")
    rows = len(features)
    severity = np.zeros((rows, len(columns)), dtype="int8")
    functional = np.zeros_like(severity)
    multi_hit = np.zeros_like(severity)
    category_counts = np.zeros((rows, len(CONSEQUENCE_NAMES)), dtype="int16")

    for column_index, column in enumerate(columns):
        normalized = (
            features[column].astype("string").fillna("WT").str.strip().str.upper()
        )
        for row_index in np.flatnonzero(normalized.ne("WT").to_numpy()):
            mutations = split_mutations(str(normalized.iloc[row_index]))
            if not mutations:
                continue
            consequences = [consequence for _, consequence in mutations]
            severity[row_index, column_index] = max(consequences)
            functional[row_index, column_index] = int(any(value >= MISSENSE for value in consequences))
            multi_hit[row_index, column_index] = int(len(mutations) >= 2)
            for consequence in consequences:
                category_counts[row_index, consequence - 1] += 1
    frame_arguments = {"index": features.index, "columns": columns, "dtype": "int8"}
    return (
        pd.DataFrame(severity, **frame_arguments),
        pd.DataFrame(functional, **frame_arguments),
        pd.DataFrame(multi_hit, **frame_arguments),
        category_counts,
    )


def select_functional_genes(
    features: pd.DataFrame,
    minimum_count: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """동의 변이를 제외한 기능 변이 환자 수로 유전자를 선택합니다."""
    _, functional, _, _ = build_matrices(features, features.columns.tolist())
    counts = functional.sum(axis=0).astype(int).to_dict()
    selected = [column for column in features.columns if counts[column] >= minimum_count]
    dropped = [column for column in features.columns if counts[column] < minimum_count]
    return selected, dropped, counts


def learn_hotspots(
    features: pd.DataFrame,
    columns: list[str],
    minimum_count: int,
    maximum_hotspots: int,
) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str], int]]:
    """동의 변이를 제외한 recurrent hotspot을 학습합니다."""
    support: Counter[tuple[str, str]] = Counter()
    for gene in columns:
        normalized = features[gene].astype("string").fillna("WT").str.strip().str.upper()
        for value in normalized[normalized.ne("WT")]:
            support.update(
                (gene, token)
                for token, consequence in split_mutations(str(value))
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
    """선택된 hotspot의 표본별 존재 여부를 만듭니다."""
    by_gene: dict[str, list[tuple[str, str]]] = {}
    for name, gene, token in hotspots:
        by_gene.setdefault(gene, []).append((name, token))
    output: dict[str, pd.Series] = {}
    for gene, definitions in by_gene.items():
        normalized = features[gene].astype("string").fillna("WT").str.strip().str.upper()
        token_sets = normalized.map(
            lambda value: frozenset(token for token, _ in split_mutations(str(value)))
        )
        for name, token in definitions:
            output[name] = token_sets.map(lambda tokens: token in tokens).astype("int8")
    return pd.DataFrame(output, index=features.index)


def learn_class_weights(
    functional: pd.DataFrame,
    labels: pd.Series,
    top_genes: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """암종별 안정화 log2 odds 기능 변이 signature를 학습합니다."""
    labels = labels.reindex(functional.index).astype("string")
    classes = sorted(labels.dropna().unique().tolist())
    class_weights: dict[str, dict[str, float]] = {}
    for class_name in classes:
        in_class = labels.eq(class_name).fillna(False)
        class_count = int(in_class.sum())
        other_count = len(labels) - class_count
        positive = functional.loc[in_class].sum(axis=0).astype("float64")
        negative = functional.loc[~in_class].sum(axis=0).astype("float64")
        class_odds = (positive + smoothing) / (class_count - positive + smoothing)
        other_odds = (negative + smoothing) / (other_count - negative + smoothing)
        log2_odds = np.log2(class_odds / other_odds).clip(0.0, max_log2_odds)
        reliability = np.sqrt((positive + negative) / (positive + negative + shrinkage))
        scores = (log2_odds * reliability).nlargest(top_genes)
        scores = scores[scores.gt(0)]
        if scores.empty:
            raise ValueError(f"{class_name}의 기능 변이 signature를 만들 수 없습니다.")
        class_weights[class_name] = scores.astype(float).to_dict()
    return classes, class_weights


def add_signatures(
    engineered: pd.DataFrame,
    functional: pd.DataFrame,
    classes: list[str],
    class_weights: dict[str, dict[str, float]],
) -> None:
    """중복 없이 클래스별 weighted와 match-count 피처를 추가합니다."""
    for class_name in classes:
        weights = class_weights[class_name]
        genes = list(weights)
        matrix = functional[genes].astype("float32")
        vector = np.array([weights[gene] for gene in genes], dtype="float32")
        engineered[f"signature_{class_name}_weighted"] = matrix.to_numpy().dot(vector)
        engineered[f"signature_{class_name}_match_count"] = matrix.sum(axis=1)


def create_oof_signatures(
    functional: pd.DataFrame,
    labels: pd.Series,
    top_genes: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
    folds: int,
    random_state: int,
) -> pd.DataFrame:
    """자기 정답 영향을 제거한 inner-fold OOF signature를 만듭니다."""
    labels = labels.reindex(functional.index)
    n_splits = min(folds, int(labels.value_counts().min()))
    if n_splits < 2:
        raise ValueError("OOF signature에는 클래스별 표본이 최소 2개 필요합니다.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    output = pd.DataFrame(index=functional.index)
    for train_positions, valid_positions in splitter.split(functional, labels):
        train = functional.iloc[train_positions]
        valid = functional.iloc[valid_positions]
        classes, weights = learn_class_weights(
            train, labels.iloc[train_positions], top_genes, smoothing, max_log2_odds, shrinkage
        )
        fold_output = pd.DataFrame(index=valid.index)
        add_signatures(fold_output, valid, classes, weights)
        for column in fold_output:
            output.loc[valid.index, column] = fold_output[column]
    return output.astype("float32")


def learn_stable_genes(
    functional: pd.DataFrame,
    labels: pd.Series,
    genes_per_class: int,
    folds: int,
    minimum_folds: int,
    maximum_genes: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
    random_state: int,
) -> tuple[list[str], dict[str, int]]:
    """여러 inner-fold에서 반복 선택된 유전자를 고릅니다."""
    labels = labels.reindex(functional.index)
    n_splits = min(folds, int(labels.value_counts().min()))
    if minimum_folds > n_splits:
        raise ValueError("min_stable_gene_folds가 실제 fold 수보다 큽니다.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    support: Counter[str] = Counter()
    for train_positions, _ in splitter.split(functional, labels):
        _, weights = learn_class_weights(
            functional.iloc[train_positions], labels.iloc[train_positions],
            genes_per_class, smoothing, max_log2_odds, shrinkage,
        )
        support.update({gene for values in weights.values() for gene in values})
    genes = sorted(
        (gene for gene, count in support.items() if count >= minimum_folds),
        key=lambda gene: (-support[gene], gene),
    )[:maximum_genes]
    return genes, dict(support)


def create_stable_onehot(features: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """안정 유전자에만 consequence multi-label one-hot을 만듭니다."""
    output: dict[str, pd.Series] = {}
    for gene in genes:
        normalized = features[gene].astype("string").fillna("WT").str.strip().str.upper()
        consequence_sets = normalized.map(
            lambda value: frozenset(value for _, value in split_mutations(str(value)))
        )
        for consequence, name in CONSEQUENCE_NAMES.items():
            output[f"stable_{gene}_{name}"] = consequence_sets.map(
                lambda values: consequence in values
            ).astype("int8")
    return pd.DataFrame(output, index=features.index)


def add_summaries(
    engineered: pd.DataFrame,
    category_counts: np.ndarray,
    functional: pd.DataFrame,
    multi_hit: pd.DataFrame,
) -> None:
    """v16/v20의 중복 burden을 하나의 상세 결과 집계로 통합합니다."""
    total = category_counts.sum(axis=1).astype("float32")
    denominator = np.maximum(total, 1.0)
    functional_count = category_counts[:, 1:].sum(axis=1).astype("float32")
    engineered["mutation_burden_log1p"] = np.log1p(total).astype("float32")
    engineered["functional_mutation_count_log1p"] = np.log1p(functional_count).astype("float32")
    engineered["functional_gene_count_log1p"] = np.log1p(functional.sum(axis=1)).astype("float32")
    engineered["multi_variant_gene_count_log1p"] = np.log1p(multi_hit.sum(axis=1)).astype("float32")
    engineered["functional_mutation_ratio"] = (functional_count / denominator).astype("float32")
    for index, name in enumerate(CONSEQUENCE_NAMES.values()):
        count = category_counts[:, index].astype("float32")
        engineered[f"consequence_{name}_count_log1p"] = np.log1p(count).astype("float32")
        engineered[f"consequence_{name}_ratio"] = (count / denominator).astype("float32")


class EMV29PreprocessingPipeline(PreprocessingPipeline):
    """v19에 이미 포함된 v16 OOF 흐름을 중복 없이 한 번만 유지합니다."""

    name = "em_v29"
    evaluation_folds = 5

    def __init__(
        self,
        min_mutation_count: int = 5,
        min_functional_mutation_count: int | None = None,
        top_genes_per_class: int = 20,
        smoothing: float = 0.5,
        max_log2_odds: float = 8.0,
        shrinkage: float = 10.0,
        min_hotspot_count: int = 5,
        max_hotspots: int = 384,
        inner_signature_folds: int = 5,
        stable_gene_folds: int = 5,
        min_stable_gene_folds: int = 4,
        onehot_genes_per_class: int = 10,
        max_stable_genes: int = 256,
        signature_random_state: int = 42,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        minimum_count = (
            min_mutation_count
            if min_functional_mutation_count is None
            else min_functional_mutation_count
        )
        integer_values = {
            "min_functional_mutation_count": (minimum_count, 1),
            "top_genes_per_class": (top_genes_per_class, 1),
            "min_hotspot_count": (min_hotspot_count, 1),
            "max_hotspots": (max_hotspots, 1),
            "inner_signature_folds": (inner_signature_folds, 2),
            "stable_gene_folds": (stable_gene_folds, 2),
            "min_stable_gene_folds": (min_stable_gene_folds, 1),
            "onehot_genes_per_class": (onehot_genes_per_class, 1),
            "max_stable_genes": (max_stable_genes, 1),
        }
        for name, (value, lower_bound) in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < lower_bound:
                raise ValueError(f"{name}는 {lower_bound} 이상의 정수여야 합니다.")
        if min_stable_gene_folds > stable_gene_folds:
            raise ValueError("min_stable_gene_folds는 stable_gene_folds 이하여야 합니다.")
        if smoothing <= 0 or max_log2_odds <= 0 or shrinkage <= 0:
            raise ValueError("signature 안정화 파라미터는 0보다 커야 합니다.")
        self.min_functional_mutation_count = minimum_count
        self.top_genes_per_class = top_genes_per_class
        self.smoothing = float(smoothing)
        self.max_log2_odds = float(max_log2_odds)
        self.shrinkage = float(shrinkage)
        self.min_hotspot_count = min_hotspot_count
        self.max_hotspots = max_hotspots
        self.inner_signature_folds = inner_signature_folds
        self.stable_gene_folds = stable_gene_folds
        self.min_stable_gene_folds = min_stable_gene_folds
        self.onehot_genes_per_class = onehot_genes_per_class
        self.max_stable_genes = max_stable_genes
        self.random_state = signature_random_state
        self.selected_gene_columns: list[str] = []
        self.dropped_rare_columns: list[str] = []
        self.functional_mutation_counts_: dict[str, int] = {}
        self.class_names: list[str] = []
        self.class_gene_weights_: dict[str, dict[str, float]] = {}
        self.hotspots_: list[tuple[str, str, str]] = []
        self.hotspot_support_: dict[tuple[str, str], int] = {}
        self.stable_genes_: list[str] = []
        self.stable_gene_support_: dict[str, int] = {}
        self.steps = (
            "동의·기능 변이 분리 및 기능 지지도 필터",
            "기능 변이 severity와 상세 consequence 집계",
            "inner-fold OOF 암종 signature",
            "기능 변이 recurrent hotspot",
            "반복 선택 안정 유전자 consequence one-hot",
        )

    def _engineer(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        severity, functional, multi_hit, counts = build_matrices(
            features, self.selected_gene_columns
        )
        engineered = severity.drop(columns=self.stable_genes_).astype("float32")
        add_summaries(engineered, counts, functional, multi_hit)
        add_signatures(engineered, functional, self.class_names, self.class_gene_weights_)
        engineered = pd.concat([
            engineered,
            create_hotspot_features(features, self.hotspots_),
            create_stable_onehot(features, self.stable_genes_),
        ], axis=1)
        return engineered, functional

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMV29PreprocessingPipeline":
        (
            self.selected_gene_columns,
            self.dropped_rare_columns,
            self.functional_mutation_counts_,
        ) = select_functional_genes(features, self.min_functional_mutation_count)
        if not self.selected_gene_columns:
            raise ValueError("최소 기능 변이 지지도를 만족하는 유전자가 없습니다.")
        self.hotspots_, self.hotspot_support_ = learn_hotspots(
            features, self.selected_gene_columns, self.min_hotspot_count, self.max_hotspots
        )
        severity, functional, multi_hit, counts = build_matrices(
            features, self.selected_gene_columns
        )
        self.class_names, self.class_gene_weights_ = learn_class_weights(
            functional, labels, self.top_genes_per_class, self.smoothing,
            self.max_log2_odds, self.shrinkage,
        )
        self.stable_genes_, self.stable_gene_support_ = learn_stable_genes(
            functional, labels, self.onehot_genes_per_class, self.stable_gene_folds,
            self.min_stable_gene_folds, self.max_stable_genes, self.smoothing,
            self.max_log2_odds, self.shrinkage, self.random_state,
        )
        engineered = severity.drop(columns=self.stable_genes_).astype("float32")
        add_summaries(engineered, counts, functional, multi_hit)
        add_signatures(engineered, functional, self.class_names, self.class_gene_weights_)
        engineered = pd.concat([
            engineered,
            create_hotspot_features(features, self.hotspots_),
            create_stable_onehot(features, self.stable_genes_),
        ], axis=1)
        super().fit(engineered, labels)
        print(
            f"[{self.name}] 기능 유전자 {len(self.selected_gene_columns)}개, "
            f"안정 one-hot {len(self.stable_genes_)}개, hotspot {len(self.hotspots_)}개"
        )
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        self.fit(features, labels)
        transformed = self.transform(features)
        _, functional, _, _ = build_matrices(features, self.selected_gene_columns)
        oof = create_oof_signatures(
            functional, labels, self.top_genes_per_class, self.smoothing,
            self.max_log2_odds, self.shrinkage, self.inner_signature_folds,
            self.random_state,
        )
        for column in oof:
            if column in transformed:
                transformed.loc[:, column] = oof[column].to_numpy()
        return transformed.astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        engineered, _ = self._engineer(features)
        return super().transform(engineered).astype("float32")

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        summary.update({
            "selected_functional_support": self.min_functional_mutation_count,
            "dropped_low_functional_support_features": len(self.dropped_rare_columns),
            "functional_gene_features": len(self.selected_gene_columns),
            "stable_onehot_genes": len(self.stable_genes_),
            "stable_consequence_features_created": 5 * len(self.stable_genes_),
            "signature_features": 2 * len(self.class_names),
            "functional_hotspot_features": len(self.hotspots_),
        })
        return summary

