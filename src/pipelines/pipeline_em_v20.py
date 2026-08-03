"""기능 변이 지지도 CV와 안정 유전자 consequence one-hot을 적용한 EM v20."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.pipelines.base import PreprocessingPipeline


SYNONYMOUS_PATTERN = re.compile(r"^([A-Z])(\d+)\1$")
MISSENSE_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")
FRAMESHIFT_PATTERN = re.compile(r"FS", re.IGNORECASE)
STOP_GAIN_PATTERN = re.compile(r"(?:\*|TER$|X$)", re.IGNORECASE)
INFRAME_PATTERN = re.compile(r"(?:DEL|INS|DUP|>|_)", re.IGNORECASE)
SAFE_NAME_PATTERN = re.compile(r"[^A-Z0-9]+")

CONSEQUENCE_WT = 0
CONSEQUENCE_SYNONYMOUS = 1
CONSEQUENCE_MISSENSE = 2
CONSEQUENCE_INFRAME = 3
CONSEQUENCE_NONSENSE = 4
CONSEQUENCE_FRAMESHIFT = 5
CONSEQUENCE_NAMES = {
    CONSEQUENCE_SYNONYMOUS: "synonymous",
    CONSEQUENCE_MISSENSE: "missense",
    CONSEQUENCE_INFRAME: "inframe",
    CONSEQUENCE_NONSENSE: "nonsense",
    CONSEQUENCE_FRAMESHIFT: "frameshift",
}


@lru_cache(maxsize=None)
def classify_mutation_token(token: str) -> int:
    """단백질 변이 토큰을 상호 배타적인 기능 결과로 분류합니다."""
    normalized = token.strip().upper()
    if not normalized or normalized in {"WT", "<NA>"}:
        return CONSEQUENCE_WT
    if SYNONYMOUS_PATTERN.fullmatch(normalized):
        return CONSEQUENCE_SYNONYMOUS
    if FRAMESHIFT_PATTERN.search(normalized):
        return CONSEQUENCE_FRAMESHIFT
    if STOP_GAIN_PATTERN.search(normalized):
        return CONSEQUENCE_NONSENSE
    if INFRAME_PATTERN.search(normalized):
        return CONSEQUENCE_INFRAME
    match = MISSENSE_PATTERN.fullmatch(normalized)
    if match and match.group(1) != match.group(3):
        return CONSEQUENCE_MISSENSE
    return CONSEQUENCE_MISSENSE


@lru_cache(maxsize=None)
def split_unique_mutations(value: str) -> tuple[tuple[str, int], ...]:
    """셀 문자열을 분리하고 중복 토큰을 제거합니다."""
    normalized = value.strip().upper()
    if not normalized or normalized in {"WT", "<NA>"}:
        return ()
    return tuple(
        (token, classify_mutation_token(token))
        for token in sorted(set(normalized.split()))
        if classify_mutation_token(token) != CONSEQUENCE_WT
    )


def build_consequence_matrices(
    features: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """유전자별 최대 결과, 기능 변이, 복수 변이와 결과별 개수를 만듭니다."""
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"결과 유형 생성에 필요한 피처가 없습니다: {sorted(missing)}")
    rows = len(features)
    severity = np.zeros((rows, len(columns)), dtype="int8")
    functional = np.zeros((rows, len(columns)), dtype="int8")
    multi_hit = np.zeros((rows, len(columns)), dtype="int8")
    category_counts = np.zeros((rows, len(CONSEQUENCE_NAMES)), dtype="int16")

    for column_position, column in enumerate(columns):
        normalized = (
            features[column].astype("string").fillna("WT").str.strip().str.upper()
        )
        for row_position in np.flatnonzero(normalized.ne("WT").to_numpy()):
            mutations = split_unique_mutations(str(normalized.iloc[row_position]))
            if not mutations:
                continue
            consequences = [consequence for _, consequence in mutations]
            severity[row_position, column_position] = max(consequences)
            functional[row_position, column_position] = int(
                any(value >= CONSEQUENCE_MISSENSE for value in consequences)
            )
            multi_hit[row_position, column_position] = int(len(mutations) >= 2)
            for consequence in consequences:
                category_counts[row_position, consequence - 1] += 1

    return (
        pd.DataFrame(severity, index=features.index, columns=columns, dtype="int8"),
        pd.DataFrame(functional, index=features.index, columns=columns, dtype="int8"),
        pd.DataFrame(multi_hit, index=features.index, columns=columns, dtype="int8"),
        category_counts,
    )


def select_genes_by_functional_count(
    features: pd.DataFrame,
    minimum_count: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """기능 변이 환자 수가 기준 이상인 유전자를 선택합니다."""
    _, functional, _, _ = build_consequence_matrices(
        features, features.columns.tolist()
    )
    counts = functional.sum(axis=0).astype(int).to_dict()
    selected = [column for column in features.columns if counts[column] >= minimum_count]
    dropped = [column for column in features.columns if counts[column] < minimum_count]
    return selected, dropped, counts


def learn_functional_hotspots(
    features: pd.DataFrame,
    columns: list[str],
    minimum_count: int,
    maximum_hotspots: int,
) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str], int]]:
    """동의 변이를 제외한 반복 기능 변이 토큰을 학습합니다."""
    support: Counter[tuple[str, str]] = Counter()
    for gene in columns:
        normalized = (
            features[gene].astype("string").fillna("WT").str.strip().str.upper()
        )
        for value in normalized[normalized.ne("WT")]:
            support.update(
                (gene, token)
                for token, consequence in split_unique_mutations(str(value))
                if consequence >= CONSEQUENCE_MISSENSE
            )
    selected = sorted(
        (pair for pair, count in support.items() if count >= minimum_count),
        key=lambda pair: (-support[pair], pair[0], pair[1]),
    )[:maximum_hotspots]
    hotspots: list[tuple[str, str, str]] = []
    for rank, (gene, token) in enumerate(selected, start=1):
        safe_token = SAFE_NAME_PATTERN.sub("_", token).strip("_") or "VARIANT"
        hotspots.append((f"hotspot_{rank:03d}_{gene}_{safe_token}", gene, token))
    return hotspots, dict(support)


def create_hotspot_matrix(
    features: pd.DataFrame,
    hotspots: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """학습 fold에서 선택된 hotspot 존재 여부를 만듭니다."""
    by_gene: dict[str, list[tuple[str, str]]] = {}
    for feature_name, gene, token in hotspots:
        by_gene.setdefault(gene, []).append((feature_name, token))
    values: dict[str, pd.Series] = {}
    for gene, definitions in by_gene.items():
        normalized = (
            features[gene].astype("string").fillna("WT").str.strip().str.upper()
        )
        token_sets = normalized.map(
            lambda value: frozenset(
                token for token, _ in split_unique_mutations(str(value))
            )
        )
        for feature_name, token in definitions:
            values[feature_name] = token_sets.map(
                lambda tokens: token in tokens
            ).astype("int8")
    return pd.DataFrame(values, index=features.index)


def add_consequence_summary_features(
    engineered: pd.DataFrame,
    category_counts: np.ndarray,
    functional: pd.DataFrame,
    multi_hit: pd.DataFrame,
) -> pd.DataFrame:
    """결과별 log-count/비율과 기능·복수 변이 부담을 추가합니다."""
    total = category_counts.sum(axis=1).astype("float32")
    denominator = np.maximum(total, 1.0)
    functional_count = category_counts[:, 1:].sum(axis=1).astype("float32")
    engineered["mutation_token_count_log1p"] = np.log1p(total).astype("float32")
    engineered["functional_mutation_count_log1p"] = np.log1p(
        functional_count
    ).astype("float32")
    engineered["functional_gene_count_log1p"] = np.log1p(
        functional.sum(axis=1).to_numpy(dtype="float32")
    ).astype("float32")
    engineered["multi_variant_gene_count_log1p"] = np.log1p(
        multi_hit.sum(axis=1).to_numpy(dtype="float32")
    ).astype("float32")
    engineered["functional_mutation_ratio"] = (
        functional_count / denominator
    ).astype("float32")
    for position, name in enumerate(CONSEQUENCE_NAMES.values()):
        count = category_counts[:, position].astype("float32")
        engineered[f"consequence_{name}_count_log1p"] = np.log1p(count).astype(
            "float32"
        )
        engineered[f"consequence_{name}_ratio"] = (count / denominator).astype(
            "float32"
        )
    return engineered


def learn_class_gene_weights(
    functional: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """기능 변이로 암종별 안정화 log2 odds 가중치를 학습합니다."""
    aligned = labels.reindex(functional.index).astype("string")
    classes = sorted(aligned.dropna().unique().tolist())
    weights_by_class: dict[str, dict[str, float]] = {}
    for class_name in classes:
        in_class = aligned.eq(class_name).fillna(False)
        class_size = int(in_class.sum())
        other_size = len(in_class) - class_size
        class_mutations = functional.loc[in_class].sum(axis=0).astype("float64")
        other_mutations = functional.loc[~in_class].sum(axis=0).astype("float64")
        class_odds = (class_mutations + smoothing) / (
            class_size - class_mutations + smoothing
        )
        other_odds = (other_mutations + smoothing) / (
            other_size - other_mutations + smoothing
        )
        log2_odds = np.log2(class_odds / other_odds).clip(0.0, max_log2_odds)
        support = class_mutations + other_mutations
        reliability = np.sqrt(support / (support + shrinkage))
        scores = (log2_odds * reliability).nlargest(top_genes_per_class)
        scores = scores[scores.gt(0)]
        if scores.empty:
            raise ValueError(f"{class_name}의 기능 변이 signature를 만들 수 없습니다.")
        weights_by_class[class_name] = {
            gene: float(score) for gene, score in scores.items()
        }
    return classes, weights_by_class


def add_class_signature_features(
    engineered: pd.DataFrame,
    functional: pd.DataFrame,
    classes: list[str],
    weights_by_class: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """기능 변이 기반 weighted/match signature를 추가합니다."""
    for class_name in classes:
        weights = weights_by_class[class_name]
        genes = list(weights)
        matrix = functional[genes].astype("float32")
        vector = np.array([weights[gene] for gene in genes], dtype="float32")
        engineered[f"signature_{class_name}_weighted"] = matrix.to_numpy().dot(vector)
        engineered[f"signature_{class_name}_match_count"] = matrix.sum(axis=1)
    return engineered


def create_oof_signature_features(
    functional: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
    folds: int,
    random_state: int,
) -> pd.DataFrame:
    """표본 정답을 제외한 inner-fold signature를 만듭니다."""
    aligned = labels.reindex(functional.index)
    n_splits = min(folds, int(aligned.value_counts().min()))
    if n_splits < 2:
        raise ValueError("OOF signature에는 클래스별 표본이 최소 2개 필요합니다.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    output = pd.DataFrame(index=functional.index)
    for train_positions, valid_positions in splitter.split(functional, aligned):
        fold_train = functional.iloc[train_positions]
        fold_valid = functional.iloc[valid_positions]
        classes, weights = learn_class_gene_weights(
            fold_train,
            aligned.iloc[train_positions],
            top_genes_per_class,
            smoothing,
            max_log2_odds,
            shrinkage,
        )
        fold_output = pd.DataFrame(index=fold_valid.index)
        add_class_signature_features(fold_output, fold_valid, classes, weights)
        for column in fold_output:
            output.loc[fold_valid.index, column] = fold_output[column]
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
    """inner-fold에서 반복 선택된 유전자만 consequence one-hot 대상으로 고릅니다."""
    aligned = labels.reindex(functional.index)
    n_splits = min(folds, int(aligned.value_counts().min()))
    if minimum_folds > n_splits:
        raise ValueError("min_stable_gene_folds는 실제 stable_gene_folds 이하여야 합니다.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    support: Counter[str] = Counter()
    for train_positions, _ in splitter.split(functional, aligned):
        _, weights = learn_class_gene_weights(
            functional.iloc[train_positions],
            aligned.iloc[train_positions],
            genes_per_class,
            smoothing,
            max_log2_odds,
            shrinkage,
        )
        fold_genes = {gene for class_weights in weights.values() for gene in class_weights}
        support.update(fold_genes)
    selected = sorted(
        (gene for gene, count in support.items() if count >= minimum_folds),
        key=lambda gene: (-support[gene], gene),
    )[:maximum_genes]
    return selected, dict(support)


def create_stable_gene_onehot(
    features: pd.DataFrame,
    stable_genes: list[str],
) -> pd.DataFrame:
    """안정 유전자에만 결과 유형별 multi-label one-hot을 생성합니다."""
    values: dict[str, pd.Series] = {}
    for gene in stable_genes:
        normalized = (
            features[gene].astype("string").fillna("WT").str.strip().str.upper()
        )
        consequence_sets = normalized.map(
            lambda value: frozenset(
                consequence for _, consequence in split_unique_mutations(str(value))
            )
        )
        for consequence, name in CONSEQUENCE_NAMES.items():
            values[f"stable_{gene}_{name}"] = consequence_sets.map(
                lambda consequences: consequence in consequences
            ).astype("int8")
    return pd.DataFrame(values, index=features.index)


class EMV20PreprocessingPipeline(PreprocessingPipeline):
    """em_v18 신호에 안정 유전자 consequence one-hot을 제한적으로 추가합니다."""

    name = "em_v20"
    evaluation_folds = 5

    def __init__(
        self,
        min_functional_mutation_count: int = 5,
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
        random_state: int = 42,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        integer_parameters = {
            "min_functional_mutation_count": (min_functional_mutation_count, 1),
            "top_genes_per_class": (top_genes_per_class, 1),
            "min_hotspot_count": (min_hotspot_count, 1),
            "max_hotspots": (max_hotspots, 1),
            "inner_signature_folds": (inner_signature_folds, 2),
            "stable_gene_folds": (stable_gene_folds, 2),
            "min_stable_gene_folds": (min_stable_gene_folds, 1),
            "onehot_genes_per_class": (onehot_genes_per_class, 1),
            "max_stable_genes": (max_stable_genes, 1),
        }
        for name, (value, minimum) in integer_parameters.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name}는 {minimum} 이상의 정수여야 합니다.")
        if min_stable_gene_folds > stable_gene_folds:
            raise ValueError("min_stable_gene_folds는 stable_gene_folds 이하여야 합니다.")
        if smoothing <= 0 or max_log2_odds <= 0 or shrinkage <= 0:
            raise ValueError("signature 안정화 파라미터는 0보다 커야 합니다.")

        self.min_functional_mutation_count = min_functional_mutation_count
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
        self.random_state = random_state
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
            "동의·기능 변이 분리",
            "fold 내부 기능 변이 지지도 선택",
            "inner-fold OOF 기능 변이 signature",
            "기능 변이 recurrent hotspot",
            "반복 선택 안정 유전자 consequence one-hot",
            "E4형 저학습률 조기 종료 모델",
        )

    def _build_features(
        self, features: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        severity, functional, multi_hit, category_counts = build_consequence_matrices(
            features, self.selected_gene_columns
        )
        engineered = severity.drop(columns=self.stable_genes_).astype("float32")
        add_consequence_summary_features(
            engineered, category_counts, functional, multi_hit
        )
        add_class_signature_features(
            engineered, functional, self.class_names, self.class_gene_weights_
        )
        engineered = pd.concat([
            engineered,
            create_hotspot_matrix(features, self.hotspots_),
            create_stable_gene_onehot(features, self.stable_genes_),
        ], axis=1)
        return engineered, functional

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "EMV20PreprocessingPipeline":
        (
            self.selected_gene_columns,
            self.dropped_rare_columns,
            self.functional_mutation_counts_,
        ) = select_genes_by_functional_count(
            features, self.min_functional_mutation_count
        )
        if not self.selected_gene_columns:
            raise ValueError("최소 기능 변이 지지도를 만족하는 유전자가 없습니다.")
        self.hotspots_, self.hotspot_support_ = learn_functional_hotspots(
            features,
            self.selected_gene_columns,
            self.min_hotspot_count,
            self.max_hotspots,
        )
        severity, functional, multi_hit, category_counts = build_consequence_matrices(
            features, self.selected_gene_columns
        )
        self.class_names, self.class_gene_weights_ = learn_class_gene_weights(
            functional,
            labels,
            self.top_genes_per_class,
            self.smoothing,
            self.max_log2_odds,
            self.shrinkage,
        )
        self.stable_genes_, self.stable_gene_support_ = learn_stable_genes(
            functional,
            labels,
            self.onehot_genes_per_class,
            self.stable_gene_folds,
            self.min_stable_gene_folds,
            self.max_stable_genes,
            self.smoothing,
            self.max_log2_odds,
            self.shrinkage,
            self.random_state,
        )
        engineered = severity.drop(columns=self.stable_genes_).astype("float32")
        add_consequence_summary_features(
            engineered, category_counts, functional, multi_hit
        )
        add_class_signature_features(
            engineered, functional, self.class_names, self.class_gene_weights_
        )
        engineered = pd.concat([
            engineered,
            create_hotspot_matrix(features, self.hotspots_),
            create_stable_gene_onehot(features, self.stable_genes_),
        ], axis=1)
        super().fit(engineered, labels)
        print(
            f"[{self.name}] 기능 지지도 {self.min_functional_mutation_count}, "
            f"유전자 {len(self.selected_gene_columns)}개, 안정 one-hot 유전자 "
            f"{len(self.stable_genes_)}개, hotspot {len(self.hotspots_)}개"
        )
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> pd.DataFrame:
        self.fit(features, labels)
        transformed = self.transform(features)
        _, functional, _, _ = build_consequence_matrices(
            features, self.selected_gene_columns
        )
        oof_signatures = create_oof_signature_features(
            functional,
            labels,
            self.top_genes_per_class,
            self.smoothing,
            self.max_log2_odds,
            self.shrinkage,
            self.inner_signature_folds,
            self.random_state,
        )
        for column in oof_signatures:
            if column in transformed:
                transformed.loc[:, column] = oof_signatures[column].to_numpy()
        return transformed.astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        engineered, _ = self._build_features(features)
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
