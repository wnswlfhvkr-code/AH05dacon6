"""동의 변이와 기능 변이를 분리하는 EM 버전 18입니다."""

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
    if not normalized or normalized == "WT" or normalized == "<NA>":
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
    # 현재 데이터의 비표준 단백질 치환 토큰은 기능 미상 missense로 보수적으로 처리합니다.
    return CONSEQUENCE_MISSENSE


@lru_cache(maxsize=None)
def split_unique_mutations(value: str) -> tuple[tuple[str, int], ...]:
    """셀의 변이를 공백으로 분리하고 중복 토큰을 한 번만 반환합니다."""
    normalized = value.strip().upper()
    if not normalized or normalized == "WT" or normalized == "<NA>":
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
    """유전자별 최대 결과 등급, 기능 변이 여부, 복수 변이 여부를 만듭니다."""
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"결과 유형 생성에 필요한 피처가 없습니다: {sorted(missing)}")

    rows = len(features)
    severity_values = np.zeros((rows, len(columns)), dtype="int8")
    functional_values = np.zeros((rows, len(columns)), dtype="int8")
    multi_hit_values = np.zeros((rows, len(columns)), dtype="int8")
    category_counts = np.zeros((rows, len(CONSEQUENCE_NAMES)), dtype="int16")

    for column_position, column in enumerate(columns):
        normalized = (
            features[column].astype("string").fillna("WT").str.strip().str.upper()
        )
        positions = np.flatnonzero(normalized.ne("WT").to_numpy())
        for row_position in positions:
            mutations = split_unique_mutations(str(normalized.iloc[row_position]))
            if not mutations:
                continue
            consequences = [consequence for _, consequence in mutations]
            severity_values[row_position, column_position] = max(consequences)
            functional_values[row_position, column_position] = int(
                any(consequence >= CONSEQUENCE_MISSENSE for consequence in consequences)
            )
            multi_hit_values[row_position, column_position] = int(len(mutations) >= 2)
            for consequence in consequences:
                category_counts[row_position, consequence - 1] += 1

    severity = pd.DataFrame(
        severity_values, index=features.index, columns=columns, dtype="int8"
    )
    functional = pd.DataFrame(
        functional_values, index=features.index, columns=columns, dtype="int8"
    )
    multi_hit = pd.DataFrame(
        multi_hit_values, index=features.index, columns=columns, dtype="int8"
    )
    return severity, functional, multi_hit, category_counts


def select_genes_by_functional_count(
    features: pd.DataFrame,
    minimum_count: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """동의 변이를 제외한 기능 변이 환자 수로 유전자를 선택합니다."""
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
    """동의 변이를 제외하고 반복 관측된 기능 변이 hotspot을 선택합니다."""
    support: Counter[tuple[str, str]] = Counter()
    for gene in columns:
        normalized = (
            features[gene].astype("string").fillna("WT").str.strip().str.upper()
        )
        for value in normalized[normalized.ne("WT")]:
            functional_tokens = {
                token
                for token, consequence in split_unique_mutations(str(value))
                if consequence >= CONSEQUENCE_MISSENSE
            }
            support.update((gene, token) for token in functional_tokens)

    selected_pairs = sorted(
        (pair for pair, count in support.items() if count >= minimum_count),
        key=lambda pair: (-support[pair], pair[0], pair[1]),
    )[:maximum_hotspots]
    hotspots: list[tuple[str, str, str]] = []
    for rank, (gene, token) in enumerate(selected_pairs, start=1):
        safe_token = SAFE_NAME_PATTERN.sub("_", token).strip("_") or "VARIANT"
        hotspots.append((f"hotspot_{rank:03d}_{gene}_{safe_token}", gene, token))
    return hotspots, dict(support)


def create_hotspot_matrix(
    features: pd.DataFrame,
    hotspots: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """학습 fold에서 선택된 기능 hotspot의 표본별 존재 여부를 만듭니다."""
    by_gene: dict[str, list[tuple[str, str]]] = {}
    for feature_name, gene, token in hotspots:
        by_gene.setdefault(gene, []).append((feature_name, token))

    values: dict[str, pd.Series] = {}
    for gene, definitions in by_gene.items():
        normalized = (
            features[gene].astype("string").fillna("WT").str.strip().str.upper()
        )
        token_sets = normalized.map(
            lambda value: frozenset(token for token, _ in split_unique_mutations(str(value)))
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
    multi_hit_gene_count = multi_hit.sum(axis=1).to_numpy(dtype="float32")

    engineered["mutation_token_count_log1p"] = np.log1p(total).astype("float32")
    engineered["functional_mutation_count_log1p"] = np.log1p(
        functional_count
    ).astype("float32")
    engineered["functional_gene_count_log1p"] = np.log1p(
        functional.sum(axis=1).to_numpy(dtype="float32")
    ).astype("float32")
    engineered["multi_variant_gene_count_log1p"] = np.log1p(
        multi_hit_gene_count
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
    functional_matrix: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """기능 변이만으로 암종별 안정화 log2 odds 가중치를 학습합니다."""
    aligned = labels.reindex(functional_matrix.index).astype("string")
    class_names = sorted(aligned.dropna().unique().tolist())
    weights_by_class: dict[str, dict[str, float]] = {}
    for class_name in class_names:
        in_class = aligned.eq(class_name).fillna(False)
        class_size = int(in_class.sum())
        other_size = len(in_class) - class_size
        class_mutations = functional_matrix.loc[in_class].sum(axis=0).astype("float64")
        other_mutations = functional_matrix.loc[~in_class].sum(axis=0).astype("float64")
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
    return class_names, weights_by_class


def add_class_signature_features(
    engineered: pd.DataFrame,
    functional_matrix: pd.DataFrame,
    class_names: list[str],
    weights_by_class: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """기능 변이 기반 weighted/match signature를 추가합니다."""
    for class_name in class_names:
        weights = weights_by_class[class_name]
        genes = list(weights)
        matrix = functional_matrix[genes].astype("float32")
        vector = np.array([weights[gene] for gene in genes], dtype="float32")
        engineered[f"signature_{class_name}_weighted"] = matrix.to_numpy().dot(vector)
        engineered[f"signature_{class_name}_match_count"] = matrix.sum(axis=1)
    return engineered


def create_oof_signature_features(
    functional_matrix: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
    folds: int,
    random_state: int,
) -> pd.DataFrame:
    """표본 자신의 정답을 제외한 inner-fold 기능 변이 signature를 만듭니다."""
    aligned = labels.reindex(functional_matrix.index)
    n_splits = min(folds, int(aligned.value_counts().min()))
    if n_splits < 2:
        raise ValueError("OOF signature에는 클래스별 표본이 최소 2개 필요합니다.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    output = pd.DataFrame(index=functional_matrix.index)
    for train_positions, valid_positions in splitter.split(functional_matrix, aligned):
        fold_train = functional_matrix.iloc[train_positions]
        fold_valid = functional_matrix.iloc[valid_positions]
        class_names, weights = learn_class_gene_weights(
            fold_train,
            aligned.iloc[train_positions],
            top_genes_per_class,
            smoothing,
            max_log2_odds,
            shrinkage,
        )
        fold_output = pd.DataFrame(index=fold_valid.index)
        add_class_signature_features(fold_output, fold_valid, class_names, weights)
        for column in fold_output:
            output.loc[fold_valid.index, column] = fold_output[column]
    return output.astype("float32")


class EMV18PreprocessingPipeline(PreprocessingPipeline):
    """동의 변이를 기능 변이 학습 신호에서 분리합니다."""

    name = "em_v18"
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
        signature_random_state: int = 42,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        integer_parameters = {
            "min_functional_mutation_count": min_functional_mutation_count,
            "top_genes_per_class": top_genes_per_class,
            "min_hotspot_count": min_hotspot_count,
            "max_hotspots": max_hotspots,
            "inner_signature_folds": inner_signature_folds,
        }
        for name, value in integer_parameters.items():
            minimum = 2 if name == "inner_signature_folds" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name}는 {minimum} 이상의 정수여야 합니다.")
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
        self.signature_random_state = signature_random_state
        self.selected_gene_columns: list[str] = []
        self.dropped_rare_columns: list[str] = []
        self.functional_mutation_counts_: dict[str, int] = {}
        self.class_names: list[str] = []
        self.class_gene_weights_: dict[str, dict[str, float]] = {}
        self.hotspots_: list[tuple[str, str, str]] = []
        self.hotspot_support_: dict[tuple[str, str], int] = {}
        self.steps = (
            "변이 토큰 분리 및 중복 제거",
            "동의·missense·nonsense·frameshift·in-frame 분리",
            "기능 변이 기준 유전자 선택",
            "inner-fold OOF 기능 변이 signature",
            "기능 변이 recurrent hotspot",
            "결과 유형 비율 및 복수 변이 집계",
        )

    def _raw_engineered_features(
        self, features: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        severity, functional, multi_hit, category_counts = build_consequence_matrices(
            features, self.selected_gene_columns
        )
        engineered = severity.astype("float32")
        add_consequence_summary_features(
            engineered, category_counts, functional, multi_hit
        )
        add_class_signature_features(
            engineered,
            functional,
            self.class_names,
            self.class_gene_weights_,
        )
        engineered = pd.concat(
            [engineered, create_hotspot_matrix(features, self.hotspots_)], axis=1
        )
        return engineered, functional

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "EMV18PreprocessingPipeline":
        (
            self.selected_gene_columns,
            self.dropped_rare_columns,
            self.functional_mutation_counts_,
        ) = select_genes_by_functional_count(
            features, self.min_functional_mutation_count
        )
        if not self.selected_gene_columns:
            raise ValueError("최소 기능 변이 빈도를 만족하는 유전자가 없습니다.")
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
        engineered = severity.astype("float32")
        add_consequence_summary_features(
            engineered, category_counts, functional, multi_hit
        )
        add_class_signature_features(
            engineered, functional, self.class_names, self.class_gene_weights_
        )
        engineered = pd.concat(
            [engineered, create_hotspot_matrix(features, self.hotspots_)], axis=1
        )
        super().fit(engineered, labels)
        print(
            f"[{self.name}] 기능 변이 유전자 {len(self.selected_gene_columns)}개, "
            f"동의 변이 제외 hotspot {len(self.hotspots_)}개"
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
            self.signature_random_state,
        )
        for column in oof_signatures:
            if column in transformed:
                transformed.loc[:, column] = oof_signatures[column].to_numpy()
        return transformed.astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        engineered, _ = self._raw_engineered_features(features)
        return super().transform(engineered).astype("float32")

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        summary.update({
            "dropped_low_functional_support_features": len(self.dropped_rare_columns),
            "functional_gene_features": len(self.selected_gene_columns),
            "consequence_summary_features": 15,
            "signature_features": 2 * len(self.class_names),
            "functional_hotspot_features": len(self.hotspots_),
        })
        return summary
