"""고차원 원시 변이와 불안정 hotspot을 축소한 과적합 완화 EM v21."""

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
    """단백질 변이 토큰을 결과 유형으로 분류합니다."""
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
    consequence_counts = np.zeros((rows, len(CONSEQUENCE_NAMES)), dtype="int16")
    for column_index, column in enumerate(columns):
        normalized = features[column].astype("string").fillna("WT").str.strip().str.upper()
        for row_index in np.flatnonzero(normalized.ne("WT").to_numpy()):
            mutations = split_mutations(str(normalized.iloc[row_index]))
            if not mutations:
                continue
            consequences = [consequence for _, consequence in mutations]
            severity[row_index, column_index] = max(consequences)
            functional[row_index, column_index] = int(any(value >= MISSENSE for value in consequences))
            multi_hit[row_index, column_index] = int(len(mutations) >= 2)
            for consequence in consequences:
                consequence_counts[row_index, consequence - 1] += 1
    arguments = {"index": features.index, "columns": columns, "dtype": "int8"}
    return (
        pd.DataFrame(severity, **arguments),
        pd.DataFrame(functional, **arguments),
        pd.DataFrame(multi_hit, **arguments),
        consequence_counts,
    )


def select_functional_genes(
    features: pd.DataFrame,
    minimum_count: int,
    maximum_raw_genes: int,
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """기능 변이 기준 전체 후보와 모델에 직접 노출할 고빈도 유전자를 구분합니다."""
    _, functional, _, _ = build_matrices(features, features.columns.tolist())
    counts = functional.sum(axis=0).astype(int).to_dict()
    eligible = [column for column in features.columns if counts[column] >= minimum_count]
    eligible.sort(key=lambda column: (-counts[column], column))
    raw_genes = eligible[:maximum_raw_genes]
    dropped = [column for column in features.columns if column not in set(eligible)]
    return eligible, raw_genes, dropped, counts


def token_support(
    features: pd.DataFrame,
    columns: list[str],
) -> Counter[tuple[str, str]]:
    """동의 변이를 제외한 기능 변이 토큰의 환자 지지도를 계산합니다."""
    support: Counter[tuple[str, str]] = Counter()
    for gene in columns:
        normalized = features[gene].astype("string").fillna("WT").str.strip().str.upper()
        for value in normalized[normalized.ne("WT")]:
            support.update(
                (gene, token)
                for token, consequence in split_mutations(str(value))
                if consequence >= MISSENSE
            )
    return support


def learn_stable_hotspots(
    features: pd.DataFrame,
    labels: pd.Series,
    columns: list[str],
    minimum_count: int,
    folds: int,
    minimum_folds: int,
    maximum_hotspots: int,
    random_state: int,
) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str], int]]:
    """여러 inner-fold에서 반복 관측된 기능 hotspot만 선택합니다."""
    labels = labels.reindex(features.index)
    n_splits = min(folds, int(labels.value_counts().min()))
    if minimum_folds > n_splits:
        raise ValueError("min_stable_hotspot_folds가 실제 fold 수보다 큽니다.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_support: Counter[tuple[str, str]] = Counter()
    for train_positions, _ in splitter.split(features, labels):
        support = token_support(features.iloc[train_positions], columns)
        fold_support.update(pair for pair, count in support.items() if count >= minimum_count)
    full_support = token_support(features, columns)
    pairs = sorted(
        (pair for pair, count in fold_support.items() if count >= minimum_folds),
        key=lambda pair: (-fold_support[pair], -full_support[pair], pair[0], pair[1]),
    )[:maximum_hotspots]
    hotspots = []
    for rank, (gene, token) in enumerate(pairs, start=1):
        safe_token = SAFE_NAME_PATTERN.sub("_", token).strip("_") or "VARIANT"
        hotspots.append((f"stable_hotspot_{rank:03d}_{gene}_{safe_token}", gene, token))
    return hotspots, dict(fold_support)


def create_hotspot_features(
    features: pd.DataFrame,
    hotspots: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """안정 hotspot의 표본별 존재 여부를 만듭니다."""
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
            output[name] = token_sets.map(lambda values: token in values).astype("int8")
    return pd.DataFrame(output, index=features.index)


def learn_class_weights(
    functional: pd.DataFrame,
    labels: pd.Series,
    top_genes: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """기능 변이 기반 암종별 안정화 log2 odds를 학습합니다."""
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


def add_signature_features(
    engineered: pd.DataFrame,
    functional: pd.DataFrame,
    classes: list[str],
    class_weights: dict[str, dict[str, float]],
) -> None:
    """weighted signature와 burden-normalized match-rate만 추가합니다."""
    burden = functional.sum(axis=1).clip(lower=1).astype("float32")
    for class_name in classes:
        weights = class_weights[class_name]
        genes = list(weights)
        matrix = functional[genes].astype("float32")
        vector = np.array([weights[gene] for gene in genes], dtype="float32")
        engineered[f"signature_{class_name}_weighted"] = matrix.to_numpy().dot(vector)
        engineered[f"signature_{class_name}_match_rate"] = (
            matrix.sum(axis=1) / burden
        ).astype("float32")


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
    """자기 정답을 제외한 inner-fold signature를 만듭니다."""
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
        add_signature_features(fold_output, valid, classes, weights)
        for column in fold_output:
            output.loc[valid.index, column] = fold_output[column]
    return output.astype("float32")


def add_compact_summaries(
    engineered: pd.DataFrame,
    consequence_counts: np.ndarray,
    functional: pd.DataFrame,
    multi_hit: pd.DataFrame,
) -> None:
    """상관성이 큰 절대 결과 개수 대신 burden과 결과 비율만 추가합니다."""
    total = consequence_counts.sum(axis=1).astype("float32")
    denominator = np.maximum(total, 1.0)
    engineered["mutation_burden_log1p"] = np.log1p(total).astype("float32")
    engineered["functional_gene_count_log1p"] = np.log1p(functional.sum(axis=1)).astype("float32")
    engineered["multi_variant_gene_count_log1p"] = np.log1p(multi_hit.sum(axis=1)).astype("float32")
    for index, name in enumerate(CONSEQUENCE_NAMES.values()):
        engineered[f"consequence_{name}_ratio"] = (
            consequence_counts[:, index].astype("float32") / denominator
        ).astype("float32")


class EMV21PreprocessingPipeline(PreprocessingPipeline):
    """원시 유전자와 hotspot 용량을 줄이고 집계 중복을 제거합니다."""

    name = "em_v21"
    evaluation_folds = 5

    def __init__(
        self,
        min_functional_mutation_count: int = 5,
        max_raw_gene_features: int = 1500,
        top_genes_per_class: int = 15,
        smoothing: float = 0.5,
        max_log2_odds: float = 8.0,
        shrinkage: float = 10.0,
        min_hotspot_count: int = 5,
        stable_hotspot_folds: int = 5,
        min_stable_hotspot_folds: int = 4,
        max_stable_hotspots: int = 128,
        inner_signature_folds: int = 5,
        random_state: int = 42,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        integer_values = {
            "min_functional_mutation_count": (min_functional_mutation_count, 1),
            "max_raw_gene_features": (max_raw_gene_features, 1),
            "top_genes_per_class": (top_genes_per_class, 1),
            "min_hotspot_count": (min_hotspot_count, 1),
            "stable_hotspot_folds": (stable_hotspot_folds, 2),
            "min_stable_hotspot_folds": (min_stable_hotspot_folds, 1),
            "max_stable_hotspots": (max_stable_hotspots, 1),
            "inner_signature_folds": (inner_signature_folds, 2),
        }
        for name, (value, minimum) in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name}는 {minimum} 이상의 정수여야 합니다.")
        if min_stable_hotspot_folds > stable_hotspot_folds:
            raise ValueError("min_stable_hotspot_folds는 stable_hotspot_folds 이하여야 합니다.")
        if smoothing <= 0 or max_log2_odds <= 0 or shrinkage <= 0:
            raise ValueError("signature 안정화 파라미터는 0보다 커야 합니다.")
        self.min_functional_mutation_count = min_functional_mutation_count
        self.max_raw_gene_features = max_raw_gene_features
        self.top_genes_per_class = top_genes_per_class
        self.smoothing = float(smoothing)
        self.max_log2_odds = float(max_log2_odds)
        self.shrinkage = float(shrinkage)
        self.min_hotspot_count = min_hotspot_count
        self.stable_hotspot_folds = stable_hotspot_folds
        self.min_stable_hotspot_folds = min_stable_hotspot_folds
        self.max_stable_hotspots = max_stable_hotspots
        self.inner_signature_folds = inner_signature_folds
        self.random_state = random_state
        self.eligible_gene_columns: list[str] = []
        self.raw_gene_columns: list[str] = []
        self.dropped_rare_columns: list[str] = []
        self.functional_mutation_counts_: dict[str, int] = {}
        self.class_names: list[str] = []
        self.class_gene_weights_: dict[str, dict[str, float]] = {}
        self.hotspots_: list[tuple[str, str, str]] = []
        self.hotspot_fold_support_: dict[tuple[str, str], int] = {}
        self.steps = (
            "동의·기능 변이 분리",
            "고빈도 원시 severity 최대 개수 제한",
            "inner-fold OOF weighted·rate signature",
            "inner-fold 반복 안정 hotspot",
            "비율 중심 compact consequence 집계",
            "consequence one-hot 제거",
        )

    def _build_features(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        severity, functional, multi_hit, counts = build_matrices(
            features, self.eligible_gene_columns
        )
        engineered = severity[self.raw_gene_columns].astype("float32")
        add_compact_summaries(engineered, counts, functional, multi_hit)
        add_signature_features(
            engineered, functional, self.class_names, self.class_gene_weights_
        )
        engineered = pd.concat([
            engineered,
            create_hotspot_features(features, self.hotspots_),
        ], axis=1)
        return engineered, functional

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMV21PreprocessingPipeline":
        (
            self.eligible_gene_columns,
            self.raw_gene_columns,
            self.dropped_rare_columns,
            self.functional_mutation_counts_,
        ) = select_functional_genes(
            features, self.min_functional_mutation_count, self.max_raw_gene_features
        )
        if not self.eligible_gene_columns:
            raise ValueError("최소 기능 변이 지지도를 만족하는 유전자가 없습니다.")
        severity, functional, multi_hit, counts = build_matrices(
            features, self.eligible_gene_columns
        )
        self.class_names, self.class_gene_weights_ = learn_class_weights(
            functional, labels, self.top_genes_per_class, self.smoothing,
            self.max_log2_odds, self.shrinkage,
        )
        self.hotspots_, self.hotspot_fold_support_ = learn_stable_hotspots(
            features, labels, self.eligible_gene_columns, self.min_hotspot_count,
            self.stable_hotspot_folds, self.min_stable_hotspot_folds,
            self.max_stable_hotspots, self.random_state,
        )
        engineered = severity[self.raw_gene_columns].astype("float32")
        add_compact_summaries(engineered, counts, functional, multi_hit)
        add_signature_features(
            engineered, functional, self.class_names, self.class_gene_weights_
        )
        engineered = pd.concat([
            engineered,
            create_hotspot_features(features, self.hotspots_),
        ], axis=1)
        super().fit(engineered, labels)
        print(
            f"[{self.name}] signature 후보 {len(self.eligible_gene_columns)}개, "
            f"원시 severity {len(self.raw_gene_columns)}개, 안정 hotspot {len(self.hotspots_)}개"
        )
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        self.fit(features, labels)
        transformed = self.transform(features)
        _, functional, _, _ = build_matrices(features, self.eligible_gene_columns)
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
        engineered, _ = self._build_features(features)
        return super().transform(engineered).astype("float32")

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        summary.update({
            "eligible_signature_genes": len(self.eligible_gene_columns),
            "raw_gene_features": len(self.raw_gene_columns),
            "removed_raw_gene_features": len(self.eligible_gene_columns) - len(self.raw_gene_columns),
            "stable_hotspot_features": len(self.hotspots_),
            "signature_features": 2 * len(self.class_names),
            "compact_summary_features": 8,
            "consequence_onehot_features": 0,
        })
        return summary
