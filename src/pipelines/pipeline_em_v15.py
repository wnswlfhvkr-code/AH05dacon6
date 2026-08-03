"""v14 전처리에 5-fold OOF 평가 정책을 추가한 EM 버전 15입니다."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.pipelines.base import PreprocessingPipeline


def create_mutation_presence_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """v12의 최소 빈도 계산을 위한 WT/변이 이진 행렬입니다."""
    return pd.DataFrame(
        {
            column: features[column].astype("string").str.strip().str.upper()
            .ne("WT").fillna(False).astype("int8")
            for column in features.columns
        },
        index=features.index,
    )


def select_genes_by_mutation_count(
    features: pd.DataFrame,
    minimum_count: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """v15 학습 데이터에서 최소 변이 횟수를 만족하는 유전자를 선택합니다."""
    mutation_counts = (
        create_mutation_presence_matrix(features).sum(axis=0).astype(int).to_dict()
    )
    selected = [column for column in features.columns if mutation_counts[column] >= minimum_count]
    dropped = [column for column in features.columns if mutation_counts[column] < minimum_count]
    return selected, dropped, mutation_counts


def learn_recurrent_hotspots(
    features: pd.DataFrame,
    columns: list[str],
    minimum_count: int,
    maximum_hotspots: int,
) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str], int]]:
    """학습 Fold에서 반복 관측된 유전자-변이 토큰을 선택합니다.

    복합 변이 문자열은 공백 단위로 나누고, 같은 표본 안에서 같은 토큰이
    반복돼도 환자 수 기준으로 한 번만 집계합니다.
    """
    missing_columns = set(columns) - set(features.columns)
    if missing_columns:
        raise ValueError(f"hotspot 생성에 필요한 피처가 없습니다: {sorted(missing_columns)}")

    support: Counter[tuple[str, str]] = Counter()
    for gene in columns:
        normalized = features[gene].astype("string").str.strip().str.upper()
        mutated = normalized[normalized.ne("WT").fillna(False)].dropna()
        for value in mutated:
            for token in set(str(value).split()):
                support[(gene, token)] += 1

    selected_pairs = sorted(
        (pair for pair, count in support.items() if count >= minimum_count),
        key=lambda pair: (-support[pair], pair[0], pair[1]),
    )[:maximum_hotspots]
    hotspots = [
        (f"hotspot_{index:04d}", gene, token)
        for index, (gene, token) in enumerate(selected_pairs)
    ]
    return hotspots, dict(support)


def create_hotspot_matrix(
    features: pd.DataFrame,
    hotspots: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """학습 Fold에서 선택한 hotspot의 표본별 존재 여부를 만듭니다."""
    by_gene: dict[str, list[tuple[str, str]]] = {}
    for feature_name, gene, token in hotspots:
        by_gene.setdefault(gene, []).append((feature_name, token))

    hotspot_columns: dict[str, pd.Series] = {}
    for gene, gene_hotspots in by_gene.items():
        if gene not in features.columns:
            raise ValueError(f"hotspot 생성에 필요한 유전자 컬럼이 없습니다: {gene}")
        normalized = features[gene].astype("string").str.strip().str.upper()
        token_sets = normalized.map(
            lambda value: frozenset()
            if pd.isna(value) or value == "WT"
            else frozenset(str(value).split())
        )
        for feature_name, token in gene_hotspots:
            hotspot_columns[feature_name] = token_sets.map(
                lambda tokens: token in tokens
            ).astype("int8")
    return pd.DataFrame(hotspot_columns, index=features.index)


def create_consequence_severity_matrix(
    features: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """v15 유전자 변이를 WT=0부터 truncating=4까지로 인코딩합니다."""
    missing_columns = set(columns) - set(features.columns)
    if missing_columns:
        raise ValueError(f"결과 유형 생성에 필요한 피처가 없습니다: {sorted(missing_columns)}")
    severity_columns: dict[str, pd.Series] = {}
    for column in columns:
        normalized = features[column].astype("string").str.strip().str.upper()
        severity = pd.Series(0, index=features.index, dtype="int8")
        severity.loc[normalized.ne("WT").fillna(False)] = 2
        severity.loc[normalized.str.fullmatch(r"([A-Z])\d+\1", na=False)] = 1
        severity.loc[normalized.str.contains(r"DEL|INS|DUP|>", regex=True, na=False)] = 3
        severity.loc[normalized.str.contains(r"FS|\*|TER", regex=True, na=False)] = 4
        severity_columns[column] = severity
    return pd.DataFrame(severity_columns, index=features.index)


def add_consequence_summary_features(
    engineered: pd.DataFrame,
    severity_matrix: pd.DataFrame,
    include_burden_log1p: bool,
) -> pd.DataFrame:
    """v15의 결과 유형별 개수와 비율 및 변이 부담을 추가합니다."""
    burden = severity_matrix.ne(0).sum(axis=1).astype("float32")
    denominator = burden.clip(lower=1.0)
    if include_burden_log1p and "mutation_burden_log1p" not in engineered:
        engineered["mutation_burden_log1p"] = np.log1p(burden).astype("float32")
    for severity, name in enumerate(
        ("synonymous", "missense", "inframe_complex", "truncating"), start=1
    ):
        count = severity_matrix.eq(severity).sum(axis=1).astype("float32")
        engineered[f"consequence_count_{name}"] = count
        engineered[f"consequence_ratio_{name}"] = (count / denominator).astype("float32")
    return engineered


def learn_class_gene_weights(
    mutation_matrix: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """v15 학습 Fold에서 암종별 안정화 log2 odds 가중치를 학습합니다."""
    aligned_labels = labels.reindex(mutation_matrix.index).astype("string")
    class_names = sorted(aligned_labels.dropna().unique().tolist())
    class_gene_weights: dict[str, dict[str, float]] = {}
    for class_name in class_names:
        in_class = aligned_labels.eq(class_name).fillna(False)
        class_size = int(in_class.sum())
        other_size = len(in_class) - class_size
        class_mutations = mutation_matrix.loc[in_class].sum(axis=0).astype("float64")
        other_mutations = mutation_matrix.loc[~in_class].sum(axis=0).astype("float64")
        class_odds = (class_mutations + smoothing) / (class_size - class_mutations + smoothing)
        other_odds = (other_mutations + smoothing) / (other_size - other_mutations + smoothing)
        log2_odds = np.log2(class_odds / other_odds).clip(lower=0.0, upper=max_log2_odds)
        total_mutations = class_mutations + other_mutations
        reliability = np.sqrt(total_mutations / (total_mutations + shrinkage))
        top_scores = (log2_odds * reliability).nlargest(top_genes_per_class)
        top_scores = top_scores[top_scores.gt(0)]
        if top_scores.empty:
            raise ValueError(f"{class_name}의 특징 유전자 그룹을 만들 수 없습니다.")
        class_gene_weights[class_name] = {
            gene: float(score) for gene, score in top_scores.items()
        }
    return class_names, class_gene_weights


def add_class_signature_features(
    engineered: pd.DataFrame,
    mutation_matrix: pd.DataFrame,
    class_names: list[str],
    class_gene_weights: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """v15의 암종별 weighted/match signature를 추가합니다."""
    for class_name in class_names:
        weights = class_gene_weights[class_name]
        genes = list(weights)
        class_matrix = mutation_matrix[genes].astype("float32")
        weight_vector = np.array([weights[gene] for gene in genes], dtype="float32")
        engineered[f"signature_{class_name}_weighted"] = class_matrix.to_numpy().dot(weight_vector)
        engineered[f"signature_{class_name}_match_count"] = class_matrix.sum(axis=1)
    return engineered


class EMV15PreprocessingPipeline(PreprocessingPipeline):
    """severity·signature와 반복 hotspot 변이 정보를 함께 사용합니다."""

    name = "em_v15"
    evaluation_folds = 5

    def __init__(
        self,
        min_mutation_count: int = 5,
        top_genes_per_class: int = 20,
        smoothing: float = 0.5,
        max_log2_odds: float = 8.0,
        shrinkage: float = 10.0,
        min_hotspot_count: int = 5,
        max_hotspots: int = 384,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        if (
            min_mutation_count < 1
            or top_genes_per_class < 1
            or min_hotspot_count < 1
            or max_hotspots < 1
        ):
            raise ValueError("변이 빈도, 클래스별 유전자 수와 hotspot 기준은 1 이상이어야 합니다.")
        if smoothing <= 0 or max_log2_odds <= 0 or shrinkage <= 0:
            raise ValueError("signature 안정화 파라미터는 0보다 커야 합니다.")
        self.min_mutation_count = min_mutation_count
        self.top_genes_per_class = top_genes_per_class
        self.smoothing = float(smoothing)
        self.max_log2_odds = float(max_log2_odds)
        self.shrinkage = float(shrinkage)
        self.min_hotspot_count = min_hotspot_count
        self.max_hotspots = max_hotspots
        self.selected_gene_columns: list[str] = []
        self.dropped_rare_columns: list[str] = []
        self.mutation_counts_: dict[str, int] = {}
        self.class_names: list[str] = []
        self.class_gene_weights_: dict[str, dict[str, float]] = {}
        self.hotspots_: list[tuple[str, str, str]] = []
        self.hotspot_support_: dict[tuple[str, str], int] = {}
        self.steps = (
            "최소 변이 빈도 필터",
            "기능 결과 심각도",
            "기능 결과 구성",
            "암종 signature",
            "학습 Fold 반복 hotspot",
        )

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        severity = create_consequence_severity_matrix(features, self.selected_gene_columns)
        mutation = severity.ne(0).astype("int8")
        engineered = severity.copy()
        add_consequence_summary_features(engineered, severity, include_burden_log1p=True)
        add_class_signature_features(engineered, mutation, self.class_names, self.class_gene_weights_)
        hotspot_matrix = create_hotspot_matrix(features, self.hotspots_)
        engineered = pd.concat([engineered, hotspot_matrix], axis=1)
        return engineered

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMV15PreprocessingPipeline":
        self.selected_gene_columns, self.dropped_rare_columns, self.mutation_counts_ = select_genes_by_mutation_count(features, self.min_mutation_count)
        self.hotspots_, self.hotspot_support_ = learn_recurrent_hotspots(
            features,
            self.selected_gene_columns,
            self.min_hotspot_count,
            self.max_hotspots,
        )
        severity = create_consequence_severity_matrix(features, self.selected_gene_columns)
        mutation = severity.ne(0).astype("int8")
        self.class_names, self.class_gene_weights_ = learn_class_gene_weights(
            mutation,
            labels,
            self.top_genes_per_class,
            self.smoothing,
            self.max_log2_odds,
            self.shrinkage,
        )
        engineered = severity.copy()
        add_consequence_summary_features(engineered, severity, include_burden_log1p=True)
        add_class_signature_features(engineered, mutation, self.class_names, self.class_gene_weights_)
        hotspot_matrix = create_hotspot_matrix(features, self.hotspots_)
        engineered = pd.concat([engineered, hotspot_matrix], axis=1)
        super().fit(engineered, labels)
        print(f"[{self.name}] 생성된 recurrent hotspot 피처: {len(self.hotspots_)}개")
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return super().transform(self._build_features(features)).astype("float32")

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        summary.update({
            "dropped_rare_features": len(self.dropped_rare_columns),
            "consequence_features": 9,
            "signature_features": 2 * len(self.class_names),
            "hotspot_features": len(self.hotspots_),
        })
        return summary
