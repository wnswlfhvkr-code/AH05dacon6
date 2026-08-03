"""em_v16과 em_v27을 중복 없이 결합한 EM v23."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.pipelines.base import PreprocessingPipeline


# 인접 TCGA 연구군은 sibling contrast에만 쓰며 원본 SUBCLASS는 변경하지 않습니다.
TCGA_RELATED_STUDY_SETS = {
    "central_nervous_system": ("GBMLGG", "LGG"),
    "kidney": ("KIPAN", "KIRC"),
    "thoracic": ("LUAD", "LUSC", "THYM"),
    "female_reproductive": ("CESC", "OV", "UCEC"),
    "digestive": ("COAD", "LIHC", "PAAD", "STES"),
    "hematolymphoid": ("DLBC", "LAML"),
    "endocrine_neuroendocrine": ("ACC", "PCPG", "THCA"),
    "male_reproductive": ("PRAD", "TGCT"),
}
PROTECTED_COMPOSITE_LABELS = {"GBMLGG", "KIPAN", "STES"}


def create_mutation_presence(features: pd.DataFrame) -> pd.DataFrame:
    """WT가 아닌 셀을 변이 존재 1로 한 번만 변환합니다."""
    return pd.DataFrame(
        {
            column: features[column].astype("string").fillna("WT")
            .str.strip().str.upper().ne("WT").astype("int8")
            for column in features.columns
        },
        index=features.index,
    )


def create_severity(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """선택 유전자를 WT=0, synonymous=1, missense=2, complex=3, truncating=4로 변환합니다."""
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"결과 유형 생성에 필요한 피처가 없습니다: {sorted(missing)}")
    output: dict[str, pd.Series] = {}
    for column in columns:
        value = features[column].astype("string").fillna("WT").str.strip().str.upper()
        severity = pd.Series(0, index=features.index, dtype="int8")
        severity.loc[value.ne("WT")] = 2
        severity.loc[value.str.fullmatch(r"([A-Z])\d+\1", na=False)] = 1
        severity.loc[value.str.contains(r"DEL|INS|DUP|>", regex=True, na=False)] = 3
        severity.loc[value.str.contains(r"FS|\*|TER", regex=True, na=False)] = 4
        output[column] = severity
    return pd.DataFrame(output, index=features.index)


def add_consequence_summary(output: pd.DataFrame, severity: pd.DataFrame) -> None:
    burden = severity.ne(0).sum(axis=1).astype("float32")
    output["mutation_burden_log1p"] = np.log1p(burden).astype("float32")
    denominator = burden.clip(lower=1)
    for level, name in enumerate(
        ("synonymous", "missense", "inframe_complex", "truncating"), start=1
    ):
        count = severity.eq(level).sum(axis=1).astype("float32")
        output[f"consequence_count_{name}"] = count
        output[f"consequence_ratio_{name}"] = (count / denominator).astype("float32")


def learn_hotspots(
    features: pd.DataFrame,
    columns: list[str],
    minimum_count: int,
    maximum_hotspots: int,
) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str], int]]:
    support: Counter[tuple[str, str]] = Counter()
    for gene in columns:
        values = features[gene].astype("string").fillna("WT").str.strip().str.upper()
        for value in values[values.ne("WT")]:
            for token in set(str(value).split()):
                support[(gene, token)] += 1
    pairs = sorted(
        (pair for pair, count in support.items() if count >= minimum_count),
        key=lambda pair: (-support[pair], pair[0], pair[1]),
    )[:maximum_hotspots]
    hotspots = [
        (f"hotspot_{index:04d}", gene, token)
        for index, (gene, token) in enumerate(pairs)
    ]
    return hotspots, dict(support)


def create_hotspot_features(
    features: pd.DataFrame,
    hotspots: list[tuple[str, str, str]],
) -> pd.DataFrame:
    by_gene: dict[str, list[tuple[str, str]]] = {}
    for feature_name, gene, token in hotspots:
        by_gene.setdefault(gene, []).append((feature_name, token))
    output: dict[str, pd.Series] = {}
    for gene, definitions in by_gene.items():
        token_sets = features[gene].astype("string").fillna("WT").str.strip().str.upper().map(
            lambda value: frozenset() if value == "WT" else frozenset(str(value).split())
        )
        for feature_name, token in definitions:
            output[feature_name] = token_sets.map(lambda tokens: token in tokens).astype("int8")
    return pd.DataFrame(output, index=features.index)


def stabilized_weights(
    mutation: pd.DataFrame,
    positive: pd.Series,
    negative: pd.Series,
    top_genes: int,
    smoothing: float,
    shrinkage: float,
    max_log2_odds: float,
) -> dict[str, float]:
    positive = positive.reindex(mutation.index).fillna(False).astype(bool)
    negative = negative.reindex(mutation.index).fillna(False).astype(bool)
    positive_count, negative_count = int(positive.sum()), int(negative.sum())
    if positive_count == 0 or negative_count == 0:
        return {}
    in_group = mutation.loc[positive].sum(axis=0).astype("float64")
    out_group = mutation.loc[negative].sum(axis=0).astype("float64")
    in_odds = (in_group + smoothing) / (positive_count - in_group + smoothing)
    out_odds = (out_group + smoothing) / (negative_count - out_group + smoothing)
    score = np.log2(in_odds / out_odds).clip(0.0, max_log2_odds)
    support = in_group + out_group
    score *= np.sqrt(support / (support + shrinkage))
    selected = score.nlargest(top_genes)
    return selected[selected.gt(0)].astype(float).to_dict()


def related_members(class_name: str) -> tuple[str, ...] | None:
    for members in TCGA_RELATED_STUDY_SETS.values():
        if class_name in members:
            return members
    return None


def learn_all_weights(
    mutation: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    top_genes_per_sibling: int,
    smoothing: float,
    shrinkage: float,
    max_log2_odds: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """동일 변이 행렬에서 class 및 related-study sibling 가중치를 학습합니다."""
    aligned = labels.reindex(mutation.index).astype("string")
    class_weights: dict[str, dict[str, float]] = {}
    sibling_weights: dict[str, dict[str, float]] = {}
    for raw_class_name in sorted(aligned.dropna().unique()):
        class_name = str(raw_class_name)
        weights = stabilized_weights(
            mutation, aligned.eq(class_name), aligned.ne(class_name),
            top_genes_per_class, smoothing, shrinkage, max_log2_odds,
        )
        if not weights:
            raise ValueError(f"{class_name}의 암종 signature를 만들 수 없습니다.")
        class_weights[class_name] = weights

        members = related_members(class_name)
        if members is None:
            continue
        positive = aligned.eq(class_name)
        negative = aligned.isin(members) & ~positive
        weights = stabilized_weights(
            mutation, positive, negative, top_genes_per_sibling,
            smoothing, shrinkage, max_log2_odds,
        )
        if not weights:
            raise ValueError(f"{class_name}의 sibling contrast를 만들 수 없습니다.")
        sibling_weights[class_name] = weights
    return class_weights, sibling_weights


def add_supervised_signatures(
    output: pd.DataFrame,
    mutation: pd.DataFrame,
    class_weights: dict[str, dict[str, float]],
    sibling_weights: dict[str, dict[str, float]],
) -> None:
    burden = mutation.sum(axis=1).clip(lower=1).to_numpy(dtype="float32")
    for class_name, weights in class_weights.items():
        genes = list(weights)
        matrix = mutation[genes].to_numpy(dtype="float32")
        vector = np.array([weights[gene] for gene in genes], dtype="float32")
        output[f"signature_{class_name}_weighted"] = matrix @ vector
        output[f"signature_{class_name}_match_count"] = matrix.sum(axis=1)
    for class_name, weights in sibling_weights.items():
        genes = list(weights)
        matrix = mutation[genes].to_numpy(dtype="float32")
        vector = np.array([weights[gene] for gene in genes], dtype="float32")
        output[f"tcga_sibling_{class_name}_weighted"] = matrix @ vector
        output[f"tcga_sibling_{class_name}_match_rate"] = matrix.sum(axis=1) / burden


def create_oof_signatures(
    mutation: pd.DataFrame,
    labels: pd.Series,
    folds: int,
    random_state: int,
    top_genes_per_class: int,
    top_genes_per_sibling: int,
    smoothing: float,
    shrinkage: float,
    max_log2_odds: float,
) -> pd.DataFrame:
    """하나의 inner-fold 분할로 두 signature 집합을 교차 적합합니다."""
    aligned = labels.reindex(mutation.index)
    n_splits = min(folds, int(aligned.value_counts().min()))
    if n_splits < 2:
        raise ValueError("OOF signature에는 클래스별 표본이 최소 2개 필요합니다.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = pd.DataFrame(index=mutation.index)
    for train_positions, valid_positions in splitter.split(mutation, aligned):
        train, valid = mutation.iloc[train_positions], mutation.iloc[valid_positions]
        class_weights, sibling_weights = learn_all_weights(
            train, aligned.iloc[train_positions], top_genes_per_class,
            top_genes_per_sibling, smoothing, shrinkage, max_log2_odds,
        )
        fold_output = pd.DataFrame(index=valid.index)
        add_supervised_signatures(fold_output, valid, class_weights, sibling_weights)
        for column in fold_output:
            result.loc[valid.index, column] = fold_output[column]
    return result.astype("float32")


class EMV23PreprocessingPipeline(PreprocessingPipeline):
    """v16 severity/class/hotspot에 v27 sibling contrast만 추가합니다."""

    name = "em_v23"
    evaluation_folds = 5

    def __init__(
        self,
        min_mutation_count: int = 5,
        max_raw_gene_features: int = 2000,
        top_genes_per_class: int = 20,
        top_genes_per_sibling: int = 20,
        smoothing: float = 0.5,
        max_log2_odds: float = 8.0,
        shrinkage: float = 10.0,
        min_hotspot_count: int = 5,
        max_hotspots: int = 384,
        inner_signature_folds: int = 5,
        random_state: int = 42,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        integer_values = {
            "min_mutation_count": (min_mutation_count, 1),
            "max_raw_gene_features": (max_raw_gene_features, 1),
            "top_genes_per_class": (top_genes_per_class, 1),
            "top_genes_per_sibling": (top_genes_per_sibling, 1),
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
        self.max_raw_gene_features = max_raw_gene_features
        self.top_genes_per_class = top_genes_per_class
        self.top_genes_per_sibling = top_genes_per_sibling
        self.smoothing = float(smoothing)
        self.max_log2_odds = float(max_log2_odds)
        self.shrinkage = float(shrinkage)
        self.min_hotspot_count = min_hotspot_count
        self.max_hotspots = max_hotspots
        self.inner_signature_folds = inner_signature_folds
        self.random_state = random_state
        self.selected_gene_columns: list[str] = []
        self.dropped_rare_columns: list[str] = []
        self.dropped_capped_columns: list[str] = []
        self.mutation_counts_: dict[str, int] = {}
        self.class_gene_weights_: dict[str, dict[str, float]] = {}
        self.sibling_gene_weights_: dict[str, dict[str, float]] = {}
        self.hotspots_: list[tuple[str, str, str]] = []
        self.hotspot_support_: dict[tuple[str, str], int] = {}
        self.steps = (
            "원본 SUBCLASS 보존",
            "공통 최소 변이 빈도와 원시 유전자 상한",
            "기능 결과 심각도와 구성",
            "공통 inner-fold OOF class/sibling signature",
            "학습 Fold 반복 hotspot",
        )

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        severity = create_severity(features, self.selected_gene_columns)
        mutation = severity.ne(0).astype("int8")
        output = severity.copy()
        add_consequence_summary(output, severity)
        add_supervised_signatures(
            output, mutation, self.class_gene_weights_, self.sibling_gene_weights_
        )
        return pd.concat(
            [output, create_hotspot_features(features, self.hotspots_)], axis=1
        )

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMV23PreprocessingPipeline":
        all_mutation = create_mutation_presence(features)
        counts = all_mutation.sum(axis=0).astype(int)
        self.mutation_counts_ = counts.to_dict()
        eligible = [
            gene for gene in features.columns if counts[gene] >= self.min_mutation_count
        ]
        self.selected_gene_columns = sorted(
            eligible, key=lambda gene: (-self.mutation_counts_[gene], gene)
        )[:self.max_raw_gene_features]
        selected = set(self.selected_gene_columns)
        eligible_set = set(eligible)
        self.dropped_rare_columns = [gene for gene in features.columns if gene not in eligible_set]
        self.dropped_capped_columns = [gene for gene in eligible if gene not in selected]
        if not self.selected_gene_columns:
            raise ValueError("최소 변이 빈도를 만족하는 유전자 피처가 없습니다.")
        self.hotspots_, self.hotspot_support_ = learn_hotspots(
            features, self.selected_gene_columns, self.min_hotspot_count, self.max_hotspots
        )
        mutation = all_mutation[self.selected_gene_columns]
        self.class_gene_weights_, self.sibling_gene_weights_ = learn_all_weights(
            mutation, labels, self.top_genes_per_class, self.top_genes_per_sibling,
            self.smoothing, self.shrinkage, self.max_log2_odds,
        )
        super().fit(self._build_features(features), labels)
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        self.fit(features, labels)
        transformed = self.transform(features)
        mutation = create_mutation_presence(features)[self.selected_gene_columns]
        oof = create_oof_signatures(
            mutation, labels, self.inner_signature_folds, self.random_state,
            self.top_genes_per_class, self.top_genes_per_sibling, self.smoothing,
            self.shrinkage, self.max_log2_odds,
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
            "dropped_rare_features": len(self.dropped_rare_columns),
            "dropped_by_feature_cap": len(self.dropped_capped_columns),
            "dropped_gene_features": len(self.dropped_rare_columns) + len(self.dropped_capped_columns),
            "selected_raw_gene_features": len(self.selected_gene_columns),
            "consequence_features": 9,
            "class_signature_features": 2 * len(self.class_gene_weights_),
            "tcga_sibling_signature_features": 2 * len(self.sibling_gene_weights_),
            "hotspot_features": len(self.hotspots_),
            "protected_composite_labels": len(PROTECTED_COMPOSITE_LABELS),
        })
        return result
