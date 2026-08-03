"""관련 TCGA 연구군 내부에서 원본 SUBCLASS를 대비하는 EM v27."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.pipelines.base import PreprocessingPipeline


# 공식 study name이 가리키는 인접 해부학 연구군입니다. 레이블 병합에 사용하지 않습니다.
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


def create_mutation_presence(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"변환에 필요한 유전자 컬럼이 없습니다: {sorted(missing)}")
    return pd.DataFrame(
        {
            column: features[column].astype("string").fillna("WT")
            .str.strip().str.upper().ne("WT").astype("int8")
            for column in columns
        },
        index=features.index,
    )


def related_set_for_label(label: str) -> tuple[str, tuple[str, ...]] | None:
    for set_name, members in TCGA_RELATED_STUDY_SETS.items():
        if label in members:
            return set_name, members
    return None


def learn_within_set_weights(
    mutation: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    smoothing: float,
    shrinkage: float,
    max_log2_odds: float,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """같은 연구군의 다른 원본 클래스만 음성군으로 사용하는 signature입니다."""
    aligned = labels.reindex(mutation.index).astype("string")
    class_names: list[str] = []
    weights: dict[str, dict[str, float]] = {}
    for class_name in sorted(aligned.dropna().unique()):
        related = related_set_for_label(str(class_name))
        if related is None:
            continue
        _, members = related
        in_related_set = aligned.isin(members).fillna(False)
        positive = aligned.eq(class_name).fillna(False)
        negative = in_related_set & ~positive
        positive_count = int(positive.sum())
        negative_count = int(negative.sum())
        if positive_count == 0 or negative_count == 0:
            continue
        class_mutations = mutation.loc[positive].sum(axis=0).astype("float64")
        sibling_mutations = mutation.loc[negative].sum(axis=0).astype("float64")
        class_odds = (class_mutations + smoothing) / (
            positive_count - class_mutations + smoothing
        )
        sibling_odds = (sibling_mutations + smoothing) / (
            negative_count - sibling_mutations + smoothing
        )
        score = np.log2(class_odds / sibling_odds).clip(0.0, max_log2_odds)
        support = class_mutations + sibling_mutations
        score *= np.sqrt(support / (support + shrinkage))
        selected = score.nlargest(top_genes_per_class)
        selected = selected[selected.gt(0)]
        if selected.empty:
            raise ValueError(f"{class_name}의 sibling contrast를 만들 수 없습니다.")
        class_names.append(str(class_name))
        weights[str(class_name)] = selected.astype(float).to_dict()
    return class_names, weights


def add_within_set_signatures(
    output: pd.DataFrame,
    mutation: pd.DataFrame,
    class_names: list[str],
    weights: dict[str, dict[str, float]],
) -> None:
    burden = mutation.sum(axis=1).clip(lower=1).to_numpy(dtype="float32")
    for class_name in class_names:
        class_weights = weights[class_name]
        genes = list(class_weights)
        matrix = mutation[genes].to_numpy(dtype="float32")
        vector = np.array([class_weights[gene] for gene in genes], dtype="float32")
        output[f"tcga_sibling_{class_name}_weighted"] = matrix @ vector
        output[f"tcga_sibling_{class_name}_match_rate"] = matrix.sum(axis=1) / burden


def create_oof_within_set_signatures(
    mutation: pd.DataFrame,
    labels: pd.Series,
    folds: int,
    random_state: int,
    top_genes_per_class: int,
    smoothing: float,
    shrinkage: float,
    max_log2_odds: float,
) -> pd.DataFrame:
    aligned = labels.reindex(mutation.index)
    n_splits = min(folds, int(aligned.value_counts().min()))
    if n_splits < 2:
        raise ValueError("OOF sibling signature에는 클래스별 표본이 최소 2개 필요합니다.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = pd.DataFrame(index=mutation.index)
    for train_positions, valid_positions in splitter.split(mutation, aligned):
        train = mutation.iloc[train_positions]
        valid = mutation.iloc[valid_positions]
        names, weights = learn_within_set_weights(
            train, aligned.iloc[train_positions], top_genes_per_class,
            smoothing, shrinkage, max_log2_odds,
        )
        fold_output = pd.DataFrame(index=valid.index)
        add_within_set_signatures(fold_output, valid, names, weights)
        for column in fold_output:
            result.loc[valid.index, column] = fold_output[column]
    return result.astype("float32")


class EMV27PreprocessingPipeline(PreprocessingPipeline):
    """TCGA 관련 study 내부의 fine-grained contrast를 추가합니다."""

    name = "em_v27"
    evaluation_folds = 5

    def __init__(
        self,
        min_mutation_count: int = 5,
        max_raw_gene_features: int = 2000,
        top_genes_per_class: int = 20,
        smoothing: float = 0.5,
        shrinkage: float = 10.0,
        max_log2_odds: float = 8.0,
        inner_contrast_folds: int = 5,
        random_state: int = 42,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        integer_values = {
            "min_mutation_count": (min_mutation_count, 1),
            "max_raw_gene_features": (max_raw_gene_features, 1),
            "top_genes_per_class": (top_genes_per_class, 1),
            "inner_contrast_folds": (inner_contrast_folds, 2),
        }
        for name, (value, minimum) in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name}는 {minimum} 이상의 정수여야 합니다.")
        if smoothing <= 0 or shrinkage <= 0 or max_log2_odds <= 0:
            raise ValueError("signature 안정화 파라미터는 0보다 커야 합니다.")
        self.min_mutation_count = min_mutation_count
        self.max_raw_gene_features = max_raw_gene_features
        self.top_genes_per_class = top_genes_per_class
        self.smoothing = float(smoothing)
        self.shrinkage = float(shrinkage)
        self.max_log2_odds = float(max_log2_odds)
        self.inner_contrast_folds = inner_contrast_folds
        self.random_state = random_state
        self.input_gene_columns: list[str] = []
        self.selected_gene_columns: list[str] = []
        self.dropped_gene_columns: list[str] = []
        self.mutation_counts_: dict[str, int] = {}
        self.contrast_class_names_: list[str] = []
        self.contrast_gene_weights_: dict[str, dict[str, float]] = {}
        self.steps = (
            "TCGA Study Abbreviation 원본 레이블 보존",
            "최소 변이 빈도와 원시 유전자 상한",
            "관련 study set 내부 one-vs-sibling contrast",
            "inner-fold OOF fine-grained mutation signature",
        )

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        mutation = create_mutation_presence(features, self.selected_gene_columns)
        output = mutation.astype("float32")
        output["mutation_burden_log1p"] = np.log1p(
            mutation.sum(axis=1).astype("float32")
        )
        add_within_set_signatures(
            output, mutation, self.contrast_class_names_, self.contrast_gene_weights_
        )
        return output

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMV27PreprocessingPipeline":
        original_labels = labels.astype("string")
        self.input_gene_columns = features.columns.tolist()
        all_mutation = create_mutation_presence(features, self.input_gene_columns)
        counts = all_mutation.sum(axis=0).astype(int)
        self.mutation_counts_ = counts.to_dict()
        eligible = [gene for gene in self.input_gene_columns if counts[gene] >= self.min_mutation_count]
        self.selected_gene_columns = sorted(
            eligible, key=lambda gene: (-self.mutation_counts_[gene], gene)
        )[:self.max_raw_gene_features]
        selected = set(self.selected_gene_columns)
        self.dropped_gene_columns = [gene for gene in self.input_gene_columns if gene not in selected]
        mutation = all_mutation[self.selected_gene_columns]
        self.contrast_class_names_, self.contrast_gene_weights_ = learn_within_set_weights(
            mutation, original_labels, self.top_genes_per_class, self.smoothing,
            self.shrinkage, self.max_log2_odds,
        )
        super().fit(self._build_features(features), labels)
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        self.fit(features, labels)
        transformed = self.transform(features)
        mutation = create_mutation_presence(features, self.selected_gene_columns)
        oof = create_oof_within_set_signatures(
            mutation, labels, self.inner_contrast_folds, self.random_state,
            self.top_genes_per_class, self.smoothing, self.shrinkage,
            self.max_log2_odds,
        )
        for column in oof:
            if column in transformed:
                transformed.loc[:, column] = oof[column].to_numpy()
        return transformed.astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return super().transform(self._build_features(features)).astype("float32")

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        summary.update({
            "selected_raw_gene_features": len(self.selected_gene_columns),
            "dropped_gene_features": len(self.dropped_gene_columns),
            "tcga_related_study_sets": len(TCGA_RELATED_STUDY_SETS),
            "tcga_contrast_classes": len(self.contrast_class_names_),
            "tcga_sibling_signature_features": 2 * len(self.contrast_class_names_),
            "protected_composite_labels": len(PROTECTED_COMPOSITE_LABELS),
        })
        return summary
