"""TCGA study name 기반 해부학적 연구군 signature를 생성하는 EM v26."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.pipelines.base import PreprocessingPipeline


# NCI GDC study name에서 유도한 모델링용 보조 그룹이며 공식 TCGA 계층은 아닙니다.
# SUBCLASS 값은 변경하지 않고 signature 학습의 보조 타깃으로만 사용합니다.
TCGA_STUDY_FAMILIES = {
    "urothelial": ("BLCA",),
    "breast": ("BRCA",),
    "female_reproductive": ("CESC", "OV", "UCEC"),
    "digestive": ("COAD", "LIHC", "PAAD", "STES"),
    "hematolymphoid": ("DLBC", "LAML"),
    "central_nervous_system": ("GBMLGG", "LGG"),
    "head_and_neck": ("HNSC",),
    "kidney": ("KIPAN", "KIRC"),
    "thoracic": ("LUAD", "LUSC", "THYM"),
    "endocrine_neuroendocrine": ("ACC", "PCPG", "THCA"),
    "male_reproductive": ("PRAD", "TGCT"),
    "mesenchymal": ("SARC",),
    "skin": ("SKCM",),
}
PROTECTED_COMPOSITE_LABELS = {"GBMLGG", "KIPAN", "STES"}


def create_mutation_presence(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """WT가 아닌 셀을 유전자 변이 존재 1로 변환합니다."""
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


def map_labels_to_families(labels: pd.Series) -> pd.Series:
    """원본 레이블을 수정하지 않고 보조 연구군 이름만 별도로 만듭니다."""
    lookup = {
        label: family
        for family, members in TCGA_STUDY_FAMILIES.items()
        for label in members
    }
    normalized = labels.astype("string")
    missing = sorted(set(normalized.dropna().unique()) - set(lookup))
    if missing:
        raise ValueError(f"TCGA 보조 연구군에 정의되지 않은 SUBCLASS: {missing}")
    return normalized.map(lookup).astype("string")


def learn_family_weights(
    mutation: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_family: int,
    smoothing: float,
    shrinkage: float,
    max_log2_odds: float,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """각 보조 연구군 대 나머지 표본의 안정화 log2 odds를 학습합니다."""
    families = map_labels_to_families(labels.reindex(mutation.index))
    # 단일 study 그룹은 기존 one-vs-rest class signature와 같으므로 제외합니다.
    family_names = sorted(
        family for family in families.dropna().unique()
        if len(TCGA_STUDY_FAMILIES[str(family)]) >= 2
    )
    weights: dict[str, dict[str, float]] = {}
    for family in family_names:
        positive = families.eq(family).fillna(False)
        positive_count = int(positive.sum())
        negative_count = len(positive) - positive_count
        in_group = mutation.loc[positive].sum(axis=0).astype("float64")
        out_group = mutation.loc[~positive].sum(axis=0).astype("float64")
        in_odds = (in_group + smoothing) / (positive_count - in_group + smoothing)
        out_odds = (out_group + smoothing) / (negative_count - out_group + smoothing)
        score = np.log2(in_odds / out_odds).clip(0.0, max_log2_odds)
        support = in_group + out_group
        score *= np.sqrt(support / (support + shrinkage))
        selected = score.nlargest(top_genes_per_family)
        selected = selected[selected.gt(0)]
        if selected.empty:
            raise ValueError(f"{family} 연구군 signature를 만들 수 없습니다.")
        weights[family] = selected.astype(float).to_dict()
    return family_names, weights


def add_family_signatures(
    output: pd.DataFrame,
    mutation: pd.DataFrame,
    family_names: list[str],
    weights: dict[str, dict[str, float]],
) -> None:
    burden = mutation.sum(axis=1).clip(lower=1).astype("float32")
    for family in family_names:
        family_weights = weights[family]
        genes = list(family_weights)
        matrix = mutation[genes].to_numpy(dtype="float32")
        vector = np.array([family_weights[gene] for gene in genes], dtype="float32")
        output[f"tcga_family_{family}_weighted"] = matrix @ vector
        output[f"tcga_family_{family}_match_rate"] = (
            matrix.sum(axis=1) / burden.to_numpy(dtype="float32")
        )


def create_oof_family_signatures(
    mutation: pd.DataFrame,
    labels: pd.Series,
    folds: int,
    random_state: int,
    top_genes_per_family: int,
    smoothing: float,
    shrinkage: float,
    max_log2_odds: float,
) -> pd.DataFrame:
    """각 표본을 제외한 inner fold에서 보조 연구군 signature를 만듭니다."""
    aligned = labels.reindex(mutation.index)
    family_labels = map_labels_to_families(aligned)
    n_splits = min(folds, int(family_labels.value_counts().min()))
    if n_splits < 2:
        raise ValueError("OOF family signature에는 연구군별 표본이 최소 2개 필요합니다.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = pd.DataFrame(index=mutation.index)
    for train_positions, valid_positions in splitter.split(mutation, family_labels):
        train = mutation.iloc[train_positions]
        valid = mutation.iloc[valid_positions]
        names, weights = learn_family_weights(
            train,
            aligned.iloc[train_positions],
            top_genes_per_family,
            smoothing,
            shrinkage,
            max_log2_odds,
        )
        fold_output = pd.DataFrame(index=valid.index)
        add_family_signatures(fold_output, valid, names, weights)
        for column in fold_output:
            result.loc[valid.index, column] = fold_output[column]
    return result.astype("float32")


class EMV26PreprocessingPipeline(PreprocessingPipeline):
    """원본 26개 레이블을 유지하며 TCGA study family signature를 추가합니다."""

    name = "em_v26"
    evaluation_folds = 5

    def __init__(
        self,
        min_mutation_count: int = 5,
        max_raw_gene_features: int = 2000,
        top_genes_per_family: int = 20,
        smoothing: float = 0.5,
        shrinkage: float = 10.0,
        max_log2_odds: float = 8.0,
        inner_family_folds: int = 5,
        random_state: int = 42,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        integer_values = {
            "min_mutation_count": (min_mutation_count, 1),
            "max_raw_gene_features": (max_raw_gene_features, 1),
            "top_genes_per_family": (top_genes_per_family, 1),
            "inner_family_folds": (inner_family_folds, 2),
        }
        for name, (value, minimum) in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name}는 {minimum} 이상의 정수여야 합니다.")
        if smoothing <= 0 or shrinkage <= 0 or max_log2_odds <= 0:
            raise ValueError("signature 안정화 파라미터는 0보다 커야 합니다.")
        self.min_mutation_count = min_mutation_count
        self.max_raw_gene_features = max_raw_gene_features
        self.top_genes_per_family = top_genes_per_family
        self.smoothing = float(smoothing)
        self.shrinkage = float(shrinkage)
        self.max_log2_odds = float(max_log2_odds)
        self.inner_family_folds = inner_family_folds
        self.random_state = random_state
        self.input_gene_columns: list[str] = []
        self.selected_gene_columns: list[str] = []
        self.dropped_gene_columns: list[str] = []
        self.mutation_counts_: dict[str, int] = {}
        self.family_names_: list[str] = []
        self.family_gene_weights_: dict[str, dict[str, float]] = {}
        self.steps = (
            "TCGA Study Abbreviation 원본 레이블 보존",
            "최소 변이 빈도와 원시 유전자 상한",
            "study name 기반 보조 해부학 연구군",
            "inner-fold OOF 연구군 mutation signature",
        )

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        mutation = create_mutation_presence(features, self.selected_gene_columns)
        output = mutation.astype("float32")
        output["mutation_burden_log1p"] = np.log1p(
            mutation.sum(axis=1).astype("float32")
        )
        add_family_signatures(
            output, mutation, self.family_names_, self.family_gene_weights_
        )
        return output

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMV26PreprocessingPipeline":
        original_labels = labels.astype("string")
        map_labels_to_families(original_labels)
        self.input_gene_columns = features.columns.tolist()
        all_mutation = create_mutation_presence(features, self.input_gene_columns)
        counts = all_mutation.sum(axis=0).astype(int)
        self.mutation_counts_ = counts.to_dict()
        eligible = [gene for gene in self.input_gene_columns if counts[gene] >= self.min_mutation_count]
        self.selected_gene_columns = sorted(
            eligible, key=lambda gene: (-self.mutation_counts_[gene], gene)
        )[:self.max_raw_gene_features]
        self.dropped_gene_columns = [gene for gene in self.input_gene_columns if gene not in set(self.selected_gene_columns)]
        mutation = all_mutation[self.selected_gene_columns]
        self.family_names_, self.family_gene_weights_ = learn_family_weights(
            mutation, labels, self.top_genes_per_family, self.smoothing,
            self.shrinkage, self.max_log2_odds,
        )
        super().fit(self._build_features(features), labels)
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        self.fit(features, labels)
        transformed = self.transform(features)
        mutation = create_mutation_presence(features, self.selected_gene_columns)
        oof = create_oof_family_signatures(
            mutation, labels, self.inner_family_folds, self.random_state,
            self.top_genes_per_family, self.smoothing, self.shrinkage,
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
            "tcga_study_family_groups": len(self.family_names_),
            "tcga_family_signature_features": 2 * len(self.family_names_),
            "protected_composite_labels": len(PROTECTED_COMPOSITE_LABELS),
        })
        return summary
