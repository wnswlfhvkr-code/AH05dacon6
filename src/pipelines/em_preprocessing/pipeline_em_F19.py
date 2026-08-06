"""EM F19: signature 확률 entropy; label 통계는 inner-fold OOF로 생성합니다."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_em_v45 import EMV45PreprocessingPipeline


FEATURE_KIND = "entropy"


def _is_mutated(value: object) -> int:
    if pd.isna(value):
        return 0
    normalized = str(value).strip().upper()
    return int(bool(normalized) and normalized not in {"WT", "<NA>"})


def _presence(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"누락된 유전자 컬럼이 있습니다: {sorted(missing)}")
    return pd.DataFrame({
        gene: features[gene].map(_is_mutated).astype("int8")
        for gene in columns
    }, index=features.index)


def _learn_signatures(
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
        inside_rate = (inside_count + smoothing) / (inside.sum() + 2.0 * smoothing)
        outside_rate = (outside_count + smoothing) / (outside.sum() + 2.0 * smoothing)
        odds = np.log2(
            (inside_rate / np.maximum(1.0 - inside_rate, 1e-12))
            / (outside_rate / np.maximum(1.0 - outside_rate, 1e-12))
        )
        stabilized = np.clip(odds, -max_log2_odds, max_log2_odds)
        stabilized *= inside_count / (inside_count + shrinkage)
        selected = np.argsort(stabilized)[::-1][:top_genes_per_class]
        positive = selected[stabilized[selected] > 0]
        weights[class_index, positive] = stabilized[positive]
        profiles[class_index] = inside_rate
    return weights, profiles


def _signature_matrices(
    mutation: pd.DataFrame,
    weights: np.ndarray,
    profiles: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = mutation.to_numpy(dtype="float64")
    weighted = matrix @ weights.T
    sample_norm = np.sqrt((matrix * matrix).sum(axis=1, keepdims=True))
    profile_norm = np.sqrt((profiles * profiles).sum(axis=1))[None, :]
    similarity = (matrix @ profiles.T) / np.maximum(sample_norm * profile_norm, 1e-12)
    if similarity.shape[1] >= 2:
        top_two = np.sort(np.partition(similarity, -2, axis=1)[:, -2:], axis=1)
        margin = top_two[:, 1] - top_two[:, 0]
    else:
        margin = similarity[:, 0]
    logits = similarity / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
    entropy /= np.log(max(similarity.shape[1], 2))
    return weighted, similarity, margin, entropy


def _select_output(
    mutation: pd.DataFrame,
    class_names: list[str],
    weights: np.ndarray,
    profiles: np.ndarray,
    temperature: float,
) -> pd.DataFrame:
    weighted, similarity, margin, entropy = _signature_matrices(
        mutation, weights, profiles, temperature
    )
    if FEATURE_KIND == "weighted":
        return pd.DataFrame(
            weighted,
            index=mutation.index,
            columns=[f"signature_weighted__{name}" for name in class_names],
            dtype="float32",
        )
    if FEATURE_KIND == "similarity":
        return pd.DataFrame(
            similarity,
            index=mutation.index,
            columns=[f"signature_similarity__{name}" for name in class_names],
            dtype="float32",
        )
    if FEATURE_KIND == "margin":
        return pd.DataFrame(
            {"signature_top2_margin": margin.astype("float32")},
            index=mutation.index,
        )
    return pd.DataFrame(
        {"signature_entropy": entropy.astype("float32")},
        index=mutation.index,
    )


class _F19DerivedFeaturePipeline(PreprocessingPipeline):
    """signature 확률 entropy를 train 통계와 inner-fold OOF로 생성합니다."""

    name = "em_F19"
    evaluation_folds = 5

    def __init__(
        self,
        min_mutation_count: int = 5,
        top_genes_per_class: int = 20,
        smoothing: float = 0.5,
        shrinkage: float = 10.0,
        max_log2_odds: float = 8.0,
        inner_signature_folds: int = 5,
        signature_temperature: float = 1.0,
        random_state: int = 42,
        **_: object,
    ) -> None:
        super().__init__()
        if min_mutation_count < 1 or top_genes_per_class < 1:
            raise ValueError("변이 빈도와 class별 유전자 수는 1 이상이어야 합니다.")
        if inner_signature_folds < 2:
            raise ValueError("inner_signature_folds는 2 이상이어야 합니다.")
        if min(smoothing, shrinkage, max_log2_odds, signature_temperature) <= 0:
            raise ValueError("signature 안정화 파라미터는 0보다 커야 합니다.")
        self.min_mutation_count = int(min_mutation_count)
        self.top_genes_per_class = int(top_genes_per_class)
        self.smoothing = float(smoothing)
        self.shrinkage = float(shrinkage)
        self.max_log2_odds = float(max_log2_odds)
        self.inner_signature_folds = int(inner_signature_folds)
        self.signature_temperature = float(signature_temperature)
        self.random_state = int(random_state)
        self.gene_columns: list[str] = []
        self.selected_genes_: list[str] = []
        self.class_names_: list[str] = []
        self.weights_ = np.empty((0, 0), dtype="float64")
        self.profiles_ = np.empty((0, 0), dtype="float64")
        self.steps = (
            "결측·WT 변이 여부 행렬",
            "fold-train 최소 변이 빈도",
            "암종별 stabilized enrichment signature",
            "학습 행 inner-fold OOF signature 확률 entropy",
            "validation·test에는 fold-train signature transform",
        )

    def _build_with_state(
        self,
        features: pd.DataFrame,
        weights: np.ndarray,
        profiles: np.ndarray,
    ) -> pd.DataFrame:
        mutation = _presence(features, self.selected_genes_)
        return _select_output(
            mutation,
            self.class_names_,
            weights,
            profiles,
            self.signature_temperature,
        )

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMF19PreprocessingPipeline":
        self.gene_columns = features.columns.tolist()
        initial = _presence(features, self.gene_columns)
        support = initial.sum(axis=0)
        self.selected_genes_ = support[
            support >= self.min_mutation_count
        ].index.tolist()
        if not self.selected_genes_:
            raise ValueError("최소 변이 빈도를 만족하는 유전자가 없습니다.")
        aligned_labels = pd.Series(labels.astype(str).to_numpy(), index=features.index)
        self.class_names_ = sorted(aligned_labels.unique().tolist())
        mutation = initial[self.selected_genes_]
        self.weights_, self.profiles_ = _learn_signatures(
            mutation,
            aligned_labels,
            self.class_names_,
            self.top_genes_per_class,
            self.smoothing,
            self.shrinkage,
            self.max_log2_odds,
        )
        engineered = _select_output(
            mutation,
            self.class_names_,
            self.weights_,
            self.profiles_,
            self.signature_temperature,
        )
        PreprocessingPipeline.fit(self, engineered, labels)
        return self

    def _oof_features(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> pd.DataFrame:
        aligned_labels = pd.Series(labels.astype(str).to_numpy(), index=features.index)
        minimum_class = int(aligned_labels.value_counts().min())
        folds = min(self.inner_signature_folds, minimum_class)
        if folds < 2:
            raise ValueError("OOF signature에는 클래스별 표본이 최소 2개 필요합니다.")
        mutation = _presence(features, self.selected_genes_)
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.random_state
        )
        output = pd.DataFrame(index=features.index)
        for train_positions, valid_positions in splitter.split(mutation, aligned_labels):
            train_mutation = mutation.iloc[train_positions]
            train_labels = aligned_labels.iloc[train_positions]
            weights, profiles = _learn_signatures(
                train_mutation,
                train_labels,
                self.class_names_,
                self.top_genes_per_class,
                self.smoothing,
                self.shrinkage,
                self.max_log2_odds,
            )
            fold_output = _select_output(
                mutation.iloc[valid_positions],
                self.class_names_,
                weights,
                profiles,
                self.signature_temperature,
            )
            for column in fold_output:
                output.loc[fold_output.index, column] = fold_output[column]
        return output.astype("float32")

    def fit_transform(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> pd.DataFrame:
        self.fit(features, labels)
        oof = self._oof_features(features, labels)
        PreprocessingPipeline.fit(self, oof, labels)
        return PreprocessingPipeline.transform(self, oof).astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        engineered = self._build_with_state(features, self.weights_, self.profiles_)
        return PreprocessingPipeline.transform(self, engineered).astype("float32")

    def summary(self) -> dict[str, int]:
        result = PreprocessingPipeline.summary(self)
        result.update({
            "selected_signature_genes": len(self.selected_genes_),
            "dropped_rare_genes": len(self.gene_columns) - len(self.selected_genes_),
            "signature_classes": len(self.class_names_),
            "oof_signature_features": len(self.feature_columns),
        })
        return result


class EMF19PreprocessingPipeline(PreprocessingPipeline):
    """EMV45를 베이스로 F19 전용 파생 피처를 결합합니다."""

    name = "em_F19"
    evaluation_folds = 5

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.v45_pipeline = EMV45PreprocessingPipeline(**parameters)
        self.derived_pipeline = _F19DerivedFeaturePipeline(**parameters)
        self.v45_feature_count_ = 0
        self.derived_feature_count_ = 0
        self.steps = (
            "EMV45 베이스 피처",
            "F19 전용 파생 피처",
            "F19 접두사로 컬럼 충돌 제거",
            "학습 행 OOF·validation/test train-fit transform",
        )

    @staticmethod
    def _combine(v45: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
        prefixed = derived.copy()
        prefixed.columns = [f"F19__{column}" for column in prefixed.columns]
        return pd.concat([v45, prefixed], axis=1)

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "EMF19PreprocessingPipeline":
        self.v45_pipeline.fit(features, labels)
        self.derived_pipeline.fit(features, labels)
        v45 = self.v45_pipeline.transform(features)
        derived = self.derived_pipeline.transform(features)
        combined = self._combine(v45, derived)
        self.v45_feature_count_ = v45.shape[1]
        self.derived_feature_count_ = derived.shape[1]
        PreprocessingPipeline.fit(self, combined, labels)
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> pd.DataFrame:
        v45 = self.v45_pipeline.fit_transform(features, labels)
        derived = self.derived_pipeline.fit_transform(features, labels)
        self.v45_feature_count_ = v45.shape[1]
        self.derived_feature_count_ = derived.shape[1]
        combined = self._combine(v45, derived)
        PreprocessingPipeline.fit(self, combined, labels)
        return PreprocessingPipeline.transform(self, combined).astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        combined = self._combine(
            self.v45_pipeline.transform(features),
            self.derived_pipeline.transform(features),
        )
        return PreprocessingPipeline.transform(self, combined).astype("float32")

    def summary(self) -> dict[str, int]:
        result = PreprocessingPipeline.summary(self)
        result.update({
            "v45_base_features": self.v45_feature_count_,
            "f19_derived_features": self.derived_feature_count_,
            "combined_before_constant_filter": (
                self.v45_feature_count_ + self.derived_feature_count_
            ),
        })
        return result
