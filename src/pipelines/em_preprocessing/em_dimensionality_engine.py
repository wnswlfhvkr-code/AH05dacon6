"""EMV46 출력을 대상으로 하는 fold-safe 차원 축소 공통 엔진."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import MiniBatchNMF, TruncatedSVD
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_em_v16 import (
    create_hotspot_matrix,
    learn_recurrent_hotspots,
)
from src.pipelines.pipeline_em_v46 import EMV46PreprocessingPipeline


STRATEGIES = {
    "frequency",
    "chi2",
    "mutual_information",
    "class_gene_union",
    "nmf",
    "svd",
    "genes_burden_hotspot",
}


def create_mutation_presence_matrix(
    features: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """원본 유전자에서 결측·WT=0, 관찰된 변이=1 행렬을 생성합니다."""
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"변이 행렬 생성에 필요한 피처가 없습니다: {sorted(missing)}")
    return pd.DataFrame(
        {
            column: features[column].astype("string").fillna("WT")
            .str.strip().str.upper().ne("WT").astype("int8")
            for column in columns
        },
        index=features.index,
    )


class EMDimensionalityEngine:
    """EMV46과 선택·분해 상태를 학습 fold에서만 fit합니다."""

    def __init__(self, strategy: str, parameters: dict[str, object]) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"지원하지 않는 차원 축소 방식입니다: {strategy}")
        self.strategy = strategy
        self.min_mutation_count = int(parameters.get("min_mutation_count", 5))
        self.k_features = int(parameters.get("k_features", 1500))
        self.top_genes_per_class = int(parameters.get("top_genes_per_class", 20))
        self.n_components = int(parameters.get("n_components", 128))
        self.max_iter = int(parameters.get("max_iter", 300))
        self.batch_size = int(parameters.get("batch_size", 256))
        self.svd_n_iter = int(parameters.get("svd_n_iter", 7))
        self.min_hotspot_count = int(parameters.get("min_hotspot_count", 5))
        self.max_hotspots = int(parameters.get("max_hotspots", 384))
        self.smoothing = float(parameters.get("smoothing", 0.5))
        self.shrinkage = float(parameters.get("shrinkage", 10.0))
        self.active_epsilon = float(parameters.get("active_epsilon", 1e-8))
        self.random_state = int(parameters.get("random_state", 42))
        for name, value in (
            ("min_mutation_count", self.min_mutation_count),
            ("k_features", self.k_features),
            ("top_genes_per_class", self.top_genes_per_class),
            ("n_components", self.n_components),
            ("max_iter", self.max_iter),
            ("batch_size", self.batch_size),
            ("svd_n_iter", self.svd_n_iter),
            ("min_hotspot_count", self.min_hotspot_count),
            ("max_hotspots", self.max_hotspots),
        ):
            if value < 1:
                raise ValueError(f"{name}는 1 이상이어야 합니다.")

        engine_only = {
            "k_features",
            "n_components",
            "max_iter",
            "batch_size",
            "svd_n_iter",
            "active_epsilon",
        }
        v46_parameters = {
            key: value for key, value in parameters.items() if key not in engine_only
        }
        self.base_pipeline = EMV46PreprocessingPipeline(**v46_parameters)
        self.raw_input_columns_: list[str] = []
        self.input_columns_: list[str] = []
        self.frequency_feature_columns_: list[str] = []
        self.selected_feature_columns_: list[str] = []
        self.h07_gene_columns_: list[str] = []
        self.dropped_rare_columns_: list[str] = []
        self.active_counts_: dict[str, int] = {}
        self.feature_scores_: dict[str, float] = {}
        self.class_selected_genes_: dict[str, list[str]] = {}
        self.hotspots_: list[tuple[str, str, str]] = []
        self.hotspot_support_: dict[tuple[str, str], int] = {}
        self.reducer_: MiniBatchNMF | TruncatedSVD | None = None
        self.effective_components_ = 0
        self.base_feature_count_ = 0

    @property
    def steps(self) -> tuple[str, ...]:
        names = {
            "frequency": "최소 활성 빈도 필터만 적용",
            "chi2": "fold-train Chi-square 상위 피처 선택",
            "mutual_information": "fold-train mutual information 상위 피처 선택",
            "class_gene_union": "암종별 enrichment 상위 유전자 합집합",
            "nmf": "MiniBatch NMF 비음수 변이 모듈",
            "svd": "Truncated SVD 잠재 피처",
            "genes_burden_hotspot": "EMV46 유전자와 burden·hotspot 결합",
        }
        return (
            "EMV46 베이스 피처 1회 생성",
            "학습 fold 최소 활성 빈도 필터",
            names[self.strategy],
            "validation/test에는 train-fit 상태로 transform만 적용",
            "GBMLGG·KIPAN·STES 원본 레이블 유지",
        )

    @staticmethod
    def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0).astype("float32")

    def _learn_frequency(self, matrix: pd.DataFrame) -> pd.DataFrame:
        self.input_columns_ = matrix.columns.tolist()
        active = matrix.abs().gt(self.active_epsilon).sum(axis=0).astype(int)
        self.active_counts_ = active.to_dict()
        self.frequency_feature_columns_ = [
            column for column in self.input_columns_
            if self.active_counts_[column] >= self.min_mutation_count
        ]
        self.dropped_rare_columns_ = [
            column for column in self.input_columns_
            if column not in self.frequency_feature_columns_
        ]
        if not self.frequency_feature_columns_:
            raise ValueError("최소 활성 빈도를 만족하는 EMV46 피처가 없습니다.")
        return matrix[self.frequency_feature_columns_]

    def _select_supervised(
        self,
        matrix: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        encoded = LabelEncoder().fit_transform(labels.astype("string"))
        if self.strategy == "chi2":
            scores, _ = chi2(matrix.clip(lower=0.0), encoded)
        else:
            scores = mutual_info_classif(
                matrix,
                encoded,
                discrete_features=False,
                random_state=self.random_state,
            )
        safe_scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        self.feature_scores_ = dict(
            zip(matrix.columns, safe_scores.astype(float), strict=True)
        )
        selected_count = min(self.k_features, matrix.shape[1])
        self.selected_feature_columns_ = sorted(
            matrix.columns,
            key=lambda column: (-self.feature_scores_[column], column),
        )[:selected_count]
        return matrix[self.selected_feature_columns_]

    def _select_class_union(
        self,
        matrix: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        gene_columns = [
            column for column in self.raw_input_columns_ if column in matrix.columns
        ]
        if not gene_columns:
            raise ValueError("EMV46 출력에서 원본 유전자 피처를 찾을 수 없습니다.")
        presence = matrix[gene_columns].ne(0.0).astype("float32")
        label_values = labels.astype("string").to_numpy()
        selected: set[str] = set()
        self.class_selected_genes_ = {}
        for class_name in sorted(pd.unique(label_values).tolist()):
            in_class = label_values == class_name
            class_size = int(in_class.sum())
            other_size = len(in_class) - class_size
            class_mutations = presence.iloc[in_class].sum(axis=0).astype("float64")
            other_mutations = presence.iloc[~in_class].sum(axis=0).astype("float64")
            class_odds = (class_mutations + self.smoothing) / (
                class_size - class_mutations + self.smoothing
            )
            other_odds = (other_mutations + self.smoothing) / (
                other_size - other_mutations + self.smoothing
            )
            log_odds = np.log2(class_odds / other_odds).clip(lower=0.0)
            support = class_mutations + other_mutations
            score = log_odds * np.sqrt(support / (support + self.shrinkage))
            top = score.nlargest(self.top_genes_per_class)
            genes = top[top.gt(0)].index.tolist()
            self.class_selected_genes_[str(class_name)] = genes
            selected.update(genes)
        self.selected_feature_columns_ = [
            column for column in gene_columns if column in selected
        ]
        if not self.selected_feature_columns_:
            raise ValueError("암종별 특징 유전자 합집합을 만들 수 없습니다.")
        return matrix[self.selected_feature_columns_]

    def _fit_reducer(self, matrix: pd.DataFrame) -> pd.DataFrame:
        maximum = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
        self.effective_components_ = min(self.n_components, max(maximum, 1))
        if self.strategy == "nmf":
            reducer_input = matrix.clip(lower=0.0)
            self.reducer_ = MiniBatchNMF(
                n_components=self.effective_components_,
                init="nndsvda",
                max_iter=self.max_iter,
                batch_size=self.batch_size,
                random_state=self.random_state,
            )
            values = self.reducer_.fit_transform(reducer_input)
            prefix = "nmf_module"
        else:
            self.reducer_ = TruncatedSVD(
                n_components=self.effective_components_,
                algorithm="randomized",
                n_iter=self.svd_n_iter,
                random_state=self.random_state,
            )
            values = self.reducer_.fit_transform(matrix)
            prefix = "svd_latent"
        return self._latent_frame(matrix.index, values, prefix)

    def _latent_frame(
        self,
        index: pd.Index,
        values: np.ndarray,
        prefix: str,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            values,
            index=index,
            columns=[
                f"{prefix}_{component:03d}"
                for component in range(self.effective_components_)
            ],
            dtype="float32",
        )

    def _genes_burden_hotspot(
        self,
        raw_features: pd.DataFrame,
        base_matrix: pd.DataFrame,
        learn: bool,
    ) -> pd.DataFrame:
        raw_presence = create_mutation_presence_matrix(
            raw_features, self.raw_input_columns_
        )
        if learn:
            self.h07_gene_columns_ = [
                gene for gene in self.raw_input_columns_
                if gene in self.frequency_feature_columns_
            ]
            if not self.h07_gene_columns_:
                raise ValueError("H07에 유지할 EMV46 원본 유전자 피처가 없습니다.")
            self.hotspots_, self.hotspot_support_ = learn_recurrent_hotspots(
                raw_features,
                self.h07_gene_columns_,
                self.min_hotspot_count,
                self.max_hotspots,
            )
            self.selected_feature_columns_ = self.h07_gene_columns_.copy()
        burden = raw_presence.sum(axis=1).astype("float32")
        engineered = base_matrix[self.h07_gene_columns_].copy()
        engineered["mutation_burden_total"] = burden
        engineered["mutation_burden_log1p"] = np.log1p(burden).astype("float32")
        engineered["mutation_burden_rate"] = (
            burden / max(len(self.raw_input_columns_), 1)
        ).astype("float32")
        hotspot = create_hotspot_matrix(raw_features, self.hotspots_).astype("float32")
        engineered = pd.concat([engineered, hotspot], axis=1)
        engineered["recurrent_hotspot_count"] = hotspot.sum(axis=1).astype("float32")
        return engineered

    def _fit_output(
        self,
        raw_features: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        self.raw_input_columns_ = raw_features.columns.tolist()
        base = self._numeric(self.base_pipeline.fit_transform(raw_features, labels))
        self.base_feature_count_ = base.shape[1]
        frequency = self._learn_frequency(base)
        self.selected_feature_columns_ = self.frequency_feature_columns_.copy()
        if self.strategy == "frequency":
            return frequency
        if self.strategy in {"chi2", "mutual_information"}:
            return self._select_supervised(frequency, labels)
        if self.strategy == "class_gene_union":
            return self._select_class_union(frequency, labels)
        if self.strategy in {"nmf", "svd"}:
            return self._fit_reducer(frequency)
        return self._genes_burden_hotspot(raw_features, base, learn=True)

    def fit_transform(
        self,
        owner: PreprocessingPipeline,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        output = self._fit_output(features, labels)
        PreprocessingPipeline.fit(owner, output, labels)
        return PreprocessingPipeline.transform(owner, output).astype("float32")

    def fit(
        self,
        owner: PreprocessingPipeline,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        self.fit_transform(owner, features, labels)

    def transform(
        self,
        owner: PreprocessingPipeline,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        if not self.input_columns_:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")
        base = self._numeric(self.base_pipeline.transform(features))
        frequency = base[self.frequency_feature_columns_]
        if self.strategy in {"nmf", "svd"}:
            if self.reducer_ is None:
                raise RuntimeError("차원 축소기가 학습되지 않았습니다.")
            reducer_input = frequency.clip(lower=0.0) if self.strategy == "nmf" else frequency
            values = self.reducer_.transform(reducer_input)
            prefix = "nmf_module" if self.strategy == "nmf" else "svd_latent"
            output = self._latent_frame(features.index, values, prefix)
        elif self.strategy == "genes_burden_hotspot":
            output = self._genes_burden_hotspot(features, base, learn=False)
        else:
            output = base[self.selected_feature_columns_]
        return PreprocessingPipeline.transform(owner, output).astype("float32")

    def summary(self, owner: PreprocessingPipeline) -> dict[str, int]:
        result = PreprocessingPipeline.summary(owner)
        result.update({
            "em_v46_base_features": self.base_feature_count_,
            "dropped_low_support_features": len(self.dropped_rare_columns_),
            "frequency_filtered_features": len(self.frequency_feature_columns_),
            "selected_features": len(self.selected_feature_columns_),
            "latent_features": self.effective_components_,
            "hotspot_features": len(self.hotspots_),
        })
        return result
