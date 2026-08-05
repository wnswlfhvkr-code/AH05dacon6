"""EMV16 공통 계산과 E1~E10의 선택 기능을 제공하는 합성 엔진입니다.

각 버전 클래스는 ``PreprocessingPipeline``을 직접 상속하고 이 엔진을
composition으로 사용합니다. 버전 간 상속과 동일 피처의 중복 생성을 피하면서
fit에서 학습한 상태만 validation/test transform에 적용합니다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.pipelines.base import PreprocessingPipeline


@dataclass(frozen=True)
class EMExperimentSpec:
    """한 실험에서 EMV16에 추가하거나 교체할 기능입니다."""

    name: str
    description: str
    pattern_support: bool = False
    token_complexity: bool = False
    comutation_pairs: bool = False
    stable_hotspots: bool = False
    pair_contrasts: bool = False
    pattern_uncertainty: bool = False
    class_gene_cap: bool = False
    deduplicate_genes: bool = False


def _presence_matrix(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """결측·WT=0, 관찰된 변이=1로 명시적으로 인코딩합니다."""
    return pd.DataFrame({
        column: features[column].astype("string").fillna("WT")
        .str.strip().str.upper().ne("WT").astype("int8")
        for column in columns
    }, index=features.index)


def _severity_matrix(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result: dict[str, pd.Series] = {}
    for column in columns:
        value = features[column].astype("string").fillna("WT").str.strip().str.upper()
        severity = pd.Series(0, index=features.index, dtype="int8")
        severity.loc[value.ne("WT")] = 2
        severity.loc[value.str.fullmatch(r"([A-Z])\d+\1", na=False)] = 1
        severity.loc[value.str.contains(r"DEL|INS|DUP|>", regex=True, na=False)] = 3
        severity.loc[value.str.contains(r"FS|\*|TER", regex=True, na=False)] = 4
        result[column] = severity
    return pd.DataFrame(result, index=features.index)


def _pattern_keys(features: pd.DataFrame) -> pd.Series:
    normalized = features.astype("string").fillna("WT").apply(
        lambda column: column.str.strip().str.upper()
    )
    hashes = pd.util.hash_pandas_object(normalized, index=False, categorize=True)
    return hashes.map(lambda value: f"{int(value):016x}")


def _normalized_tokens(value: object) -> frozenset[str]:
    if pd.isna(value):
        return frozenset()
    normalized = str(value).strip().upper()
    if not normalized or normalized == "WT":
        return frozenset()
    return frozenset(normalized.split())


def _learn_class_weights(
    mutation: pd.DataFrame,
    labels: pd.Series,
    top_genes: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    aligned = labels.reindex(mutation.index).astype(str)
    class_names = sorted(aligned.unique().tolist())
    weights: dict[str, dict[str, float]] = {}
    for class_name in class_names:
        inside = aligned.eq(class_name)
        outside = ~inside
        inside_count = mutation.loc[inside].sum(axis=0).astype("float64")
        outside_count = mutation.loc[outside].sum(axis=0).astype("float64")
        inside_rate = (inside_count + smoothing) / (inside.sum() + 2 * smoothing)
        outside_rate = (outside_count + smoothing) / (outside.sum() + 2 * smoothing)
        odds = np.log2(
            (inside_rate / (1.0 - inside_rate))
            / (outside_rate / (1.0 - outside_rate))
        ).clip(-max_log2_odds, max_log2_odds)
        support_shrinkage = inside_count / (inside_count + shrinkage)
        stabilized = odds * support_shrinkage
        selected = stabilized.sort_values(ascending=False).head(top_genes)
        weights[class_name] = {
            gene: float(weight) for gene, weight in selected.items()
        }
    return class_names, weights


def _add_signature_features(
    engineered: pd.DataFrame,
    mutation: pd.DataFrame,
    class_names: list[str],
    weights: dict[str, dict[str, float]],
) -> None:
    for class_name in class_names:
        class_weights = weights[class_name]
        genes = [gene for gene in class_weights if gene in mutation]
        if not genes:
            engineered[f"signature_{class_name}_weighted"] = 0.0
            engineered[f"signature_{class_name}_match_count"] = 0.0
            continue
        matrix = mutation[genes].astype("float32")
        vector = np.asarray([class_weights[gene] for gene in genes], dtype="float32")
        engineered[f"signature_{class_name}_weighted"] = matrix.to_numpy().dot(vector)
        engineered[f"signature_{class_name}_match_count"] = matrix.sum(axis=1)


def _add_consequence_summary(
    engineered: pd.DataFrame,
    severity: pd.DataFrame,
) -> None:
    mutated = severity.ne(0)
    burden = mutated.sum(axis=1).astype("float32")
    engineered["mutation_burden"] = burden
    engineered["mutation_burden_log1p"] = np.log1p(burden)
    engineered["severity_sum"] = severity.sum(axis=1).astype("float32")
    engineered["severity_max"] = severity.max(axis=1).astype("float32")
    for level, name in ((1, "synonymous"), (2, "missense"), (3, "indel"), (4, "truncating")):
        count = severity.eq(level).sum(axis=1).astype("float32")
        engineered[f"{name}_count"] = count
        engineered[f"{name}_ratio"] = count / burden.clip(lower=1)


class EMFeatureEngine:
    """EMV16 핵심과 실험별 추가 피처의 fit/transform 상태를 관리합니다."""

    def __init__(self, spec: EMExperimentSpec, parameters: dict) -> None:
        self.spec = spec
        self.min_mutation_count = int(parameters.get("min_mutation_count", 5))
        self.top_genes_per_class = int(parameters.get("top_genes_per_class", 20))
        self.smoothing = float(parameters.get("smoothing", 0.5))
        self.max_log2_odds = float(parameters.get("max_log2_odds", 8.0))
        self.shrinkage = float(parameters.get("shrinkage", 10.0))
        self.min_hotspot_count = int(parameters.get("min_hotspot_count", 5))
        self.max_hotspots = int(parameters.get("max_hotspots", 384))
        self.inner_signature_folds = int(parameters.get("inner_signature_folds", 5))
        self.random_state = int(parameters.get("random_state", 42))
        self.max_class_genes = int(parameters.get("max_class_genes", 1500))
        self.max_comutation_pairs = int(parameters.get("max_comutation_pairs", 96))
        self.comutation_gene_pool = int(parameters.get("comutation_gene_pool", 160))
        self.min_comutation_count = int(parameters.get("min_comutation_count", 5))
        self.min_comutation_lift = float(parameters.get("min_comutation_lift", 1.5))
        self.stable_hotspot_folds = int(parameters.get("stable_hotspot_folds", 5))
        self.min_stable_hotspot_folds = int(parameters.get("min_stable_hotspot_folds", 4))
        if min(
            self.min_mutation_count,
            self.top_genes_per_class,
            self.min_hotspot_count,
            self.max_hotspots,
            self.inner_signature_folds,
        ) < 1:
            raise ValueError("빈도·피처 수·fold 파라미터는 1 이상이어야 합니다.")
        if self.inner_signature_folds < 2:
            raise ValueError("inner_signature_folds는 2 이상이어야 합니다.")

        self.selected_genes: list[str] = []
        self.dropped_rare_genes: list[str] = []
        self.dropped_duplicate_genes: list[str] = []
        self.class_names: list[str] = []
        self.class_weights: dict[str, dict[str, float]] = {}
        self.hotspots: list[tuple[str, str, str]] = []
        self.comutation_pairs: list[tuple[str, str, str]] = []
        self.pattern_counts: dict[str, int] = {}
        self.pattern_uncertainty: dict[str, tuple[float, float, float, float]] = {}

    @property
    def steps(self) -> tuple[str, ...]:
        steps = [
            "EMV16 최소 변이 빈도 필터",
            "EMV16 consequence severity·burden",
            "EMV16 inner-fold OOF class signature",
            "EMV16 recurrent hotspot",
        ]
        additions = {
            "pattern_support": "동일 원시 패턴 train support",
            "token_complexity": "multi-hit token complexity",
            "comutation_pairs": "train co-mutation pair",
            "stable_hotspots": "fold 안정 hotspot으로 교체",
            "pair_contrasts": "KIRC-KIPAN·LGG-GBMLGG signature contrast",
            "pattern_uncertainty": "OOF 패턴 충돌 불확실성",
            "class_gene_cap": "클래스 판별 유전자 cap",
            "deduplicate_genes": "완전 중복 유전자 제거",
        }
        for field, description in additions.items():
            if getattr(self.spec, field):
                steps.append(description)
        return tuple(steps)

    def _apply_class_gene_cap(
        self,
        mutation: pd.DataFrame,
        labels: pd.Series,
    ) -> list[str]:
        if not self.spec.class_gene_cap or len(mutation.columns) <= self.max_class_genes:
            return mutation.columns.tolist()
        _, weights = _learn_class_weights(
            mutation,
            labels,
            max(10, self.top_genes_per_class * 3),
            self.smoothing,
            self.max_log2_odds,
            self.shrinkage,
        )
        selected: list[str] = []
        for class_weights in weights.values():
            for gene in class_weights:
                if gene not in selected:
                    selected.append(gene)
        frequency_order = mutation.sum(axis=0).sort_values(ascending=False).index
        for gene in frequency_order:
            if gene not in selected:
                selected.append(str(gene))
            if len(selected) >= self.max_class_genes:
                break
        return selected[: self.max_class_genes]

    def _deduplicate_genes(self, mutation: pd.DataFrame) -> list[str]:
        if not self.spec.deduplicate_genes:
            return mutation.columns.tolist()
        duplicate_mask = mutation.T.duplicated(keep="first")
        self.dropped_duplicate_genes = mutation.columns[duplicate_mask].tolist()
        return mutation.columns[~duplicate_mask].tolist()

    def _token_support(
        self,
        features: pd.DataFrame,
        row_positions: np.ndarray | None = None,
    ) -> Counter[tuple[str, str]]:
        source = features if row_positions is None else features.iloc[row_positions]
        support: Counter[tuple[str, str]] = Counter()
        for gene in self.selected_genes:
            for value in source[gene]:
                for token in _normalized_tokens(value):
                    support[(gene, token)] += 1
        return support

    def _learn_hotspots(self, features: pd.DataFrame, labels: pd.Series) -> None:
        support = self._token_support(features)
        eligible = {
            pair for pair, count in support.items()
            if count >= self.min_hotspot_count
        }
        if self.spec.stable_hotspots:
            minimum_class = int(labels.value_counts().min())
            folds = min(self.stable_hotspot_folds, minimum_class)
            stable_counts: Counter[tuple[str, str]] = Counter()
            if folds >= 2:
                splitter = StratifiedKFold(
                    n_splits=folds, shuffle=True, random_state=self.random_state
                )
                for train_positions, _ in splitter.split(features, labels):
                    fold_support = self._token_support(features, train_positions)
                    for pair, count in fold_support.items():
                        if count >= self.min_hotspot_count:
                            stable_counts[pair] += 1
                eligible = {
                    pair for pair in eligible
                    if stable_counts[pair] >= min(self.min_stable_hotspot_folds, folds)
                }
        ordered = sorted(eligible, key=lambda pair: (-support[pair], pair))
        self.hotspots = [
            (f"hotspot_{index:04d}", gene, token)
            for index, (gene, token) in enumerate(ordered[: self.max_hotspots])
        ]

    def _learn_comutations(self, mutation: pd.DataFrame) -> None:
        self.comutation_pairs = []
        if not self.spec.comutation_pairs:
            return
        counts = mutation.sum(axis=0).sort_values(ascending=False)
        genes = counts.head(self.comutation_gene_pool).index.tolist()
        rows = max(len(mutation), 1)
        candidates: list[tuple[float, int, str, str]] = []
        for left, right in combinations(genes, 2):
            joint = int((mutation[left].astype(bool) & mutation[right].astype(bool)).sum())
            if joint < self.min_comutation_count:
                continue
            expected = (float(counts[left]) * float(counts[right])) / rows
            lift = joint / max(expected, 1e-9)
            if lift >= self.min_comutation_lift:
                candidates.append((lift, joint, left, right))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        self.comutation_pairs = [
            (f"comut_{index:04d}", left, right)
            for index, (_, _, left, right) in enumerate(
                candidates[: self.max_comutation_pairs]
            )
        ]

    def _learn_pattern_state(self, features: pd.DataFrame, labels: pd.Series) -> None:
        if not (self.spec.pattern_support or self.spec.pattern_uncertainty):
            self.pattern_counts = {}
            self.pattern_uncertainty = {}
            return
        keys = _pattern_keys(features).astype(str)
        self.pattern_counts = keys.value_counts().astype(int).to_dict()
        if self.spec.pattern_uncertainty:
            self.pattern_uncertainty = self._make_uncertainty_lookup(keys, labels)

    def _make_uncertainty_lookup(
        self,
        keys: pd.Series,
        labels: pd.Series,
    ) -> dict[str, tuple[float, float, float, float]]:
        frame = pd.DataFrame({
            "key": keys.to_numpy(),
            "label": labels.astype(str).to_numpy(),
        })
        class_count = max(len(self.class_names), 2)
        lookup: dict[str, tuple[float, float, float, float]] = {}
        for key, group in frame.groupby("key", sort=False):
            counts = group["label"].value_counts().to_numpy(dtype="float64")
            probabilities = counts / counts.sum()
            entropy = float(
                -(probabilities * np.log(probabilities)).sum() / np.log(class_count)
            )
            lookup[str(key)] = (
                float(np.log1p(len(group))),
                entropy,
                float(probabilities.max()),
                float(len(counts) > 1),
            )
        return lookup

    def fit(
        self,
        pipeline: PreprocessingPipeline,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        all_genes = features.columns.tolist()
        initial = _presence_matrix(features, all_genes)
        counts = initial.sum(axis=0)
        selected = counts[counts >= self.min_mutation_count].index.tolist()
        self.dropped_rare_genes = counts[counts < self.min_mutation_count].index.tolist()
        mutation = initial[selected]
        selected = self._apply_class_gene_cap(mutation, labels)
        mutation = mutation[selected]
        selected = self._deduplicate_genes(mutation)
        self.selected_genes = selected
        mutation = mutation[self.selected_genes]
        self.class_names, self.class_weights = _learn_class_weights(
            mutation,
            labels,
            self.top_genes_per_class,
            self.smoothing,
            self.max_log2_odds,
            self.shrinkage,
        )
        self._learn_hotspots(features, labels)
        self._learn_comutations(mutation)
        self._learn_pattern_state(features, labels)
        engineered = self._build_features(features)
        PreprocessingPipeline.fit(pipeline, engineered, labels)

    def _add_hotspots(self, engineered: pd.DataFrame, features: pd.DataFrame) -> None:
        token_cache: dict[str, pd.Series] = {}
        for feature_name, gene, token in self.hotspots:
            if gene not in token_cache:
                token_cache[gene] = features[gene].map(_normalized_tokens)
            engineered[feature_name] = token_cache[gene].map(
                lambda tokens: token in tokens
            ).astype("int8")

    def _add_token_complexity(
        self,
        engineered: pd.DataFrame,
        features: pd.DataFrame,
    ) -> None:
        if not self.spec.token_complexity:
            return
        token_counts = pd.DataFrame({
            gene: features[gene].map(lambda value: len(_normalized_tokens(value)))
            for gene in self.selected_genes
        }, index=features.index)
        total = token_counts.sum(axis=1).astype("float32")
        multi = token_counts.gt(1).sum(axis=1).astype("float32")
        engineered["token_event_count"] = total
        engineered["multi_hit_gene_count"] = multi
        engineered["token_excess_count"] = token_counts.sub(1).clip(lower=0).sum(axis=1)
        engineered["max_gene_token_count"] = token_counts.max(axis=1).astype("float32")
        engineered["multi_hit_ratio"] = multi / engineered["mutation_burden"].clip(lower=1)

    def _add_pattern_features(
        self,
        engineered: pd.DataFrame,
        features: pd.DataFrame,
        uncertainty_lookup: dict[str, tuple[float, float, float, float]] | None = None,
    ) -> None:
        keys = _pattern_keys(features).astype(str)
        if self.spec.pattern_support:
            support = keys.map(self.pattern_counts).fillna(0).astype("float32")
            engineered["pattern_support_log1p"] = np.log1p(support)
            engineered["pattern_seen_multiple"] = support.ge(2).astype("int8")
        if self.spec.pattern_uncertainty:
            lookup = self.pattern_uncertainty if uncertainty_lookup is None else uncertainty_lookup
            values = keys.map(lambda key: lookup.get(str(key), (0.0, 0.0, 0.0, 0.0)))
            matrix = np.asarray(values.tolist(), dtype="float32")
            engineered["pattern_label_support_log1p"] = matrix[:, 0]
            engineered["pattern_label_entropy"] = matrix[:, 1]
            engineered["pattern_label_max_probability"] = matrix[:, 2]
            engineered["pattern_label_conflict"] = matrix[:, 3]

    def _add_pair_contrasts(self, engineered: pd.DataFrame) -> None:
        if not self.spec.pair_contrasts:
            return
        for left, right in (("KIRC", "KIPAN"), ("LGG", "GBMLGG")):
            left_weighted = f"signature_{left}_weighted"
            right_weighted = f"signature_{right}_weighted"
            left_count = f"signature_{left}_match_count"
            right_count = f"signature_{right}_match_count"
            if left_weighted in engineered and right_weighted in engineered:
                engineered[f"pair_{left}_{right}_weighted_diff"] = (
                    engineered[left_weighted] - engineered[right_weighted]
                )
                engineered[f"pair_{left}_{right}_count_diff"] = (
                    engineered[left_count] - engineered[right_count]
                )

    def _build_features(
        self,
        features: pd.DataFrame,
        class_weights: dict[str, dict[str, float]] | None = None,
        uncertainty_lookup: dict[str, tuple[float, float, float, float]] | None = None,
    ) -> pd.DataFrame:
        severity = _severity_matrix(features, self.selected_genes)
        mutation = severity.ne(0).astype("int8")
        engineered = severity.copy()
        _add_consequence_summary(engineered, severity)
        _add_signature_features(
            engineered,
            mutation,
            self.class_names,
            self.class_weights if class_weights is None else class_weights,
        )
        self._add_hotspots(engineered, features)
        self._add_token_complexity(engineered, features)
        for feature_name, left, right in self.comutation_pairs:
            engineered[feature_name] = (
                mutation[left].astype(bool) & mutation[right].astype(bool)
            ).astype("int8")
        self._add_pattern_features(engineered, features, uncertainty_lookup)
        self._add_pair_contrasts(engineered)
        return engineered

    def transform(
        self,
        pipeline: PreprocessingPipeline,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        engineered = self._build_features(features)
        return PreprocessingPipeline.transform(pipeline, engineered).astype("float32")

    def fit_transform(
        self,
        pipeline: PreprocessingPipeline,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        self.fit(pipeline, features, labels)
        transformed = self.transform(pipeline, features)
        minimum_class = int(labels.value_counts().min())
        folds = min(self.inner_signature_folds, minimum_class)
        if folds < 2:
            return transformed
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.random_state
        )
        for train_positions, valid_positions in splitter.split(features, labels):
            fold_features = features.iloc[train_positions]
            fold_labels = labels.iloc[train_positions]
            valid_features = features.iloc[valid_positions]
            fold_mutation = _presence_matrix(fold_features, self.selected_genes)
            _, fold_weights = _learn_class_weights(
                fold_mutation,
                fold_labels,
                self.top_genes_per_class,
                self.smoothing,
                self.max_log2_odds,
                self.shrinkage,
            )
            uncertainty = None
            if self.spec.pattern_uncertainty:
                uncertainty = self._make_uncertainty_lookup(
                    _pattern_keys(fold_features).astype(str), fold_labels
                )
            fold_engineered = self._build_features(
                valid_features,
                class_weights=fold_weights,
                uncertainty_lookup=uncertainty,
            )
            fold_transformed = PreprocessingPipeline.transform(
                pipeline, fold_engineered
            ).astype("float32")
            signature_columns = [
                column for column in transformed.columns
                if column.startswith("signature_") or column.startswith("pair_")
            ]
            if self.spec.pattern_uncertainty:
                signature_columns += [
                    column for column in transformed.columns
                    if column.startswith("pattern_label_")
                ]
            transformed.loc[features.index[valid_positions], signature_columns] = (
                fold_transformed[signature_columns].to_numpy()
            )
        return transformed.astype("float32")

    def summary(self, pipeline: PreprocessingPipeline) -> dict[str, int]:
        result = PreprocessingPipeline.summary(pipeline)
        result.update({
            "selected_gene_features": len(self.selected_genes),
            "dropped_rare_features": len(self.dropped_rare_genes),
            "dropped_duplicate_gene_features": len(self.dropped_duplicate_genes),
            "hotspot_features": len(self.hotspots),
            "comutation_features": len(self.comutation_pairs),
        })
        return result
