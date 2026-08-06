"""EMV45와 F01~F19 고유 로직을 한 파일에서 중복 없이 결합한 F20."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_em_v45 import EMV45PreprocessingPipeline


FEATURE_NAMES = tuple(f"F{index:02d}" for index in range(1, 20))
SIGNATURE_FEATURE_NAMES = ("F16", "F17", "F18", "F19")
_SPLICE = re.compile(r"SPLICE|SPL_|IVS|(?:[+-]\d+)", re.IGNORECASE)
_FRAMESHIFT = re.compile(r"FS", re.IGNORECASE)
_NONSENSE = re.compile(r"\*|TER$|STOP", re.IGNORECASE)
_MISSENSE = re.compile(r"^([A-Z*])\d+([A-Z*])$", re.IGNORECASE)


@dataclass
class _SharedMatrices:
    event_counts: pd.DataFrame
    consequence_event_summary: pd.DataFrame
    consequence_presence: dict[str, pd.DataFrame]


@dataclass
class _SignatureState:
    selected_genes: list[str]
    class_names: list[str]
    weights: np.ndarray
    profiles: np.ndarray
    top_genes_per_class: int
    smoothing: float
    shrinkage: float
    max_log2_odds: float
    folds: int
    temperature: float
    random_state: int


def _tokens(value: object) -> tuple[str, ...]:
    """결측·WT를 제거하고 셀 안의 중복 변이 토큰을 한 번만 유지합니다."""
    if pd.isna(value):
        return ()
    normalized = str(value).strip().upper()
    if not normalized or normalized in {"WT", "<NA>"}:
        return ()
    return tuple(sorted(set(normalized.split())))


def _consequence_values(
    tokens: tuple[str, ...],
) -> tuple[int, int, int, int, bool, bool, bool]:
    """F04 이벤트 수와 F13~F15 유전자 여부를 한 번에 판정합니다."""
    event_counts = [0, 0, 0, 0]
    truncating = False
    gene_missense = False
    gene_splice = False
    for token in tokens:
        is_splice = bool(_SPLICE.search(token))
        is_frameshift = bool(_FRAMESHIFT.search(token))
        is_nonsense = bool(_NONSENSE.search(token))
        match = _MISSENSE.fullmatch(token)
        is_missense = bool(
            match and match.group(1) != match.group(2) and not is_nonsense
        )
        truncating = truncating or is_frameshift or is_nonsense
        gene_missense = gene_missense or is_missense
        gene_splice = gene_splice or is_splice
        if is_splice:
            event_counts[3] += 1
        elif is_frameshift:
            event_counts[2] += 1
        elif is_nonsense:
            event_counts[1] += 1
        else:
            if is_missense:
                event_counts[0] += 1
    return (*event_counts, truncating, gene_missense, gene_splice)


def _shared_matrices(
    features: pd.DataFrame,
    gene_columns: list[str],
) -> _SharedMatrices:
    missing = set(gene_columns) - set(features.columns)
    if missing:
        raise ValueError(f"누락된 유전자 컬럼이 있습니다: {sorted(missing)}")
    event_data: dict[str, pd.Series] = {}
    consequence_event_summary = pd.DataFrame(
        0,
        index=features.index,
        columns=("missense", "nonsense", "frameshift", "splice"),
        dtype="int32",
    )
    consequence_presence_data: dict[str, dict[str, pd.Series]] = {
        name: {} for name in ("truncating", "missense", "splice")
    }
    for gene in gene_columns:
        parsed = features[gene].map(_tokens)
        event_data[gene] = parsed.map(len).astype("int16")
        classified = parsed.map(_consequence_values)
        for position, name in enumerate(consequence_event_summary.columns):
            consequence_event_summary[name] += classified.map(
                lambda values, index=position: values[index]
            ).astype("int32")
        for position, name in enumerate(consequence_presence_data, start=4):
            consequence_presence_data[name][gene] = classified.map(
                lambda values, index=position: values[index]
            ).astype("int8")
    return _SharedMatrices(
        event_counts=pd.DataFrame(event_data, index=features.index),
        consequence_event_summary=consequence_event_summary,
        consequence_presence={
            name: pd.DataFrame(values, index=features.index)
            for name, values in consequence_presence_data.items()
        },
    )


def _learn_hotspots(
    features: pd.DataFrame,
    gene_columns: list[str],
    minimum_count: int,
    maximum_hotspots: int,
) -> list[tuple[str, str]]:
    support: Counter[tuple[str, str]] = Counter()
    for gene in gene_columns:
        for value in features[gene]:
            for token in _tokens(value):
                support[(gene, token)] += 1
    eligible = [pair for pair, count in support.items() if count >= minimum_count]
    return sorted(eligible, key=lambda pair: (-support[pair], pair))[
        :maximum_hotspots
    ]


def _hotspot_outputs(
    features: pd.DataFrame,
    hotspots: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = list(dict.fromkeys(gene for gene, _ in hotspots))
    token_cache = {gene: features[gene].map(_tokens) for gene in genes}
    total = pd.Series(0, index=features.index, dtype="int32")
    grouped: dict[str, set[str]] = {}
    for gene, token in hotspots:
        total += token_cache[gene].map(lambda values: token in values).astype("int32")
        grouped.setdefault(gene, set()).add(token)
    per_gene = pd.DataFrame(
        {
            f"gene_hotspot__{gene}": token_cache[gene].map(
                lambda values, selected=frozenset(tokens): not selected.isdisjoint(values)
            ).astype("int8")
            for gene, tokens in grouped.items()
        },
        index=features.index,
    )
    return (
        pd.DataFrame({"hotspot_mutation_count": total}, index=features.index),
        per_gene,
    )


def _learn_signature_arrays(
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
        inside_rate = (inside_count + smoothing) / (
            inside.sum() + 2.0 * smoothing
        )
        outside_rate = (outside_count + smoothing) / (
            outside.sum() + 2.0 * smoothing
        )
        log_odds = np.log2(
            (inside_rate / np.maximum(1.0 - inside_rate, 1e-12))
            / (outside_rate / np.maximum(1.0 - outside_rate, 1e-12))
        )
        stabilized = np.clip(log_odds, -max_log2_odds, max_log2_odds)
        stabilized *= inside_count / (inside_count + shrinkage)
        ranked = np.argsort(stabilized)[::-1][:top_genes_per_class]
        positive = ranked[stabilized[ranked] > 0]
        weights[class_index, positive] = stabilized[positive]
        profiles[class_index] = inside_rate
    return weights, profiles


def _signature_outputs(
    mutation: pd.DataFrame,
    class_names: list[str],
    weights: np.ndarray,
    profiles: np.ndarray,
    temperature: float,
) -> dict[str, pd.DataFrame]:
    matrix = mutation.to_numpy(dtype="float64")
    weighted = matrix @ weights.T
    sample_norm = np.sqrt((matrix * matrix).sum(axis=1, keepdims=True))
    profile_norm = np.sqrt((profiles * profiles).sum(axis=1))[None, :]
    similarity = (matrix @ profiles.T) / np.maximum(
        sample_norm * profile_norm, 1e-12
    )
    if similarity.shape[1] >= 2:
        top_two = np.sort(
            np.partition(similarity, -2, axis=1)[:, -2:], axis=1
        )
        margin = top_two[:, 1] - top_two[:, 0]
    else:
        margin = similarity[:, 0]
    logits = similarity / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
    entropy /= np.log(max(similarity.shape[1], 2))
    return {
        "F16": pd.DataFrame(
            weighted,
            index=mutation.index,
            columns=[f"signature_weighted__{name}" for name in class_names],
            dtype="float32",
        ),
        "F17": pd.DataFrame(
            similarity,
            index=mutation.index,
            columns=[f"signature_similarity__{name}" for name in class_names],
            dtype="float32",
        ),
        "F18": pd.DataFrame(
            {"signature_top2_margin": margin.astype("float32")},
            index=mutation.index,
        ),
        "F19": pd.DataFrame(
            {"signature_entropy": entropy.astype("float32")},
            index=mutation.index,
        ),
    }


class EMF20PreprocessingPipeline(PreprocessingPipeline):
    """EMV45 한 번과 통합된 F01~F19 계산 한 번을 결합합니다."""

    name = "em_F20"
    evaluation_folds = 5

    def __init__(
        self,
        feature_parameters: dict[str, dict[str, object]] | None = None,
        **parameters: object,
    ) -> None:
        super().__init__()
        overrides = {
            str(name).upper(): dict(values)
            for name, values in (feature_parameters or {}).items()
        }
        unknown = set(overrides) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"지원하지 않는 F20 구성 피처입니다: {sorted(unknown)}")
        self.base_parameters = dict(parameters)
        self.feature_configs = {
            name: {**parameters, **overrides.get(name, {})}
            for name in FEATURE_NAMES
        }
        self.v45_pipeline = EMV45PreprocessingPipeline(**parameters)
        self.gene_columns_: list[str] = []
        self.selected_genes_: dict[str, list[str]] = {}
        self.hotspot_states_: dict[tuple[int, int], list[tuple[str, str]]] = {}
        self.feature_hotspot_keys_: dict[str, tuple[int, int]] = {}
        self.signature_states_: dict[tuple[object, ...], _SignatureState] = {}
        self.feature_signature_keys_: dict[str, tuple[object, ...]] = {}
        self.derived_columns_: dict[str, list[str]] = {
            name: [] for name in FEATURE_NAMES
        }
        self.v45_feature_count_ = 0
        self.derived_feature_counts_: dict[str, int] = {
            name: 0 for name in FEATURE_NAMES
        }
        self.steps = (
            "EMV45 베이스 피처 1회 생성",
            "공통 토큰 수·consequence 행렬 1회 계산",
            "F05·F12 공통 hotspot 상태 재사용",
            "F16~F19 공통 inner-fold OOF signature 계산 재사용",
            "F01~F19 접두사로 컬럼 충돌 제거",
            "원본 SUBCLASS 유지",
        )

    def _int_config(self, feature: str, key: str, default: int) -> int:
        value = int(self.feature_configs[feature].get(key, default))
        if value < 1:
            raise ValueError(f"{feature}.{key}는 1 이상이어야 합니다.")
        return value

    def _learn_feature_states(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        shared: _SharedMatrices,
    ) -> None:
        mutated_support = shared.event_counts.gt(0).sum(axis=0)
        multi_support = shared.event_counts.gt(1).sum(axis=0)
        self.selected_genes_ = {}
        for feature in ("F09", "F10"):
            minimum = self._int_config(feature, "min_feature_support", 2)
            self.selected_genes_[feature] = mutated_support[
                mutated_support >= minimum
            ].index.tolist()
        minimum = self._int_config("F11", "min_feature_support", 2)
        self.selected_genes_["F11"] = multi_support[
            multi_support >= minimum
        ].index.tolist()
        consequence_map = {
            "F13": shared.consequence_presence["truncating"].gt(0),
            "F14": shared.consequence_presence["missense"].gt(0),
            "F15": shared.consequence_presence["splice"].gt(0),
        }
        for feature, matrix in consequence_map.items():
            minimum = self._int_config(feature, "min_feature_support", 2)
            support = matrix.sum(axis=0)
            self.selected_genes_[feature] = support[
                support >= minimum
            ].index.tolist()
            if not self.selected_genes_[feature]:
                raise ValueError(f"{feature} 지지도를 만족하는 유전자가 없습니다.")

        self.hotspot_states_ = {}
        self.feature_hotspot_keys_ = {}
        for feature in ("F05", "F12"):
            minimum = self._int_config(feature, "min_hotspot_count", 5)
            maximum = self._int_config(feature, "max_hotspots", 384)
            if minimum < 2:
                raise ValueError(f"{feature}.min_hotspot_count는 2 이상이어야 합니다.")
            key = (minimum, maximum)
            self.feature_hotspot_keys_[feature] = key
            if key not in self.hotspot_states_:
                self.hotspot_states_[key] = _learn_hotspots(
                    features, self.gene_columns_, minimum, maximum
                )
        if not self.hotspot_states_[self.feature_hotspot_keys_["F12"]]:
            raise ValueError("F12 최소 지지도를 만족하는 hotspot이 없습니다.")

        presence = shared.event_counts.gt(0).astype("int8")
        aligned_labels = pd.Series(
            labels.astype(str).to_numpy(), index=features.index
        )
        class_names = sorted(aligned_labels.unique().tolist())
        self.signature_states_ = {}
        self.feature_signature_keys_ = {}
        for feature in SIGNATURE_FEATURE_NAMES:
            config = self.feature_configs[feature]
            key = (
                int(config.get("min_mutation_count", 5)),
                int(config.get("top_genes_per_class", 20)),
                float(config.get("smoothing", 0.5)),
                float(config.get("shrinkage", 10.0)),
                float(config.get("max_log2_odds", 8.0)),
                int(config.get("inner_signature_folds", 5)),
                float(config.get("signature_temperature", 1.0)),
                int(config.get("random_state", 42)),
            )
            self.feature_signature_keys_[feature] = key
            if key in self.signature_states_:
                continue
            (
                minimum,
                top_genes,
                smoothing,
                shrinkage,
                max_odds,
                folds,
                temperature,
                random_state,
            ) = key
            if minimum < 1 or top_genes < 1 or folds < 2:
                raise ValueError("signature 빈도·유전자 수·fold 설정이 잘못되었습니다.")
            if min(smoothing, shrinkage, max_odds, temperature) <= 0:
                raise ValueError("signature 안정화 파라미터는 0보다 커야 합니다.")
            support = presence.sum(axis=0)
            selected = support[support >= minimum].index.tolist()
            if not selected:
                raise ValueError("signature 최소 변이 빈도를 만족하는 유전자가 없습니다.")
            mutation = presence[selected]
            weights, profiles = _learn_signature_arrays(
                mutation,
                aligned_labels,
                class_names,
                top_genes,
                smoothing,
                shrinkage,
                max_odds,
            )
            self.signature_states_[key] = _SignatureState(
                selected_genes=selected,
                class_names=class_names,
                weights=weights,
                profiles=profiles,
                top_genes_per_class=top_genes,
                smoothing=smoothing,
                shrinkage=shrinkage,
                max_log2_odds=max_odds,
                folds=folds,
                temperature=temperature,
                random_state=random_state,
            )

    def _non_signature_features(
        self,
        features: pd.DataFrame,
        shared: _SharedMatrices,
    ) -> dict[str, pd.DataFrame]:
        counts = shared.event_counts
        total = counts.sum(axis=1).astype("float32")
        active = counts.gt(0).sum(axis=1).astype("float32")
        matrix = counts.to_numpy(dtype="float64")
        probabilities = matrix / np.maximum(total.to_numpy(), 1.0)[:, None]
        entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
        normalized_entropy = entropy / np.maximum(
            np.log(np.maximum(active.to_numpy(), 2)), 1e-12
        )
        consequence = shared.consequence_presence
        hotspot_cache = {
            key: _hotspot_outputs(features, hotspots)
            for key, hotspots in self.hotspot_states_.items()
        }
        selected09 = self.selected_genes_["F09"]
        selected10 = self.selected_genes_["F10"]
        selected11 = self.selected_genes_["F11"]
        selected13 = self.selected_genes_["F13"]
        selected14 = self.selected_genes_["F14"]
        selected15 = self.selected_genes_["F15"]
        return {
            "F01": pd.DataFrame(
                {"mutated_gene_count": active}, index=features.index
            ),
            "F02": pd.DataFrame(
                {"mutation_event_count": total}, index=features.index
            ),
            "F03": pd.DataFrame(
                {"mutation_burden_log1p": np.log1p(total).astype("float32")},
                index=features.index,
            ),
            "F04": pd.DataFrame(
                {
                    f"{name}_event_count": shared.consequence_event_summary[name]
                    for name in ("missense", "nonsense", "frameshift", "splice")
                },
                index=features.index,
            ),
            "F05": hotspot_cache[self.feature_hotspot_keys_["F05"]][0],
            "F06": pd.DataFrame(
                {"multi_hit_gene_count": counts.gt(1).sum(axis=1)},
                index=features.index,
            ),
            "F07": pd.DataFrame(
                {"no_mutation_sample": total.eq(0).astype("int8")},
                index=features.index,
            ),
            "F08": pd.DataFrame(
                {
                    "mutation_event_concentration": matrix.max(axis=1)
                    / np.maximum(total.to_numpy(), 1.0),
                    "mutation_gene_entropy": normalized_entropy,
                    "mutated_gene_diversity_ratio": active
                    / max(len(self.gene_columns_), 1),
                },
                index=features.index,
                dtype="float32",
            ),
            "F09": pd.DataFrame(
                {
                    f"gene_mutated__{gene}": counts[gene].gt(0).astype("int8")
                    for gene in selected09
                },
                index=features.index,
            ),
            "F10": pd.DataFrame(
                {
                    f"gene_mutation_count__{gene}": counts[gene].astype("int16")
                    for gene in selected10
                },
                index=features.index,
            ),
            "F11": pd.DataFrame(
                {
                    f"gene_multi_hit__{gene}": counts[gene].gt(1).astype("int8")
                    for gene in selected11
                },
                index=features.index,
            ),
            "F12": hotspot_cache[self.feature_hotspot_keys_["F12"]][1],
            "F13": consequence["truncating"][selected13].rename(
                columns=lambda gene: f"gene_truncating__{gene}"
            ).astype("int8"),
            "F14": consequence["missense"][selected14].gt(0).rename(
                columns=lambda gene: f"gene_missense__{gene}"
            ).astype("int8"),
            "F15": consequence["splice"][selected15].gt(0).rename(
                columns=lambda gene: f"gene_splice__{gene}"
            ).astype("int8"),
        }

    def _full_signature_features(
        self,
        shared: _SharedMatrices,
    ) -> dict[str, pd.DataFrame]:
        presence = shared.event_counts.gt(0).astype("int8")
        cache: dict[tuple[object, ...], dict[str, pd.DataFrame]] = {}
        for key, state in self.signature_states_.items():
            cache[key] = _signature_outputs(
                presence[state.selected_genes],
                state.class_names,
                state.weights,
                state.profiles,
                state.temperature,
            )
        return {
            feature: cache[self.feature_signature_keys_[feature]][feature]
            for feature in SIGNATURE_FEATURE_NAMES
        }

    def _oof_signature_features(
        self,
        shared: _SharedMatrices,
        labels: pd.Series,
    ) -> dict[str, pd.DataFrame]:
        presence = shared.event_counts.gt(0).astype("int8")
        aligned_labels = pd.Series(
            labels.astype(str).to_numpy(), index=presence.index
        )
        cache: dict[tuple[object, ...], dict[str, pd.DataFrame]] = {}
        for key, state in self.signature_states_.items():
            minimum_class = int(aligned_labels.value_counts().min())
            folds = min(state.folds, minimum_class)
            if folds < 2:
                raise ValueError("OOF signature에는 클래스별 표본이 최소 2개 필요합니다.")
            mutation = presence[state.selected_genes]
            output = {
                feature: pd.DataFrame(index=presence.index)
                for feature in SIGNATURE_FEATURE_NAMES
            }
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=state.random_state,
            )
            for train_positions, valid_positions in splitter.split(
                mutation, aligned_labels
            ):
                weights, profiles = _learn_signature_arrays(
                    mutation.iloc[train_positions],
                    aligned_labels.iloc[train_positions],
                    state.class_names,
                    state.top_genes_per_class,
                    state.smoothing,
                    state.shrinkage,
                    state.max_log2_odds,
                )
                fold_output = _signature_outputs(
                    mutation.iloc[valid_positions],
                    state.class_names,
                    weights,
                    profiles,
                    state.temperature,
                )
                for feature, frame in fold_output.items():
                    for column in frame:
                        output[feature].loc[frame.index, column] = frame[column]
            cache[key] = {
                feature: frame.astype("float32")
                for feature, frame in output.items()
            }
        return {
            feature: cache[self.feature_signature_keys_[feature]][feature]
            for feature in SIGNATURE_FEATURE_NAMES
        }

    def _derived_features(
        self,
        features: pd.DataFrame,
        shared: _SharedMatrices,
        labels: pd.Series | None = None,
        use_oof: bool = False,
    ) -> dict[str, pd.DataFrame]:
        derived = self._non_signature_features(features, shared)
        if use_oof:
            if labels is None:
                raise ValueError("OOF signature 생성에는 labels가 필요합니다.")
            derived.update(self._oof_signature_features(shared, labels))
        else:
            derived.update(self._full_signature_features(shared))
        return derived

    def _learn_derived_columns(
        self,
        derived: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        filtered: dict[str, pd.DataFrame] = {}
        for feature in FEATURE_NAMES:
            frame = derived[feature]
            columns = frame.columns[frame.nunique(dropna=False) > 1].tolist()
            self.derived_columns_[feature] = columns
            filtered[feature] = frame[columns]
        return filtered

    def _select_derived_columns(
        self,
        derived: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        return {
            feature: derived[feature][self.derived_columns_[feature]]
            for feature in FEATURE_NAMES
        }

    @staticmethod
    def _combine(
        v45: pd.DataFrame,
        derived: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        parts = [v45]
        for feature in FEATURE_NAMES:
            frame = derived[feature].copy()
            frame.columns = [f"{feature}__{column}" for column in frame.columns]
            parts.append(frame)
        return pd.concat(parts, axis=1)

    def _update_counts(
        self,
        v45: pd.DataFrame,
        derived: dict[str, pd.DataFrame],
    ) -> None:
        self.v45_feature_count_ = v45.shape[1]
        self.derived_feature_counts_ = {
            feature: derived[feature].shape[1] for feature in FEATURE_NAMES
        }

    def _fit_integrated_state(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> _SharedMatrices:
        self.gene_columns_ = features.columns.tolist()
        shared = _shared_matrices(features, self.gene_columns_)
        self._learn_feature_states(features, labels, shared)
        return shared

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EMF20PreprocessingPipeline":
        self.v45_pipeline.fit(features, labels)
        shared = self._fit_integrated_state(features, labels)
        v45 = self.v45_pipeline.transform(features)
        derived = self._learn_derived_columns(
            self._derived_features(features, shared)
        )
        self._update_counts(v45, derived)
        PreprocessingPipeline.fit(self, self._combine(v45, derived), labels)
        return self

    def fit_transform(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        v45 = self.v45_pipeline.fit_transform(features, labels)
        shared = self._fit_integrated_state(features, labels)
        derived = self._learn_derived_columns(
            self._derived_features(features, shared, labels, use_oof=True)
        )
        self._update_counts(v45, derived)
        combined = self._combine(v45, derived)
        PreprocessingPipeline.fit(self, combined, labels)
        return PreprocessingPipeline.transform(self, combined).astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        shared = _shared_matrices(features, self.gene_columns_)
        derived = self._select_derived_columns(
            self._derived_features(features, shared)
        )
        combined = self._combine(
            self.v45_pipeline.transform(features),
            derived,
        )
        return PreprocessingPipeline.transform(self, combined).astype("float32")

    def summary(self) -> dict[str, int]:
        result = PreprocessingPipeline.summary(self)
        result.update({
            "v45_base_features": self.v45_feature_count_,
            "f01_f19_derived_features": sum(self.derived_feature_counts_.values()),
            "included_feature_groups": len(self.derived_feature_counts_),
            "shared_hotspot_states": len(self.hotspot_states_),
            "shared_signature_states": len(self.signature_states_),
            "combined_before_constant_filter": (
                self.v45_feature_count_ + sum(self.derived_feature_counts_.values())
            ),
        })
        return result
