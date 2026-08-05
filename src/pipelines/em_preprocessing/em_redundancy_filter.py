"""EMV46용 저빈도·완전 중복·고상관 제거 공통 엔진."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_em_v46 import EMV46PreprocessingPipeline


class EMV46RedundancyFilterEngine:
    """V46 피처를 정해진 순서로 줄이고 학습 fold의 선택 상태를 보존합니다."""

    def __init__(self, parameters: dict[str, object]) -> None:
        self.min_active_count = int(parameters.get("min_active_count", 5))
        self.correlation_threshold = float(parameters.get("correlation_threshold", 0.90))
        self.correlation_max_features = int(parameters.get("correlation_max_features", 3000))
        self.correlation_block_size = int(parameters.get("correlation_block_size", 256))
        self.stability_folds = int(parameters.get("stability_folds", 5))
        self.stability_top_fraction = float(parameters.get("stability_top_fraction", 0.20))
        self.active_epsilon = float(parameters.get("active_epsilon", 1e-8))
        self.random_state = int(parameters.get("random_state", 42))
        if self.min_active_count < 1:
            raise ValueError("min_active_count는 1 이상이어야 합니다.")
        if not 0.0 < self.correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold는 0보다 크고 1 이하여야 합니다.")
        if self.correlation_max_features < 2 or self.correlation_block_size < 2:
            raise ValueError("상관 후보 수와 block 크기는 2 이상이어야 합니다.")
        if self.stability_folds < 2:
            raise ValueError("stability_folds는 2 이상이어야 합니다.")
        if not 0.0 < self.stability_top_fraction <= 1.0:
            raise ValueError("stability_top_fraction은 0보다 크고 1 이하여야 합니다.")

        v46_parameters = {
            key: value
            for key, value in parameters.items()
            if key not in {
                "min_active_count",
                "correlation_threshold",
                "correlation_max_features",
                "correlation_block_size",
                "stability_folds",
                "stability_top_fraction",
                "active_epsilon",
            }
        }
        self.v46_pipeline = EMV46PreprocessingPipeline(**v46_parameters)
        self.raw_missing_rates_: dict[str, float] = {}
        self.dropped_constant_: list[str] = []
        self.dropped_rare_: list[str] = []
        self.dropped_exact_duplicate_: list[str] = []
        self.dropped_high_correlation_: list[str] = []
        self.correlation_candidate_count_ = 0
        self.high_correlation_group_count_ = 0

    @property
    def steps(self) -> tuple[str, ...]:
        return (
            "EMV46 피처 1회 생성",
            "상수 피처 제거",
            "fold-train 최소 활성 빈도 필터",
            "완전히 동일한 피처 제거",
            f"|상관계수| >= {self.correlation_threshold:.2f} 후보 탐색",
            "클래스 연관성·fold 안정성·빈도·해석성·결측 편향 순으로 대표 유지",
            "validation/test에는 train-fit 선택 열만 적용",
            "원본 SUBCLASS 유지",
        )

    @staticmethod
    def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        return numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

    @staticmethod
    def _association_scores(frame: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
        encoded = LabelEncoder().fit_transform(labels.astype(str))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores, _ = f_classif(frame.to_numpy(dtype="float32", copy=False), encoded)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return dict(zip(frame.columns, scores.astype(float), strict=True))

    def _stability_scores(
        self, frame: pd.DataFrame, labels: pd.Series
    ) -> dict[str, float]:
        minimum_class = int(labels.value_counts().min())
        folds = min(self.stability_folds, minimum_class)
        if folds < 2:
            return {column: 0.0 for column in frame.columns}
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.random_state
        )
        selected = np.zeros(frame.shape[1], dtype="float64")
        top_count = max(1, int(np.ceil(frame.shape[1] * self.stability_top_fraction)))
        for train_index, _ in splitter.split(frame, labels):
            fold_scores = self._association_scores(
                frame.iloc[train_index], labels.iloc[train_index]
            )
            values = np.asarray([fold_scores[column] for column in frame.columns])
            top_indices = np.argpartition(values, -top_count)[-top_count:]
            selected[top_indices] += 1.0
        selected /= folds
        return dict(zip(frame.columns, selected, strict=True))

    @staticmethod
    def _interpretability_score(column: str, raw_genes: set[str]) -> float:
        upper = column.upper()
        if upper in raw_genes:
            return 5.0
        if "HOTSPOT" in upper or "CONSEQUENCE" in upper:
            return 4.0
        if any(token in upper for token in ("MUTATION_COUNT", "MULTI_HIT", "GENE_MUTATED")):
            return 4.0
        if any(token in upper for token in ("BURDEN", "COUNT", "RATIO", "ENTROPY")):
            return 3.0
        if "SIGNATURE" in upper or "PATTERN" in upper:
            return 2.0
        return 1.0

    @staticmethod
    def _feature_missing_rate(
        column: str, raw_missing_rates: dict[str, float]
    ) -> float:
        upper = column.upper()
        if upper in raw_missing_rates:
            return raw_missing_rates[upper]
        tokens = re.findall(r"[A-Z0-9]+", upper)
        matched = [raw_missing_rates[token] for token in tokens if token in raw_missing_rates]
        return max(matched, default=0.0)

    def _priority(
        self,
        column: str,
        association: dict[str, float],
        stability: dict[str, float],
        support: dict[str, int],
        interpretability: dict[str, float],
        missing_rate: dict[str, float],
    ) -> tuple[float, ...]:
        return (
            float(association[column]),
            float(stability[column]),
            float(support[column]),
            float(interpretability[column]),
            -float(missing_rate[column]),
            -float(len(column)),
        )

    @staticmethod
    def _fingerprint(values: np.ndarray) -> bytes:
        contiguous = np.ascontiguousarray(values, dtype="float32")
        return hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()

    def _drop_exact_duplicates(
        self,
        frame: pd.DataFrame,
        priority,
    ) -> tuple[pd.DataFrame, list[str]]:
        buckets: dict[bytes, list[str]] = defaultdict(list)
        for column in frame.columns:
            buckets[self._fingerprint(frame[column].to_numpy())].append(column)
        dropped: list[str] = []
        for candidates in buckets.values():
            if len(candidates) < 2:
                continue
            groups: list[list[str]] = []
            for column in candidates:
                values = frame[column].to_numpy()
                for group in groups:
                    if np.array_equal(values, frame[group[0]].to_numpy(), equal_nan=True):
                        group.append(column)
                        break
                else:
                    groups.append([column])
            for group in groups:
                if len(group) > 1:
                    winner = max(group, key=priority)
                    dropped.extend(column for column in group if column != winner)
        return frame.drop(columns=dropped), dropped

    def _high_correlation_groups(
        self,
        frame: pd.DataFrame,
        priority,
    ) -> list[list[str]]:
        if frame.shape[1] < 2:
            self.correlation_candidate_count_ = frame.shape[1]
            return []
        candidate_rank = sorted(
            frame.columns,
            key=priority,
            reverse=True,
        )[: self.correlation_max_features]
        self.correlation_candidate_count_ = len(candidate_rank)
        values = frame[candidate_rank].to_numpy(dtype="float32", copy=True)
        values -= values.mean(axis=0, keepdims=True)
        scale = values.std(axis=0, ddof=1, keepdims=True)
        scale[scale <= self.active_epsilon] = 1.0
        values /= scale
        values = np.nan_to_num(values, copy=False)
        denominator = max(len(frame) - 1, 1)

        parent = np.arange(len(candidate_rank), dtype="int32")

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        width = len(candidate_rank)
        for start in range(0, width, self.correlation_block_size):
            stop = min(start + self.correlation_block_size, width)
            correlations = values[:, start:stop].T @ values / denominator
            for local_index in range(stop - start):
                global_index = start + local_index
                matches = np.flatnonzero(
                    np.abs(correlations[local_index, global_index + 1 :])
                    >= self.correlation_threshold
                )
                for offset in matches:
                    union(global_index, global_index + 1 + int(offset))

        components: dict[int, list[str]] = defaultdict(list)
        for index, column in enumerate(candidate_rank):
            components[find(index)].append(column)
        return [group for group in components.values() if len(group) > 1]

    def _learn_selection(
        self,
        owner: PreprocessingPipeline,
        transformed: pd.DataFrame,
        raw_features: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        frame = self._numeric(transformed)
        self.dropped_constant_ = frame.columns[frame.nunique(dropna=False) <= 1].tolist()
        frame = frame.drop(columns=self.dropped_constant_)

        active_support = frame.abs().gt(self.active_epsilon).sum(axis=0).astype(int)
        self.dropped_rare_ = active_support[active_support < self.min_active_count].index.tolist()
        frame = frame.drop(columns=self.dropped_rare_)
        if frame.empty:
            raise ValueError("최소 활성 빈도를 만족하는 EMV46 피처가 없습니다.")

        association = self._association_scores(frame, labels)
        stability = self._stability_scores(frame, labels)
        support = frame.abs().gt(self.active_epsilon).sum(axis=0).astype(int).to_dict()
        raw_genes = {str(column).upper() for column in raw_features.columns}
        interpretability = {
            column: self._interpretability_score(column, raw_genes)
            for column in frame.columns
        }
        missing_rate = {
            column: self._feature_missing_rate(column, self.raw_missing_rates_)
            for column in frame.columns
        }
        priority = lambda column: self._priority(
            column, association, stability, support, interpretability, missing_rate
        )

        frame, self.dropped_exact_duplicate_ = self._drop_exact_duplicates(frame, priority)
        groups = self._high_correlation_groups(frame, priority)
        self.high_correlation_group_count_ = len(groups)
        self.dropped_high_correlation_ = []
        for group in groups:
            winner = max(group, key=priority)
            self.dropped_high_correlation_.extend(
                column for column in group if column != winner
            )
        frame = frame.drop(columns=sorted(set(self.dropped_high_correlation_)))
        PreprocessingPipeline.fit(owner, frame, labels)
        self._print_summary(owner.name)
        return PreprocessingPipeline.transform(owner, frame).astype("float32")

    def fit(
        self,
        owner: PreprocessingPipeline,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        self.fit_transform(owner, features, labels)

    def fit_transform(
        self,
        owner: PreprocessingPipeline,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        self.raw_missing_rates_ = {
            str(column).upper(): float(features[column].isna().mean())
            for column in features.columns
        }
        transformed = self.v46_pipeline.fit_transform(features, labels)
        return self._learn_selection(owner, transformed, features, labels)

    def transform(
        self, owner: PreprocessingPipeline, features: pd.DataFrame
    ) -> pd.DataFrame:
        transformed = self._numeric(self.v46_pipeline.transform(features))
        return PreprocessingPipeline.transform(owner, transformed).astype("float32")

    def _print_summary(self, name: str) -> None:
        print(f"[{name}] 상수로 삭제된 피처: {len(self.dropped_constant_)}개")
        print(f"[{name}] 저빈도로 삭제된 피처: {len(self.dropped_rare_)}개")
        print(f"[{name}] 완전 중복으로 삭제된 피처: {len(self.dropped_exact_duplicate_)}개")
        print(f"[{name}] 고상관으로 삭제된 피처: {len(set(self.dropped_high_correlation_))}개")

    def summary(self, owner: PreprocessingPipeline) -> dict[str, int]:
        result = PreprocessingPipeline.summary(owner)
        result.update({
            "dropped_preexisting_constant_features": len(self.dropped_constant_),
            "dropped_low_frequency_features": len(self.dropped_rare_),
            "dropped_exact_duplicate_features": len(self.dropped_exact_duplicate_),
            "dropped_high_correlation_features": len(set(self.dropped_high_correlation_)),
            "correlation_candidates": self.correlation_candidate_count_,
            "high_correlation_groups": self.high_correlation_group_count_,
        })
        return result
