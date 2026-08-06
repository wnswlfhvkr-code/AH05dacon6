"""베이스라인에 fold 안정형 유전자 중복 제거를 적용한 EM v47입니다.

모든 선택 기준은 ``fit``에 전달된 학습 데이터에서만 계산합니다. 검증/시험
데이터는 학습 시 결정된 열 목록을 그대로 적용받으므로 데이터 누수가 없습니다.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold

from src.pipelines.base import PreprocessingPipeline


class EMV47PreprocessingPipeline(PreprocessingPipeline):
    """변이 여부 행렬에서 안정적인 양의 고상관 중복 유전자만 제거합니다."""

    name = "em_v47"
    evaluation_folds = 1

    def __init__(
        self,
        minimum_mutation_support: int = 5,
        phi_threshold: float = 0.98,
        jaccard_threshold: float = 0.95,
        stability_folds: int = 5,
        minimum_stability_folds: int = 4,
        correlation_block_size: int = 256,
        random_state: int = 42,
        **_: object,
    ) -> None:
        super().__init__()
        if minimum_mutation_support < 1:
            raise ValueError("minimum_mutation_support는 1 이상이어야 합니다.")
        if not 0.0 <= phi_threshold <= 1.0:
            raise ValueError("phi_threshold는 0과 1 사이여야 합니다.")
        if not 0.0 <= jaccard_threshold <= 1.0:
            raise ValueError("jaccard_threshold는 0과 1 사이여야 합니다.")
        if stability_folds < 2:
            raise ValueError("stability_folds는 2 이상이어야 합니다.")
        if minimum_stability_folds < 1 or minimum_stability_folds > stability_folds:
            raise ValueError(
                "minimum_stability_folds는 1 이상 stability_folds 이하여야 합니다."
            )
        if correlation_block_size < 16:
            raise ValueError("correlation_block_size는 16 이상이어야 합니다.")

        self.minimum_mutation_support = int(minimum_mutation_support)
        self.phi_threshold = float(phi_threshold)
        self.jaccard_threshold = float(jaccard_threshold)
        self.stability_folds = int(stability_folds)
        self.minimum_stability_folds = int(minimum_stability_folds)
        self.correlation_block_size = int(correlation_block_size)
        self.random_state = int(random_state)
        self.steps = (
            "결측값/WT를 0으로 통합하고 변이 여부를 1로 이진화",
            "상수 및 최소 변이 빈도 미달 피처 제거",
            "완전히 동일한 유전자 피처 제거",
            "블록 단위 양의 Phi 상관 및 Jaccard 중복 후보 탐색",
            "학습 fold 안정성으로 중복 후보 확정",
            "안정성·변이 빈도 기준 대표 유전자 유지",
        )

        self.input_columns_: list[str] = []
        self.dropped_low_support_columns: list[str] = []
        self.dropped_exact_duplicate_columns: list[str] = []
        self.dropped_high_correlation_columns: list[str] = []
        self.exact_duplicate_groups_: list[list[str]] = []
        self.high_correlation_groups_: list[list[str]] = []
        self.correlation_candidate_count_: int = 0
        self.stable_correlation_edge_count_: int = 0
        self.actual_stability_folds_: int = 0

    @staticmethod
    def _mutation_matrix(features: pd.DataFrame) -> np.ndarray:
        """결측/WT는 0, 그 밖의 셀 문자열은 변이 1로 변환합니다."""
        normalized = features.astype("string").fillna("WT")
        normalized = normalized.apply(lambda column: column.str.strip().str.upper())
        wild_type_tokens = {"", "WT", "NAN", "NONE", "<NA>"}
        return (~normalized.isin(wild_type_tokens)).to_numpy(dtype=np.uint8)

    @staticmethod
    def _exact_duplicate_groups(matrix: np.ndarray) -> list[list[int]]:
        signatures: dict[bytes, list[int]] = defaultdict(list)
        packed = np.packbits(matrix, axis=0)
        for index in range(matrix.shape[1]):
            signatures[packed[:, index].tobytes()].append(index)
        return [indices for indices in signatures.values() if len(indices) > 1]

    def _correlation_candidates(self, matrix: np.ndarray) -> list[tuple[int, int]]:
        """전체 p×p 상관행렬을 만들지 않고 임계값 통과 쌍만 반환합니다."""
        row_count, column_count = matrix.shape
        if row_count < 2 or column_count < 2:
            return []

        encoded = sparse.csc_matrix(matrix, dtype=np.float32)
        support = np.asarray(encoded.sum(axis=0)).ravel().astype(np.float64)
        block = self.correlation_block_size
        candidates: list[tuple[int, int]] = []

        for left_start in range(0, column_count, block):
            left_end = min(left_start + block, column_count)
            left_support = support[left_start:left_end, None]
            left = encoded[:, left_start:left_end]
            for right_start in range(left_start, column_count, block):
                right_end = min(right_start + block, column_count)
                right_support = support[None, right_start:right_end]
                right = encoded[:, right_start:right_end]
                both = (left.T @ right).toarray().astype(np.float64, copy=False)

                numerator = row_count * both - left_support * right_support
                denominator = np.sqrt(
                    left_support
                    * right_support
                    * (row_count - left_support)
                    * (row_count - right_support)
                )
                phi = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator),
                    where=denominator > 0,
                )
                union = left_support + right_support - both
                jaccard = np.divide(
                    both,
                    union,
                    out=np.zeros_like(both),
                    where=union > 0,
                )
                selected = (phi >= self.phi_threshold) & (
                    jaccard >= self.jaccard_threshold
                )
                if left_start == right_start:
                    selected &= np.triu(np.ones_like(selected, dtype=bool), k=1)

                for left_local, right_local in np.argwhere(selected):
                    candidates.append(
                        (left_start + int(left_local), right_start + int(right_local))
                    )
        return candidates

    def _stable_edges(
        self,
        matrix: np.ndarray,
        labels: pd.Series,
        candidates: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
        if not candidates:
            return [], {}

        encoded_labels = pd.Series(labels).reset_index(drop=True)
        minimum_class_size = int(encoded_labels.value_counts().min())
        fold_count = min(self.stability_folds, minimum_class_size)
        if fold_count < 2:
            self.actual_stability_folds_ = 1
            return candidates, {edge: 1 for edge in candidates}

        self.actual_stability_folds_ = fold_count
        required = min(self.minimum_stability_folds, fold_count)
        edge_counts = np.zeros(len(candidates), dtype=np.int16)
        left_indices = np.asarray([edge[0] for edge in candidates], dtype=np.int32)
        right_indices = np.asarray([edge[1] for edge in candidates], dtype=np.int32)
        splitter = StratifiedKFold(
            n_splits=fold_count, shuffle=True, random_state=self.random_state
        )

        for train_indices, _ in splitter.split(matrix, encoded_labels):
            fold_matrix = matrix[train_indices]
            row_count = fold_matrix.shape[0]
            for start in range(0, len(candidates), 1024):
                end = min(start + 1024, len(candidates))
                left = fold_matrix[:, left_indices[start:end]].astype(np.float64)
                right = fold_matrix[:, right_indices[start:end]].astype(np.float64)
                left_support = left.sum(axis=0)
                right_support = right.sum(axis=0)
                both = (left * right).sum(axis=0)
                numerator = row_count * both - left_support * right_support
                denominator = np.sqrt(
                    left_support
                    * right_support
                    * (row_count - left_support)
                    * (row_count - right_support)
                )
                phi = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator),
                    where=denominator > 0,
                )
                union = left_support + right_support - both
                jaccard = np.divide(
                    both,
                    union,
                    out=np.zeros_like(both),
                    where=union > 0,
                )
                edge_counts[start:end] += (
                    (phi >= self.phi_threshold)
                    & (jaccard >= self.jaccard_threshold)
                )

        stable = [
            edge for edge, count in zip(candidates, edge_counts) if count >= required
        ]
        counts = {
            edge: int(count)
            for edge, count in zip(candidates, edge_counts)
            if count >= required
        }
        return stable, counts

    @staticmethod
    def _connected_components(
        column_count: int, edges: list[tuple[int, int]]
    ) -> list[list[int]]:
        parent = list(range(column_count))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for left, right in edges:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        groups: dict[int, list[int]] = defaultdict(list)
        involved = {node for edge in edges for node in edge}
        for node in involved:
            groups[find(node)].append(node)
        return [sorted(group) for group in groups.values() if len(group) > 1]

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "EMV47PreprocessingPipeline":
        self.input_columns_ = features.columns.tolist()
        matrix = self._mutation_matrix(features)
        support = matrix.sum(axis=0)

        constant_mask = (support == 0) | (support == matrix.shape[0])
        low_support_mask = (~constant_mask) & (
            support < self.minimum_mutation_support
        )
        self.dropped_constant_columns = [
            column
            for column, dropped in zip(self.input_columns_, constant_mask)
            if dropped
        ]
        self.dropped_low_support_columns = [
            column
            for column, dropped in zip(self.input_columns_, low_support_mask)
            if dropped
        ]

        eligible_mask = ~(constant_mask | low_support_mask)
        eligible_columns = [
            column
            for column, keep in zip(self.input_columns_, eligible_mask)
            if keep
        ]
        eligible_matrix = matrix[:, eligible_mask]

        exact_groups = self._exact_duplicate_groups(eligible_matrix)
        exact_drop_indices = {index for group in exact_groups for index in group[1:]}
        self.exact_duplicate_groups_ = [
            [eligible_columns[index] for index in group] for group in exact_groups
        ]
        self.dropped_exact_duplicate_columns = [
            eligible_columns[index] for index in sorted(exact_drop_indices)
        ]
        unique_indices = [
            index
            for index in range(len(eligible_columns))
            if index not in exact_drop_indices
        ]
        unique_columns = [eligible_columns[index] for index in unique_indices]
        unique_matrix = eligible_matrix[:, unique_indices]

        candidates = self._correlation_candidates(unique_matrix)
        self.correlation_candidate_count_ = len(candidates)
        stable_edges, edge_stability = self._stable_edges(
            unique_matrix, labels, candidates
        )
        self.stable_correlation_edge_count_ = len(stable_edges)
        correlation_groups = self._connected_components(
            len(unique_columns), stable_edges
        )
        unique_support = unique_matrix.sum(axis=0)
        node_stability: dict[int, int] = defaultdict(int)
        for (left, right), count in edge_stability.items():
            node_stability[left] += count
            node_stability[right] += count

        correlation_drop_indices: set[int] = set()
        self.high_correlation_groups_ = []
        for group in correlation_groups:
            representative = max(
                group,
                key=lambda index: (
                    node_stability[index],
                    int(unique_support[index]),
                    float(
                        unique_support[index]
                        * (unique_matrix.shape[0] - unique_support[index])
                    ),
                    -index,
                ),
            )
            correlation_drop_indices.update(set(group) - {representative})
            self.high_correlation_groups_.append(
                [unique_columns[index] for index in group]
            )

        self.dropped_high_correlation_columns = [
            unique_columns[index] for index in sorted(correlation_drop_indices)
        ]
        self.feature_columns = [
            column
            for index, column in enumerate(unique_columns)
            if index not in correlation_drop_indices
        ]
        self.categorical_columns = []
        self.label_encoder.fit(labels)

        print(f"[em_v47] 상수 피처 제거: {len(self.dropped_constant_columns)}개")
        print(
            "[em_v47] 최소 변이 빈도 미달 제거: "
            f"{len(self.dropped_low_support_columns)}개"
        )
        print(
            "[em_v47] 완전 중복 피처 제거: "
            f"{len(self.dropped_exact_duplicate_columns)}개"
        )
        print(
            "[em_v47] fold 안정형 고상관 피처 제거: "
            f"{len(self.dropped_high_correlation_columns)}개"
        )
        print(f"[em_v47] 최종 피처: {len(self.feature_columns)}개")
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"누락된 피처가 있습니다: {sorted(missing)}")
        matrix = self._mutation_matrix(features[self.feature_columns])
        return pd.DataFrame(matrix, index=features.index, columns=self.feature_columns)

    def summary(self) -> dict[str, int]:
        return {
            "input_features": len(self.input_columns_),
            "dropped_constant_features": len(self.dropped_constant_columns),
            "dropped_low_support_features": len(self.dropped_low_support_columns),
            "dropped_exact_duplicate_features": len(
                self.dropped_exact_duplicate_columns
            ),
            "correlation_candidates": self.correlation_candidate_count_,
            "stable_correlation_edges": self.stable_correlation_edge_count_,
            "dropped_high_correlation_features": len(
                self.dropped_high_correlation_columns
            ),
            "remaining_features": len(self.feature_columns),
        }
