"""JH v03: F0 + F1 + F2 유전자×consequence multi-hot."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.pipelines.pipeline_jh_v02 import (
    JHV02PreprocessingPipeline,
)
from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_jh_v04 import (
    CONSEQUENCE_TYPES,
    parse_wide_mutations,
)


def build_f2_matrix(
    events: pd.DataFrame,
    sample_count: int,
    gene_columns: list[str],
) -> tuple[
    sparse.csr_matrix,
    list[str],
]:
    gene_to_index = {
        gene: index
        for index, gene
        in enumerate(gene_columns)
    }

    consequence_to_index = {
        consequence: index
        for index, consequence
        in enumerate(
            CONSEQUENCE_TYPES
        )
    }

    feature_names = [
        f"{gene}__{consequence}"
        for gene in gene_columns
        for consequence in CONSEQUENCE_TYPES
    ]

    if events.empty:
        return (
            sparse.csr_matrix(
                (
                    sample_count,
                    len(feature_names),
                ),
                dtype=np.float32,
            ),
            feature_names,
        )

    selected = (
        events[
            events["gene"].isin(
                gene_to_index
            )
            & events[
                "consequence"
            ].isin(
                consequence_to_index
            )
        ][
            [
                "row_index",
                "gene",
                "consequence",
            ]
        ]
        .drop_duplicates()
    )

    rows = (
        selected["row_index"]
        .to_numpy(dtype=int)
    )

    gene_indices = (
        selected["gene"]
        .map(gene_to_index)
        .to_numpy(dtype=int)
    )

    consequence_indices = (
        selected["consequence"]
        .map(consequence_to_index)
        .to_numpy(dtype=int)
    )

    columns = (
        gene_indices
        * len(CONSEQUENCE_TYPES)
        + consequence_indices
    )

    matrix = sparse.csr_matrix(
        (
            np.ones(
                len(selected),
                dtype=np.float32,
            ),
            (
                rows,
                columns,
            ),
        ),
        shape=(
            sample_count,
            len(feature_names),
        ),
        dtype=np.float32,
    )

    matrix.data[:] = 1.0
    matrix.eliminate_zeros()

    return matrix, feature_names


class JHV03PreprocessingPipeline(PreprocessingPipeline):
    """v02에 유전자×consequence F2를 추가합니다."""

    name = "jh_v03"

    def __init__(self, **kwargs: object) -> None:
        self.base_pipeline = JHV02PreprocessingPipeline(**kwargs)

        self.f2_active_mask: (
            np.ndarray | None
        ) = None

        self.f2_all_names: list[str] = []

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "JHV03PreprocessingPipeline":
        self.base_pipeline.fit(
            features,
            labels,
        )
        self.gene_columns = self.base_pipeline.gene_columns

        events = parse_wide_mutations(
            features
        )

        f2, self.f2_all_names = (
            build_f2_matrix(
                events=events,
                sample_count=len(features),
                gene_columns=(
                    self.gene_columns
                ),
            )
        )

        self.f2_active_mask = (
            np.asarray(
                f2.getnnz(axis=0)
            ).ravel()
            > 0
        )

        return self

    def transform(
        self,
        features: pd.DataFrame,
    ) -> sparse.csr_matrix:
        if self.f2_active_mask is None:
            raise RuntimeError(
                "fit을 먼저 실행해야 합니다."
            )

        base = self.base_pipeline.transform(
            features
        )

        events = parse_wide_mutations(
            features
        )

        f2, _ = build_f2_matrix(
            events=events,
            sample_count=len(features),
            gene_columns=self.gene_columns,
        )

        return sparse.hstack(
            [
                base,
                f2[
                    :,
                    self.f2_active_mask,
                ],
            ],
            format="csr",
            dtype=np.float32,
        )

    def summary(self) -> dict[str, int]:
        result = self.base_pipeline.summary()

        f2_count = (
            int(
                self.f2_active_mask.sum()
            )
            if self.f2_active_mask
            is not None
            else 0
        )

        result["f2_features"] = f2_count
        result["remaining_features"] += (
            f2_count
        )

        return result

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> sparse.csr_matrix:
        return self.fit(features, labels).transform(features)

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        return self.base_pipeline.encode_labels(labels)

    def decode_labels(self, labels: np.ndarray) -> np.ndarray:
        return self.base_pipeline.decode_labels(labels)
