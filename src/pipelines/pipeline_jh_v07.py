"""JH v07: F0+F1+global F3+exact hotspot F4, support 10."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.pipelines.pipeline_jh_v04 import (
    JHV04PreprocessingPipeline,
    parse_wide_mutations,
)
from src.pipelines.base import PreprocessingPipeline


def build_f4_tokens(
    events: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "row_index",
                "f4_feature",
            ]
        )

    result = events[
        [
            "row_index",
            "gene",
            "exact_event",
        ]
    ].dropna().copy()

    result["f4_feature"] = (
        result["gene"].astype(str)
        + "__EXACT="
        + result[
            "exact_event"
        ].astype(str)
    )

    return (
        result[
            [
                "row_index",
                "f4_feature",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_f4_matrix(
    tokens: pd.DataFrame,
    sample_count: int,
    vocabulary: dict[str, int],
) -> sparse.csr_matrix:
    selected = tokens[
        tokens["f4_feature"].isin(
            vocabulary
        )
    ].drop_duplicates(
        subset=[
            "row_index",
            "f4_feature",
        ]
    )

    rows = selected[
        "row_index"
    ].to_numpy(dtype=int)

    columns = (
        selected["f4_feature"]
        .map(vocabulary)
        .to_numpy(dtype=int)
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
            len(vocabulary),
        ),
        dtype=np.float32,
    )

    matrix.data[:] = 1.0
    matrix.eliminate_zeros()

    return matrix


class JHV07PreprocessingPipeline(PreprocessingPipeline):
    """v04에 Fold-Train exact hotspot F4를 추가합니다."""

    name = "jh_v07"
    f4_min_support = 10

    def __init__(
        self,
        f4_min_support: int | None = None,
        **kwargs: object,
    ) -> None:
        self.base_pipeline = JHV04PreprocessingPipeline(**kwargs)

        if f4_min_support is not None:
            self.f4_min_support = int(
                f4_min_support
            )

        self.f4_vocabulary: dict[
            str,
            int,
        ] = {}

        self.f4_support: dict[
            str,
            int,
        ] = {}

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "JHV07PreprocessingPipeline":
        self.base_pipeline.fit(
            features,
            labels,
        )

        events = parse_wide_mutations(
            features
        )

        tokens = build_f4_tokens(
            events
        )

        support = (
            tokens
            .groupby("f4_feature")[
                "row_index"
            ]
            .nunique()
            .sort_values(
                ascending=False
            )
        )

        selected = support[
            support
            >= self.f4_min_support
        ]

        selected_names = sorted(
            selected.index.tolist()
        )

        self.f4_vocabulary = {
            name: column
            for column, name
            in enumerate(
                selected_names
            )
        }

        self.f4_support = {
            name: int(value)
            for name, value
            in selected.items()
        }

        return self

    def transform(
        self,
        features: pd.DataFrame,
    ) -> sparse.csr_matrix:
        base = self.base_pipeline.transform(
            features
        )

        events = parse_wide_mutations(
            features
        )

        tokens = build_f4_tokens(
            events
        )

        f4 = build_f4_matrix(
            tokens=tokens,
            sample_count=len(features),
            vocabulary=(
                self.f4_vocabulary
            ),
        )

        return sparse.hstack(
            [
                base,
                f4,
            ],
            format="csr",
            dtype=np.float32,
        )

    def summary(self) -> dict[str, int]:
        result = self.base_pipeline.summary()

        f4_count = len(
            self.f4_vocabulary
        )

        result.update({
            "f4_features": f4_count,
            "f4_min_support": int(
                self.f4_min_support
            ),
            "remaining_features": (
                result[
                    "remaining_features"
                ]
                + f4_count
            ),
        })

        return result

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> sparse.csr_matrix:
        return self.fit(features, labels).transform(features)

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        return self.base_pipeline.encode_labels(labels)

    def decode_labels(self, labels: np.ndarray) -> np.ndarray:
        return self.base_pipeline.decode_labels(labels)
