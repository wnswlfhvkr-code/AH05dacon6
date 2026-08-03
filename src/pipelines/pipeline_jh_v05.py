"""JH v05: F0+F1+gene-aware F3, Fold-Train support 3."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.pipelines.pipeline_jh_v02 import (
    JHV02PreprocessingPipeline,
)
from src.pipelines.pipeline_jh_v04 import (
    AMINO_ACIDS,
    POSITION_BIN_EDGES,
    POSITION_BIN_LABELS,
    parse_wide_mutations,
)
from src.pipelines.base import PreprocessingPipeline


def build_gene_f3_tokens(
    events: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "row_index",
                "f3_feature",
            ]
        )

    work = events[
        [
            "row_index",
            "gene",
            "aa_from",
            "aa_to",
            "position",
        ]
    ].copy()

    work["aa_from"] = (
        work["aa_from"]
        .astype("string")
        .str.upper()
    )

    work["aa_to"] = (
        work["aa_to"]
        .astype("string")
        .str.upper()
    )

    work["position_bin"] = pd.cut(
        pd.to_numeric(
            work["position"],
            errors="coerce",
        ),
        bins=POSITION_BIN_EDGES,
        labels=POSITION_BIN_LABELS,
        include_lowest=True,
    ).astype("string")

    valid_amino_acids = set(
        AMINO_ACIDS
    )

    frames = []

    aa_from_mask = (
        work["aa_from"].isin(
            valid_amino_acids
        )
    )

    aa_from = work.loc[
        aa_from_mask,
        [
            "row_index",
            "gene",
            "aa_from",
        ],
    ].copy()

    aa_from["f3_feature"] = (
        aa_from["gene"]
        + "__AA_FROM="
        + aa_from["aa_from"]
    )

    frames.append(
        aa_from[
            [
                "row_index",
                "f3_feature",
            ]
        ]
    )

    aa_to_mask = (
        work["aa_to"].isin(
            valid_amino_acids
        )
    )

    aa_to = work.loc[
        aa_to_mask,
        [
            "row_index",
            "gene",
            "aa_to",
        ],
    ].copy()

    aa_to["f3_feature"] = (
        aa_to["gene"]
        + "__AA_TO="
        + aa_to["aa_to"]
    )

    frames.append(
        aa_to[
            [
                "row_index",
                "f3_feature",
            ]
        ]
    )

    aa_change_mask = (
        work["aa_from"].isin(
            valid_amino_acids
        )
        & work["aa_to"].isin(
            valid_amino_acids
        )
    )

    aa_change = work.loc[
        aa_change_mask,
        [
            "row_index",
            "gene",
            "aa_from",
            "aa_to",
        ],
    ].copy()

    aa_change["f3_feature"] = (
        aa_change["gene"]
        + "__AA_CHANGE="
        + aa_change["aa_from"]
        + ">"
        + aa_change["aa_to"]
    )

    frames.append(
        aa_change[
            [
                "row_index",
                "f3_feature",
            ]
        ]
    )

    position_mask = (
        work["position_bin"].notna()
        & work["position_bin"].ne(
            "<NA>"
        )
    )

    position = work.loc[
        position_mask,
        [
            "row_index",
            "gene",
            "position_bin",
        ],
    ].copy()

    position["f3_feature"] = (
        position["gene"]
        + "__POSITION_BIN="
        + position["position_bin"]
    )

    frames.append(
        position[
            [
                "row_index",
                "f3_feature",
            ]
        ]
    )

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .drop_duplicates(
            subset=[
                "row_index",
                "f3_feature",
            ]
        )
        .reset_index(drop=True)
    )


def build_token_matrix(
    token_table: pd.DataFrame,
    sample_count: int,
    vocabulary: dict[str, int],
    token_column: str,
) -> sparse.csr_matrix:
    selected = token_table[
        token_table[
            token_column
        ].isin(vocabulary)
    ].drop_duplicates(
        subset=[
            "row_index",
            token_column,
        ]
    )

    rows = selected[
        "row_index"
    ].to_numpy(dtype=int)

    columns = (
        selected[token_column]
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


class JHV05PreprocessingPipeline(PreprocessingPipeline):
    """v02에 Fold-Train support를 통과한 gene-aware F3를 추가합니다."""

    name = "jh_v05"
    min_support = 3

    def __init__(
        self,
        min_support: int | None = None,
        **kwargs: object,
    ) -> None:
        self.base_pipeline = JHV02PreprocessingPipeline(**kwargs)

        if min_support is not None:
            self.min_support = int(
                min_support
            )

        self.f3_vocabulary: dict[
            str,
            int,
        ] = {}

        self.f3_support: dict[
            str,
            int,
        ] = {}

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "JHV05PreprocessingPipeline":
        self.base_pipeline.fit(
            features,
            labels,
        )

        events = parse_wide_mutations(
            features
        )

        tokens = build_gene_f3_tokens(
            events
        )

        support = (
            tokens
            .groupby("f3_feature")[
                "row_index"
            ]
            .nunique()
            .sort_values(
                ascending=False
            )
        )

        selected = support[
            support >= self.min_support
        ]

        selected_names = sorted(
            selected.index.tolist()
        )

        self.f3_vocabulary = {
            name: column
            for column, name
            in enumerate(
                selected_names
            )
        }

        self.f3_support = {
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

        tokens = build_gene_f3_tokens(
            events
        )

        f3 = build_token_matrix(
            token_table=tokens,
            sample_count=len(features),
            vocabulary=(
                self.f3_vocabulary
            ),
            token_column="f3_feature",
        )

        return sparse.hstack(
            [
                base,
                f3,
            ],
            format="csr",
            dtype=np.float32,
        )

    def summary(self) -> dict[str, int]:
        result = self.base_pipeline.summary()

        f3_count = len(
            self.f3_vocabulary
        )

        result.update({
            "f3_features": f3_count,
            "f3_min_support": int(
                self.min_support
            ),
            "remaining_features": (
                result[
                    "remaining_features"
                ]
                + f3_count
            ),
        })

        return result

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> sparse.csr_matrix:
        return self.fit(features, labels).transform(features)

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        return self.base_pipeline.encode_labels(labels)

    def decode_labels(self, labels: np.ndarray) -> np.ndarray:
        return self.base_pipeline.decode_labels(labels)
