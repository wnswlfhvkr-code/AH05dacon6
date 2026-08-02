import numpy as np
import pandas as pd
from scipy import sparse

from src.pipelines.pipeline_jh_v01 import (
    JHV01PreprocessingPipeline,
    build_profile_groups,
    make_class_burden_strata,
    parse_event,
    parse_wide_mutations,
    split_mutation_tokens,
)


def sample_features() -> pd.DataFrame:
    return pd.DataFrame({
        "TP53": ["WT", "R175H", "R213X", "R175H R248Q", None, "R175H"],
        "KRAS": ["G12D", "WT", "G12D", "WT", None, "WT"],
        "BRAF": ["WT", "V600E", "WT", "V600E", None, "V600E"],
    })


def test_parser_normalizes_and_classifies_events() -> None:
    assert split_mutation_tokens("WT R175H R175H") == ["R175H"]
    assert parse_event("R213X")["exact_event"] == "R213*"
    assert parse_event("R213X")["consequence"] == "STOP"
    assert parse_event("K16fs")["consequence"] == "FRAMESHIFT"
    assert parse_event("E746_A750del")["consequence"] == "COMPLEX"


def test_v01_pipeline_builds_sparse_f0_f1_f3_features() -> None:
    features = sample_features()
    labels = pd.Series(["A", "A", "B", "B", "A", "B"])
    pipeline = JHV01PreprocessingPipeline()
    transformed = pipeline.fit_transform(features, labels)
    repeated = pipeline.transform(features)

    assert sparse.isspmatrix_csr(transformed)
    assert transformed.shape == repeated.shape
    assert transformed.shape[0] == len(features)
    assert pipeline.summary()["f1_features"] == 22
    assert pipeline.summary()["f0_features"] == 3
    assert pipeline.summary()["f3_features"] > 0
    assert np.isfinite(transformed.data).all()


def test_profile_groups_keep_identical_profiles_together() -> None:
    features = sample_features()
    events = parse_wide_mutations(features)
    groups = build_profile_groups(events, len(features))
    assert groups[1] == groups[5]
    assert groups[0] != groups[1]


def test_class_burden_strata_are_valid_for_splits() -> None:
    labels = pd.Series(["A"] * 6 + ["B"] * 6)
    burden = np.array([1, 1, 7, 7, 20, 20] * 2)
    strata, bins = make_class_burden_strata(labels, burden, n_splits=2)
    assert len(strata) == len(labels)
    assert len(bins) == len(labels)
    assert strata.value_counts().min() >= 2
