import numpy as np
import pandas as pd
from scipy import sparse

from src.pipelines.pipeline_jh_v04 import (
    JHV04PreprocessingPipeline,
    build_profile_groups,
    make_class_burden_strata,
    parse_event,
    parse_wide_mutations,
    split_mutation_tokens,
)
from src.pipelines.pipeline_jh_v09 import JHV09PreprocessingPipeline
from src.pipelines.pipeline_jh_v10 import JHV10PreprocessingPipeline


def sample_features() -> pd.DataFrame:
    return pd.DataFrame({
        "TP53": ["WT", "R175H", "R213X", "R175H R248Q", None, "R175H"],
        "KRAS": ["G12D", "WT", "G12D", "WT", None, "WT"],
        "BRAF": ["WT", "V600E", "WT", "V600E", None, "V600E"],
    })


def test_parser_normalizes_and_classifies_events() -> None:
    assert split_mutation_tokens("WT R175H R175H") == ["R175H"]
    assert split_mutation_tokens("MISSING") == []
    assert split_mutation_tokens("WT MISSING R175H") == ["R175H"]
    assert parse_event("R213X")["exact_event"] == "R213*"
    assert parse_event("R213X")["consequence"] == "STOP"
    assert parse_event("K16fs")["consequence"] == "FRAMESHIFT"
    assert parse_event("E746_A750del")["consequence"] == "COMPLEX"


def test_v04_pipeline_builds_sparse_f0_f1_f3_features() -> None:
    features = sample_features()
    labels = pd.Series(["A", "A", "B", "B", "A", "B"])
    pipeline = JHV04PreprocessingPipeline()
    transformed = pipeline.fit_transform(features, labels)
    repeated = pipeline.transform(features)

    assert sparse.isspmatrix_csr(transformed)
    assert transformed.shape == repeated.shape
    assert transformed.shape[0] == len(features)
    assert pipeline.summary()["f1_features"] == 22
    assert pipeline.summary()["f0_features"] == 3
    assert pipeline.summary()["f3_features"] > 0
    assert np.isfinite(transformed.data).all()


def test_missing_token_is_not_counted_as_mutation() -> None:
    features = pd.DataFrame({"TP53": ["MISSING", "WT", None, "R175H"]})
    labels = pd.Series(["A", "A", "B", "B"])
    pipeline = JHV04PreprocessingPipeline()
    transformed = pipeline.fit_transform(features, labels)

    assert pipeline.summary()["f0_features"] == 1
    assert transformed[:3, 0].nnz == 0
    assert transformed[3, 0] == 1


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


def test_v09_applies_fold_fitted_tfidf_and_keeps_f1() -> None:
    features = sample_features()
    labels = pd.Series(["A", "A", "B", "B", "A", "B"])
    pipeline = JHV09PreprocessingPipeline(f4_min_support=1)
    transformed = pipeline.fit_transform(features, labels)
    repeated = pipeline.transform(features)

    assert sparse.isspmatrix_csr(transformed)
    assert transformed.shape == repeated.shape
    assert transformed.shape[0] == len(features)
    assert pipeline.summary()["f1_features"] == 22
    assert pipeline.summary()["tfidf_token_features"] > 0
    assert np.isfinite(transformed.data).all()


def test_v10_reduces_tfidf_and_appends_f1_as_dense_features() -> None:
    features = sample_features()
    labels = pd.Series(["A", "A", "B", "B", "A", "B"])
    pipeline = JHV10PreprocessingPipeline(
        f4_min_support=1,
        svd_components=2,
        svd_n_iter=3,
        svd_random_state=7,
    )
    transformed = pipeline.fit_transform(features, labels)
    repeated = pipeline.transform(features)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == repeated.shape == (len(features), 24)
    assert pipeline.summary()["f1_features"] == 22
    assert pipeline.summary()["svd_components"] == 2
    assert pipeline.summary()["final_features"] == 24
    assert np.isfinite(transformed).all()
