import pandas as pd

from src.mutation_features import (
    MutationFeatureEngineer,
    mutation_position,
    mutation_type,
    normalize_variant,
)


def test_variant_normalization_and_parsing() -> None:
    assert normalize_variant(" p.V600E ") == "V600E"
    assert normalize_variant(None) == "MISSING"
    assert mutation_type("p.V600E") == "SUB"
    assert mutation_type("p.R213*") == "STOP"
    assert mutation_type("p.E746_A750del") == "DEL"
    assert mutation_type("p.G12fs") == "FS"
    assert mutation_position("p.E746_A750del") == 746


def test_fold_safe_marker_selection_and_transform_schema() -> None:
    features = pd.DataFrame(
        {
            "APC": ["p.R145*", "p.E1309fs", "p.R876*", "WT", "WT", "WT", "WT", "WT"],
            "PTEN": ["WT", "WT", "WT", "WT", "p.R130*", "p.G129E", "p.R173C", "WT"],
            "TP53": ["p.R175H", "WT", "p.R248Q", "WT", "p.R175H", "WT", "p.R248Q", "WT"],
        }
    )
    labels = pd.Series(["COAD"] * 4 + ["UCEC"] * 4)
    engineer = MutationFeatureEngineer(
        min_class_rate=0.50,
        min_class_count=2,
        min_rate_difference=0.25,
        min_lift=1.5,
        max_markers_per_class=3,
    )

    transformed = engineer.fit_transform(features, labels)
    marker_table = engineer.get_marker_table()

    assert {"COAD", "UCEC"} == set(marker_table["SUBCLASS"])
    assert "APC" in set(marker_table.loc[marker_table["SUBCLASS"] == "COAD", "gene"])
    assert "PTEN" in set(marker_table.loc[marker_table["SUBCLASS"] == "UCEC", "gene"])
    assert "marker_COAD_score" in transformed
    assert "marker_UCEC_score" in transformed
    assert transformed.loc[0, "marker_COAD_score"] > transformed.loc[4, "marker_COAD_score"]
    assert transformed.loc[4, "marker_UCEC_score"] > transformed.loc[0, "marker_UCEC_score"]

    unseen = features.copy()
    unseen.loc[0, "APC"] = "p.Q999delinsR"
    unseen_transformed = engineer.transform(unseen)
    assert transformed.columns.tolist() == unseen_transformed.columns.tolist()


def test_transform_does_not_require_labels() -> None:
    train = pd.DataFrame(
        {
            "BRAF": ["p.V600E", "p.V600E", "WT", "WT"],
            "IDH1": ["WT", "WT", "p.R132H", "p.R132H"],
        }
    )
    labels = pd.Series(["THCA", "THCA", "LGG", "LGG"])
    test = pd.DataFrame(
        {
            "BRAF": ["p.V600K", "WT"],
            "IDH1": ["WT", "p.R132C"],
        }
    )
    engineer = MutationFeatureEngineer(
        min_class_rate=0.50,
        min_class_count=1,
        min_rate_difference=0.25,
        min_lift=1.5,
    ).fit(train, labels)

    transformed = engineer.transform(test)

    assert len(transformed) == len(test)
    assert not transformed.isna().any().any()
