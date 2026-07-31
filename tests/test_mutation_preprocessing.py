import pandas as pd

from src.mutation_preprocessing import (
    MutationStringPreprocessor,
    mutation_position,
    mutation_type,
    normalize_variant,
    split_mutation_events,
)


def test_single_cell_is_split_into_individual_events() -> None:
    assert split_mutation_events("L26V L24V") == ("L26V", "L24V")
    assert split_mutation_events("WT") == ()
    assert split_mutation_events(None) == ()


def test_normalization_keeps_missing_distinct_from_wt() -> None:
    assert normalize_variant(" p.V600E ") == "V600E"
    assert normalize_variant("WT") == "WT"
    assert normalize_variant(None) == "MISSING"
    assert mutation_type("p.R213*") == "STOP"
    assert mutation_type("p.E746_A750del") == "DEL"
    assert mutation_position("p.E746_A750del") == 746


def test_long_transform_preserves_raw_and_missing_records() -> None:
    frame = pd.DataFrame(
        {
            "ID": ["S1", "S2"],
            "SUBCLASS": ["COAD", "UCEC"],
            "APC": ["L26V L24V", "WT"],
            "PTEN": [None, "p.R130*"],
        }
    )
    preprocessor = MutationStringPreprocessor().fit(frame)
    transformed = preprocessor.transform(frame)

    mutations = transformed.loc[transformed["status"].eq("MUTATION")]
    missing = transformed.loc[transformed["status"].eq("MISSING")]

    assert mutations["raw_variant"].tolist() == ["L26V", "L24V", "p.R130*"]
    assert mutations["normalized_variant"].tolist() == ["L26V", "L24V", "R130*"]
    assert mutations["event_order"].tolist() == [1, 2, 1]
    assert len(missing) == 1
    assert missing.iloc[0]["gene"] == "PTEN"
    assert missing.iloc[0]["normalized_variant"] == "MISSING"
    assert not transformed["raw_cell"].isna().all()


def test_audit_reports_preprocessing_quality() -> None:
    frame = pd.DataFrame(
        {
            "ID": ["S1", "S2", "S3"],
            "SUBCLASS": ["COAD", "UCEC", "LGG"],
            "APC": ["L26V L24V", "WT", "WT"],
            "PTEN": [None, "p.R130*", "WT"],
        }
    )
    preprocessor = MutationStringPreprocessor().fit(frame)
    transformed = preprocessor.transform(frame)
    audit = preprocessor.audit(frame, transformed)

    assert audit["total_cells"] == 6
    assert audit["mutation_cells"] == 2
    assert audit["mutation_events"] == 3
    assert audit["multi_event_cells"] == 1
    assert audit["missing_cells"] == 1
    assert audit["wt_or_no_mutation_cells"] == 3
    assert audit["position_parse_rate"] == 1.0
