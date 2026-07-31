import pandas as pd

from src.preprocessing_ablation import (
    build_mutation_documents,
    build_structural_features,
)


def test_split_documents_recover_individual_events() -> None:
    frame = pd.DataFrame(
        {
            "BRAF": ["V600E V512E", "WT"],
            "PTEN": ["WT", "R130*"],
        }
    )
    raw = build_mutation_documents(
        frame,
        ["BRAF", "PTEN"],
        split_events=False,
    )
    split = build_mutation_documents(
        frame,
        ["BRAF", "PTEN"],
        split_events=True,
    )

    assert "E=BRAF:V600E_V512E" in raw[0]
    assert "GT=BRAF:OTHER" in raw[0]
    assert "E=BRAF:V600E" in split[0]
    assert "E=BRAF:V512E" in split[0]
    assert split[0].count("GT=BRAF:SUB") == 2


def test_structural_features_count_events_and_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "BRAF": ["V600E V512E", "WT"],
            "PTEN": [None, "R130*"],
        }
    )
    features = build_structural_features(frame, ["BRAF", "PTEN"])

    assert features.loc[0, "mutation_gene_count"] == 1
    assert features.loc[0, "event_count"] == 2
    assert features.loc[0, "multi_event_gene_count"] == 1
    assert features.loc[0, "type_SUB_count"] == 2
    assert features.loc[0, "missing_count"] == 1
    assert features.loc[1, "type_STOP_count"] == 1

