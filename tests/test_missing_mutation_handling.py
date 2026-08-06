import pandas as pd

from src.pipelines.em_preprocessing.em_feature_engine import (
    _pattern_keys,
    _presence_matrix,
    _severity_matrix,
)
from src.pipelines.pipeline_em_v16 import (
    create_consequence_severity_matrix,
    create_mutation_presence_matrix,
    create_raw_mutation_pattern_keys,
)


def test_missing_mutation_is_explicitly_encoded_as_zero() -> None:
    features = pd.DataFrame({"TP53": [None, "WT", "R175H"]})

    v16_presence = create_mutation_presence_matrix(features)
    engine_presence = _presence_matrix(features, ["TP53"])
    v16_severity = create_consequence_severity_matrix(features, ["TP53"])
    engine_severity = _severity_matrix(features, ["TP53"])

    assert v16_presence["TP53"].tolist() == [0, 0, 1]
    assert engine_presence["TP53"].tolist() == [0, 0, 1]
    assert v16_severity["TP53"].tolist() == [0, 0, 2]
    assert engine_severity["TP53"].tolist() == [0, 0, 2]


def test_missing_and_wt_have_the_same_pattern_key() -> None:
    features = pd.DataFrame({"TP53": [None, "WT", "R175H"]})

    v16_keys = create_raw_mutation_pattern_keys(features)
    engine_keys = _pattern_keys(features)

    assert v16_keys.iloc[0] == v16_keys.iloc[1]
    assert engine_keys.iloc[0] == engine_keys.iloc[1]
    assert v16_keys.iloc[0] != v16_keys.iloc[2]
    assert engine_keys.iloc[0] != engine_keys.iloc[2]
