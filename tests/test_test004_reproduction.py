from importlib import import_module
from pathlib import Path

import yaml

from src.pipelines.preprocessing_registry import PIPELINES
from src.pipelines.pipeline_jsj_v1 import _mutation_type


EXPECTED = {
    1: ("TEST_004_1", "jsj_v3", 0.3714981583),
    2: ("TEST_004_2", "jsj_v4", 0.3741003172),
    3: ("TEST_004_3", "jsj_v5", 0.3756133965),
    4: ("TEST_004_4", "jsj_v6", 0.3768261636),
    5: ("TEST_004_5", "jsj_v7", 0.3863356794),
    6: ("TEST_004_6", "jsj_v8", 0.3893423841),
}


def test_test004_configs_specs_and_pipelines_are_connected() -> None:
    root = Path(__file__).parents[1]
    for number, (spec_name, pipeline_name, public_score) in EXPECTED.items():
        with (root / "configs" / f"test_004_{number}.yaml").open(
            encoding="utf-8"
        ) as file:
            config = yaml.safe_load(file)
        spec = import_module(f"src.models.baseline.{spec_name}").SPEC
        assert config["project"]["experiment_name"] == spec_name.lower()
        assert config["model"]["spec"] == spec_name
        assert config["preprocessing"]["name"] == pipeline_name
        assert pipeline_name in PIPELINES
        assert config["record"]["historical_public_macro_f1"] == public_score
        assert spec["historical_public_macro_f1"] == public_score


def test_pipeline_strategy_matches_model_spec() -> None:
    for number, (spec_name, pipeline_name, _) in EXPECTED.items():
        pipeline = PIPELINES[pipeline_name]()
        spec = import_module(f"src.models.baseline.{spec_name}").SPEC
        if number == 1:
            assert spec["postprocessing"]["mode"] == "none"
            continue
        assert pipeline.conflict_mode == spec["postprocessing"]["mode"]
        assert pipeline.conflict_weights == {
            "KIRC_KIPAN": spec["postprocessing"]["KIRC_KIPAN"],
            "LGG_GBMLGG": spec["postprocessing"]["LGG_GBMLGG"],
        }
        if number in {5, 6}:
            assert pipeline.conflict_specialist_c == spec["postprocessing"][
                "specialist_c"
            ]
            assert pipeline.conflict_right_offset == spec["postprocessing"][
                "right_offset"
            ]
        if number == 6:
            assert pipeline.auxiliary_specialist == spec["postprocessing"][
                "auxiliary"
            ]


def test_historical_text_token_classification_is_preserved() -> None:
    """제출 당시 TF-IDF 문서와 달라지는 회귀를 막습니다."""
    assert _mutation_type("D623D") == "OTHER"
    assert _mutation_type("R123X") == "OTHER"
    assert _mutation_type("R123*") == "STOP"
    assert _mutation_type("c.123A>G") == "SUB"
