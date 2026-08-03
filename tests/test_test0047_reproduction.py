from pathlib import Path

import yaml

from src.models.baseline.TEST_004_7 import SPEC
from src.pipelines.pipeline_jsj_v9 import JSJV9PreprocessingPipeline
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_test0047_config_spec_and_pipeline_are_connected():
    with (ROOT / "configs" / "test_004_7.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    pipeline = create_preprocessing_pipeline(config["preprocessing"])

    assert config["model"]["spec"] == SPEC["name"]
    assert isinstance(pipeline, JSJV9PreprocessingPipeline)
    assert config["model"]["team_pipelines"] == SPEC["team_pipeline_weights"]
    assert pipeline.team_experts["pipelines"] == SPEC["team_pipeline_weights"]


def test_test0047_uses_no_external_data():
    with (ROOT / "configs" / "test_004_7.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["record"]["external_data_used"] is False
    assert set(config["model"]["pair_experts"]) == {
        "temperature",
        "KIRC_KIPAN",
        "LGG_GBMLGG",
    }
