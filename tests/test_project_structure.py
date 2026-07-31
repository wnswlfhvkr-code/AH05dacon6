from pathlib import Path

import yaml


def test_baseline_config_matches_notebook_parameters() -> None:
    root = Path(__file__).parents[1]
    config_path = root / "configs" / "baseline.yaml"
    assert config_path.is_file()

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["project"]["experiment_name"] == "baseline"
    assert config["project"]["seed"] == 42
    assert config["model"]["n_estimators"] == 100
    assert config["model"]["learning_rate"] == 0.1
    assert config["model"]["max_depth"] == 6
    assert config["model"]["eval_metric"] == "mlogloss"
    assert config["preprocessing"]["name"] == "baseline"


def test_baseline_pipeline_has_descriptive_filename() -> None:
    root = Path(__file__).parents[1]

    assert (root / "src" / "pipelines" / "pipeline_baseline.py").is_file()
    assert not (root / "src" / "pipelines" / "v1.py").exists()


def test_test_001_config_uses_em_v1_pipeline() -> None:
    root = Path(__file__).parents[1]
    config_path = root / "configs" / "test_001.yaml"

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert (root / "src" / "pipelines" / "pipeline_em_v1.py").is_file()
    assert config["preprocessing"]["name"] == "em_v1"


def test_test_004_config_uses_jsj_v2_pipeline() -> None:
    root = Path(__file__).parents[1]
    config_path = root / "configs" / "test_004.yaml"

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert (root / "src" / "pipelines" / "pipeline_jsj_v2.py").is_file()
    assert config["project"]["experiment_name"] == "test_004"
    assert config["preprocessing"]["name"] == "jsj_v2"
