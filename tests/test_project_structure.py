from pathlib import Path

import pytest
import yaml

from src.pipelines.preprocessing_registry import PIPELINES


ROOT = Path(__file__).parents[1]
CONFIG_PATHS = sorted((ROOT / "configs").rglob("*.yaml"))


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


@pytest.mark.parametrize(
    "config_path",
    CONFIG_PATHS,
    ids=lambda path: str(path.relative_to(ROOT / "configs")),
)
def test_config_uses_registered_pipeline(config_path: Path) -> None:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    pipeline_name = config["preprocessing"]["name"]
    assert pipeline_name in PIPELINES
    assert (ROOT / "src" / "pipelines" / f"pipeline_{pipeline_name}.py").is_file()
