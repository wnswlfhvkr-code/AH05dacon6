from pathlib import Path


def test_xgboost_baseline_config_exists() -> None:
    root = Path(__file__).parents[1]
    assert (root / "configs" / "xgboost_baseline.yaml").is_file()
