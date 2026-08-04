import ast
import importlib.util
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

    if config_path.parent.name == "ensembles":
        experiment_name = config["project"]["experiment_name"]
        suffix = experiment_name.removeprefix("test_002_")
        assert (ROOT / "src" / "ensembles" / f"train_jh_{suffix}.py").is_file()
        return

    pipeline_name = config["preprocessing"]["name"]
    assert pipeline_name in PIPELINES
    pipeline_module = PIPELINES[pipeline_name].__module__
    module_spec = importlib.util.find_spec(pipeline_module)
    assert module_spec is not None
    assert module_spec.origin is not None
    assert Path(module_spec.origin).is_file()


def test_test_005_config_matches_selected_em_v19_e4_condition() -> None:
    expected_model = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5.0,
        "subsample": 0.75,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.5,
        "reg_lambda": 10.0,
        "early_stopping_rounds": 30,
    }
    with (ROOT / "configs" / "test_005.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["preprocessing"]["name"] == "em_v19"
    assert {
        key: config["model"][key]
        for key in expected_model
    } == expected_model
def test_test_004_config_uses_jsj_v2_pipeline() -> None:
    config_path = ROOT / "configs" / "test_004.yaml"

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert (ROOT / "src" / "pipelines" / "pipeline_jsj_v2.py").is_file()
    assert config["project"]["experiment_name"] == "test_004"
    assert config["preprocessing"]["name"] == "jsj_v2"


def test_test_003_config_uses_registered_jyp_pipeline() -> None:
    config_path = ROOT / "configs" / "test_003.yaml"

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["model"]["name"] == "xgboost"
    pipeline_name = config["preprocessing"]["name"]
    assert pipeline_name.startswith("jyp_f")
    assert PIPELINES[pipeline_name].__module__.endswith(
        f"pipeline_{pipeline_name}"
    )


def test_jyp_pipelines_are_self_contained_in_their_dedicated_package() -> None:
    source = ROOT / "src"
    pipelines = source / "pipelines"
    jyp = pipelines / "jyp_preprocessing"
    selectable_files = sorted(path.name for path in jyp.glob("pipeline_jyp_*.py"))

    assert selectable_files == [
        "pipeline_jyp_f0.py",
        "pipeline_jyp_f0_no_raw.py",
        "pipeline_jyp_f1.py",
        "pipeline_jyp_f10.py",
        "pipeline_jyp_f11.py",
        "pipeline_jyp_f2.py",
        "pipeline_jyp_f3.py",
        "pipeline_jyp_f3_no_raw.py",
        "pipeline_jyp_f3_position.py",
        "pipeline_jyp_f4.py",
        "pipeline_jyp_f4_no_raw.py",
        "pipeline_jyp_f5.py",
        "pipeline_jyp_f5_no_raw.py",
        "pipeline_jyp_f5_no_raw_missmask.py",
        "pipeline_jyp_f5_selective_no_raw.py",
        "pipeline_jyp_f6.py",
        "pipeline_jyp_f7.py",
        "pipeline_jyp_f8.py",
        "pipeline_jyp_f9.py",
        "pipeline_jyp_raw.py",
    ]
    assert not list(pipelines.glob("pipeline_jyp_*.py"))
    assert not (jyp / "validation").exists()
    assert not (jyp / "pipeline_base.py").exists()
    assert not (jyp / "mutation_parser.py").exists()
    assert not list(jyp.glob("feature_f*.py"))


def test_jyp_pipeline_files_do_not_import_each_other() -> None:
    jyp = ROOT / "src" / "pipelines" / "jyp_preprocessing"

    for path in jyp.glob("pipeline_jyp_*.py"):
        syntax_tree = ast.parse(path.read_text(encoding="utf-8"))
        cross_imports = [
            node.module
            for node in ast.walk(syntax_tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("src.pipelines.jyp_preprocessing.pipeline_jyp_")
        ]
        assert not cross_imports, f"{path.name} has JYP cross-imports: {cross_imports}"


def test_shared_src_has_no_jyp_experiment_entrypoints() -> None:
    source = ROOT / "src"
    excluded = [
        source / "analyze_burden_shift.py",
        source / "export_preprocessing_optimizer_results.py",
        source / "optimize_preprocessing.py",
        source / "validate_preprocessing_all.py",
        source / "pipelines" / "pipeline_jyp_v1.py",
    ]

    assert not [path for path in excluded if path.exists()]


def test_shared_trainer_uses_fit_transform_for_validation_training_data() -> None:
    train_path = ROOT / "src" / "train.py"
    syntax_tree = ast.parse(train_path.read_text(encoding="utf-8"))
    train_x_assignments = [
        node.value
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "train_x"
            for target in node.targets
        )
    ]

    assert len(train_x_assignments) == 1
    call = train_x_assignments[0]
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Attribute)
    assert isinstance(call.func.value, ast.Name)
    assert call.func.value.id == "validation_preprocessor"
    assert call.func.attr == "fit_transform"
    assert [
        argument.id if isinstance(argument, ast.Name) else None
        for argument in call.args
    ] == ["train_features", "train_labels"]
