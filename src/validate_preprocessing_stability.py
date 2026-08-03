"""전처리 하나를 3-seed CV로 반복 검증해 안정성 결과를 저장합니다."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.models import MODEL_BUILDERS
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline


ARTIFACT_SCHEMA_VERSION = 1


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError("YAML 최상위 설정은 매핑이어야 합니다.")
    return loaded


def _require_mapping(config: Mapping[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"설정에 {key} 매핑이 필요합니다.")
    return dict(value)


def _normalize_seeds(values: object) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("stability_validation.seeds는 정수 시퀀스여야 합니다.")
    seeds = tuple(values)
    if not seeds:
        raise ValueError("stability_validation.seeds는 한 개 이상이어야 합니다.")
    if any(
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        for seed in seeds
    ):
        raise TypeError("각 seed는 정수여야 합니다.")
    normalized = tuple(int(seed) for seed in seeds)
    if len(set(normalized)) != len(normalized):
        raise ValueError("stability_validation.seeds에 중복값이 있습니다.")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_hash(splits: Sequence[tuple[np.ndarray, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for _, valid_indices in splits:
        digest.update(np.asarray(valid_indices, dtype="<i8").tobytes())
        digest.update(b"\xff")
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _implementation_identity(
    *,
    project_root: Path,
    preprocessing_config: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
    preprocessor = create_preprocessing_pipeline(dict(preprocessing_config))
    pipeline_source = inspect.getsourcefile(type(preprocessor))
    if pipeline_source is None:
        raise RuntimeError("전처리 파이프라인 소스 파일을 확인할 수 없습니다.")
    source_paths = {
        "stability_runner": Path(__file__).resolve(),
        "preprocessing_registry": (
            project_root / "src" / "pipelines" / "preprocessing_registry.py"
        ).resolve(),
        "preprocessing_base": (
            project_root / "src" / "pipelines" / "base.py"
        ).resolve(),
        "preprocessing_pipeline": Path(pipeline_source).resolve(),
        "model_registry": (project_root / "src" / "models" / "__init__.py").resolve(),
        "xgboost_model": (
            project_root / "src" / "models" / "xgboost_model.py"
        ).resolve(),
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"구현 소스 파일을 확인할 수 없습니다: {missing}")
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    runtime_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    return source_hashes, runtime_versions


def _contract_hash(
    config: Mapping[str, object],
    *,
    train_sha256: str,
    source_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
) -> str:
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "data": config["data"],
        "model": config["model"],
        "preprocessing": config["preprocessing"],
        "stability_validation": config["stability_validation"],
        "train_sha256": train_sha256,
        "source_hashes": source_hashes,
        "runtime_versions": runtime_versions,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _score_predictions(
    truth: Sequence[object],
    predictions: Sequence[object],
    *,
    class_labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    macro_f1 = float(
        f1_score(
            truth,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )
    )
    class_f1 = np.asarray(
        f1_score(
            truth,
            predictions,
            labels=class_labels,
            average=None,
            zero_division=0,
        ),
        dtype=np.float64,
    )
    return macro_f1, class_f1


def _validate_config(
    config: Mapping[str, object],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    tuple[int, ...],
    int,
]:
    model_config = _require_mapping(config, "model")
    preprocessing_config = _require_mapping(config, "preprocessing")
    validation_config = _require_mapping(config, "stability_validation")
    if model_config.get("name") != "xgboost":
        raise ValueError("현재 안정성 검증은 고정 기준 모델 xgboost만 지원합니다.")
    if not isinstance(preprocessing_config.get("name"), str):
        raise ValueError("preprocessing.name이 필요합니다.")

    seeds = _normalize_seeds(validation_config.get("seeds"))
    n_splits = validation_config.get("n_splits")
    if (
        isinstance(n_splits, (bool, np.bool_))
        or not isinstance(n_splits, (int, np.integer))
        or int(n_splits) < 2
    ):
        raise ValueError("stability_validation.n_splits는 2 이상의 정수여야 합니다.")
    if validation_config.get("metric", "macro_f1") != "macro_f1":
        raise ValueError("현재 안정성 검증 지표는 macro_f1만 지원합니다.")

    reference_results = validation_config.get("reference_results", {})
    if not isinstance(reference_results, Mapping):
        raise ValueError("stability_validation.reference_results는 매핑이어야 합니다.")
    for name, raw_reference in reference_results.items():
        if not isinstance(raw_reference, Mapping):
            raise ValueError(f"reference_results.{name}은 매핑이어야 합니다.")
        try:
            reference_mean = float(raw_reference["mean_seed_oof_macro_f1"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"reference_results.{name}.mean_seed_oof_macro_f1이 필요합니다."
            ) from error
        if not np.isfinite(reference_mean):
            raise ValueError(
                f"reference_results.{name}.mean_seed_oof_macro_f1은 유한해야 합니다."
            )
    stability_reference = validation_config.get("stability_reference")
    if stability_reference is not None:
        if not isinstance(stability_reference, Mapping):
            raise ValueError("stability_validation.stability_reference는 매핑이어야 합니다.")
        reference_name = stability_reference.get("name")
        if reference_name not in reference_results:
            raise ValueError("stability_reference.name에 해당하는 참고 결과가 없습니다.")
        selected_reference = reference_results[reference_name]
        for key in ("seed_oof_std", "fold_std"):
            try:
                value = float(selected_reference[key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"reference_results.{reference_name}.{key}가 필요합니다."
                ) from error
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"reference_results.{reference_name}.{key}는 유한한 0 이상 값이어야 합니다."
                )
        if float(stability_reference.get("max_seed_std_increase", -1.0)) < 0.0:
            raise ValueError("max_seed_std_increase는 0 이상이어야 합니다.")
        if float(stability_reference.get("max_fold_std_increase", -1.0)) < 0.0:
            raise ValueError("max_fold_std_increase는 0 이상이어야 합니다.")

    create_preprocessing_pipeline(dict(preprocessing_config))
    try:
        MODEL_BUILDERS["xgboost"](model_config, seeds[0])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("XGBoost 모델 설정이 올바르지 않습니다.") from error
    return (
        model_config,
        preprocessing_config,
        validation_config,
        seeds,
        int(n_splits),
    )


def _load_checkpoint(
    path: Path,
    *,
    expected_contract_hash: str,
    resume: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not resume or not path.exists():
        return [], []
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("contract_hash") != expected_contract_hash:
        raise RuntimeError(
            "기존 checkpoint의 데이터·설정·구현이 현재 검증과 다릅니다. "
            "--force로 현재 검증 산출물만 초기화한 뒤 다시 실행하세요."
        )
    fold_results = checkpoint.get("fold_results", [])
    oof_records = checkpoint.get("oof_records", [])
    if not isinstance(fold_results, list) or not isinstance(oof_records, list):
        raise RuntimeError("checkpoint 형식이 올바르지 않습니다.")
    fold_keys = [
        (int(row["seed"]), int(row["fold"])) for row in fold_results
    ]
    if len(fold_keys) != len(set(fold_keys)):
        raise RuntimeError("checkpoint에 중복된 seed/Fold 결과가 있습니다.")
    completed_folds = set(fold_keys)
    if any(
        (int(row["seed"]), int(row["fold"])) not in completed_folds
        for row in oof_records
    ):
        raise RuntimeError("checkpoint OOF에 완료되지 않은 Fold 기록이 있습니다.")
    return fold_results, oof_records


def _reference_comparisons(
    *,
    candidate_mean: float,
    reference_results: Mapping[str, object],
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    for name, raw_reference in reference_results.items():
        if not isinstance(raw_reference, Mapping):
            raise ValueError(f"reference_results.{name}은 매핑이어야 합니다.")
        reference = dict(raw_reference)
        reference_mean = float(reference["mean_seed_oof_macro_f1"])
        comparisons[str(name)] = {
            "reference_mean_seed_oof_macro_f1": reference_mean,
            "candidate_minus_reference": candidate_mean - reference_mean,
            "comparison_type": "stored_reference_not_same_run_paired",
        }
    return comparisons


def run_stability_validation(
    config: Mapping[str, object],
    *,
    project_root: Path,
    force: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
) -> dict[str, object]:
    if force and dry_run:
        raise ValueError("--force와 --dry-run은 함께 사용할 수 없습니다.")
    (
        model_config,
        preprocessing_config,
        validation_config,
        seeds,
        n_splits,
    ) = _validate_config(config)
    data_config = _require_mapping(config, "data")
    project_config = _require_mapping(config, "project")

    raw_dir = project_root / str(data_config["raw_dir"])
    processed_dir = project_root / str(data_config["processed_dir"])
    train_path = raw_dir / str(data_config["train_file"])
    target_column = str(data_config["target_column"])
    id_column = str(data_config["id_column"])
    output_stem = str(
        validation_config.get(
            "output_stem",
            project_config.get("experiment_name", "preprocessing_stability"),
        )
    )
    output_path = processed_dir / f"{output_stem}.json"
    fold_path = processed_dir / f"{output_stem}_folds.csv"
    oof_path = processed_dir / f"{output_stem}_oof.csv"
    checkpoint_path = processed_dir / f"{output_stem}_checkpoint.json"
    processed_dir.mkdir(parents=True, exist_ok=True)

    owned_outputs = (output_path, fold_path, oof_path, checkpoint_path)
    if force:
        for path in owned_outputs:
            if path.exists():
                path.unlink()

    if not train_path.exists():
        raise FileNotFoundError(f"학습 데이터가 없습니다: {train_path}")
    train_sha256 = _sha256_file(train_path)
    source_hashes, runtime_versions = _implementation_identity(
        project_root=project_root,
        preprocessing_config=preprocessing_config,
    )
    contract_hash = _contract_hash(
        config,
        train_sha256=train_sha256,
        source_hashes=source_hashes,
        runtime_versions=runtime_versions,
    )

    if output_path.exists():
        completed = json.loads(output_path.read_text(encoding="utf-8"))
        if completed.get("status") == "complete":
            if completed.get("contract_hash") != contract_hash:
                raise RuntimeError(
                    "기존 완료 결과가 현재 데이터·설정·구현과 다릅니다. "
                    "--force로 현재 검증 산출물만 초기화한 뒤 다시 실행하세요."
                )
            missing_artifacts = [
                str(path.relative_to(project_root))
                for path in (fold_path, oof_path)
                if not path.exists()
            ]
            if missing_artifacts:
                raise RuntimeError(
                    "완료 JSON에 대응하는 상세 산출물이 없습니다: "
                    f"{missing_artifacts}. --force로 다시 실행하세요."
                )
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            print(
                json.dumps(
                    {
                        "status": "already_complete",
                        "output": str(output_path.relative_to(project_root)),
                        "summary": completed.get("summary"),
                    },
                    ensure_ascii=False,
                )
            )
            return completed

    train = pd.read_csv(train_path)
    missing_columns = [
        column for column in (target_column, id_column) if column not in train.columns
    ]
    if missing_columns:
        raise ValueError(f"학습 데이터 필수 열이 없습니다: {missing_columns}")
    features = train.drop(columns=[target_column, id_column])
    labels = train[target_column]
    identifiers = train[id_column]
    if features.columns.has_duplicates:
        raise ValueError("학습 데이터에 중복 유전자 열 이름이 있습니다.")
    if labels.isna().any():
        raise ValueError("타겟에 결측값이 있습니다.")
    class_labels, class_counts = np.unique(labels.to_numpy(), return_counts=True)
    if int(class_counts.min()) < n_splits:
        raise ValueError("각 클래스 표본 수는 n_splits 이상이어야 합니다.")

    if dry_run:
        result = {
            "status": "dry_run_passed",
            "pipeline": preprocessing_config["name"],
            "model": model_config["name"],
            "seeds": list(seeds),
            "n_splits": n_splits,
            "training_count": len(seeds) * n_splits,
            "rows": len(features),
            "gene_features": features.shape[1],
            "classes": len(class_labels),
            "output": str(output_path.relative_to(project_root)),
        }
        print(json.dumps(result, ensure_ascii=False))
        return result

    resume = bool(validation_config.get("resume", True))
    show_progress = bool(validation_config.get("show_progress", True)) and not quiet
    fold_results, oof_records = _load_checkpoint(
        checkpoint_path,
        expected_contract_hash=contract_hash,
        resume=resume,
    )
    completed_folds = {
        (int(row["seed"]), int(row["fold"])) for row in fold_results
    }
    if show_progress and completed_folds:
        print(
            json.dumps({"resumed_folds": len(completed_folds)}, ensure_ascii=False),
            flush=True,
        )

    model_builder = MODEL_BUILDERS["xgboost"]
    split_hashes: dict[str, str] = {}
    started_all = time.perf_counter()
    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        splits = list(splitter.split(features, labels.to_numpy()))
        split_hashes[str(seed)] = _split_hash(splits)
        for fold_number, (train_indices, valid_indices) in enumerate(
            splits, start=1
        ):
            if (seed, fold_number) in completed_folds:
                continue
            fold_started = time.perf_counter()
            train_x = features.iloc[train_indices]
            valid_x = features.iloc[valid_indices]
            train_y = labels.iloc[train_indices]
            valid_y = labels.iloc[valid_indices]

            preprocessor = create_preprocessing_pipeline(
                dict(preprocessing_config)
            )
            encoded_train_x = preprocessor.fit_transform(train_x, train_y)
            encoded_valid_x = preprocessor.transform(valid_x)
            encoded_train_y = preprocessor.encode_labels(train_y)
            model = model_builder(model_config, seed)
            model.fit(encoded_train_x, encoded_train_y)
            predictions = preprocessor.decode_labels(
                model.predict(encoded_valid_x)
            )
            macro_f1, class_f1 = _score_predictions(
                valid_y.to_numpy(), predictions, class_labels=class_labels
            )
            elapsed = time.perf_counter() - fold_started
            fold_result = {
                "seed": seed,
                "fold": fold_number,
                "macro_f1": macro_f1,
                "feature_count": int(encoded_train_x.shape[1]),
                "elapsed_seconds": elapsed,
                "class_f1": {
                    str(label): float(score)
                    for label, score in zip(class_labels, class_f1, strict=True)
                },
            }
            fold_results.append(fold_result)
            for local_index, row_index in enumerate(valid_indices):
                oof_records.append(
                    {
                        "seed": seed,
                        "fold": fold_number,
                        "row_index": int(row_index),
                        "ID": str(identifiers.iloc[row_index]),
                        "y_true": str(valid_y.iloc[local_index]),
                        "y_pred": str(predictions[local_index]),
                    }
                )
            checkpoint = {
                "status": "running",
                "contract_hash": contract_hash,
                "fold_results": fold_results,
                "oof_records": oof_records,
            }
            _write_json_atomic(checkpoint_path, checkpoint)
            if show_progress:
                print(
                    json.dumps(
                        {
                            "completed": {
                                "seed": seed,
                                "fold": fold_number,
                                "macro_f1": macro_f1,
                                "feature_count": int(encoded_train_x.shape[1]),
                                "elapsed_seconds": round(elapsed, 2),
                            },
                            "progress": (
                                f"{len(fold_results)}/{len(seeds) * n_splits}"
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    fold_results = sorted(
        fold_results, key=lambda row: (int(row["seed"]), int(row["fold"]))
    )
    if len(fold_results) != len(seeds) * n_splits:
        raise RuntimeError("완료된 Fold 수가 설정과 일치하지 않습니다.")
    oof_frame = (
        pd.DataFrame(oof_records)
        .sort_values(["seed", "row_index"])
        .reset_index(drop=True)
    )
    expected_oof_rows = len(seeds) * len(features)
    if len(oof_frame) != expected_oof_rows:
        raise RuntimeError(
            f"OOF 행 수가 다릅니다: {len(oof_frame)} != {expected_oof_rows}"
        )
    if oof_frame.duplicated(["seed", "row_index"]).any():
        raise RuntimeError("OOF에 중복 seed/row_index가 있습니다.")

    seed_results: list[dict[str, object]] = []
    seed_class_scores: list[np.ndarray] = []
    for seed in seeds:
        seed_oof = oof_frame[oof_frame["seed"] == seed].sort_values(
            "row_index"
        )
        if seed_oof["row_index"].tolist() != list(range(len(features))):
            raise RuntimeError(f"seed {seed}의 OOF 범위가 완전하지 않습니다.")
        macro_f1, class_f1 = _score_predictions(
            seed_oof["y_true"],
            seed_oof["y_pred"],
            class_labels=class_labels,
        )
        seed_class_scores.append(class_f1)
        seed_fold_scores = np.asarray(
            [
                row["macro_f1"]
                for row in fold_results
                if int(row["seed"]) == seed
            ],
            dtype=np.float64,
        )
        seed_results.append(
            {
                "seed": seed,
                "oof_macro_f1": macro_f1,
                "fold_mean_macro_f1": float(seed_fold_scores.mean()),
                "fold_std_macro_f1": float(seed_fold_scores.std(ddof=0)),
                "fold_min_macro_f1": float(seed_fold_scores.min()),
                "fold_max_macro_f1": float(seed_fold_scores.max()),
                "class_f1": {
                    str(label): float(score)
                    for label, score in zip(class_labels, class_f1, strict=True)
                },
            }
        )

    seed_scores = np.asarray(
        [row["oof_macro_f1"] for row in seed_results], dtype=np.float64
    )
    fold_scores = np.asarray(
        [row["macro_f1"] for row in fold_results], dtype=np.float64
    )
    mean_class_f1 = np.mean(np.vstack(seed_class_scores), axis=0)
    worst_class_index = int(np.argmin(mean_class_f1))
    candidate_mean = float(seed_scores.mean())
    candidate_seed_std = float(seed_scores.std(ddof=0))
    candidate_fold_std = float(fold_scores.std(ddof=0))
    reference_results = dict(validation_config.get("reference_results", {}))
    comparisons = _reference_comparisons(
        candidate_mean=candidate_mean,
        reference_results=reference_results,
    )

    raw_stability_config = validation_config.get("stability_reference")
    if raw_stability_config is None:
        stability_config: dict[str, object] = {}
        reference_name = None
        score_delta = None
        seed_std_increase = None
        fold_std_increase = None
        seed_std_check = None
        fold_std_check = None
        stability_comparable = None
        promotion_passed = None
        decision = "standalone_validation"
    else:
        stability_config = dict(raw_stability_config)
        reference_name = str(stability_config["name"])
        stability_reference = dict(reference_results[reference_name])
        score_delta = candidate_mean - float(
            stability_reference["mean_seed_oof_macro_f1"]
        )
        seed_std_increase = candidate_seed_std - float(
            stability_reference["seed_oof_std"]
        )
        fold_std_increase = candidate_fold_std - float(
            stability_reference["fold_std"]
        )
        seed_std_check = seed_std_increase <= float(
            stability_config["max_seed_std_increase"]
        )
        fold_std_check = fold_std_increase <= float(
            stability_config["max_fold_std_increase"]
        )
        stability_comparable = bool(seed_std_check and fold_std_check)
        promotion_passed = bool(score_delta > 0.0 and stability_comparable)
        decision = "promote_candidate" if promotion_passed else "keep_reference"

    summary = {
        "mean_seed_oof_macro_f1": candidate_mean,
        "seed_oof_std": candidate_seed_std,
        "fold_mean_macro_f1": float(fold_scores.mean()),
        "fold_std_macro_f1": candidate_fold_std,
        "fold_min_macro_f1": float(fold_scores.min()),
        "fold_max_macro_f1": float(fold_scores.max()),
        "worst_mean_class": str(class_labels[worst_class_index]),
        "worst_mean_class_f1": float(mean_class_f1[worst_class_index]),
        "stability_reference": reference_name,
        "score_delta_vs_stability_reference": score_delta,
        "seed_std_increase_vs_reference": seed_std_increase,
        "fold_std_increase_vs_reference": fold_std_increase,
        "seed_std_increase_within_limit": seed_std_check,
        "fold_std_increase_within_limit": fold_std_check,
        "stability_comparable": stability_comparable,
        "promotion_passed": promotion_passed,
        "decision": decision,
        "elapsed_seconds": float(time.perf_counter() - started_all),
    }
    result = {
        "status": "complete",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "contract_hash": contract_hash,
        "train_sha256": train_sha256,
        "source_hashes": source_hashes,
        "runtime_versions": runtime_versions,
        "pipeline_name": preprocessing_config["name"],
        "seeds": list(seeds),
        "n_splits": n_splits,
        "training_count": len(fold_results),
        "splitter": "StratifiedKFold(shuffle=True, random_state=seed)",
        "split_hashes": split_hashes,
        "model": model_config,
        "preprocessing": preprocessing_config,
        "holdout_reference": validation_config.get("holdout_reference", {}),
        "reference_results": reference_results,
        "reference_comparisons": comparisons,
        "stability_rule": stability_config,
        "summary": summary,
        "seed_results": seed_results,
        "fold_results": fold_results,
        "artifacts": {
            "result_json": str(output_path.relative_to(project_root)),
            "fold_csv": str(fold_path.relative_to(project_root)),
            "oof_csv": str(oof_path.relative_to(project_root)),
        },
    }
    pd.DataFrame(fold_results).drop(columns=["class_f1"]).to_csv(
        fold_path, index=False, encoding="utf-8-sig"
    )
    oof_frame.to_csv(oof_path, index=False, encoding="utf-8-sig")
    _write_json_atomic(output_path, result)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_path.relative_to(project_root)),
                "summary": summary,
            },
            ensure_ascii=False,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="전처리 하나를 3-seed CV로 반복 검증해 안정성을 저장합니다."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    config_path = (
        args.config
        if args.config.is_absolute()
        else project_root / args.config
    )
    config = _load_yaml(config_path)
    run_stability_validation(
        config,
        project_root=project_root,
        force=args.force,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
