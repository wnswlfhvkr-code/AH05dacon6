"""전처리 안정성, XGBoost 확률 결합 및 내부 피처 결합을 검증합니다."""

from __future__ import annotations

import argparse
import gc
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
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.models import MODEL_BUILDERS
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline


ARTIFACT_SCHEMA_VERSION = 2
XGB_BLEND_ARTIFACT_SCHEMA_VERSION = 2
XGB_INTERNAL_FUSION_ARTIFACT_SCHEMA_VERSION = 3
DEFAULT_OVERFIT_THRESHOLD = 0.1

EM24_INTERNAL_SUMMARY_COLUMNS = (
    "mutated_gene_count_log1p",
    "mutation_token_count_log1p",
    "functional_mutation_count_log1p",
    "functional_gene_count_log1p",
    "multi_variant_gene_count_log1p",
    "functional_mutation_ratio",
    "consequence_synonymous_count_log1p",
    "consequence_synonymous_ratio",
    "consequence_missense_count_log1p",
    "consequence_missense_ratio",
    "consequence_inframe_count_log1p",
    "consequence_inframe_ratio",
    "consequence_nonsense_count_log1p",
    "consequence_nonsense_ratio",
    "consequence_frameshift_count_log1p",
    "consequence_frameshift_ratio",
)
EM24_INTERNAL_BLOCKS = (
    "summary",
    "signature_all",
    "signature_functional",
)
EM16_INTERNAL_SUMMARY_COLUMNS = (
    "mutation_burden_log1p",
    "consequence_count_synonymous",
    "consequence_ratio_synonymous",
    "consequence_count_missense",
    "consequence_ratio_missense",
    "consequence_count_inframe_complex",
    "consequence_ratio_inframe_complex",
    "consequence_count_truncating",
    "consequence_ratio_truncating",
)
INTERNAL_COMPANION_BLOCKS = {
    "em_v16": ("severity", "summary", "signature", "hotspot"),
    "em_v24": EM24_INTERNAL_BLOCKS,
}
INTERNAL_COMPANION_PREFIXES = {"em_v16": "EM16", "em_v24": "EM24"}


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


def _resolve_overfit_threshold(
    validation_config: Mapping[str, object],
    *,
    section_name: str,
) -> float:
    """Train-Validation Macro F1 차이의 과적합 판정 임계값을 검증합니다."""
    raw_threshold = validation_config.get(
        "overfit_threshold", DEFAULT_OVERFIT_THRESHOLD
    )
    if isinstance(raw_threshold, (bool, np.bool_)) or not isinstance(
        raw_threshold, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{section_name}.overfit_threshold는 0 이상의 숫자여야 합니다.")
    threshold = float(raw_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"{section_name}.overfit_threshold는 유한한 0 이상 값이어야 합니다.")
    return threshold


def _fold_overfit_metrics(
    train_macro_f1: float,
    validation_macro_f1: float,
    *,
    threshold: float,
) -> dict[str, object]:
    """한 Fold의 Train-Validation 차이와 과적합 판정을 반환합니다."""
    train_score = float(train_macro_f1)
    validation_score = float(validation_macro_f1)
    if not np.isfinite(train_score) or not np.isfinite(validation_score):
        raise ValueError("과적합 계산에 사용하는 Macro F1은 유한해야 합니다.")
    gap = train_score - validation_score
    return {
        "train_macro_f1": train_score,
        "validation_macro_f1": validation_score,
        "train_validation_gap": gap,
        "overfit_threshold": threshold,
        "is_overfitting": bool(gap > threshold),
    }


def _summarize_overfit_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    threshold: float,
) -> dict[str, object]:
    """여러 Fold의 과적합 지표를 동일한 결과 스키마로 집계합니다."""
    if not rows:
        raise ValueError("과적합을 집계할 Fold 결과가 없습니다.")
    train_scores = np.asarray(
        [float(row["train_macro_f1"]) for row in rows], dtype=np.float64
    )
    validation_scores = np.asarray(
        [
            float(
                row["validation_macro_f1"]
                if "validation_macro_f1" in row
                else row["macro_f1"]
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.isfinite(train_scores).all() or not np.isfinite(validation_scores).all():
        raise ValueError("과적합 집계에 사용하는 Macro F1은 유한해야 합니다.")
    gaps = train_scores - validation_scores
    mean_gap = float(gaps.mean())
    return {
        "mean_train_macro_f1": float(train_scores.mean()),
        "mean_validation_macro_f1": float(validation_scores.mean()),
        "mean_train_validation_gap": mean_gap,
        "min_train_validation_gap": float(gaps.min()),
        "max_train_validation_gap": float(gaps.max()),
        "overfit_threshold": threshold,
        "overfit_fold_count": int((gaps > threshold).sum()),
        "total_fold_count": int(len(rows)),
        "is_overfitting": bool(mean_gap > threshold),
        "overfit_rule": "mean_train_validation_gap > overfit_threshold",
    }


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
    _resolve_overfit_threshold(
        validation_config, section_name="stability_validation"
    )

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
    overfit_threshold = _resolve_overfit_threshold(
        validation_config, section_name="stability_validation"
    )
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
            "overfit_threshold": overfit_threshold,
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
            train_predictions = preprocessor.decode_labels(
                model.predict(encoded_train_x)
            )
            predictions = preprocessor.decode_labels(
                model.predict(encoded_valid_x)
            )
            train_macro_f1, _ = _score_predictions(
                train_y.to_numpy(), train_predictions, class_labels=class_labels
            )
            macro_f1, class_f1 = _score_predictions(
                valid_y.to_numpy(), predictions, class_labels=class_labels
            )
            overfit_metrics = _fold_overfit_metrics(
                train_macro_f1,
                macro_f1,
                threshold=overfit_threshold,
            )
            elapsed = time.perf_counter() - fold_started
            fold_result = {
                "seed": seed,
                "fold": fold_number,
                "macro_f1": macro_f1,
                **overfit_metrics,
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
                                "train_macro_f1": train_macro_f1,
                                "train_validation_gap": overfit_metrics[
                                    "train_validation_gap"
                                ],
                                "is_overfitting": overfit_metrics["is_overfitting"],
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
        seed_fold_results = [
            row for row in fold_results if int(row["seed"]) == seed
        ]
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
                **_summarize_overfit_rows(
                    seed_fold_results,
                    threshold=overfit_threshold,
                ),
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
        **_summarize_overfit_rows(
            fold_results,
            threshold=overfit_threshold,
        ),
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


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """NPZ를 같은 디렉터리에서 원자적으로 교체합니다."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def _artifact_token(value: str) -> str:
    token = "".join(character if character.isalnum() else "_" for character in value)
    token = token.strip("_")
    if not token:
        raise ValueError("산출물 이름으로 사용할 수 없는 빈 이름입니다.")
    return token


def _safe_output_stem(value: object) -> str:
    """산출물이 processed_dir 밖으로 벗어나지 않는 단일 파일명을 검증합니다."""
    stem = str(value)
    if (
        not stem
        or stem != stem.strip()
        or stem in {".", ".."}
        or Path(stem).is_absolute()
        or "/" in stem
        or "\\" in stem
        or any(not (character.isalnum() or character in {"_", "-"}) for character in stem)
    ):
        raise ValueError(
            "output_stem/experiment_name은 문자·숫자·밑줄·하이픈만 포함한 "
            "단일 파일명이어야 합니다."
        )
    return stem


def _validate_probability_matrix(
    probabilities: np.ndarray,
    *,
    expected_rows: int,
    expected_columns: int,
    context: str,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (expected_rows, expected_columns):
        raise RuntimeError(
            f"{context} 확률 행렬 크기가 다릅니다: {values.shape} != "
            f"({expected_rows}, {expected_columns})"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(f"{context} 확률 행렬에 NaN 또는 무한대가 있습니다.")
    if float(values.min(initial=0.0)) < -1e-7:
        raise RuntimeError(f"{context} 확률 행렬에 음수가 있습니다.")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"{context} 클래스 확률 합이 1이 아닙니다.")
    return values


def _align_probability_columns(
    probabilities: np.ndarray,
    *,
    model: object,
    preprocessor: object,
    global_class_labels: np.ndarray,
    context: str,
) -> np.ndarray:
    """Fold 내부 정수 클래스 순서를 전체 원본 SUBCLASS 순서로 정렬합니다."""
    model_classes = np.asarray(getattr(model, "classes_", []))
    values = _validate_probability_matrix(
        probabilities,
        expected_rows=len(probabilities),
        expected_columns=len(model_classes),
        context=context,
    )
    decoded_classes = np.asarray(
        preprocessor.decode_labels(model_classes.astype(int)), dtype=str
    )
    if len(decoded_classes) != len(set(decoded_classes.tolist())):
        raise RuntimeError(f"{context} 모델 클래스에 중복 원본 라벨이 있습니다.")
    expected = np.asarray(global_class_labels, dtype=str)
    if set(decoded_classes.tolist()) != set(expected.tolist()):
        missing = sorted(set(expected.tolist()) - set(decoded_classes.tolist()))
        unexpected = sorted(set(decoded_classes.tolist()) - set(expected.tolist()))
        raise RuntimeError(
            f"{context} 원본 SUBCLASS 집합이 다릅니다. "
            f"누락={missing}, 예상 외={unexpected}"
        )
    source_index = {label: index for index, label in enumerate(decoded_classes)}
    aligned = values[:, [source_index[label] for label in expected]]
    return _validate_probability_matrix(
        aligned,
        expected_rows=len(aligned),
        expected_columns=len(expected),
        context=f"{context} 정렬 후",
    ).astype(np.float32)


def _normalize_enabled_names(
    values: object,
    *,
    available: Mapping[str, object],
) -> tuple[str, ...]:
    if values is None:
        names = tuple(str(name) for name in available)
    else:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError("xgb_blend_validation.enabled_candidates는 이름 시퀀스여야 합니다.")
        names = tuple(str(name) for name in values)
    if not names:
        raise ValueError("enabled_candidates는 한 개 이상이어야 합니다.")
    if len(set(names)) != len(names):
        raise ValueError("enabled_candidates에 중복 이름이 있습니다.")
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"enabled_candidates에 정의되지 않은 후보가 있습니다: {unknown}")
    return names


def _validate_xgb_blend_config(
    config: Mapping[str, object],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, dict[str, float]],
    tuple[str, ...],
    tuple[str, ...],
    str,
    tuple[int, ...],
    int,
]:
    model_config = _require_mapping(config, "model")
    validation_config = _require_mapping(config, "xgb_blend_validation")
    if model_config.get("name") != "xgboost":
        raise ValueError("XGBoost 확률 결합은 model.name=xgboost만 지원합니다.")
    seeds = _normalize_seeds(validation_config.get("seeds"))
    raw_n_splits = validation_config.get("n_splits")
    if (
        isinstance(raw_n_splits, (bool, np.bool_))
        or not isinstance(raw_n_splits, (int, np.integer))
        or int(raw_n_splits) < 2
    ):
        raise ValueError("xgb_blend_validation.n_splits는 2 이상의 정수여야 합니다.")
    if validation_config.get("metric", "macro_f1") != "macro_f1":
        raise ValueError("XGBoost 확률 결합 지표는 macro_f1만 지원합니다.")
    _resolve_overfit_threshold(
        validation_config, section_name="xgb_blend_validation"
    )

    raw_members = validation_config.get("members")
    if not isinstance(raw_members, Mapping) or not raw_members:
        raise ValueError("xgb_blend_validation.members 매핑이 필요합니다.")
    members: dict[str, dict[str, object]] = {}
    member_tokens: set[str] = set()
    for raw_alias, raw_pipeline_config in raw_members.items():
        alias = str(raw_alias)
        if not isinstance(raw_pipeline_config, Mapping):
            raise ValueError(f"members.{alias}는 전처리 설정 매핑이어야 합니다.")
        pipeline_config = dict(raw_pipeline_config)
        if not isinstance(pipeline_config.get("name"), str):
            raise ValueError(f"members.{alias}.name이 필요합니다.")
        token = _artifact_token(alias)
        if token in member_tokens:
            raise ValueError("members 이름을 산출물 이름으로 변환했을 때 충돌합니다.")
        member_tokens.add(token)
        create_preprocessing_pipeline(pipeline_config)
        members[alias] = pipeline_config

    raw_candidates = validation_config.get("candidates")
    if not isinstance(raw_candidates, Mapping) or not raw_candidates:
        raise ValueError("xgb_blend_validation.candidates 매핑이 필요합니다.")
    candidates: dict[str, dict[str, float]] = {}
    candidate_tokens: set[str] = set()
    for raw_name, raw_candidate in raw_candidates.items():
        name = str(raw_name)
        if not isinstance(raw_candidate, Mapping):
            raise ValueError(f"candidates.{name}은 매핑이어야 합니다.")
        raw_weights = raw_candidate.get("weights")
        if not isinstance(raw_weights, Mapping) or len(raw_weights) < 2:
            raise ValueError(f"candidates.{name}.weights에는 두 개 이상 멤버가 필요합니다.")
        weights: dict[str, float] = {}
        for raw_alias, raw_weight in raw_weights.items():
            alias = str(raw_alias)
            if alias not in members:
                raise ValueError(f"candidates.{name}에 정의되지 않은 멤버 {alias}가 있습니다.")
            if isinstance(raw_weight, (bool, np.bool_)):
                raise ValueError(f"candidates.{name}.{alias} 가중치는 양수여야 합니다.")
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"candidates.{name}.{alias} 가중치는 숫자여야 합니다."
                ) from error
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"candidates.{name}.{alias} 가중치는 유한한 양수여야 합니다.")
            weights[alias] = weight
        if not np.isclose(sum(weights.values()), 1.0, rtol=0.0, atol=1e-9):
            raise ValueError(f"candidates.{name} 가중치 합은 정확히 1이어야 합니다.")
        token = _artifact_token(name)
        if token in candidate_tokens:
            raise ValueError("candidate 이름을 산출물 이름으로 변환했을 때 충돌합니다.")
        candidate_tokens.add(token)
        candidates[name] = weights

    enabled_candidates = _normalize_enabled_names(
        validation_config.get("enabled_candidates"), available=candidates
    )
    baseline_member = str(validation_config.get("baseline_member", "jyp_f9"))
    if baseline_member not in members:
        raise ValueError("baseline_member가 members에 없습니다.")
    for candidate_name in enabled_candidates:
        if baseline_member not in candidates[candidate_name]:
            raise ValueError(
                f"활성 후보 {candidate_name}에 baseline_member={baseline_member}가 없습니다."
            )
    required_members = tuple(
        alias
        for alias in members
        if alias == baseline_member
        or any(alias in candidates[name] for name in enabled_candidates)
    )
    if validation_config.get("submission_policy", "screening_only") not in {
        "screening_only",
        "write_local_candidates",
    }:
        raise ValueError(
            "submission_policy는 screening_only 또는 write_local_candidates여야 합니다."
        )
    try:
        MODEL_BUILDERS["xgboost"](model_config, seeds[0])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("XGBoost 모델 설정이 올바르지 않습니다.") from error
    return (
        model_config,
        validation_config,
        members,
        candidates,
        enabled_candidates,
        required_members,
        baseline_member,
        seeds,
        int(raw_n_splits),
    )


def _blend_implementation_identity(
    *,
    project_root: Path,
    members: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, str], dict[str, str]]:
    source_paths = {
        "stability_runner": Path(__file__).resolve(),
        "preprocessing_registry": (
            project_root / "src" / "pipelines" / "preprocessing_registry.py"
        ).resolve(),
        "preprocessing_base": (
            project_root / "src" / "pipelines" / "base.py"
        ).resolve(),
        "model_registry": (project_root / "src" / "models" / "__init__.py").resolve(),
        "xgboost_model": (
            project_root / "src" / "models" / "xgboost_model.py"
        ).resolve(),
    }
    for alias, pipeline_config in members.items():
        pipeline = create_preprocessing_pipeline(dict(pipeline_config))
        source = inspect.getsourcefile(type(pipeline))
        if source is None:
            raise RuntimeError(f"{alias} 전처리 파이프라인 소스를 확인할 수 없습니다.")
        source_paths[f"pipeline__{_artifact_token(alias)}"] = Path(source).resolve()
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"구현 소스 파일을 확인할 수 없습니다: {missing}")
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    runtime_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    return source_hashes, runtime_versions


def _xgb_blend_contract_hash(
    config: Mapping[str, object],
    *,
    data_hashes: Mapping[str, str],
    source_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
) -> str:
    validation_config = {
        key: value
        for key, value in dict(config["xgb_blend_validation"]).items()
        if key not in {"resume", "show_progress"}
    }
    payload = {
        "artifact_schema_version": XGB_BLEND_ARTIFACT_SCHEMA_VERSION,
        "project": config["project"],
        "data": config["data"],
        "model": config["model"],
        "xgb_blend_validation": validation_config,
        "data_hashes": data_hashes,
        "source_hashes": source_hashes,
        "runtime_versions": runtime_versions,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _blend_probabilities(
    weights: Mapping[str, float],
    probabilities: Mapping[str, np.ndarray],
    *,
    context: str,
) -> np.ndarray:
    combined: np.ndarray | None = None
    for alias, weight in weights.items():
        values = np.asarray(probabilities[alias], dtype=np.float64)
        combined = values * weight if combined is None else combined + values * weight
    if combined is None:
        raise RuntimeError(f"{context}에 결합할 확률이 없습니다.")
    return _validate_probability_matrix(
        combined,
        expected_rows=combined.shape[0],
        expected_columns=combined.shape[1],
        context=context,
    ).astype(np.float32)


def _score_probability(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_labels: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    predictions = class_labels[np.asarray(probabilities).argmax(axis=1)]
    macro_f1, class_f1 = _score_predictions(
        truth, predictions, class_labels=class_labels
    )
    return macro_f1, class_f1, predictions


def _normalize_internal_blocks(
    values: object,
    *,
    candidate: str,
    companion_name: str = "em_v24",
) -> tuple[str, ...]:
    available = INTERNAL_COMPANION_BLOCKS.get(companion_name)
    if available is None:
        raise ValueError(f"지원하지 않는 내부 결합 전처리입니다: {companion_name}")
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"candidates.{candidate}.companion_blocks는 이름 시퀀스여야 합니다.")
    blocks = tuple(str(value) for value in values)
    if len(set(blocks)) != len(blocks):
        raise ValueError(f"candidates.{candidate}.companion_blocks에 중복값이 있습니다.")
    unknown = sorted(set(blocks) - set(available))
    if unknown:
        raise ValueError(
            f"candidates.{candidate}.companion_blocks에 지원하지 않는 블록이 있습니다: "
            f"{unknown}"
        )
    return tuple(block for block in available if block in blocks)


def _select_internal_companion_columns(
    frame: pd.DataFrame,
    blocks: Sequence[str],
    *,
    companion_name: str,
) -> list[str]:
    normalized = _normalize_internal_blocks(
        blocks, candidate="runtime", companion_name=companion_name
    )
    columns = frame.columns.astype(str).tolist()
    selected: list[str] = []
    if companion_name == "em_v24":
        if "summary" in normalized:
            missing = [
                name for name in EM24_INTERNAL_SUMMARY_COLUMNS if name not in columns
            ]
            if missing:
                raise RuntimeError(f"EM24 summary 열이 누락되었습니다: {missing}")
            selected.extend(EM24_INTERNAL_SUMMARY_COLUMNS)
        if "signature_all" in normalized:
            selected.extend(
                name for name in columns if name.startswith("signature_all_")
            )
        if "signature_functional" in normalized:
            selected.extend(
                name for name in columns if name.startswith("signature_functional_")
            )
    elif companion_name == "em_v16":
        reserved = set(EM16_INTERNAL_SUMMARY_COLUMNS)
        signature_columns = [
            name for name in columns if name.startswith("signature_")
        ]
        hotspot_columns = [name for name in columns if name.startswith("hotspot_")]
        reserved.update(signature_columns)
        reserved.update(hotspot_columns)
        severity_columns = [name for name in columns if name not in reserved]
        block_columns = {
            "severity": severity_columns,
            "summary": [
                name for name in EM16_INTERNAL_SUMMARY_COLUMNS if name in columns
            ],
            "signature": signature_columns,
            "hotspot": hotspot_columns,
        }
        for block in normalized:
            if not block_columns[block]:
                raise RuntimeError(f"EM16 {block} 열이 없습니다.")
            selected.extend(block_columns[block])
    if not selected and normalized:
        raise RuntimeError(f"{companion_name} 내부 결합 열을 선택하지 못했습니다.")
    if len(selected) != len(set(selected)):
        raise RuntimeError(f"{companion_name} 내부 결합 열 선택에 중복이 있습니다.")
    return selected


def _select_em24_internal_columns(
    frame: pd.DataFrame,
    blocks: Sequence[str],
) -> list[str]:
    """EM24 전체 행렬에서 내부 결합 대상으로 승인된 저차원 블록만 선택합니다."""
    return _select_internal_companion_columns(
        frame, blocks, companion_name="em_v24"
    )


def _build_internal_fusion_matrix(
    f9_matrix: object,
    companion_frame: pd.DataFrame,
    blocks: Sequence[str],
    *,
    companion_name: str = "em_v24",
) -> tuple[sparse.csr_matrix, list[str]]:
    """F9 CSR을 유지하면서 선택한 동반 전처리 블록을 CSR로 결합합니다."""
    if not isinstance(companion_frame, pd.DataFrame):
        raise TypeError("동반 전처리 결과는 pandas DataFrame이어야 합니다.")
    f9_csr = sparse.csr_matrix(f9_matrix, dtype=np.float32)
    if f9_csr.shape[0] != len(companion_frame):
        raise RuntimeError("F9과 동반 전처리 행 수가 다릅니다.")
    selected = _select_internal_companion_columns(
        companion_frame, blocks, companion_name=companion_name
    )
    if not selected:
        return f9_csr, []
    suffix_values = companion_frame.loc[:, selected].to_numpy(
        dtype=np.float32, copy=False
    )
    if not np.isfinite(suffix_values).all():
        raise RuntimeError("내부 결합 피처에 NaN 또는 무한대가 있습니다.")
    suffix = sparse.csr_matrix(suffix_values, dtype=np.float32)
    combined = sparse.hstack([f9_csr, suffix], format="csr", dtype=np.float32)
    prefix = INTERNAL_COMPANION_PREFIXES[companion_name]
    return combined, [f"{prefix}__{name}" for name in selected]


def _validate_xgb_internal_fusion_config(
    config: Mapping[str, object],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    str,
    dict[str, tuple[str, ...]],
    tuple[str, ...],
    str,
    tuple[int, ...],
    int,
]:
    model_config = _require_mapping(config, "model")
    validation_config = _require_mapping(config, "xgb_internal_fusion_validation")
    if model_config.get("name") != "xgboost":
        raise ValueError("XGBoost 내부 결합은 model.name=xgboost만 지원합니다.")
    seeds = _normalize_seeds(validation_config.get("seeds"))
    raw_n_splits = validation_config.get("n_splits")
    if (
        isinstance(raw_n_splits, (bool, np.bool_))
        or not isinstance(raw_n_splits, (int, np.integer))
        or int(raw_n_splits) < 2
    ):
        raise ValueError("xgb_internal_fusion_validation.n_splits는 2 이상의 정수여야 합니다.")
    if validation_config.get("metric", "macro_f1") != "macro_f1":
        raise ValueError("XGBoost 내부 결합 지표는 macro_f1만 지원합니다.")
    _resolve_overfit_threshold(
        validation_config, section_name="xgb_internal_fusion_validation"
    )

    f9_config = _require_mapping(validation_config, "f9_preprocessing")
    has_generic = "companion_preprocessing" in validation_config
    has_legacy = "em24_preprocessing" in validation_config
    if has_generic and has_legacy:
        raise ValueError(
            "companion_preprocessing과 legacy em24_preprocessing을 함께 쓸 수 없습니다."
        )
    companion_key = (
        "companion_preprocessing" if has_generic else "em24_preprocessing"
    )
    companion_config = _require_mapping(validation_config, companion_key)
    companion_name = str(companion_config.get("name", ""))
    if f9_config.get("name") != "jyp_f9":
        raise ValueError("f9_preprocessing.name은 jyp_f9이어야 합니다.")
    if companion_name not in INTERNAL_COMPANION_BLOCKS:
        raise ValueError("companion_preprocessing.name은 em_v16 또는 em_v24여야 합니다.")
    if has_legacy and companion_name != "em_v24":
        raise ValueError("legacy em24_preprocessing에는 em_v24만 사용할 수 있습니다.")
    create_preprocessing_pipeline(dict(f9_config))
    create_preprocessing_pipeline(dict(companion_config))

    raw_candidates = validation_config.get("candidates")
    if not isinstance(raw_candidates, Mapping) or not raw_candidates:
        raise ValueError("xgb_internal_fusion_validation.candidates 매핑이 필요합니다.")
    candidates: dict[str, tuple[str, ...]] = {}
    candidate_tokens: set[str] = set()
    for raw_name, raw_candidate in raw_candidates.items():
        name = str(raw_name)
        if not isinstance(raw_candidate, Mapping):
            raise ValueError(f"candidates.{name}은 매핑이어야 합니다.")
        token = _artifact_token(name)
        if token in candidate_tokens:
            raise ValueError("candidate 이름을 산출물 이름으로 변환했을 때 충돌합니다.")
        candidate_tokens.add(token)
        has_generic_blocks = "companion_blocks" in raw_candidate
        has_legacy_blocks = "em24_blocks" in raw_candidate
        if has_generic_blocks and has_legacy_blocks:
            raise ValueError(
                f"candidates.{name}에 companion_blocks와 em24_blocks를 함께 쓸 수 없습니다."
            )
        raw_blocks = raw_candidate.get(
            "companion_blocks",
            raw_candidate.get("em24_blocks", ()),
        )
        candidates[name] = _normalize_internal_blocks(
            raw_blocks, candidate=name, companion_name=companion_name
        )
    enabled_candidates = _normalize_enabled_names(
        validation_config.get("enabled_candidates"), available=candidates
    )
    baseline_candidate = str(validation_config.get("baseline_candidate", "f9"))
    if baseline_candidate not in candidates:
        raise ValueError("baseline_candidate가 candidates에 없습니다.")
    if candidates[baseline_candidate]:
        raise ValueError("baseline_candidate는 동반 전처리 블록을 포함할 수 없습니다.")
    if baseline_candidate not in enabled_candidates:
        raise ValueError("baseline_candidate는 enabled_candidates에 포함되어야 합니다.")
    if len(enabled_candidates) < 2:
        raise ValueError(
            "내부 결합 검증에는 baseline 외 비교 후보가 한 개 이상 필요합니다."
        )
    if validation_config.get("submission_policy", "screening_only") not in {
        "screening_only",
        "write_local_candidates",
    }:
        raise ValueError(
            "submission_policy는 screening_only 또는 write_local_candidates여야 합니다."
        )
    try:
        MODEL_BUILDERS["xgboost"](model_config, seeds[0])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("XGBoost 모델 설정이 올바르지 않습니다.") from error
    return (
        model_config,
        validation_config,
        f9_config,
        companion_config,
        companion_name,
        candidates,
        enabled_candidates,
        baseline_candidate,
        seeds,
        int(raw_n_splits),
    )


def _xgb_internal_fusion_contract_hash(
    config: Mapping[str, object],
    *,
    data_hashes: Mapping[str, str],
    source_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
) -> str:
    validation_config = {
        key: value
        for key, value in dict(config["xgb_internal_fusion_validation"]).items()
        if key not in {"resume", "show_progress"}
    }
    payload = {
        "artifact_schema_version": XGB_INTERNAL_FUSION_ARTIFACT_SCHEMA_VERSION,
        "project": config["project"],
        "data": config["data"],
        "model": config["model"],
        "xgb_internal_fusion_validation": validation_config,
        "data_hashes": data_hashes,
        "source_hashes": source_hashes,
        "runtime_versions": runtime_versions,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_internal_fusion_checkpoint(
    *,
    json_path: Path,
    npz_path: Path,
    expected_contract_hash: str,
    expected_shapes: Mapping[str, tuple[int, ...]],
    resume: bool,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    arrays = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name, shape in expected_shapes.items()
    }
    if not resume or (not json_path.exists() and not npz_path.exists()):
        return [], arrays
    if not json_path.exists() or not npz_path.exists():
        raise RuntimeError("내부 결합 checkpoint 일부만 남아 있습니다. --force로 초기화하세요.")
    checkpoint = json.loads(json_path.read_text(encoding="utf-8"))
    if checkpoint.get("contract_hash") != expected_contract_hash:
        raise RuntimeError(
            "기존 내부 결합 checkpoint 계약이 현재 데이터·설정·구현과 다릅니다. "
            "--force로 초기화하세요."
        )
    expected_npz_hash = checkpoint.get("checkpoint_npz_sha256")
    if not isinstance(expected_npz_hash, str) or not expected_npz_hash:
        raise RuntimeError("내부 결합 checkpoint NPZ 해시가 없습니다.")
    if _sha256_file(npz_path) != expected_npz_hash:
        raise RuntimeError("내부 결합 checkpoint NPZ 무결성 검증에 실패했습니다.")
    raw_results = checkpoint.get("candidate_fold_results", [])
    if not isinstance(raw_results, list):
        raise RuntimeError("내부 결합 checkpoint 결과 형식이 올바르지 않습니다.")
    with np.load(npz_path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(expected_shapes):
            raise RuntimeError("내부 결합 checkpoint 배열 구성이 다릅니다.")
        for name, shape in expected_shapes.items():
            values = np.asarray(loaded[name], dtype=np.float32)
            if values.shape != shape:
                raise RuntimeError(f"checkpoint {name} 배열 크기가 다릅니다.")
            arrays[name] = values.copy()
    keys = [
        (int(row["seed"]), int(row["fold"]), str(row["name"]))
        for row in raw_results
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("내부 결합 checkpoint에 중복 학습 단위가 있습니다.")
    return raw_results, arrays


def _load_blend_checkpoint(
    *,
    json_path: Path,
    npz_path: Path,
    expected_contract_hash: str,
    expected_shapes: Mapping[str, tuple[int, ...]],
    resume: bool,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    arrays = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name, shape in expected_shapes.items()
    }
    if not resume or (not json_path.exists() and not npz_path.exists()):
        return [], arrays
    if not json_path.exists() or not npz_path.exists():
        raise RuntimeError(
            "XGBoost 결합 checkpoint 일부만 남아 있습니다. --force로 초기화하세요."
        )
    checkpoint = json.loads(json_path.read_text(encoding="utf-8"))
    if checkpoint.get("contract_hash") != expected_contract_hash:
        raise RuntimeError(
            "기존 XGBoost 결합 checkpoint 계약이 현재 데이터·설정·구현과 다릅니다. "
            "--force로 초기화하세요."
        )
    expected_npz_hash = checkpoint.get("checkpoint_npz_sha256")
    if not isinstance(expected_npz_hash, str) or not expected_npz_hash:
        raise RuntimeError("XGBoost 결합 checkpoint NPZ 해시가 없습니다.")
    if _sha256_file(npz_path) != expected_npz_hash:
        raise RuntimeError("XGBoost 결합 checkpoint NPZ 무결성 검증에 실패했습니다.")
    raw_results = checkpoint.get("member_fold_results", [])
    if not isinstance(raw_results, list):
        raise RuntimeError("XGBoost 결합 checkpoint 결과 형식이 올바르지 않습니다.")
    with np.load(npz_path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(expected_shapes):
            raise RuntimeError("XGBoost 결합 checkpoint 배열 구성이 다릅니다.")
        for name, shape in expected_shapes.items():
            values = np.asarray(loaded[name], dtype=np.float32)
            if values.shape != shape:
                raise RuntimeError(f"checkpoint {name} 배열 크기가 다릅니다.")
            arrays[name] = values.copy()
    keys = [
        (int(row["seed"]), int(row["fold"]), str(row["name"]))
        for row in raw_results
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("XGBoost 결합 checkpoint에 중복 학습 단위가 있습니다.")
    return raw_results, arrays


def run_xgboost_blend_validation(
    config: Mapping[str, object],
    *,
    project_root: Path,
    force: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
) -> dict[str, object]:
    """고정 XGBoost와 동일 Fold로 전처리별 확률을 한 번씩 학습·결합합니다."""
    if force and dry_run:
        raise ValueError("--force와 --dry-run은 함께 사용할 수 없습니다.")
    (
        model_config,
        validation_config,
        members,
        candidates,
        enabled_candidates,
        required_members,
        baseline_member,
        seeds,
        n_splits,
    ) = _validate_xgb_blend_config(config)
    overfit_threshold = _resolve_overfit_threshold(
        validation_config, section_name="xgb_blend_validation"
    )
    data_config = _require_mapping(config, "data")
    project_config = _require_mapping(config, "project")

    raw_dir = project_root / str(data_config["raw_dir"])
    processed_dir = project_root / str(data_config["processed_dir"])
    train_path = raw_dir / str(data_config["train_file"])
    test_path = raw_dir / str(data_config["test_file"])
    submission_path = raw_dir / str(data_config["submission_file"])
    required_paths = {
        "train": train_path,
        "test": test_path,
        "sample_submission": submission_path,
    }
    missing_paths = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"필수 데이터 파일이 없습니다: {missing_paths}")

    output_stem = _safe_output_stem(
        validation_config.get(
            "output_stem",
            project_config.get("experiment_name", "test_003_xgb_blend"),
        )
    )
    output_json = processed_dir / f"{output_stem}.json"
    scores_csv = processed_dir / f"{output_stem}_scores.csv"
    probabilities_npz = processed_dir / f"{output_stem}_probabilities.npz"
    checkpoint_json = processed_dir / f"{output_stem}_checkpoint.json"
    checkpoint_npz = processed_dir / f"{output_stem}_checkpoint.npz"
    submission_outputs = {
        name: processed_dir / f"{output_stem}_{_artifact_token(name)}_submission.csv"
        for name in enabled_candidates
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    owned_outputs = [
        output_json,
        scores_csv,
        probabilities_npz,
        checkpoint_json,
        checkpoint_npz,
        *submission_outputs.values(),
    ]
    owned_outputs.extend(
        path
        for path in processed_dir.glob(f"{output_stem}_*_submission.csv")
        if path not in owned_outputs
    )
    if force:
        for path in owned_outputs:
            if path.exists():
                path.unlink()

    data_hashes = {name: _sha256_file(path) for name, path in required_paths.items()}
    active_member_configs = {alias: members[alias] for alias in required_members}
    source_hashes, runtime_versions = _blend_implementation_identity(
        project_root=project_root, members=active_member_configs
    )
    contract_hash = _xgb_blend_contract_hash(
        config,
        data_hashes=data_hashes,
        source_hashes=source_hashes,
        runtime_versions=runtime_versions,
    )
    submission_policy = str(
        validation_config.get("submission_policy", "screening_only")
    )
    expected_final_artifacts = [scores_csv, probabilities_npz]
    if submission_policy == "write_local_candidates":
        expected_final_artifacts.extend(submission_outputs.values())
    if output_json.exists():
        completed = json.loads(output_json.read_text(encoding="utf-8"))
        if completed.get("status") == "complete":
            if completed.get("contract_hash") != contract_hash:
                raise RuntimeError(
                    "기존 완료 결과가 현재 XGBoost 결합 계약과 다릅니다. --force로 다시 실행하세요."
                )
            missing = [
                str(path.relative_to(project_root))
                for path in expected_final_artifacts
                if not path.exists()
            ]
            if missing:
                raise RuntimeError(f"완료 결과의 산출물이 누락되었습니다: {missing}")
            for path in (checkpoint_json, checkpoint_npz):
                if path.exists():
                    path.unlink()
            print(
                json.dumps(
                    {
                        "status": "already_complete",
                        "output": str(output_json.relative_to(project_root)),
                        "summary": completed.get("summary"),
                    },
                    ensure_ascii=False,
                )
            )
            return completed

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(submission_path)
    target_column = str(data_config["target_column"])
    id_column = str(data_config["id_column"])
    train_required = [column for column in (target_column, id_column) if column not in train]
    test_required = [id_column] if id_column not in test else []
    submission_required = [
        column for column in (target_column, id_column) if column not in sample_submission
    ]
    if train_required or test_required or submission_required:
        raise ValueError(
            "필수 열이 없습니다: "
            f"train={train_required}, test={test_required}, submission={submission_required}"
        )
    features = train.drop(columns=[target_column, id_column])
    test_features = test.drop(columns=[id_column])
    if train[target_column].isna().any():
        raise ValueError("타겟에 결측값이 있습니다.")
    labels = train[target_column].astype(str)
    train_ids = train[id_column].astype(str)
    test_ids = test[id_column].astype(str)
    if features.columns.has_duplicates or test_features.columns.has_duplicates:
        raise ValueError("데이터에 중복 유전자 열 이름이 있습니다.")
    if features.columns.tolist() != test_features.columns.tolist():
        raise ValueError("Train/Test 유전자 열과 순서가 다릅니다.")
    if len(sample_submission) != len(test) or not sample_submission[id_column].astype(
        str
    ).reset_index(drop=True).equals(test_ids.reset_index(drop=True)):
        raise ValueError("sample_submission과 Test의 ID 및 순서가 다릅니다.")
    class_labels, class_counts = np.unique(labels.to_numpy(), return_counts=True)
    class_labels = class_labels.astype(str)
    if int(class_counts.min()) < n_splits:
        raise ValueError("각 원본 SUBCLASS 표본 수는 n_splits 이상이어야 합니다.")
    training_count = len(seeds) * n_splits * len(required_members)
    if dry_run:
        result = {
            "status": "dry_run_passed",
            "mode": "xgboost_probability_blend",
            "model": model_config["name"],
            "members": list(required_members),
            "enabled_candidates": list(enabled_candidates),
            "baseline_member": baseline_member,
            "seeds": list(seeds),
            "n_splits": n_splits,
            "training_count": training_count,
            "rows": len(features),
            "test_rows": len(test_features),
            "gene_features": features.shape[1],
            "classes": class_labels.tolist(),
            "overfit_threshold": overfit_threshold,
            "submission_policy": submission_policy,
            "output": str(output_json.relative_to(project_root)),
        }
        print(json.dumps(result, ensure_ascii=False))
        return result

    seed_index = {seed: index for index, seed in enumerate(seeds)}
    member_index = {name: index for index, name in enumerate(required_members)}
    expected_checkpoint_shapes = {
        "member_oof": (
            len(seeds), len(required_members), len(features), len(class_labels)
        ),
        "member_test_by_fold": (
            len(seeds),
            len(required_members),
            n_splits,
            len(test_features),
            len(class_labels),
        ),
        "member_train_by_fold": (
            len(seeds),
            len(required_members),
            n_splits,
            len(features),
            len(class_labels),
        ),
        "fold_ids": (len(seeds), len(features)),
    }
    resume = bool(validation_config.get("resume", True))
    member_fold_results, checkpoint_arrays = _load_blend_checkpoint(
        json_path=checkpoint_json,
        npz_path=checkpoint_npz,
        expected_contract_hash=contract_hash,
        expected_shapes=expected_checkpoint_shapes,
        resume=resume,
    )
    member_oof = checkpoint_arrays["member_oof"]
    member_test_by_fold = checkpoint_arrays["member_test_by_fold"]
    member_train_by_fold = checkpoint_arrays["member_train_by_fold"]
    fold_ids = checkpoint_arrays["fold_ids"]
    completed_units = {
        (int(row["seed"]), int(row["fold"]), str(row["name"]))
        for row in member_fold_results
    }
    show_progress = bool(validation_config.get("show_progress", True)) and not quiet
    if show_progress and completed_units:
        print(json.dumps({"resumed_units": len(completed_units)}, ensure_ascii=False))

    split_hashes: dict[str, str] = {}
    model_builder = MODEL_BUILDERS["xgboost"]
    started_all = time.perf_counter()
    label_values = labels.to_numpy()
    for seed in seeds:
        current_seed_index = seed_index[seed]
        splits = list(
            StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            ).split(features, labels.to_numpy())
        )
        split_hashes[str(seed)] = _split_hash(splits)
        expected_fold_ids = np.empty(len(features), dtype=np.float32)
        for fold_number, (_, valid_indices) in enumerate(splits, start=1):
            expected_fold_ids[valid_indices] = fold_number
        stored_fold_ids = fold_ids[current_seed_index]
        if np.isfinite(stored_fold_ids).any() and not np.array_equal(
            stored_fold_ids, expected_fold_ids
        ):
            raise RuntimeError(f"seed {seed} checkpoint의 Fold 배정이 다릅니다.")
        fold_ids[current_seed_index] = expected_fold_ids

        for fold_number, (train_indices, valid_indices) in enumerate(splits, start=1):
            train_x = features.iloc[train_indices]
            valid_x = features.iloc[valid_indices]
            train_y = labels.iloc[train_indices]
            valid_y = labels.iloc[valid_indices].to_numpy()
            for alias in required_members:
                unit = (seed, fold_number, alias)
                if unit in completed_units:
                    continue
                unit_started = time.perf_counter()
                preprocessor = create_preprocessing_pipeline(dict(members[alias]))
                encoded_train_x = preprocessor.fit_transform(train_x, train_y)
                encoded_valid_x = preprocessor.transform(valid_x)
                encoded_test_x = preprocessor.transform(test_features)
                encoded_train_y = preprocessor.encode_labels(train_y)
                model = model_builder(model_config, seed)
                model.fit(encoded_train_x, encoded_train_y)
                train_probability = _align_probability_columns(
                    model.predict_proba(encoded_train_x),
                    model=model,
                    preprocessor=preprocessor,
                    global_class_labels=class_labels,
                    context=f"seed={seed}, fold={fold_number}, member={alias}, train",
                )
                valid_probability = _align_probability_columns(
                    model.predict_proba(encoded_valid_x),
                    model=model,
                    preprocessor=preprocessor,
                    global_class_labels=class_labels,
                    context=f"seed={seed}, fold={fold_number}, member={alias}, validation",
                )
                test_probability = _align_probability_columns(
                    model.predict_proba(encoded_test_x),
                    model=model,
                    preprocessor=preprocessor,
                    global_class_labels=class_labels,
                    context=f"seed={seed}, fold={fold_number}, member={alias}, test",
                )
                current_member_index = member_index[alias]
                member_oof[
                    current_seed_index, current_member_index, valid_indices
                ] = valid_probability
                member_test_by_fold[
                    current_seed_index, current_member_index, fold_number - 1
                ] = test_probability
                member_train_by_fold[
                    current_seed_index,
                    current_member_index,
                    fold_number - 1,
                    train_indices,
                ] = train_probability
                train_macro_f1, _, _ = _score_probability(
                    train_y.to_numpy(), train_probability, class_labels=class_labels
                )
                macro_f1, class_f1, _ = _score_probability(
                    valid_y, valid_probability, class_labels=class_labels
                )
                overfit_metrics = _fold_overfit_metrics(
                    train_macro_f1,
                    macro_f1,
                    threshold=overfit_threshold,
                )
                member_fold_results.append(
                    {
                        "record_type": "member",
                        "name": alias,
                        "pipeline_name": members[alias]["name"],
                        "seed": seed,
                        "fold": fold_number,
                        "macro_f1": macro_f1,
                        **overfit_metrics,
                        "feature_count": int(encoded_train_x.shape[1]),
                        "elapsed_seconds": time.perf_counter() - unit_started,
                        "class_f1": {
                            label: float(score)
                            for label, score in zip(class_labels, class_f1, strict=True)
                        },
                    }
                )
                completed_units.add(unit)
                _write_npz_atomic(checkpoint_npz, checkpoint_arrays)
                _write_json_atomic(
                    checkpoint_json,
                    {
                        "status": "running",
                        "contract_hash": contract_hash,
                        "checkpoint_npz_sha256": _sha256_file(checkpoint_npz),
                        "member_fold_results": member_fold_results,
                    },
                )
                if show_progress:
                    print(
                        json.dumps(
                            {
                                "completed": {
                                    "seed": seed,
                                    "fold": fold_number,
                                    "member": alias,
                                    "macro_f1": macro_f1,
                                    "train_macro_f1": train_macro_f1,
                                    "train_validation_gap": overfit_metrics[
                                        "train_validation_gap"
                                    ],
                                    "is_overfitting": overfit_metrics[
                                        "is_overfitting"
                                    ],
                                    "feature_count": int(encoded_train_x.shape[1]),
                                },
                                "progress": f"{len(completed_units)}/{training_count}",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                del (
                    preprocessor,
                    encoded_train_x,
                    encoded_valid_x,
                    encoded_test_x,
                    encoded_train_y,
                    model,
                    train_probability,
                    valid_probability,
                    test_probability,
                )
                gc.collect()

    if len(completed_units) != training_count:
        raise RuntimeError("완료된 XGBoost 멤버 학습 수가 설정과 일치하지 않습니다.")
    if not np.isfinite(member_oof).all() or not np.isfinite(member_test_by_fold).all():
        raise RuntimeError("완료 후 멤버 OOF/Test 확률에 빈 값이 있습니다.")
    for current_seed_index, seed in enumerate(seeds):
        for fold_number in range(1, n_splits + 1):
            train_mask = fold_ids[current_seed_index] != fold_number
            if not np.isfinite(
                member_train_by_fold[
                    current_seed_index, :, fold_number - 1, train_mask
                ]
            ).all():
                raise RuntimeError(
                    f"seed={seed}, fold={fold_number} 멤버 Train 확률에 빈 값이 있습니다."
                )
    member_test = member_test_by_fold.mean(axis=2, dtype=np.float64).astype(np.float32)
    member_fold_lookup = {
        (int(row["seed"]), int(row["fold"]), str(row["name"])): row
        for row in member_fold_results
    }

    score_rows: list[dict[str, object]] = []
    seed_results: list[dict[str, object]] = []
    candidate_oof: dict[str, np.ndarray] = {
        name: np.empty(
            (len(seeds), len(features), len(class_labels)), dtype=np.float32
        )
        for name in enabled_candidates
    }
    candidate_test: dict[str, np.ndarray] = {
        name: np.empty(
            (len(seeds), len(test_features), len(class_labels)), dtype=np.float32
        )
        for name in enabled_candidates
    }
    for seed in seeds:
        current_seed_index = seed_index[seed]
        member_probability = {
            alias: member_oof[current_seed_index, member_index[alias]]
            for alias in required_members
        }
        member_test_probability = {
            alias: member_test[current_seed_index, member_index[alias]]
            for alias in required_members
        }
        baseline_probability = member_probability[baseline_member]
        baseline_seed_score, baseline_class_f1, _ = _score_probability(
            label_values, baseline_probability, class_labels=class_labels
        )
        seed_entry: dict[str, object] = {
            "seed": seed,
            "baseline_member": baseline_member,
            "baseline_oof_macro_f1": baseline_seed_score,
            "baseline_class_f1": {
                label: float(score)
                for label, score in zip(class_labels, baseline_class_f1, strict=True)
            },
            "members": {},
            "candidates": {},
        }
        for fold_number in range(1, n_splits + 1):
            mask = fold_ids[current_seed_index] == fold_number
            train_mask = ~mask
            truth = label_values[mask]
            train_truth = label_values[train_mask]
            fold_baseline_score, _, _ = _score_probability(
                truth, baseline_probability[mask], class_labels=class_labels
            )
            for alias in required_members:
                score, _, _ = _score_probability(
                    truth, member_probability[alias][mask], class_labels=class_labels
                )
                source_row = member_fold_lookup[
                    (seed, fold_number, alias)
                ]
                stored_train_probability = member_train_by_fold[
                    current_seed_index,
                    member_index[alias],
                    fold_number - 1,
                    train_mask,
                ]
                train_score, _, _ = _score_probability(
                    train_truth,
                    stored_train_probability,
                    class_labels=class_labels,
                )
                if not np.isclose(score, float(source_row["macro_f1"])) or not np.isclose(
                    train_score, float(source_row["train_macro_f1"])
                ):
                    raise RuntimeError(
                        f"seed={seed}, fold={fold_number}, member={alias} "
                        "checkpoint 점수와 확률이 일치하지 않습니다."
                    )
                overfit_metrics = _fold_overfit_metrics(
                    train_score,
                    score,
                    threshold=overfit_threshold,
                )
                score_rows.append(
                    {
                        "record_type": "member",
                        "name": alias,
                        "pipeline_name": members[alias]["name"],
                        "seed": seed,
                        "fold": fold_number,
                        "macro_f1": score,
                        **overfit_metrics,
                        "baseline_macro_f1": fold_baseline_score,
                        "delta_vs_baseline": score - fold_baseline_score,
                        "feature_count": source_row["feature_count"],
                        "elapsed_seconds": source_row["elapsed_seconds"],
                    }
                )
        for alias in required_members:
            member_seed_score, member_class_f1, _ = _score_probability(
                label_values,
                member_probability[alias],
                class_labels=class_labels,
            )
            member_seed_rows = [
                row
                for row in score_rows
                if row["record_type"] == "member"
                and row["name"] == alias
                and int(row["seed"]) == seed
            ]
            seed_entry["members"][alias] = {
                "oof_macro_f1": member_seed_score,
                "class_f1": {
                    label: float(value)
                    for label, value in zip(
                        class_labels, member_class_f1, strict=True
                    )
                },
                **_summarize_overfit_rows(
                    member_seed_rows,
                    threshold=overfit_threshold,
                ),
            }
        for candidate_name in enabled_candidates:
            weights = candidates[candidate_name]
            combined_oof = _blend_probabilities(
                weights,
                member_probability,
                context=f"seed={seed}, candidate={candidate_name}, OOF",
            )
            combined_test = _blend_probabilities(
                weights,
                member_test_probability,
                context=f"seed={seed}, candidate={candidate_name}, Test",
            )
            candidate_oof[candidate_name][current_seed_index] = combined_oof
            candidate_test[candidate_name][current_seed_index] = combined_test
            score, class_f1, _ = _score_probability(
                label_values, combined_oof, class_labels=class_labels
            )
            candidate_seed_rows: list[dict[str, object]] = []
            for fold_number in range(1, n_splits + 1):
                mask = fold_ids[current_seed_index] == fold_number
                train_mask = ~mask
                truth = label_values[mask]
                train_truth = label_values[train_mask]
                baseline_fold_score, _, _ = _score_probability(
                    truth, baseline_probability[mask], class_labels=class_labels
                )
                candidate_fold_score, _, _ = _score_probability(
                    truth, combined_oof[mask], class_labels=class_labels
                )
                train_member_probability = {
                    alias: member_train_by_fold[
                        current_seed_index,
                        member_index[alias],
                        fold_number - 1,
                        train_mask,
                    ]
                    for alias in weights
                }
                combined_train = _blend_probabilities(
                    weights,
                    train_member_probability,
                    context=(
                        f"seed={seed}, fold={fold_number}, "
                        f"candidate={candidate_name}, Train"
                    ),
                )
                candidate_train_score, _, _ = _score_probability(
                    train_truth,
                    combined_train,
                    class_labels=class_labels,
                )
                overfit_metrics = _fold_overfit_metrics(
                    candidate_train_score,
                    candidate_fold_score,
                    threshold=overfit_threshold,
                )
                score_row = {
                    "record_type": "candidate",
                    "name": candidate_name,
                    "pipeline_name": "+".join(weights),
                    "seed": seed,
                    "fold": fold_number,
                    "macro_f1": candidate_fold_score,
                    **overfit_metrics,
                    "baseline_macro_f1": baseline_fold_score,
                    "delta_vs_baseline": candidate_fold_score - baseline_fold_score,
                    "feature_count": None,
                    "elapsed_seconds": None,
                }
                score_rows.append(score_row)
                candidate_seed_rows.append(score_row)
            seed_entry["candidates"][candidate_name] = {
                "oof_macro_f1": score,
                "delta_vs_baseline": score - baseline_seed_score,
                "class_f1": {
                    label: float(value)
                    for label, value in zip(class_labels, class_f1, strict=True)
                },
                **_summarize_overfit_rows(
                    candidate_seed_rows,
                    threshold=overfit_threshold,
                ),
            }
        seed_results.append(seed_entry)

    summary_members: dict[str, object] = {}
    for alias in required_members:
        member_rows = [
            row
            for row in score_rows
            if row["record_type"] == "member" and row["name"] == alias
        ]
        member_seed_scores = np.asarray(
            [row["members"][alias]["oof_macro_f1"] for row in seed_results],
            dtype=np.float64,
        )
        summary_members[alias] = {
            "pipeline_name": members[alias]["name"],
            "mean_seed_oof_macro_f1": float(member_seed_scores.mean()),
            "seed_oof_std": float(member_seed_scores.std(ddof=0)),
            **_summarize_overfit_rows(
                member_rows,
                threshold=overfit_threshold,
            ),
        }

    summary_candidates: dict[str, object] = {}
    for candidate_name in enabled_candidates:
        scores = np.asarray(
            [
                row["candidates"][candidate_name]["oof_macro_f1"]
                for row in seed_results
            ],
            dtype=np.float64,
        )
        deltas = np.asarray(
            [
                row["candidates"][candidate_name]["delta_vs_baseline"]
                for row in seed_results
            ],
            dtype=np.float64,
        )
        fold_deltas = np.asarray(
            [
                row["delta_vs_baseline"]
                for row in score_rows
                if row["record_type"] == "candidate" and row["name"] == candidate_name
            ],
            dtype=np.float64,
        )
        candidate_rows = [
            row
            for row in score_rows
            if row["record_type"] == "candidate" and row["name"] == candidate_name
        ]
        summary_candidates[candidate_name] = {
            "weights": candidates[candidate_name],
            "mean_seed_oof_macro_f1": float(scores.mean()),
            "seed_oof_std": float(scores.std(ddof=0)),
            "mean_delta_vs_baseline": float(deltas.mean()),
            "min_seed_delta_vs_baseline": float(deltas.min()),
            "improved_seed_count": int((deltas > 0.0).sum()),
            "won_fold_count": int((fold_deltas > 0.0).sum()),
            "total_fold_count": int(len(fold_deltas)),
            **_summarize_overfit_rows(
                candidate_rows,
                threshold=overfit_threshold,
            ),
        }
    baseline_scores = np.asarray(
        [row["baseline_oof_macro_f1"] for row in seed_results], dtype=np.float64
    )
    best_candidate = max(
        enabled_candidates,
        key=lambda name: float(summary_candidates[name]["mean_seed_oof_macro_f1"]),
    )
    summary = {
        "baseline_member": baseline_member,
        "baseline_mean_seed_oof_macro_f1": float(baseline_scores.mean()),
        "baseline_seed_oof_std": float(baseline_scores.std(ddof=0)),
        "overfit_threshold": overfit_threshold,
        "baseline_is_overfitting": summary_members[baseline_member][
            "is_overfitting"
        ],
        "best_candidate": best_candidate,
        "best_candidate_mean_seed_oof_macro_f1": summary_candidates[best_candidate][
            "mean_seed_oof_macro_f1"
        ],
        "best_candidate_mean_delta_vs_baseline": summary_candidates[best_candidate][
            "mean_delta_vs_baseline"
        ],
        "members": summary_members,
        "candidates": summary_candidates,
        "submission_policy": submission_policy,
        "submission_files_created": submission_policy == "write_local_candidates",
        "elapsed_seconds": float(time.perf_counter() - started_all),
    }

    probability_arrays: dict[str, np.ndarray] = {
        "classes": class_labels.astype(str),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "train_ids": train_ids.to_numpy(dtype=str),
        "test_ids": test_ids.to_numpy(dtype=str),
        "fold_ids": fold_ids.astype(np.int16),
    }
    for alias in required_members:
        token = _artifact_token(alias)
        probability_arrays[f"member_oof__{token}"] = member_oof[
            :, member_index[alias]
        ]
        probability_arrays[f"member_test__{token}"] = member_test[
            :, member_index[alias]
        ]
    for candidate_name in enabled_candidates:
        token = _artifact_token(candidate_name)
        probability_arrays[f"candidate_oof__{token}"] = candidate_oof[candidate_name]
        probability_arrays[f"candidate_test__{token}"] = candidate_test[candidate_name]
    _write_npz_atomic(probabilities_npz, probability_arrays)
    pd.DataFrame(score_rows).sort_values(
        ["seed", "fold", "record_type", "name"]
    ).to_csv(scores_csv, index=False, encoding="utf-8-sig")

    written_submissions: dict[str, str] = {}
    if submission_policy == "write_local_candidates":
        for candidate_name in enabled_candidates:
            mean_test_probability = candidate_test[candidate_name].mean(
                axis=0, dtype=np.float64
            )
            predictions = class_labels[mean_test_probability.argmax(axis=1)]
            submission = sample_submission.copy()
            submission[target_column] = predictions
            output = submission_outputs[candidate_name]
            submission.to_csv(output, index=False, encoding="utf-8-sig")
            written_submissions[candidate_name] = str(output.relative_to(project_root))

    result = {
        "status": "complete",
        "mode": "xgboost_probability_blend",
        "artifact_schema_version": XGB_BLEND_ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "contract_hash": contract_hash,
        "data_hashes": data_hashes,
        "source_hashes": source_hashes,
        "runtime_versions": runtime_versions,
        "class_order": class_labels.tolist(),
        "class_order_source": "sorted original train.csv SUBCLASS labels",
        "label_remapping": False,
        "model": model_config,
        "members": {alias: members[alias] for alias in required_members},
        "candidates": {
            name: {"weights": candidates[name]} for name in enabled_candidates
        },
        "baseline_member": baseline_member,
        "overfit_threshold": overfit_threshold,
        "seeds": list(seeds),
        "n_splits": n_splits,
        "training_count": training_count,
        "splitter": "StratifiedKFold(shuffle=True, random_state=seed)",
        "split_hashes": split_hashes,
        "summary": summary,
        "seed_results": seed_results,
        "artifacts": {
            "result_json": str(output_json.relative_to(project_root)),
            "scores_csv": str(scores_csv.relative_to(project_root)),
            "probabilities_npz": str(probabilities_npz.relative_to(project_root)),
            "submissions": written_submissions,
        },
    }
    _write_json_atomic(output_json, result)
    for path in (checkpoint_json, checkpoint_npz):
        if path.exists():
            path.unlink()
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_json.relative_to(project_root)),
                "summary": summary,
            },
            ensure_ascii=False,
        )
    )
    return result


def run_xgboost_internal_fusion_validation(
    config: Mapping[str, object],
    *,
    project_root: Path,
    force: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
) -> dict[str, object]:
    """F9과 선택된 동반 전처리 피처를 결합해 단일 XGBoost로 평가합니다."""
    if force and dry_run:
        raise ValueError("--force와 --dry-run은 함께 사용할 수 없습니다.")
    (
        model_config,
        validation_config,
        f9_config,
        companion_config,
        companion_name,
        candidates,
        enabled_candidates,
        baseline_candidate,
        seeds,
        n_splits,
    ) = _validate_xgb_internal_fusion_config(config)
    overfit_threshold = _resolve_overfit_threshold(
        validation_config, section_name="xgb_internal_fusion_validation"
    )
    data_config = _require_mapping(config, "data")
    project_config = _require_mapping(config, "project")

    raw_dir = project_root / str(data_config["raw_dir"])
    processed_dir = project_root / str(data_config["processed_dir"])
    train_path = raw_dir / str(data_config["train_file"])
    test_path = raw_dir / str(data_config["test_file"])
    submission_path = raw_dir / str(data_config["submission_file"])
    required_paths = {
        "train": train_path,
        "test": test_path,
        "sample_submission": submission_path,
    }
    missing_paths = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"필수 데이터 파일이 없습니다: {missing_paths}")

    output_stem = _safe_output_stem(
        validation_config.get(
            "output_stem",
            project_config.get("experiment_name", "test_003_xgb_internal_fusion"),
        )
    )
    output_json = processed_dir / f"{output_stem}.json"
    scores_csv = processed_dir / f"{output_stem}_scores.csv"
    probabilities_npz = processed_dir / f"{output_stem}_probabilities.npz"
    checkpoint_json = processed_dir / f"{output_stem}_checkpoint.json"
    checkpoint_npz = processed_dir / f"{output_stem}_checkpoint.npz"
    submission_outputs = {
        name: processed_dir / f"{output_stem}_{_artifact_token(name)}_submission.csv"
        for name in enabled_candidates
        if name != baseline_candidate
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    owned_outputs = [
        output_json,
        scores_csv,
        probabilities_npz,
        checkpoint_json,
        checkpoint_npz,
        *submission_outputs.values(),
    ]
    owned_outputs.extend(
        path
        for path in processed_dir.glob(f"{output_stem}_*_submission.csv")
        if path not in owned_outputs
    )
    if force:
        for path in owned_outputs:
            if path.exists():
                path.unlink()

    data_hashes = {name: _sha256_file(path) for name, path in required_paths.items()}
    source_hashes, runtime_versions = _blend_implementation_identity(
        project_root=project_root,
        members={"jyp_f9": f9_config, companion_name: companion_config},
    )
    contract_hash = _xgb_internal_fusion_contract_hash(
        config,
        data_hashes=data_hashes,
        source_hashes=source_hashes,
        runtime_versions=runtime_versions,
    )
    submission_policy = str(
        validation_config.get("submission_policy", "screening_only")
    )
    expected_final_artifacts = [scores_csv, probabilities_npz]
    if submission_policy == "write_local_candidates":
        expected_final_artifacts.extend(submission_outputs.values())
    if output_json.exists():
        completed = json.loads(output_json.read_text(encoding="utf-8"))
        if completed.get("status") == "complete":
            if completed.get("contract_hash") != contract_hash:
                raise RuntimeError(
                    "기존 완료 결과가 현재 내부 결합 계약과 다릅니다. --force로 다시 실행하세요."
                )
            missing = [
                str(path.relative_to(project_root))
                for path in expected_final_artifacts
                if not path.exists()
            ]
            if missing:
                raise RuntimeError(f"완료 결과의 산출물이 누락되었습니다: {missing}")
            raw_artifact_hashes = completed.get("artifact_hashes")
            if not isinstance(raw_artifact_hashes, Mapping):
                raise RuntimeError("완료 결과에 산출물 해시가 없습니다.")
            expected_hashes = {
                "scores_csv": scores_csv,
                "probabilities_npz": probabilities_npz,
            }
            for name, path in expected_hashes.items():
                expected_hash = raw_artifact_hashes.get(name)
                if not isinstance(expected_hash, str) or _sha256_file(path) != expected_hash:
                    raise RuntimeError(f"완료 결과의 {name} 무결성 검증에 실패했습니다.")
            raw_submission_hashes = raw_artifact_hashes.get("submissions", {})
            if not isinstance(raw_submission_hashes, Mapping):
                raise RuntimeError("완료 결과의 submission 해시 형식이 올바르지 않습니다.")
            if submission_policy == "write_local_candidates":
                for name, path in submission_outputs.items():
                    expected_hash = raw_submission_hashes.get(name)
                    if (
                        not isinstance(expected_hash, str)
                        or _sha256_file(path) != expected_hash
                    ):
                        raise RuntimeError(
                            f"완료 결과의 {name} submission 무결성 검증에 실패했습니다."
                        )
            for path in (checkpoint_json, checkpoint_npz):
                if path.exists():
                    path.unlink()
            print(
                json.dumps(
                    {
                        "status": "already_complete",
                        "output": str(output_json.relative_to(project_root)),
                        "summary": completed.get("summary"),
                    },
                    ensure_ascii=False,
                )
            )
            return completed

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(submission_path)
    target_column = str(data_config["target_column"])
    id_column = str(data_config["id_column"])
    train_required = [column for column in (target_column, id_column) if column not in train]
    test_required = [id_column] if id_column not in test else []
    submission_required = [
        column for column in (target_column, id_column) if column not in sample_submission
    ]
    if train_required or test_required or submission_required:
        raise ValueError(
            "필수 열이 없습니다: "
            f"train={train_required}, test={test_required}, submission={submission_required}"
        )
    features = train.drop(columns=[target_column, id_column])
    test_features = test.drop(columns=[id_column])
    if train[target_column].isna().any():
        raise ValueError("타겟에 결측값이 있습니다.")
    labels = train[target_column].astype(str)
    label_values = labels.to_numpy()
    train_ids = train[id_column].astype(str)
    test_ids = test[id_column].astype(str)
    if features.columns.has_duplicates or test_features.columns.has_duplicates:
        raise ValueError("데이터에 중복 유전자 열 이름이 있습니다.")
    if features.columns.tolist() != test_features.columns.tolist():
        raise ValueError("Train/Test 유전자 열과 순서가 다릅니다.")
    if len(sample_submission) != len(test) or not sample_submission[id_column].astype(
        str
    ).reset_index(drop=True).equals(test_ids.reset_index(drop=True)):
        raise ValueError("sample_submission과 Test의 ID 및 순서가 다릅니다.")
    class_labels, class_counts = np.unique(label_values, return_counts=True)
    class_labels = class_labels.astype(str)
    if int(class_counts.min()) < n_splits:
        raise ValueError("각 원본 SUBCLASS 표본 수는 n_splits 이상이어야 합니다.")

    training_count = len(seeds) * n_splits * len(enabled_candidates)
    preprocessing_fit_count = len(seeds) * n_splits * 2
    if dry_run:
        result = {
            "status": "dry_run_passed",
            "mode": "xgboost_internal_feature_fusion",
            "model": model_config["name"],
            "candidates": {
                name: list(candidates[name]) for name in enabled_candidates
            },
            "baseline_candidate": baseline_candidate,
            "seeds": list(seeds),
            "n_splits": n_splits,
            "training_count": training_count,
            "preprocessing_fit_count": preprocessing_fit_count,
            "rows": len(features),
            "test_rows": len(test_features),
            "gene_features": features.shape[1],
            "classes": class_labels.tolist(),
            "overfit_threshold": overfit_threshold,
            "submission_policy": submission_policy,
            "output": str(output_json.relative_to(project_root)),
        }
        print(json.dumps(result, ensure_ascii=False))
        return result

    seed_index = {seed: index for index, seed in enumerate(seeds)}
    candidate_index = {
        name: index for index, name in enumerate(enabled_candidates)
    }
    expected_checkpoint_shapes = {
        "candidate_oof": (
            len(seeds), len(enabled_candidates), len(features), len(class_labels)
        ),
        "candidate_test_by_fold": (
            len(seeds),
            len(enabled_candidates),
            n_splits,
            len(test_features),
            len(class_labels),
        ),
        "fold_ids": (len(seeds), len(features)),
    }
    resume = bool(validation_config.get("resume", True))
    candidate_fold_results, checkpoint_arrays = _load_internal_fusion_checkpoint(
        json_path=checkpoint_json,
        npz_path=checkpoint_npz,
        expected_contract_hash=contract_hash,
        expected_shapes=expected_checkpoint_shapes,
        resume=resume,
    )
    candidate_oof = checkpoint_arrays["candidate_oof"]
    candidate_test_by_fold = checkpoint_arrays["candidate_test_by_fold"]
    fold_ids = checkpoint_arrays["fold_ids"]
    completed_units = {
        (int(row["seed"]), int(row["fold"]), str(row["name"]))
        for row in candidate_fold_results
    }
    expected_units = {
        (seed, fold_number, candidate_name)
        for seed in seeds
        for fold_number in range(1, n_splits + 1)
        for candidate_name in enabled_candidates
    }
    unexpected_units = sorted(completed_units - expected_units)
    if unexpected_units:
        raise RuntimeError(
            f"내부 결합 checkpoint에 현재 설정 밖 학습 단위가 있습니다: {unexpected_units}"
        )
    show_progress = bool(validation_config.get("show_progress", True)) and not quiet
    if show_progress and completed_units:
        print(json.dumps({"resumed_units": len(completed_units)}, ensure_ascii=False))

    split_hashes: dict[str, str] = {}
    model_builder = MODEL_BUILDERS["xgboost"]
    started_all = time.perf_counter()
    for seed in seeds:
        current_seed_index = seed_index[seed]
        splits = list(
            StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            ).split(features, label_values)
        )
        split_hashes[str(seed)] = _split_hash(splits)
        expected_fold_ids = np.empty(len(features), dtype=np.float32)
        for fold_number, (_, valid_indices) in enumerate(splits, start=1):
            expected_fold_ids[valid_indices] = fold_number
        stored_fold_ids = fold_ids[current_seed_index]
        if np.isfinite(stored_fold_ids).any() and not np.array_equal(
            stored_fold_ids, expected_fold_ids
        ):
            raise RuntimeError(f"seed {seed} checkpoint의 Fold 배정이 다릅니다.")
        fold_ids[current_seed_index] = expected_fold_ids

        for row in candidate_fold_results:
            if int(row["seed"]) != seed:
                continue
            fold_number = int(row["fold"])
            candidate_name = str(row["name"])
            valid_indices = splits[fold_number - 1][1]
            current_candidate_index = candidate_index[candidate_name]
            resumed_valid_probability = _validate_probability_matrix(
                candidate_oof[
                    current_seed_index, current_candidate_index, valid_indices
                ],
                expected_rows=len(valid_indices),
                expected_columns=len(class_labels),
                context=(
                    f"checkpoint seed={seed}, fold={fold_number}, "
                    f"candidate={candidate_name}, validation"
                ),
            )
            _validate_probability_matrix(
                candidate_test_by_fold[
                    current_seed_index,
                    current_candidate_index,
                    fold_number - 1,
                ],
                expected_rows=len(test_features),
                expected_columns=len(class_labels),
                context=(
                    f"checkpoint seed={seed}, fold={fold_number}, "
                    f"candidate={candidate_name}, test"
                ),
            )
            resumed_score, _, _ = _score_probability(
                label_values[valid_indices],
                resumed_valid_probability,
                class_labels=class_labels,
            )
            if not np.isclose(
                resumed_score,
                float(row["macro_f1"]),
                rtol=0.0,
                atol=1e-8,
            ):
                raise RuntimeError(
                    f"checkpoint seed={seed}, fold={fold_number}, "
                    f"candidate={candidate_name} 점수가 확률과 다릅니다."
                )

        for fold_number, (train_indices, valid_indices) in enumerate(splits, start=1):
            fold_units = {
                (seed, fold_number, name) for name in enabled_candidates
            }
            if fold_units.issubset(completed_units):
                continue
            train_x = features.iloc[train_indices]
            valid_x = features.iloc[valid_indices]
            train_y = labels.iloc[train_indices]
            valid_y = labels.iloc[valid_indices].to_numpy()
            preprocessing_started = time.perf_counter()
            f9 = create_preprocessing_pipeline(dict(f9_config))
            companion = create_preprocessing_pipeline(dict(companion_config))
            f9_train = f9.fit_transform(train_x, train_y)
            f9_valid = f9.transform(valid_x)
            f9_test = f9.transform(test_features)
            companion_train = companion.fit_transform(train_x, train_y)
            companion_valid = companion.transform(valid_x)
            companion_test = companion.transform(test_features)
            preprocessing_seconds = time.perf_counter() - preprocessing_started
            companion_columns = companion_train.columns.astype(str).tolist()
            if (
                companion_columns != companion_valid.columns.astype(str).tolist()
                or companion_columns != companion_test.columns.astype(str).tolist()
            ):
                raise RuntimeError("동반 전처리 Train/Validation/Test 열과 순서가 다릅니다.")
            f9_classes = np.asarray(f9.label_encoder.classes_, dtype=str)
            companion_classes = np.asarray(companion.label_encoder.classes_, dtype=str)
            if not np.array_equal(f9_classes, companion_classes):
                raise RuntimeError("F9과 동반 전처리의 원본 SUBCLASS 순서가 다릅니다.")
            f9_feature_names = f9.get_feature_names_out().astype(str).tolist()
            encoded_train_y = f9.encode_labels(train_y)

            for candidate_name in enabled_candidates:
                unit = (seed, fold_number, candidate_name)
                if unit in completed_units:
                    continue
                unit_started = time.perf_counter()
                blocks = candidates[candidate_name]
                encoded_train_x, suffix_names = _build_internal_fusion_matrix(
                    f9_train,
                    companion_train,
                    blocks,
                    companion_name=companion_name,
                )
                encoded_valid_x, valid_suffix_names = _build_internal_fusion_matrix(
                    f9_valid,
                    companion_valid,
                    blocks,
                    companion_name=companion_name,
                )
                encoded_test_x, test_suffix_names = _build_internal_fusion_matrix(
                    f9_test,
                    companion_test,
                    blocks,
                    companion_name=companion_name,
                )
                if suffix_names != valid_suffix_names or suffix_names != test_suffix_names:
                    raise RuntimeError("내부 결합 Train/Validation/Test suffix가 다릅니다.")
                if encoded_train_x.shape[1] != len(f9_feature_names) + len(suffix_names):
                    raise RuntimeError("내부 결합 피처 수와 스키마가 다릅니다.")
                model = model_builder(model_config, seed)
                model.fit(encoded_train_x, encoded_train_y)
                train_probability = _align_probability_columns(
                    model.predict_proba(encoded_train_x),
                    model=model,
                    preprocessor=f9,
                    global_class_labels=class_labels,
                    context=(
                        f"seed={seed}, fold={fold_number}, candidate={candidate_name}, train"
                    ),
                )
                valid_probability = _align_probability_columns(
                    model.predict_proba(encoded_valid_x),
                    model=model,
                    preprocessor=f9,
                    global_class_labels=class_labels,
                    context=(
                        f"seed={seed}, fold={fold_number}, candidate={candidate_name}, validation"
                    ),
                )
                test_probability = _align_probability_columns(
                    model.predict_proba(encoded_test_x),
                    model=model,
                    preprocessor=f9,
                    global_class_labels=class_labels,
                    context=(
                        f"seed={seed}, fold={fold_number}, candidate={candidate_name}, test"
                    ),
                )
                current_candidate_index = candidate_index[candidate_name]
                candidate_oof[
                    current_seed_index, current_candidate_index, valid_indices
                ] = valid_probability
                candidate_test_by_fold[
                    current_seed_index,
                    current_candidate_index,
                    fold_number - 1,
                ] = test_probability
                train_score, _, _ = _score_probability(
                    train_y.to_numpy(), train_probability, class_labels=class_labels
                )
                macro_f1, class_f1, _ = _score_probability(
                    valid_y, valid_probability, class_labels=class_labels
                )
                overfit_metrics = _fold_overfit_metrics(
                    train_score,
                    macro_f1,
                    threshold=overfit_threshold,
                )
                candidate_fold_results.append(
                    {
                        "record_type": "candidate",
                        "name": candidate_name,
                        "companion_blocks": list(blocks),
                        "seed": seed,
                        "fold": fold_number,
                        "macro_f1": macro_f1,
                        **overfit_metrics,
                        "feature_count": int(encoded_train_x.shape[1]),
                        "f9_feature_count": len(f9_feature_names),
                        "companion_suffix_feature_count": len(suffix_names),
                        "preprocessing_seconds": preprocessing_seconds,
                        "elapsed_seconds": time.perf_counter() - unit_started,
                        "class_f1": {
                            label: float(score)
                            for label, score in zip(class_labels, class_f1, strict=True)
                        },
                    }
                )
                completed_units.add(unit)
                _write_npz_atomic(checkpoint_npz, checkpoint_arrays)
                _write_json_atomic(
                    checkpoint_json,
                    {
                        "status": "running",
                        "contract_hash": contract_hash,
                        "checkpoint_npz_sha256": _sha256_file(checkpoint_npz),
                        "candidate_fold_results": candidate_fold_results,
                    },
                )
                if show_progress:
                    print(
                        json.dumps(
                            {
                                "completed": {
                                    "seed": seed,
                                    "fold": fold_number,
                                    "candidate": candidate_name,
                                    "macro_f1": macro_f1,
                                    "train_macro_f1": train_score,
                                    "train_validation_gap": overfit_metrics[
                                        "train_validation_gap"
                                    ],
                                    "is_overfitting": overfit_metrics[
                                        "is_overfitting"
                                    ],
                                    "feature_count": int(encoded_train_x.shape[1]),
                                    "companion_suffix_features": len(suffix_names),
                                    "preprocessing_seconds": round(
                                        preprocessing_seconds, 2
                                    ),
                                },
                                "progress": f"{len(completed_units)}/{training_count}",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                del (
                    encoded_train_x,
                    encoded_valid_x,
                    encoded_test_x,
                    model,
                    train_probability,
                    valid_probability,
                    test_probability,
                )
                gc.collect()
            del (
                f9,
                companion,
                f9_train,
                f9_valid,
                f9_test,
                companion_train,
                companion_valid,
                companion_test,
            )
            gc.collect()

    if len(completed_units) != training_count:
        raise RuntimeError("완료된 내부 결합 학습 수가 설정과 일치하지 않습니다.")
    if not np.isfinite(candidate_oof).all() or not np.isfinite(
        candidate_test_by_fold
    ).all():
        raise RuntimeError("완료 후 내부 결합 OOF/Test 확률에 빈 값이 있습니다.")
    candidate_test = candidate_test_by_fold.mean(axis=2, dtype=np.float64).astype(
        np.float32
    )

    fold_lookup = {
        (int(row["seed"]), int(row["fold"]), str(row["name"])): row
        for row in candidate_fold_results
    }
    score_rows: list[dict[str, object]] = []
    seed_results: list[dict[str, object]] = []
    for seed in seeds:
        current_seed_index = seed_index[seed]
        baseline_probability = candidate_oof[
            current_seed_index, candidate_index[baseline_candidate]
        ]
        baseline_score, baseline_class_f1, _ = _score_probability(
            label_values, baseline_probability, class_labels=class_labels
        )
        seed_entry: dict[str, object] = {
            "seed": seed,
            "baseline_candidate": baseline_candidate,
            "baseline_oof_macro_f1": baseline_score,
            "baseline_class_f1": {
                label: float(score)
                for label, score in zip(class_labels, baseline_class_f1, strict=True)
            },
            "candidates": {},
        }
        for candidate_name in enabled_candidates:
            probability = candidate_oof[
                current_seed_index, candidate_index[candidate_name]
            ]
            score, class_f1, _ = _score_probability(
                label_values, probability, class_labels=class_labels
            )
            seed_candidate_rows = [
                fold_lookup[(seed, fold_number, candidate_name)]
                for fold_number in range(1, n_splits + 1)
            ]
            seed_entry["candidates"][candidate_name] = {
                "oof_macro_f1": score,
                "delta_vs_baseline": score - baseline_score,
                "class_f1": {
                    label: float(value)
                    for label, value in zip(class_labels, class_f1, strict=True)
                },
                **_summarize_overfit_rows(
                    seed_candidate_rows,
                    threshold=overfit_threshold,
                ),
            }
            for fold_number in range(1, n_splits + 1):
                row = fold_lookup[(seed, fold_number, candidate_name)]
                baseline_row = fold_lookup[(seed, fold_number, baseline_candidate)]
                score_rows.append(
                    {
                        "record_type": "candidate",
                        "name": candidate_name,
                        "companion_blocks": "+".join(candidates[candidate_name]),
                        "seed": seed,
                        "fold": fold_number,
                        "macro_f1": row["macro_f1"],
                        "train_macro_f1": row["train_macro_f1"],
                        "validation_macro_f1": row["validation_macro_f1"],
                        "train_validation_gap": row["train_validation_gap"],
                        "overfit_threshold": row["overfit_threshold"],
                        "is_overfitting": row["is_overfitting"],
                        "baseline_macro_f1": baseline_row["macro_f1"],
                        "delta_vs_baseline": (
                            float(row["macro_f1"]) - float(baseline_row["macro_f1"])
                        ),
                        "feature_count": row["feature_count"],
                        "f9_feature_count": row["f9_feature_count"],
                        "companion_suffix_feature_count": row[
                            "companion_suffix_feature_count"
                        ],
                        "preprocessing_seconds": row["preprocessing_seconds"],
                        "elapsed_seconds": row["elapsed_seconds"],
                    }
                )
        seed_results.append(seed_entry)

    baseline_scores = np.asarray(
        [row["baseline_oof_macro_f1"] for row in seed_results], dtype=np.float64
    )
    summary_candidates: dict[str, object] = {}
    for candidate_name in enabled_candidates:
        scores = np.asarray(
            [
                row["candidates"][candidate_name]["oof_macro_f1"]
                for row in seed_results
            ],
            dtype=np.float64,
        )
        deltas = scores - baseline_scores
        candidate_rows = [
            row for row in score_rows if row["name"] == candidate_name
        ]
        fold_deltas = np.asarray(
            [row["delta_vs_baseline"] for row in candidate_rows], dtype=np.float64
        )
        summary_candidates[candidate_name] = {
            "companion_blocks": list(candidates[candidate_name]),
            "mean_seed_oof_macro_f1": float(scores.mean()),
            "seed_oof_std": float(scores.std(ddof=0)),
            "mean_delta_vs_baseline": float(deltas.mean()),
            "min_seed_delta_vs_baseline": float(deltas.min()),
            "improved_seed_count": int((deltas > 0.0).sum()),
            "won_fold_count": int((fold_deltas > 0.0).sum()),
            "total_fold_count": int(len(fold_deltas)),
            **_summarize_overfit_rows(
                candidate_rows,
                threshold=overfit_threshold,
            ),
            "feature_count_min": int(
                min(int(row["feature_count"]) for row in candidate_rows)
            ),
            "feature_count_max": int(
                max(int(row["feature_count"]) for row in candidate_rows)
            ),
        }
    nonbaseline_candidates = [
        name for name in enabled_candidates if name != baseline_candidate
    ]
    best_nonbaseline_candidate = max(
        nonbaseline_candidates,
        key=lambda name: float(summary_candidates[name]["mean_seed_oof_macro_f1"]),
    )
    best_overall_candidate = max(
        enabled_candidates,
        key=lambda name: float(summary_candidates[name]["mean_seed_oof_macro_f1"]),
    )
    promotion_candidate = (
        best_nonbaseline_candidate
        if float(
            summary_candidates[best_nonbaseline_candidate]["mean_delta_vs_baseline"]
        ) > 0.0
        else None
    )
    summary = {
        "baseline_candidate": baseline_candidate,
        "baseline_mean_seed_oof_macro_f1": float(baseline_scores.mean()),
        "baseline_seed_oof_std": float(baseline_scores.std(ddof=0)),
        "overfit_threshold": overfit_threshold,
        "baseline_is_overfitting": summary_candidates[baseline_candidate][
            "is_overfitting"
        ],
        "best_candidate": best_overall_candidate,
        "best_candidate_mean_seed_oof_macro_f1": summary_candidates[best_overall_candidate][
            "mean_seed_oof_macro_f1"
        ],
        "best_candidate_mean_delta_vs_baseline": summary_candidates[best_overall_candidate][
            "mean_delta_vs_baseline"
        ],
        "best_nonbaseline_candidate": best_nonbaseline_candidate,
        "best_nonbaseline_mean_seed_oof_macro_f1": summary_candidates[
            best_nonbaseline_candidate
        ]["mean_seed_oof_macro_f1"],
        "best_nonbaseline_mean_delta_vs_baseline": summary_candidates[
            best_nonbaseline_candidate
        ]["mean_delta_vs_baseline"],
        "promotion_candidate": promotion_candidate,
        "candidates": summary_candidates,
        "submission_policy": submission_policy,
        "submission_files_created": submission_policy == "write_local_candidates",
        "elapsed_seconds": float(time.perf_counter() - started_all),
    }

    probability_arrays: dict[str, np.ndarray] = {
        "classes": class_labels.astype(str),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "train_ids": train_ids.to_numpy(dtype=str),
        "test_ids": test_ids.to_numpy(dtype=str),
        "fold_ids": fold_ids.astype(np.int16),
    }
    for candidate_name in enabled_candidates:
        token = _artifact_token(candidate_name)
        probability_arrays[f"candidate_oof__{token}"] = candidate_oof[
            :, candidate_index[candidate_name]
        ]
        probability_arrays[f"candidate_test__{token}"] = candidate_test[
            :, candidate_index[candidate_name]
        ]
    _write_npz_atomic(probabilities_npz, probability_arrays)
    pd.DataFrame(score_rows).sort_values(
        ["seed", "fold", "name"]
    ).to_csv(scores_csv, index=False, encoding="utf-8-sig")

    written_submissions: dict[str, str] = {}
    if submission_policy == "write_local_candidates":
        for candidate_name, output in submission_outputs.items():
            mean_test_probability = candidate_test[
                :, candidate_index[candidate_name]
            ].mean(axis=0, dtype=np.float64)
            predictions = class_labels[mean_test_probability.argmax(axis=1)]
            submission = sample_submission.copy()
            submission[target_column] = predictions
            submission.to_csv(output, index=False, encoding="utf-8-sig")
            written_submissions[candidate_name] = str(output.relative_to(project_root))

    artifact_hashes = {
        "scores_csv": _sha256_file(scores_csv),
        "probabilities_npz": _sha256_file(probabilities_npz),
        "submissions": {
            name: _sha256_file(submission_outputs[name])
            for name in written_submissions
        },
    }

    result = {
        "status": "complete",
        "mode": "xgboost_internal_feature_fusion",
        "artifact_schema_version": XGB_INTERNAL_FUSION_ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "contract_hash": contract_hash,
        "data_hashes": data_hashes,
        "source_hashes": source_hashes,
        "runtime_versions": runtime_versions,
        "class_order": class_labels.tolist(),
        "class_order_source": "sorted original train.csv SUBCLASS labels",
        "label_remapping": False,
        "model": model_config,
        "f9_preprocessing": f9_config,
        "companion_name": companion_name,
        "companion_preprocessing": companion_config,
        "candidates": {
            name: {"companion_blocks": list(candidates[name])}
            for name in enabled_candidates
        },
        "baseline_candidate": baseline_candidate,
        "overfit_threshold": overfit_threshold,
        "seeds": list(seeds),
        "n_splits": n_splits,
        "training_count": training_count,
        "preprocessing_fit_count": preprocessing_fit_count,
        "splitter": "StratifiedKFold(shuffle=True, random_state=seed)",
        "split_hashes": split_hashes,
        "summary": summary,
        "seed_results": seed_results,
        "artifacts": {
            "result_json": str(output_json.relative_to(project_root)),
            "scores_csv": str(scores_csv.relative_to(project_root)),
            "probabilities_npz": str(probabilities_npz.relative_to(project_root)),
            "submissions": written_submissions,
        },
        "artifact_hashes": artifact_hashes,
    }
    _write_json_atomic(output_json, result)
    for path in (checkpoint_json, checkpoint_npz):
        if path.exists():
            path.unlink()
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_json.relative_to(project_root)),
                "summary": summary,
            },
            ensure_ascii=False,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "전처리 안정성, 동일 XGBoost 확률 결합 또는 내부 피처 결합을 검증합니다."
        )
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
    if "xgb_internal_fusion_validation" in config:
        run_xgboost_internal_fusion_validation(
            config,
            project_root=project_root,
            force=args.force,
            dry_run=args.dry_run,
            quiet=args.quiet,
        )
    elif "xgb_blend_validation" in config:
        run_xgboost_blend_validation(
            config,
            project_root=project_root,
            force=args.force,
            dry_run=args.dry_run,
            quiet=args.quiet,
        )
    else:
        run_stability_validation(
            config,
            project_root=project_root,
            force=args.force,
            dry_run=args.dry_run,
            quiet=args.quiet,
        )


if __name__ == "__main__":
    main()
