"""TEST_004_1~4를 원본 CSV에서 재학습하여 제출 파일로 재현합니다.

Public 점수는 Dacon의 비공개 정답으로만 계산되므로 이 실행기는 5-Fold OOF,
예측 CSV와 재현 메타데이터를 생성합니다. 모든 전처리는 Fold 학습 데이터에서만
fit하며 test.csv는 transform/predict에만 사용합니다.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import yaml

from src.pipelines.preprocessing_registry import create_preprocessing_pipeline
from src.pipelines.pipeline_jsj_v3 import add_e1_features, make_tree_features


CONFLICT_PAIRS = (("KIRC", "KIPAN"), ("LGG", "GBMLGG"))


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_spec(name: str) -> dict:
    module = importlib.import_module(f"src.models.baseline.{name}")
    return dict(module.SPEC)


def make_tree_model(seed: int, e1: bool, quick: bool):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise ImportError("`pip install -r requirements.txt`를 먼저 실행하세요.") from error
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=30 if quick else 1800,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=18,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.15,
        reg_lambda=0.50,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def balanced_sample_weights(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    mapping = {
        label: len(labels) / (len(unique_labels) * count)
        for label, count in zip(unique_labels, counts)
    }
    return np.asarray([mapping[label] for label in labels], dtype=np.float32)


def dataset_signature(train: pd.DataFrame, test: pd.DataFrame, genes: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(genes).encode("utf-8"))
    digest.update(train["ID"].astype(str).str.cat(sep="\n").encode("utf-8"))
    digest.update(test["ID"].astype(str).str.cat(sep="\n").encode("utf-8"))
    return digest.hexdigest()


def redistribute(scores, pair_probabilities, pair_indices, strategy):
    mode = strategy["mode"]
    if mode == "none":
        return scores.argmax(axis=1)
    adjusted = scores.copy()
    initial_top = scores.argmax(axis=1)
    strict_prediction = initial_top.copy()
    for pair_name, (left, right) in pair_indices.items():
        weight = float(strategy[pair_name])
        if weight <= 0:
            continue
        active = (initial_top == left) | (initial_top == right)
        if not np.any(active):
            continue
        pair_total = scores[active, left] + scores[active, right]
        base_right = scores[active, right] / np.maximum(pair_total, 1e-12)
        right_share = (
            (1.0 - weight) * base_right
            + weight * pair_probabilities[pair_name][active]
        )
        adjusted[active, left] = pair_total * (1.0 - right_share)
        adjusted[active, right] = pair_total * right_share
        if mode == "strict":
            strict_prediction[active] = np.where(right_share >= 0.5, right, left)
    return strict_prediction if mode == "strict" else adjusted.argmax(axis=1)


def validate_pipeline_strategy(pipeline, spec):
    if spec["postprocessing"]["mode"] == "none":
        return
    if getattr(pipeline, "conflict_mode", None) != spec["postprocessing"]["mode"]:
        raise ValueError("pipeline의 충돌 보정 방식과 model spec이 일치하지 않습니다.")
    if getattr(pipeline, "conflict_weights", None) != {
        key: spec["postprocessing"][key]
        for key in ("KIRC_KIPAN", "LGG_GBMLGG")
    }:
        raise ValueError("pipeline의 충돌 보정 가중치와 model spec이 일치하지 않습니다.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="구조/실행 확인용 저비용 설정입니다. 기록 점수 비교에는 사용하지 않습니다.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="공통 LightGBM OOF 캐시가 있어도 다시 학습합니다.",
    )
    args = parser.parse_args()
    started = time.time()
    config = load_config(args.config)
    spec = load_spec(config["model"]["spec"])
    if spec["name"].lower() != config["project"]["experiment_name"].lower():
        raise ValueError("config의 experiment_name과 model spec 이름이 다릅니다.")

    data = config["data"]
    raw_dir = Path(data["raw_dir"])
    output_dir = Path(data["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(raw_dir / data["train_file"])
    test = pd.read_csv(raw_dir / data["test_file"])
    submission = pd.read_csv(raw_dir / data["submission_file"])
    target = data["target_column"]
    identifier = data["id_column"]
    genes = [column for column in train.columns if column not in {target, identifier}]
    if genes != [column for column in test.columns if column != identifier]:
        raise ValueError("train/test 유전자 열과 순서가 일치하지 않습니다.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(train[target].astype(str))
    classes = encoder.classes_
    n_classes = len(classes)
    class_to_index = {name: index for index, name in enumerate(classes)}
    pair_indices = {
        f"{left}_{right}": (class_to_index[left], class_to_index[right])
        for left, right in CONFLICT_PAIRS
    }
    features = train[genes]
    test_features = test[genes]
    n_splits = 2 if args.quick else int(config["model"].get("n_splits", 5))
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config["project"]["seed"]
    )
    fold_ids = np.full(len(train), -1, dtype=np.int8)
    svm_oof_scores = np.zeros((len(train), n_classes), dtype=np.float32)
    svm_test_scores = np.zeros((len(test), n_classes), dtype=np.float32)
    base_tree_oof = np.zeros((len(train), n_classes), dtype=np.float32)
    base_tree_test = np.zeros((len(test), n_classes), dtype=np.float32)
    e1_tree_oof = np.zeros((len(train), n_classes), dtype=np.float32)
    e1_tree_test = np.zeros((len(test), n_classes), dtype=np.float32)
    specialist_oof = {
        pair_name: np.zeros(len(train), dtype=np.float32) for pair_name in pair_indices
    }
    specialist_test = {
        pair_name: np.zeros(len(test), dtype=np.float32) for pair_name in pair_indices
    }

    signature = dataset_signature(train, test, genes)
    cache_path = output_dir / "test_004_tree_probabilities.npz"
    cache_meta_path = output_dir / "test_004_tree_probabilities.json"
    cache_valid = False
    if not args.quick and cache_path.exists() and cache_meta_path.exists():
        cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        cache_valid = (
            cache_meta.get("dataset_signature") == signature
            and cache_meta.get("classes") == classes.tolist()
        )
    if cache_valid and not args.refresh_cache:
        cache = np.load(cache_path)
        base_tree_oof[:] = cache["base_tree_oof"]
        base_tree_test[:] = cache["base_tree_test"]
        e1_tree_oof[:] = cache["e1_tree_oof"]
        e1_tree_test[:] = cache["e1_tree_test"]
        print("공통 LightGBM OOF 캐시를 불러왔습니다.", flush=True)
    else:
        from lightgbm import early_stopping, log_evaluation

        print("원 제출의 base/E1 LightGBM 피처를 생성합니다.", flush=True)
        tree_base = make_tree_features(features)
        tree_base_test = make_tree_features(test_features)
        tree_e1 = add_e1_features(tree_base)
        tree_e1_test = add_e1_features(tree_base_test)
        base_columns = tree_base.columns[tree_base.nunique(dropna=False) > 1]
        e1_columns = tree_e1.columns[tree_e1.nunique(dropna=False) > 1]
        tree_base = tree_base[base_columns]
        tree_base_test = tree_base_test[base_columns]
        tree_e1 = tree_e1[e1_columns]
        tree_e1_test = tree_e1_test[e1_columns]
        tree_seeds = (42,) if args.quick else (42, 2026)
        for tree_name, tree_train, tree_test, oof_store, test_store in (
            ("base", tree_base, tree_base_test, base_tree_oof, base_tree_test),
            ("e1", tree_e1, tree_e1_test, e1_tree_oof, e1_tree_test),
        ):
            for seed in tree_seeds:
                tree_splitter = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=seed
                )
                for fold, (train_index, valid_index) in enumerate(
                    tree_splitter.split(tree_train, y), 1
                ):
                    print(
                        f"{tree_name} LightGBM seed={seed} fold={fold}/{n_splits}",
                        flush=True,
                    )
                    model = make_tree_model(
                        seed + fold, e1=tree_name == "e1", quick=args.quick
                    )
                    fit_kwargs = {
                        "sample_weight": balanced_sample_weights(y[train_index]),
                    }
                    if not args.quick:
                        fit_kwargs.update(
                            {
                                "eval_set": [(tree_train.iloc[valid_index], y[valid_index])],
                                "eval_metric": "multi_logloss",
                                "callbacks": [
                                    early_stopping(150, verbose=False),
                                    log_evaluation(0),
                                ],
                            }
                        )
                    model.fit(tree_train.iloc[train_index], y[train_index], **fit_kwargs)
                    oof_store[valid_index] += (
                        model.predict_proba(tree_train.iloc[valid_index])
                        / len(tree_seeds)
                    )
                    test_store += (
                        model.predict_proba(tree_test)
                        / (n_splits * len(tree_seeds))
                    )
                    del model
                    gc.collect()
        if not args.quick:
            np.savez_compressed(
                cache_path,
                base_tree_oof=base_tree_oof,
                base_tree_test=base_tree_test,
                e1_tree_oof=e1_tree_oof,
                e1_tree_test=e1_tree_test,
            )
            cache_meta_path.write_text(
                json.dumps(
                    {
                        "dataset_signature": signature,
                        "classes": classes.tolist(),
                        "base_feature_count": len(base_columns),
                        "e1_feature_count": len(e1_columns),
                        "seeds": list(tree_seeds),
                        "n_splits": n_splits,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        del tree_base, tree_base_test, tree_e1, tree_e1_test
        gc.collect()

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(np.zeros(len(y)), y), 1
    ):
        print(f"fold={fold}/{n_splits} 시작", flush=True)
        fold_ids[valid_index] = fold - 1
        pipeline_config = dict(config["preprocessing"])
        if args.quick:
            pipeline_config["word_max_features"] = 5_000
            pipeline_config["char_max_features"] = 5_000
        pipeline = create_preprocessing_pipeline(pipeline_config)
        train_bundle = pipeline.fit_transform(
            features.iloc[train_index], train[target].iloc[train_index]
        )
        valid_bundle = pipeline.transform(features.iloc[valid_index])
        test_bundle = pipeline.transform(test_features)
        validate_pipeline_strategy(pipeline, spec)

        svm = LinearSVC(
            C=spec["linear_c"], class_weight="balanced", random_state=42
        )
        svm.fit(train_bundle.text, y[train_index])
        svm_oof_scores[valid_index] = svm.decision_function(valid_bundle.text)
        svm_test_scores += svm.decision_function(test_bundle.text) / n_splits

        fold_y = y[train_index]
        for pair_name, (left, right) in pair_indices.items():
            pair_mask = (fold_y == left) | (fold_y == right)
            specialist = LinearSVC(C=0.2, class_weight="balanced", random_state=42)
            specialist.fit(
                train_bundle.text[pair_mask], (fold_y[pair_mask] == right).astype(int)
            )
            specialist_oof[pair_name][valid_index] = expit(
                specialist.decision_function(valid_bundle.text)
            )
            specialist_test[pair_name] += expit(
                specialist.decision_function(test_bundle.text)
            ) / n_splits

        del (
            pipeline,
            train_bundle,
            valid_bundle,
            test_bundle,
            svm,
        )
        gc.collect()

    svm_oof = softmax(svm_oof_scores / spec["temperature"], axis=1)
    svm_test = softmax(svm_test_scores / spec["temperature"], axis=1)
    base_oof = (
        spec["linear_weight"] * svm_oof
        + spec["base_tree_weight"] * base_tree_oof
        + spec["e1_tree_weight"] * e1_tree_oof
    )
    base_test = (
        spec["linear_weight"] * svm_test
        + spec["base_tree_weight"] * base_tree_test
        + spec["e1_tree_weight"] * e1_tree_test
    )
    multipliers = np.asarray(spec["class_multipliers"], dtype=np.float64)
    corrected_oof = base_oof * multipliers
    corrected_test = base_test * multipliers
    oof_prediction = redistribute(
        corrected_oof, specialist_oof, pair_indices, spec["postprocessing"]
    )
    test_prediction = redistribute(
        corrected_test, specialist_test, pair_indices, spec["postprocessing"]
    )
    fold_scores = [
        f1_score(y[fold_ids == fold], oof_prediction[fold_ids == fold], average="macro")
        for fold in range(n_splits)
    ]
    oof_score = f1_score(y, oof_prediction, average="macro")
    experiment_name = config["project"]["experiment_name"]
    submission[target] = encoder.inverse_transform(test_prediction)
    submission_path = output_dir / f"{experiment_name}_submission.csv"
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")
    result = {
        "experiment": experiment_name,
        "config": str(args.config),
        "pipeline": config["preprocessing"]["name"],
        "model_spec": config["model"]["spec"],
        "quick_mode": args.quick,
        "oof_macro_f1": float(oof_score),
        "fold_macro_f1": [float(score) for score in fold_scores],
        "historical_oof_macro_f1": spec["historical_oof_macro_f1"],
        "historical_public_macro_f1": spec["historical_public_macro_f1"],
        "submission": str(submission_path),
        "external_data_used": False,
        "test_used_for_fit": False,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / f"{experiment_name}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
