"""TEST_004_7을 원자료부터 재현하고 제출 CSV를 생성합니다.

실행: python -m src.reproduce_test_004_7 --config configs/test_004_7.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import yaml

from src.models.baseline.TEST_004_7 import SPEC
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline
from src.reproduce_test_004 import dataset_signature


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def pair_right(probabilities, left, right, temperature, offset):
    left_probability = np.clip(probabilities[:, left], 1e-12, 1.0)
    right_probability = np.clip(probabilities[:, right], 1e-12, 1.0)
    log_odds = (np.log(right_probability) - np.log(left_probability)) / temperature
    return expit(log_odds - offset)


def balanced_pair_prediction(base, cache, team_k, team_g, indices, settings):
    adjusted = base.copy()
    initial = base.argmax(axis=1)
    k_left, k_right = indices["KIRC_KIPAN"]
    g_left, g_right = indices["LGG_GBMLGG"]

    active = (initial == k_left) | (initial == k_right)
    total = base[active, k_left] + base[active, k_right]
    base_right = base[active, k_right] / np.maximum(total, 1e-12)
    weight = settings["KIRC_KIPAN"]["weight"]
    right = (
        (0.60 - weight) * base_right
        + 0.10 * cache["k_specialist"][active]
        + 0.30 * cache["em14_k"][active]
        + weight * team_k[active]
    )
    adjusted[active, k_left] = total * (1.0 - right)
    adjusted[active, k_right] = total * right

    active = (initial == g_left) | (initial == g_right)
    total = base[active, g_left] + base[active, g_right]
    weight = settings["LGG_GBMLGG"]["weight"]
    right = (
        (1.0 - weight) * cache["g_specialist"][active]
        + weight * team_g[active]
    )
    adjusted[active, g_left] = total * (1.0 - right)
    adjusted[active, g_right] = total * right
    return adjusted.argmax(axis=1)


def ensure_base_cache(root: Path, config: dict, output: Path) -> Path:
    path = output / "test_004_6_ensemble_inputs.npz"
    if path.exists():
        return path
    command = [
        sys.executable,
        "-m",
        "src.reproduce_test_004",
        "--config",
        config["model"]["base_config"],
    ]
    print("TEST_004_6 기반 확률 캐시가 없어 먼저 생성합니다.", flush=True)
    subprocess.run(command, cwd=root, check=True)
    if not path.exists():
        raise FileNotFoundError(f"기반 확률 캐시가 생성되지 않았습니다: {path}")
    return path


def train_team_pipeline(
    pipeline_name,
    parameters,
    features,
    test_features,
    labels,
    y,
    classes,
    splits,
    signature,
    output,
):
    cache_path = output / f"test_004_7_{pipeline_name}_e4.npz"
    meta_path = output / f"test_004_7_{pipeline_name}_e4.json"
    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("dataset_signature") == signature and meta.get("parameters") == parameters:
            print(f"{pipeline_name} E4 캐시를 사용합니다.", flush=True)
            return np.load(cache_path)

    oof = np.zeros((len(features), len(classes)), dtype=np.float32)
    test_probability = np.zeros((len(test_features), len(classes)), dtype=np.float32)
    fold_scores = []
    started = time.time()
    for fold, (train_index, valid_index) in enumerate(splits, 1):
        print(f"{pipeline_name} fold={fold}/{len(splits)}", flush=True)
        pipeline = create_preprocessing_pipeline({"name": pipeline_name})
        fold_train = pipeline.fit_transform(features.iloc[train_index], labels.iloc[train_index])
        fold_valid = pipeline.transform(features.iloc[valid_index])
        fold_test = pipeline.transform(test_features)
        model = XGBClassifier(random_state=42, n_jobs=-1, **parameters)
        model.fit(
            fold_train,
            y[train_index],
            eval_set=[(fold_valid, y[valid_index])],
            verbose=False,
        )
        oof[valid_index] = model.predict_proba(fold_valid)
        test_probability += model.predict_proba(fold_test) / len(splits)
        fold_score = f1_score(y[valid_index], oof[valid_index].argmax(axis=1), average="macro")
        fold_scores.append(float(fold_score))
        del pipeline, fold_train, fold_valid, fold_test, model
        gc.collect()

    np.savez_compressed(cache_path, oof=oof, test=test_probability, classes=classes)
    meta_path.write_text(
        json.dumps(
            {
                "pipeline": pipeline_name,
                "model": "xgboost_e4",
                "dataset_signature": signature,
                "parameters": parameters,
                "oof_macro_f1": float(f1_score(y, oof.argmax(axis=1), average="macro")),
                "fold_macro_f1": fold_scores,
                "elapsed_seconds": time.time() - started,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return np.load(cache_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/test_004_7.yaml"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config = load_yaml(args.config)
    if config["model"]["spec"] != SPEC["name"]:
        raise ValueError("config와 TEST_004_7 모델 사양이 일치하지 않습니다.")

    pipeline = create_preprocessing_pipeline(dict(config["preprocessing"]))
    if pipeline.team_experts["pipelines"] != config["model"]["team_pipelines"]:
        raise ValueError("pipeline_jsj_v9과 config의 팀 파이프라인 가중치가 다릅니다.")
    if args.check_only:
        print("TEST_004_7 설정·모델·pipeline_jsj_v9 연결 확인 완료")
        return

    data = config["data"]
    raw = root / data["raw_dir"]
    output = root / data["processed_dir"]
    output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(raw / data["train_file"])
    test = pd.read_csv(raw / data["test_file"])
    submission = pd.read_csv(raw / data["submission_file"])
    target, identifier = data["target_column"], data["id_column"]
    genes = [column for column in train.columns if column not in {target, identifier}]
    features, test_features = train[genes], test[genes]
    labels = train[target].astype(str)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = encoder.classes_.astype(str)
    signature = dataset_signature(train, test, genes)
    base_cache = np.load(ensure_base_cache(root, config, output))
    if str(base_cache["dataset_signature"].item()) != signature:
        raise ValueError("TEST_004_6 캐시와 현재 데이터가 일치하지 않습니다. 캐시를 삭제 후 재실행하세요.")
    if not np.array_equal(base_cache["classes"].astype(str), classes):
        raise ValueError("기반 모델과 현재 데이터의 클래스 순서가 다릅니다.")

    splitter = list(
        StratifiedKFold(
            n_splits=config["model"]["n_splits"],
            shuffle=True,
            random_state=config["project"]["seed"],
        ).split(features, y)
    )
    parameters = dict(config["model"]["xgboost"])
    team = {
        name: train_team_pipeline(
            name, parameters, features, test_features, labels, y, classes,
            splitter, signature, output,
        )
        for name in config["model"]["team_pipelines"]
    }
    weights = config["model"]["team_pipelines"]
    team_oof = sum(weights[name] * team[name]["oof"] for name in weights)
    team_test = sum(weights[name] * team[name]["test"] for name in weights)
    class_index = {name: index for index, name in enumerate(classes)}
    pairs = {
        "KIRC_KIPAN": (class_index["KIRC"], class_index["KIPAN"]),
        "LGG_GBMLGG": (class_index["LGG"], class_index["GBMLGG"]),
    }
    settings = config["model"]["pair_experts"]
    temperature = settings["temperature"]
    k_oof = pair_right(team_oof, *pairs["KIRC_KIPAN"], temperature, settings["KIRC_KIPAN"]["right_offset"])
    k_test = pair_right(team_test, *pairs["KIRC_KIPAN"], temperature, settings["KIRC_KIPAN"]["right_offset"])
    g_oof = pair_right(team_oof, *pairs["LGG_GBMLGG"], temperature, settings["LGG_GBMLGG"]["right_offset"])
    g_test = pair_right(team_test, *pairs["LGG_GBMLGG"], temperature, settings["LGG_GBMLGG"]["right_offset"])
    oof_cache = {
        "k_specialist": base_cache["k_specialist_oof"],
        "g_specialist": base_cache["g_specialist_oof"],
        "em14_k": base_cache["em14_k_oof"],
    }
    test_cache = {
        "k_specialist": base_cache["k_specialist_test"],
        "g_specialist": base_cache["g_specialist_test"],
        "em14_k": base_cache["em14_k_test"],
    }
    oof_prediction = balanced_pair_prediction(base_cache["corrected_oof"], oof_cache, k_oof, g_oof, pairs, settings)
    test_prediction = balanced_pair_prediction(base_cache["corrected_test"], test_cache, k_test, g_test, pairs, settings)
    fold_ids = base_cache["fold_ids"]
    fold_scores = [
        f1_score(y[fold_ids == fold], oof_prediction[fold_ids == fold], average="macro")
        for fold in range(config["model"]["n_splits"])
    ]
    score = f1_score(y, oof_prediction, average="macro")
    submission[target] = encoder.inverse_transform(test_prediction)
    submission_path = output / "TEST_004_7_submission.csv"
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")
    result = {
        "experiment": "TEST_004_7",
        "pipeline": "pipeline_jsj_v9.py",
        "oof_macro_f1": float(score),
        "fold_macro_f1": [float(value) for value in fold_scores],
        "historical_oof_macro_f1": SPEC["historical_oof_macro_f1"],
        "public_macro_f1": None,
        "submission": str(submission_path),
        "external_data_used": False,
        "test_used_for_fit": False,
    }
    (output / "TEST_004_7_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
