"""Build the compact probability bundle used by TEST_004_8 and TEST_004_9.

The source arrays are fold-safe OOF/test predictions produced only from the
competition data.  The compact bundle removes unrelated intermediate arrays so
the two historical submissions can be reproduced without the full experiment
cache directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "TEST_004"
BUNDLE = MODEL_DIR / "vote_select_inputs.npz"
MANIFEST = MODEL_DIR / "vote_select_inputs.json"


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 1e-7, None)
    return values / values.sum(axis=1, keepdims=True)


def id_hash(frame: pd.DataFrame) -> str:
    payload = "\n".join(frame["ID"].astype(str)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    cache = np.load(PROCESSED / "test_004_6_ensemble_inputs.npz")
    multi_v8 = np.load(PROCESSED / "JSJ_MultiPair_v8_predictions.npz")
    multi_v9 = np.load(PROCESSED / "JSJ_MultiPair_v9_predictions.npz")
    full_v6 = np.load(PROCESSED / "JSJ_FullMeta_v6_probabilities.npz")
    full_v7 = np.load(PROCESSED / "JSJ_FullMeta_v7_probabilities.npz")
    lgb = np.load(PROCESSED / "JSJ_LGBMeta_v1_probabilities.npz")
    cpem = np.load(PROCESSED / "JSJ_CPEMProxy_v1_probabilities.npz")
    train = pd.read_csv(RAW / "train.csv", usecols=["ID"])
    test = pd.read_csv(RAW / "test.csv", usecols=["ID"])

    classes = cache["classes"].astype(str)
    y = cache["y"].astype(np.int16)
    folds = cache["fold_ids"].astype(np.int8)
    if not (
        np.array_equal(y, multi_v8["y"])
        and np.array_equal(y, multi_v9["y"])
        and np.array_equal(folds, multi_v8["fold_ids"])
        and np.array_equal(folds, multi_v9["fold_ids"])
        and np.array_equal(classes, multi_v8["classes"].astype(str))
        and np.array_equal(classes, multi_v9["classes"].astype(str))
    ):
        raise ValueError("OOF cache labels, folds, or class order do not match")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {
        "y": y,
        "fold_ids": folds,
        "classes": classes,
        "v1_base_oof": multi_v8["oof_labels"].astype(np.int16),
        "v1_base_test": multi_v8["test_labels"].astype(np.int16),
        "v2_base_oof": multi_v9["oof_labels"].astype(np.int16),
        "v2_base_test": multi_v9["test_labels"].astype(np.int16),
        "full_v6_oof": normalize(full_v6["oof"]),
        "full_v6_test": normalize(full_v6["test"]),
        "full_v7_oof": normalize(full_v7["oof"]),
        "full_v7_test": normalize(full_v7["test"]),
        "lgb_oof": normalize(lgb["best_oof"]),
        "lgb_test": normalize(lgb["best_test"]),
        "corrected_oof": normalize(cache["corrected_oof"]),
        "corrected_test": normalize(cache["corrected_test"]),
        "cpem_oof": normalize(cpem["f003_leaf1_oof"]),
        "cpem_test": normalize(cpem["f003_leaf1_test"]),
    }
    np.savez_compressed(BUNDLE, **arrays)
    digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    manifest = {
        "artifact": str(BUNDLE.relative_to(ROOT)),
        "sha256": digest,
        "train_rows": len(train),
        "test_rows": len(test),
        "class_count": len(classes),
        "train_id_sha256": id_hash(train),
        "test_id_sha256": id_hash(test),
        "external_data_used": False,
        "test_labels_used": False,
        "description": "Compact fold-safe OOF/test probability inputs for TEST_004_8 and TEST_004_9",
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
