from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.test_007.evaluate_test_007_specialization as evaluator


CLASSES = [f"C{i:02d}" for i in range(26)]
CANDIDATES = ["nested_primary", "nested_robustness", "nested_specialization"]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_integrity(path: Path, payload: dict) -> None:
    _write_json(path, {"payload": payload, "sha256": evaluator._canonical_hash(payload)})


def _uniform_probability() -> np.ndarray:
    return np.full((3, 6201, 26), 1.0 / 26.0, dtype=np.float32)


def _make_complete(root: Path, role_to_lane: dict[str, str] | None = None) -> Path:
    root.mkdir()
    role_to_lane = role_to_lane or {role: role for role in evaluator.FINAL_ROLES}
    selections = {
        role: evaluator.LANE_CANDIDATE_NAMES[lane]
        for role, lane in role_to_lane.items()
    }
    evaluator.CANONICAL_TEST = root / "test.csv"
    pd.DataFrame({"ID": ["R0000"], "x": [1]}).to_csv(evaluator.CANONICAL_TEST, index=False)
    evaluator.CANONICAL_SUBMISSION = root / "sample_submission.csv"
    evaluator.CANONICAL_SUBMISSION.write_text("ID,SUBCLASS\n", encoding="utf-8")
    evaluator.CANONICAL_TRAIN = root / "train.csv"
    pd.DataFrame({"ID": ["A"], "SUBCLASS": ["C00"], "x": [0]}).to_csv(evaluator.CANONICAL_TRAIN, index=False)
    digest = "a" * 64
    protected_dir = root / "protected"; protected_dir.mkdir()
    protected_rows = []
    for index in range(69):
        protected_path = protected_dir / f"file_{index}.py"
        protected_path.write_text(f"# {index}\n", encoding="utf-8")
        current = evaluator._sha256(protected_path)
        protected_rows.append({"path": str(protected_path), "before": current, "after": current, "unchanged": True})
    _write_json(
        root / "protected_file_hashes_before_after.json",
        {
            "unchanged": True,
            "files": protected_rows,
        },
    )
    _write_json(root / "class_names.json", CLASSES)

    assignments = pd.concat(
        [
            pd.DataFrame(
                {
                    "seed": seed,
                    "row_index": np.arange(6201),
                    "ID": [f"R{index:04d}" for index in range(6201)],
                    "fold": np.arange(6201) % 5,
                }
            )
            for seed in evaluator.EXPECTED_SEEDS
        ],
        ignore_index=True,
    )
    assignments.to_csv(root / "fold_assignments.csv", index=False)
    _write_json(
        root / "io_audit.json",
        {
            "events": [
                {"stage": "nested_oof", "action": "read", "path": "train.csv", "evidence_kind": "verified"}
            ],
            "summary": {
                "oof_test_reads": 0,
                "nested_test_reads": 0,
                "oof_submission_reads": 0,
                "nested_submission_reads": 0,
            },
        },
    )

    np.save(root / "candidate_probability.npy", _uniform_probability())
    assignments.to_csv(root / "candidate_alignment.csv", index=False)
    _write_json(
        root / "candidate_manifest.json",
        {
            "candidates": [
                {
                    "name": name,
                    "probability_path": "candidate_probability.npy",
                    "alignment_path": "candidate_alignment.csv",
                    "class_names": CLASSES,
                    "probability_sha256": evaluator._sha256(root / "candidate_probability.npy"),
                    "alignment_sha256": evaluator._sha256(root / "candidate_alignment.csv"),
                }
                for name in CANDIDATES
            ],
        },
    )
    np.save(root / "nested_oof_probability.npy", _uniform_probability())
    _write_json(
        root / "nested_cv_metrics.json",
        {
            "outer_evaluations": [
                {"seed": seed, "fold": fold, "macro_f1": 0.5}
                for seed in evaluator.EXPECTED_SEEDS
                for fold in evaluator.EXPECTED_FOLDS
            ]
        },
    )
    _write_json(
        root / "class_metrics.json",
        {"classes": [{"class_name": name, "macro_f1": 0.5} for name in CLASSES]},
    )
    _write_json(
        root / "subgroup_metrics.json",
        {
            "subgroups": [
                {"dimension": "burden", "subgroup": "low", "macro_f1": 0.5},
                {"dimension": "novelty", "subgroup": "known", "macro_f1": 0.5},
            ]
        },
    )
    _write_json(
        root / "paired_comparisons.json",
        {
            "comparisons": [
                {
                    "candidate": "nested_specialization",
                    "baseline": "nested_primary",
                    "delta": 0.001,
                    "se_seed": 0.002,
                    "se_boot": 0.003,
                    "epsilon": 0.003,
                    "ci_lower": -0.002,
                    "ci_upper": 0.004,
                }
            ]
        },
    )
    _write_json(
        root / "collapse_constraints.json",
        {
            "overall_pass": True,
            "constraints": {
                name: {"pass": True, "repeated_collapse_count": 0}
                for name in ("seed", "class", "burden", "novelty")
            },
            "by_candidate": {
                candidate: {
                    "by_dimension": {
                        dimension: {
                            "repeated": [],
                            "evidence": {
                                ("overall" if dimension == "seed" else "C00" if dimension == "class" else "burden:0"): {
                                    "se_seed": 0.01, "se_boot": 0.02, "epsilon": 0.02,
                                    "ci_lower": -0.01, "ci_upper": 0.01,
                                    "support": {str(seed): 10 for seed in evaluator.EXPECTED_SEEDS},
                                }
                            },
                        }
                        for dimension in ("seed", "class", "subgroup")
                    }
                }
                for candidate in CANDIDATES
            },
        },
    )
    _write_json(
        root / "ensemble_ablation.json",
        {
            "complete": True,
            "ablations": [
                {"name": "without_specialist", "macro_f1": 0.49, "delta": -0.01}
            ],
        },
    )
    _write_json(
        root / "final_selection.json",
        {
            "selections": selections,
            "selection_evidence": {
                "selection_order": [
                    "mean_nested_oof",
                    "one_se_eligibility",
                    "overfit_gap",
                    "simpler_structure",
                ],
                "gap_used_only_within_one_se": True,
                "one_se": {
                    "best_mean": 0.5,
                    "best_se": 0.01,
                    "threshold": 0.49,
                    "eligible_candidates": CANDIDATES,
                },
                "evidence": ["OOF 평균으로 1-SE 집합을 만든 뒤 gap을 적용했다."],
            },
        },
    )
    raw_path = root / "nested" / "raw" / "verified"; raw_path.mkdir(parents=True)
    for name in evaluator.EXPECTED_NESTED_RAW:
        (raw_path / name).write_bytes(b"{}")
    source_path = root / "used_source.py"; source_path.write_text("# source\n", encoding="utf-8")
    used_sources = {str(source_path): evaluator._sha256(source_path)}
    pipeline_source = evaluator.REPO_ROOT / "src/pipelines/jyp_preprocessing/pipeline_jyp_f9.py"
    nested_sources = {
        **used_sources,
        "src/pipelines/jyp_preprocessing/pipeline_jyp_f9.py": evaluator._sha256(pipeline_source),
    }
    nested_source_bundle = evaluator._canonical_hash(nested_sources)
    _write_json(raw_path / "nested_run_manifest.json", {
        "verified": True,
        "trusted_input_hashes": {
            "sources": nested_sources,
            "source_bundle": nested_source_bundle,
        },
    })
    raw_hashes = {name: evaluator._sha256(raw_path / name) for name in evaluator.EXPECTED_NESTED_RAW}
    raw_manifest_hash = raw_hashes["nested_run_manifest.json"]
    frozen_candidates, bundle_inventory = [], {}
    source_hash = evaluator._canonical_hash(used_sources)
    train_hash = evaluator._sha256(evaluator.CANONICAL_TRAIN)
    class_hash = evaluator._sha256(root / "class_names.json")
    fold_hash = evaluator._sha256(root / "fold_assignments.csv")
    for candidate in CANDIDATES:
        runs, bundle_inventory[candidate] = {}, {}
        for seed in evaluator.EXPECTED_SEEDS:
            run_dir = root / "runs" / candidate / str(seed); run_dir.mkdir(parents=True)
            bundle = run_dir / "fold_bundles.pkl"; bundle.write_bytes(f"{candidate}-{seed}".encode())
            bundle_hash = evaluator._sha256(bundle)
            run_manifest_path = run_dir / "run_manifest.json"
            _write_json(run_manifest_path, {"bundle_safe_for_infer": True, "artifact_bundle_hash": bundle_hash})
            runs[str(seed)] = {
                "run_manifest_path": str(run_manifest_path), "run_manifest_hash": evaluator._sha256(run_manifest_path),
                "bundle_safe_for_infer": True, "artifact_bundle_hash": bundle_hash, "train_data_hash": train_hash,
                "class_names_hash": class_hash, "source_hash": source_hash, "fold_assignments_hash": fold_hash,
                "runtime_metrics_hash": evaluator._canonical_hash({"candidate": candidate, "seed": seed}),
                "library_versions": {"python": "fixture"}, "external_runtime_snapshot": {"device": "fixture"},
            }
            bundle_inventory[candidate][str(seed)] = bundle_hash
        frozen_candidates.append({"name": candidate, "config_hash": evaluator._canonical_hash({"candidate": candidate}), "runs": runs})
    role_alias = dict(evaluator.LANE_CANDIDATE_NAMES)
    selected_variants = [
        {"strategy": role, "seed": seed, "fold": fold,
         "variant": {"kind": "single", "models": [alias], "weights": [1.0], "temperatures": [1.0]}}
        for role, alias in role_alias.items() for seed in evaluator.EXPECTED_SEEDS for fold in evaluator.EXPECTED_FOLDS
    ]
    frozen = {
            "frozen": True,
            "class_names": CLASSES,
            "selection_uses_test": False,
            "thresholds_frozen": True,
            "source_hash": evaluator._canonical_hash(used_sources),
            "train_hash": evaluator._sha256(evaluator.CANONICAL_TRAIN),
            "fold_hash": evaluator._sha256(root / "fold_assignments.csv"),
            "bundle_hash": evaluator._canonical_hash(bundle_inventory),
            "used_source_files": used_sources,
            "candidates": frozen_candidates,
            "selection_hash": evaluator._sha256(root / "final_selection.json"),
            "candidate_manifest_hash": evaluator._sha256(root / "candidate_manifest.json"),
            "nested_probability_hash": evaluator._sha256(root / "nested_oof_probability.npy"),
            "selections": selections,
            "role_to_lane": role_to_lane,
            "inference_source_base_files": nested_sources,
            "inference_source_base_bundle": nested_source_bundle,
            "inference_source_files": {
                **nested_sources,
                evaluator.FINALIZER_SOURCE_FILE: evaluator._sha256(
                    evaluator.REPO_ROOT / evaluator.FINALIZER_SOURCE_FILE
                ),
            },
            "nested_raw_path": str(raw_path),
            "nested_raw_manifest_hash": raw_manifest_hash,
            "nested_raw_identity": {"hashes": raw_hashes},
            "selected_variants": selected_variants,
        }
    frozen["inference_source_hash"] = evaluator._canonical_hash(frozen["inference_source_files"])
    _write_json(root / "frozen_manifest.json", frozen)
    generation = root / "generations" / "fixture"; generation.mkdir(parents=True)
    (generation / "candidate_manifest.json").write_bytes((root / "candidate_manifest.json").read_bytes())
    _write_json(generation / "generation_manifest.json", {"phase": "evidence", "hashes": {"candidate_manifest.json": evaluator._sha256(root / "candidate_manifest.json")}})
    synthetic_test_schema = {"columns": ["ID", "x"], "dtypes": ["str", "int64"]}
    test_contract = {"row_count": 1, "id_order_hash": evaluator._canonical_hash(["R0000"]), "schema_hash": evaluator._canonical_hash(synthetic_test_schema)}
    freeze_hash = evaluator._sha256(root / "frozen_manifest.json")
    test_bytes = b"ID,x\nR0000,1\n"; test_content_hash = evaluator.hashlib.sha256(test_bytes).hexdigest()
    checkpoint_root = root / "inference_checkpoints"; preflight_dir = checkpoint_root / "preflight"; prediction_dir = checkpoint_root / "predictions"; test_cache = checkpoint_root / "test_cache"
    preflight_dir.mkdir(parents=True); prediction_dir.mkdir(); test_cache.mkdir()
    (test_cache / "test.csv.bytes").write_bytes(test_bytes)
    _write_integrity(test_cache / "receipt.json", {"schema_version": 1, "freeze_hash": freeze_hash, "content_hash": test_content_hash, "byte_count": len(test_bytes)})
    candidate_map = {candidate["name"]: candidate for candidate in frozen_candidates}
    preflight_hashes, prediction_hashes = {}, {}
    for alias in CANDIDATES:
        for seed in evaluator.EXPECTED_SEEDS:
            stem = evaluator._checkpoint_name(alias, seed)
            preflight_receipt = preflight_dir / f"{stem}.json"
            _write_integrity(preflight_receipt, evaluator._run_identity(freeze_hash, candidate_map[alias], seed))
            preflight_hashes[stem] = evaluator._sha256(preflight_receipt)
            probability_path = prediction_dir / f"{stem}.npz"
            class_index = CANDIDATES.index(alias)
            probability = np.full((1, 26), 1e-6)
            probability[0, class_index] = 1.0
            probability /= probability.sum(axis=1, keepdims=True)
            np.savez_compressed(probability_path, **{f"fold_{fold}": probability for fold in evaluator.EXPECTED_FOLDS})
            prediction_identity = {"schema_version": 1, "freeze_hash": freeze_hash, "test_hash": test_content_hash,
                                   "class_hash": evaluator._canonical_hash(CLASSES), "alias": alias, "seed": seed,
                                   "bundle_hash": candidate_map[alias]["runs"][str(seed)]["artifact_bundle_hash"]}
            prediction_receipt = prediction_dir / f"{stem}.json"
            _write_integrity(prediction_receipt, {"identity": prediction_identity, "npz_hash": evaluator._sha256(probability_path)})
            prediction_hashes[stem] = {"receipt_hash": evaluator._sha256(prediction_receipt), "npz_hash": evaluator._sha256(probability_path)}
    prediction_manifest_hash = evaluator._canonical_hash(prediction_hashes)
    checkpoint_payload = {"schema_version": 1, "freeze_hash": freeze_hash, "test_content_hash": test_content_hash,
                          "class_hash": evaluator._canonical_hash(CLASSES), "run_count": len(CANDIDATES) * len(evaluator.EXPECTED_SEEDS),
                          "preflight_receipts": preflight_hashes, "predictions": prediction_hashes,
                          "prediction_manifest_hash": prediction_manifest_hash}
    checkpoint_manifest = checkpoint_root / "inference_checkpoint_manifest.json"; _write_integrity(checkpoint_manifest, checkpoint_payload)
    for role in sorted(evaluator.FINAL_ROLES):
        class_index = list(evaluator.LANE_CANDIDATE_NAMES).index(role_to_lane[role])
        (root / f"{role}.csv").write_text(f"ID,SUBCLASS\nR0000,C{class_index:02d}\n", encoding="utf-8")
    _write_json(
        root / "final_submissions.json",
        {
            "submissions": [
                {
                    "role": role,
                    "path": f"{role}.csv",
                    "sha256": evaluator._sha256(root / f"{role}.csv"),
                    "row_count": 1,
                    "columns": ["ID", "SUBCLASS"],
                    "candidate": frozen["selections"][role],
                    "members": 15,
                }
                for role in sorted(evaluator.FINAL_ROLES)
            ],
            "freeze_hash": freeze_hash, "test_content_hash": test_content_hash,
            "checkpoint_manifest_hash": evaluator._sha256(checkpoint_manifest),
            "prediction_manifest_hash": prediction_manifest_hash,
            "test_contract": test_contract,
        },
    )
    _write_json(root / "finalization_state.json", {"state": "documented", "freeze_hash": freeze_hash, "test_read_count": 1, "canonical_test_read_count": 1, "fit_after_test": 0, "warm_after_test": 0, "bundle_load_after_test_count": len(CANDIDATES) * len(evaluator.EXPECTED_SEEDS), "selection_or_tuning_after_test": 0, "test_content_hash": test_content_hash, "generation": str(generation), "generation_manifest_hash": evaluator._sha256(generation / "generation_manifest.json"), "test_contract": test_contract})
    (root / "final_report_ko.md").write_text(
        "# 최종 모델 전문화 보고서\n\n"
        "학습 데이터만 사용한 완전 중첩 교차검증 결과를 기록한다. "
        "주력 모델과 강건형 후보 및 충돌 암종 전문 후보의 선택 근거를 "
        "세 시드와 다섯 외부 폴드의 지표로 비교했다. 테스트 데이터는 선택에 사용하지 않았다.\n",
        encoding="utf-8",
    )
    _write_json(
        root / "leakage_checklist.json",
        {
            "complete": True,
            "all_passed": True,
            "test_used_for_selection": False,
            "items": [
                {"name": "outer validation isolation", "passed": True},
                {"name": "Test selection isolation", "passed": True},
            ],
        },
    )
    return root


def _check(report: dict, name: str) -> dict:
    return next(item for item in report["checks"] if item["name"] == name)


def test_complete_fixture_passes_and_cli_requires_complete(tmp_path, capsys):
    root = _make_complete(tmp_path / "artifacts")
    report = evaluator.evaluate(root).report(require_complete=True)
    assert report["status"] == "PASS"
    assert report["failed_checks"] == []
    assert evaluator.main(["--artifact-root", str(root), "--require-complete"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "PASS"


def test_nonidentity_role_to_lane_replays_distinct_lane_probabilities(tmp_path):
    mapping = {
        "primary": "specialization_challenger",
        "robustness_challenger": "primary",
        "specialization_challenger": "robustness_challenger",
    }
    root = _make_complete(tmp_path / "artifacts", role_to_lane=mapping)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert report["status"] == "PASS"
    expected = {
        "primary": "C02",
        "robustness_challenger": "C00",
        "specialization_challenger": "C01",
    }
    for role, label in expected.items():
        assert pd.read_csv(root / f"{role}.csv")["SUBCLASS"].tolist() == [label]


def test_frozen_role_to_lane_must_match_selection_candidate_names(tmp_path):
    root = _make_complete(tmp_path / "artifacts")
    path = root / "frozen_manifest.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    frozen["role_to_lane"] = {
        "primary": "robustness_challenger",
        "robustness_challenger": "primary",
        "specialization_challenger": "specialization_challenger",
    }
    _write_json(path, frozen)
    assert _check(evaluator.evaluate(root).report(True), "frozen_manifest")["passed"] is False


def test_missing_artifact_fails_complete_but_readiness_exits_zero(tmp_path, capsys):
    root = _make_complete(tmp_path / "artifacts")
    (root / "paired_comparisons.json").unlink()
    report = evaluator.evaluate(root).report(require_complete=True)
    assert report["status"] == "FAIL"
    assert _check(report, "paired_comparisons")["passed"] is False
    assert evaluator.main(["--artifact-root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["complete"] is False
    assert evaluator.main(["--artifact-root", str(root), "--require-complete"]) == 1


@pytest.mark.parametrize("failure", ["shape", "row_sum"])
def test_bad_candidate_probability_fails(tmp_path, failure):
    root = _make_complete(tmp_path / "artifacts")
    probability = _uniform_probability()
    if failure == "shape":
        probability = probability[:, :-1, :]
    else:
        probability[0, 0] = 0.5
    np.save(root / "candidate_probability.npy", probability)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "candidate_oof_probabilities")["passed"] is False


def test_test_audit_contamination_fails(tmp_path):
    root = _make_complete(tmp_path / "artifacts")
    audit = json.loads((root / "io_audit.json").read_text(encoding="utf-8"))
    audit["events"].append(
        {"stage": "nested_oof_selection", "action": "read", "path": "data/raw/test.csv"}
    )
    _write_json(root / "io_audit.json", audit)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "io_audit")["passed"] is False


def test_forged_state_and_final_manifest_cannot_override_test_cache(tmp_path):
    root = _make_complete(tmp_path / "artifacts")
    forged_hash = "f" * 64
    state = json.loads((root / "finalization_state.json").read_text(encoding="utf-8")); state["test_content_hash"] = forged_hash
    _write_json(root / "finalization_state.json", state)
    final = json.loads((root / "final_submissions.json").read_text(encoding="utf-8")); final["test_content_hash"] = forged_hash
    _write_json(root / "final_submissions.json", final)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "inference_checkpoints")["passed"] is False


def test_forged_zero_bundle_load_count_cannot_pass_complete_checkpoints(tmp_path):
    root = _make_complete(tmp_path / "artifacts")
    state = json.loads((root / "finalization_state.json").read_text(encoding="utf-8")); state["bundle_load_after_test_count"] = 0
    _write_json(root / "finalization_state.json", state)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "inference_checkpoints")["passed"] is False


def test_forged_submission_member_count_fails(tmp_path):
    root = _make_complete(tmp_path / "artifacts")
    final = json.loads((root / "final_submissions.json").read_text(encoding="utf-8")); final["submissions"][0]["members"] = 14
    _write_json(root / "final_submissions.json", final)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "final_submissions")["passed"] is False


def test_valid_class_label_with_refreshed_sha_fails_checkpoint_replay(tmp_path):
    root = _make_complete(tmp_path / "artifacts")
    final = json.loads((root / "final_submissions.json").read_text(encoding="utf-8")); row = final["submissions"][0]
    submission_path = root / row["path"]
    frame = pd.read_csv(submission_path); frame.loc[0, "SUBCLASS"] = "C01"; frame.to_csv(submission_path, index=False)
    row["sha256"] = evaluator._sha256(submission_path); _write_json(root / "final_submissions.json", final)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "inference_checkpoints")["passed"] is False


def test_forged_checkpoint_hash_chain_still_rejects_out_of_range_probability(tmp_path):
    root = _make_complete(tmp_path / "artifacts")
    checkpoint_root = root / "inference_checkpoints"; manifest_path = checkpoint_root / "inference_checkpoint_manifest.json"
    stem = evaluator._checkpoint_name(CANDIDATES[0], evaluator.EXPECTED_SEEDS[0])
    probability_path = checkpoint_root / "predictions" / f"{stem}.npz"
    with np.load(probability_path, allow_pickle=False) as original:
        values = {name: np.asarray(original[name]).copy() for name in original.files}
    values["fold_0"][0, 0], values["fold_0"][0, 1] = -0.1, 1.1
    np.savez_compressed(probability_path, **values)
    receipt_path = checkpoint_root / "predictions" / f"{stem}.json"
    receipt_payload = evaluator._integrity_payload(receipt_path); receipt_payload["npz_hash"] = evaluator._sha256(probability_path)
    _write_integrity(receipt_path, receipt_payload)
    manifest = evaluator._integrity_payload(manifest_path)
    manifest["predictions"][stem] = {"receipt_hash": evaluator._sha256(receipt_path), "npz_hash": evaluator._sha256(probability_path)}
    manifest["prediction_manifest_hash"] = evaluator._canonical_hash(manifest["predictions"])
    _write_integrity(manifest_path, manifest)
    final = json.loads((root / "final_submissions.json").read_text(encoding="utf-8"))
    final["checkpoint_manifest_hash"] = evaluator._sha256(manifest_path); final["prediction_manifest_hash"] = manifest["prediction_manifest_hash"]
    _write_json(root / "final_submissions.json", final)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "inference_checkpoints")["passed"] is False


@pytest.mark.parametrize("canonical_name", ["test", "submission"])
@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_evaluator_rejects_canonical_alias_before_submission_reader(tmp_path, monkeypatch, canonical_name, alias_kind):
    root = _make_complete(tmp_path / "artifacts")
    canonical = evaluator.CANONICAL_TEST if canonical_name == "test" else evaluator.CANONICAL_SUBMISSION
    alias = root / f"forbidden_{canonical_name}_{alias_kind}.csv"
    try:
        if alias_kind == "symlink": os.symlink(canonical, alias)
        else: os.link(canonical, alias)
    except OSError as error:
        pytest.skip(f"{alias_kind} unavailable: {error}")
    final = json.loads((root / "final_submissions.json").read_text(encoding="utf-8")); final["submissions"][0]["path"] = alias.name
    _write_json(root / "final_submissions.json", final)
    original = evaluator.pd.read_csv
    def guarded_reader(path, *args, **kwargs):
        if Path(path).resolve() == canonical.resolve():
            raise AssertionError("canonical reader was reached")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(evaluator.pd, "read_csv", guarded_reader)
    with pytest.raises(ValueError, match="aliases canonical"):
        evaluator._final_submissions(root)


@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink"])
def test_evaluator_rejects_inventory_alias_before_hash(tmp_path, monkeypatch, alias_kind):
    forbidden = tmp_path / "test.csv"
    forbidden.write_bytes(b"forbidden")
    alias = forbidden
    if alias_kind != "direct":
        alias = tmp_path / f"source_{alias_kind}.py"
        try:
            if alias_kind == "symlink":
                alias.symlink_to(forbidden)
            else:
                alias.hardlink_to(forbidden)
        except OSError as error:
            pytest.skip(f"{alias_kind} unavailable: {error}")
    sources = {str(alias): "0" * 64}
    bundle = evaluator._canonical_hash(sources)
    frozen = {
        "inference_source_base_files": sources,
        "inference_source_base_bundle": bundle,
        "inference_source_files": sources,
        "inference_source_hash": evaluator._canonical_hash(sources),
    }
    raw_manifest = {"trusted_input_hashes": {"sources": sources, "source_bundle": bundle}}
    monkeypatch.setattr(
        evaluator, "_trusted_forbidden", lambda: (forbidden.resolve(), (tmp_path / "submission.csv").resolve())
    )
    monkeypatch.setattr(
        evaluator, "_sha256", lambda path: (_ for _ in ()).throw(AssertionError("hash reader reached"))
    )
    with pytest.raises(ValueError, match="aliases canonical"):
        evaluator._validated_inference_sources(frozen, raw_manifest)


def test_evaluator_rejects_transitive_pipeline_drift(tmp_path, monkeypatch):
    root = _make_complete(tmp_path / "artifacts")
    pipeline = (evaluator.REPO_ROOT / "src/pipelines/jyp_preprocessing/pipeline_jyp_f9.py").resolve()
    original_sha256 = evaluator._sha256
    original_read_csv = evaluator._read_csv
    canonical_reads = []

    def drift(path):
        if Path(path).resolve() == pipeline:
            return "0" * 64
        return original_sha256(Path(path))

    def tracked_read_csv(path, **kwargs):
        if Path(path).resolve() == evaluator.CANONICAL_TEST.resolve():
            canonical_reads.append(path)
        return original_read_csv(path, **kwargs)

    monkeypatch.setattr(evaluator, "_sha256", drift)
    monkeypatch.setattr(evaluator, "_read_csv", tracked_read_csv)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "frozen_manifest")["passed"] is False
    assert canonical_reads == []


def test_protected_hash_mismatch_fails(tmp_path):
    root = _make_complete(tmp_path / "artifacts")
    path = root / "protected_file_hashes_before_after.json"
    hashes = json.loads(path.read_text(encoding="utf-8"))
    hashes["files"][0]["after"] = "b" * 64
    _write_json(path, hashes)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "protected_file_hashes")["passed"] is False


def test_repeated_collapse_failure_fails(tmp_path):
    root = _make_complete(tmp_path / "artifacts")
    path = root / "collapse_constraints.json"
    collapse = json.loads(path.read_text(encoding="utf-8"))
    collapse["constraints"]["novelty"] = {
        "pass": False,
        "repeated_collapse_count": 2,
    }
    _write_json(path, collapse)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "collapse_constraints")["passed"] is False


def test_gap_before_one_se_selection_fails(tmp_path):
    root = _make_complete(tmp_path / "artifacts")
    path = root / "final_selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    selection["selection_evidence"]["selection_order"] = [
        "mean_nested_oof",
        "overfit_gap",
        "one_se_eligibility",
        "simpler_structure",
    ]
    selection["selection_evidence"]["gap_used_only_within_one_se"] = False
    _write_json(path, selection)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "final_selection")["passed"] is False


@pytest.mark.parametrize("field", ["source_hash", "train_hash", "fold_hash", "bundle_hash"])
def test_arbitrary_top_hash_replacement_is_rejected(tmp_path, field):
    root = _make_complete(tmp_path / "artifacts")
    path = root / "frozen_manifest.json"; frozen = json.loads(path.read_text())
    frozen[field] = "f" * 64; _write_json(path, frozen)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "frozen_manifest")["passed"] is False


def test_bundle_safe_for_infer_false_is_rejected(tmp_path):
    root = _make_complete(tmp_path / "artifacts")
    frozen_path = root / "frozen_manifest.json"; frozen = json.loads(frozen_path.read_text())
    run = frozen["candidates"][0]["runs"]["42"]
    run["bundle_safe_for_infer"] = False; _write_json(frozen_path, frozen)
    report = evaluator.evaluate(root).report(require_complete=True)
    assert _check(report, "frozen_manifest")["passed"] is False
