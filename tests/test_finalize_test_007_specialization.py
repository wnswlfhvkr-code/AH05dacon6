from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml

from src.test_007 import finalize_test_007_specialization as finalizer
from src.test_007 import run_test_007_nested as nested
from src.test_007.evaluate_test_007_specialization import evaluate


def _probability(prediction: list[int], classes: int = 3) -> np.ndarray:
    result = np.full((len(prediction), classes), .01)
    result[np.arange(len(prediction)), prediction] = .98
    return result / result.sum(axis=1, keepdims=True)


def test_repeated_collapse_requires_two_of_three_seeds():
    repeated, names = finalizer.repeated_collapse(
        {"burden:2": {42: -.02, 2026: -.03, 777: .01}, "novelty:2": {42: -.02, 2026: 0, 777: 0}},
        epsilon=.01,
    )
    assert repeated is True
    assert names == ["burden:2"]


def test_protected_before_hashes_accepts_collector_metadata_schema():
    digest = hashlib.sha256(b"protected").hexdigest()
    payload = {
        "before": {"src/protected.py": {"sha256": digest, "bytes": 9}},
        "files": [
            {
                "path": "src/protected.py",
                "before": digest,
                "after": digest,
                "unchanged": True,
            }
        ],
    }
    assert finalizer._protected_before_hashes(payload) == {
        "src/protected.py": digest
    }


def test_protected_before_hashes_rejects_schema_disagreement():
    first = hashlib.sha256(b"first").hexdigest()
    second = hashlib.sha256(b"second").hexdigest()
    with pytest.raises(ValueError, match="inventories disagree"):
        finalizer._protected_before_hashes(
            {
                "before": {"src/protected.py": {"sha256": first, "bytes": 5}},
                "files": [{"path": "src/protected.py", "before": second}],
            }
        )


def test_paired_bootstrap_epsilon_is_maximum_and_finite():
    labels = np.tile(np.arange(3), 20)
    baseline = np.stack([_probability(labels.tolist()) for _ in range(3)])
    candidate = baseline.copy()
    candidate[:, [0, 3, 6], :] = _probability([1, 1, 1])
    result = finalizer.paired_bootstrap(labels, candidate, baseline, iterations=50)
    assert result["epsilon"] == max(result["se_seed"], result["se_boot"])
    assert result["ci_lower"] <= result["ci_upper"]


def test_paired_bootstrap_uses_shared_row_draw_across_seeds(monkeypatch):
    labels = np.tile(np.arange(3), 5)
    probability = np.stack([_probability(labels.tolist()) for _ in range(3)])
    calls: list[np.ndarray] = []

    def capture(current_labels, current_probability, classes):
        if len(current_labels) == len(labels):
            calls.append(np.asarray(current_labels).copy())
        return 0.5

    monkeypatch.setattr(finalizer, "_macro", capture)
    finalizer.paired_bootstrap(labels, probability, probability, iterations=2)
    # Six seed-level calls precede draws. Within each draw candidate/baseline and
    # all three seeds receive the identical class-stratified row sample.
    draw_calls = calls[6:]
    assert len(draw_calls) == 12
    assert all(np.array_equal(draw_calls[0], value) for value in draw_calls[:6])
    assert all(np.array_equal(draw_calls[6], value) for value in draw_calls[6:])


def test_select_roles_applies_gap_only_inside_one_se():
    stats = {
        "primary": {"mean": .50, "se": .02, "gap": .01, "complexity": 1, "no_collapse": False, "noninferior": True, "worst_floor": .30, "collision_evidence": False, "net_rescue": 0},
        "robustness": {"mean": .49, "se": .01, "gap": .02, "complexity": 2, "no_collapse": False, "noninferior": True, "worst_floor": .40, "collision_evidence": False, "net_rescue": 0},
        "specialization": {"mean": .47, "se": .01, "gap": .01, "complexity": 3, "no_collapse": True, "noninferior": True, "worst_floor": .35, "collision_evidence": True, "net_rescue": 8},
    }
    roles, evidence = finalizer.select_roles(stats)
    assert roles == {"primary": "primary", "robustness": "robustness", "specialization": "specialization"}
    assert evidence["one_se"]["eligible_candidates"] == ["primary", "robustness"]
    assert evidence["gap_used_only_within_one_se"] is True
    assert evidence["role_status"] == {
        "primary": "best_available",
        "robustness_challenger": "qualified",
        "specialization_challenger": "qualified",
    }
    assert evidence["portfolio_constraints_pass"] is True


def test_select_primary_does_not_require_challenger_fields():
    primary, evidence = finalizer.select_primary({
        "a": {"mean": .5, "se": .01, "gap": .02, "complexity": 2},
        "b": {"mean": .495, "se": .01, "gap": .01, "complexity": 1},
    })
    assert primary == "b"
    assert evidence["one_se"]["eligible_candidates"] == ["a", "b"]


def test_select_roles_preserves_distinct_exploratory_fallbacks():
    stats = {
        name: {"mean": .5 - index * .001, "se": .01, "gap": 0, "complexity": index, "no_collapse": False, "noninferior": False, "worst_floor": .3 - index * .01, "collision_evidence": False, "net_rescue": 0}
        for index, name in enumerate(("primary", "robustness_challenger", "specialization_challenger"))
    }
    roles, evidence = finalizer.select_roles(stats)
    assert roles == {
        "primary": "primary",
        "robustness": "robustness_challenger",
        "specialization": "specialization_challenger",
    }
    assert len(set(roles.values())) == 3
    assert evidence["role_status"]["robustness_challenger"] == "exploratory_failed_constraints"
    assert evidence["role_status"]["specialization_challenger"] == "exploratory_failed_constraints"
    assert evidence["portfolio_constraints_pass"] is False
    assert evidence["oracle_collapse"] == {
        "diagnostic_only": True,
        "gating": False,
        "note": "generic per-slice oracle-best collapse 수치는 보존하지만 primary/challenger 자격을 차단하지 않는다.",
    }


def test_apply_variant_replay_matches_exactly():
    rows, classes = 10, 3
    assignments = pd.concat([
        pd.DataFrame({"seed": seed, "row_index": np.arange(rows), "ID": [f"I{i}" for i in range(rows)], "fold": np.arange(rows) % 5})
        for seed in finalizer.SEEDS
    ], ignore_index=True)
    a = np.stack([_probability((np.arange(rows) % classes).tolist()) for _ in finalizer.SEEDS])
    b = np.stack([np.roll(value, 1, axis=1) for value in a])
    variant = {"kind": "blend", "models": ["a", "b"], "weights": [0.7, 0.3], "temperatures": [1.0, 1.0]}
    records = {(lane, seed, fold): {"selected_variant": variant} for lane in finalizer.LANES for seed in finalizer.SEEDS for fold in finalizer.FOLDS}
    stored = {lane: np.stack([nested._apply_variant(variant, {"a": a[index], "b": b[index]}) for index in range(3)]) for lane in finalizer.LANES}
    labels = np.arange(rows) % classes
    replay = finalizer._replay_nested({"a": a, "b": b}, records, assignments, stored, labels)
    assert np.array_equal(replay["primary"], stored["primary"])


def _replay_drift_fixture(alias="catboost"):
    rows, classes = 10, 3
    assignments = pd.concat([
        pd.DataFrame({"seed": seed, "row_index": np.arange(rows), "ID": [f"I{i}" for i in range(rows)], "fold": np.arange(rows) % 5})
        for seed in finalizer.SEEDS
    ], ignore_index=True)
    base = np.stack([_probability((np.arange(rows) % classes).tolist()) for _ in finalizer.SEEDS])
    variant = {"kind": "blend", "models": [alias], "weights": [1.0], "temperatures": [1.0]}
    records = {(lane, seed, fold): {"selected_variant": variant} for lane in finalizer.LANES for seed in finalizer.SEEDS for fold in finalizer.FOLDS}
    stored = {lane: base.copy() for lane in finalizer.LANES}
    labels = np.arange(rows) % classes
    return assignments, base, records, stored, labels


def test_replay_uses_nested_authority_for_bounded_catboost_gpu_drift():
    assignments, base, records, stored, labels = _replay_drift_fixture()
    for probability in stored.values():
        probability[:, 0, 0] -= 5e-7
        probability[:, 0, 1] += 5e-7
    audit = []
    replay = finalizer._replay_nested(
        {"catboost": base}, records, assignments, stored, labels, audit
    )
    assert np.array_equal(replay["primary"], stored["primary"])
    assert len(audit) == 45
    assert len({(row["strategy"], row["seed"], row["fold"]) for row in audit}) == 45
    assert any(not row["strict_probability_match"] for row in audit)
    assert all(row["argmax_disagreement_count"] == 0 for row in audit)


def test_replay_rejects_probability_drift_for_deterministic_alias():
    assignments, base, records, stored, labels = _replay_drift_fixture("deterministic")
    for probability in stored.values():
        probability[:, 0, 0] -= 1e-3
        probability[:, 0, 1] += 1e-3
    with pytest.raises(ValueError, match="deterministic base OOF replay"):
        finalizer._replay_nested(
            {"deterministic": base}, records, assignments, stored, labels
        )


def test_replay_rejects_catboost_argmax_change():
    assignments, base, records, stored, labels = _replay_drift_fixture()
    for probability in stored.values():
        probability[:, 0, [0, 1]] = probability[:, 0, [1, 0]]
    with pytest.raises(ValueError, match="argmax mismatch"):
        finalizer._replay_nested(
            {"catboost": base}, records, assignments, stored, labels
        )


def test_replay_rejects_excessive_catboost_probability_drift():
    assignments, base, records, stored, labels = _replay_drift_fixture()
    for probability in stored.values():
        probability[:, 0, 0] -= 6e-3
        probability[:, 0, 1] += 6e-3
    with pytest.raises(ValueError, match="drift exceeds bounds"):
        finalizer._replay_nested(
            {"catboost": base}, records, assignments, stored, labels
        )


def test_replay_rejects_catboost_mean_drift_even_when_max_is_bounded():
    assignments, base, records, stored, labels = _replay_drift_fixture()
    for probability in stored.values():
        probability[:, :, 0] -= 1e-6
        probability[:, :, 1] += 1e-6
    with pytest.raises(ValueError, match="drift exceeds bounds"):
        finalizer._replay_nested(
            {"catboost": base}, records, assignments, stored, labels
        )


def _write_nested_source(path: Path, tamper: bool = False) -> None:
    path.mkdir()
    np.save(path / "nested_oof_probability.npy", np.ones((1, 1, 1)))
    np.savez(path / "nested_strategy_probability.npz", primary=np.ones((1, 1, 1)))
    np.savez(path / "nested_inner_selection_probability.npz", fixture=np.ones((1, 1)))
    for name in ("nested_inner_selection_records.json", "nested_cv_metrics.json", "nested_selection_records.json"):
        (path / name).write_text("{}", encoding="utf-8")
    (path / "nested_io_audit.json").write_text(json.dumps({"events": [{"action": "read", "path": "train.csv"}], "test_accessed": False, "submission_reads": 0}), encoding="utf-8")
    hashes = {name: finalizer._sha256(path / name) for name in finalizer.RAW_FILES if name != "nested_run_manifest.json"}
    if tamper:
        hashes["nested_cv_metrics.json"] = "0" * 64
    (path / "nested_run_manifest.json").write_text(json.dumps({"test_accessed": False, "outer_validation_used_for_selection": False, "run_identity_hash": "abc", "source_bundle_before": "bundle", "artifact_hashes": hashes}), encoding="utf-8")


def _write_completion_attestation(path: Path, raw: Path) -> dict:
    manifest = json.loads((raw / "nested_run_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "created_at": "2026-08-05T21:57:42+09:00",
        "scope": "post_nested_pre_finalization_external_trust_anchor",
        "selection_uses_test": False,
        "nested_run_manifest_sha256": finalizer._sha256(raw / "nested_run_manifest.json"),
        "nested_run_identity_hash": manifest["run_identity_hash"],
        "nested_source_bundle": manifest["source_bundle_before"],
        "artifact_hashes": manifest["artifact_hashes"],
        "validation": {
            "source_unchanged": True,
            "test_reads": 0,
            "submission_reads": 0,
            "transcript_records_verified": 45,
            "strategy_probability_shape": [3, 6201, 26],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_completion_attestation_binds_manifest_and_all_nested_artifacts(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw"
    _write_nested_source(raw)
    attestation = tmp_path / "attestation.json"
    manifest = _write_completion_attestation(attestation, raw)
    monkeypatch.setattr(finalizer, "NESTED_COMPLETION_ATTESTATION", attestation)
    monkeypatch.setattr(
        finalizer,
        "NESTED_COMPLETION_ATTESTATION_SHA256",
        finalizer._sha256(attestation),
    )
    verified = finalizer._verify_nested_completion_attestation(raw, manifest, [])
    assert verified["manifest_sha256"] == finalizer._sha256(
        raw / "nested_run_manifest.json"
    )
    assert verified["strategy_probability_sha256"] == manifest["artifact_hashes"][
        "nested_strategy_probability.npz"
    ]


def test_completion_attestation_rejects_nested_artifact_tamper(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_nested_source(raw)
    attestation = tmp_path / "attestation.json"
    manifest = _write_completion_attestation(attestation, raw)
    monkeypatch.setattr(finalizer, "NESTED_COMPLETION_ATTESTATION", attestation)
    monkeypatch.setattr(
        finalizer,
        "NESTED_COMPLETION_ATTESTATION_SHA256",
        finalizer._sha256(attestation),
    )
    with (raw / "nested_strategy_probability.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        finalizer._verify_nested_completion_attestation(raw, manifest, [])


def test_completion_attestation_rejects_manifest_digest_tamper(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_nested_source(raw)
    attestation = tmp_path / "attestation.json"
    manifest = _write_completion_attestation(attestation, raw)
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    payload["nested_run_manifest_sha256"] = "0" * 64
    attestation.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(finalizer, "NESTED_COMPLETION_ATTESTATION", attestation)
    monkeypatch.setattr(
        finalizer,
        "NESTED_COMPLETION_ATTESTATION_SHA256",
        finalizer._sha256(attestation),
    )
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        finalizer._verify_nested_completion_attestation(raw, manifest, [])


def _valid_replay_audit() -> dict:
    records = [
        {
            "strategy": lane,
            "seed": seed,
            "fold": fold,
            "models": ["catboost"],
            "known_nondeterministic_models": ["catboost"],
            "strict_probability_match": True,
            "max_abs_probability_drift": 0.0,
            "mean_abs_probability_drift": 0.0,
            "argmax_disagreement_count": 0,
            "macro_f1_equal": True,
            "probability_authority": "stored_hashed_nested_outer_lane_probability",
        }
        for lane in finalizer.LANES
        for seed in finalizer.SEEDS
        for fold in finalizer.FOLDS
    ]
    return {
        "probability_authority": "stored_hashed_nested_outer_lane_probability",
        "selection_metric_authority": "stored_hashed_nested_outer_lane_probability",
        "production_inference_realization": "hashed_base_oof_fold_bundles",
        "semantic_equivalence_count": 45,
        "argmax_disagreement_count": 0,
        "max_observed_abs_drift": 0.0,
        "max_observed_mean_drift": 0.0,
        "records": records,
    }


def test_replay_audit_persists_identically_to_candidate_and_selection(tmp_path):
    audit = _valid_replay_audit()
    (tmp_path / "candidate_manifest.json").write_text(
        json.dumps({"nested_base_oof_replay_audit": audit}), encoding="utf-8"
    )
    (tmp_path / "final_selection.json").write_text(
        json.dumps(
            {"selection_evidence": {"nested_base_oof_replay_audit": audit}}
        ),
        encoding="utf-8",
    )
    finalizer._verify_persisted_replay_audit(tmp_path, audit)
    selection = json.loads((tmp_path / "final_selection.json").read_text())
    selection["selection_evidence"]["nested_base_oof_replay_audit"]["records"].pop()
    (tmp_path / "final_selection.json").write_text(json.dumps(selection))
    with pytest.raises(RuntimeError, match="persisted identically"):
        finalizer._verify_persisted_replay_audit(tmp_path, audit)


def test_primary_npy_must_be_bitwise_identical_to_nested_npz_primary(tmp_path):
    expected = np.ones((3, 2, 3), dtype=np.float64) / 3
    np.save(tmp_path / "nested_oof_probability.npy", expected)
    finalizer._verify_primary_probability_identity(tmp_path, expected)
    changed = expected.copy()
    changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], np.inf)
    np.save(tmp_path / "nested_oof_probability.npy", changed)
    with pytest.raises(ValueError, match="NPY differs from NPZ"):
        finalizer._verify_primary_probability_identity(tmp_path, expected)


def test_nested_snapshot_rejects_hash_tamper_before_copy(tmp_path):
    source, root = tmp_path / "source", tmp_path / "root"
    _write_nested_source(source, tamper=True); root.mkdir()
    with pytest.raises(ValueError, match="hash mismatch"):
        finalizer._snapshot_nested(root, source)
    assert not (root / "nested" / "raw" / "abc").exists()


def test_nested_snapshot_is_immutable_and_detects_later_tamper(tmp_path):
    source, root = tmp_path / "source", tmp_path / "root"
    _write_nested_source(source); root.mkdir()
    destination, _ = finalizer._snapshot_nested(root, source)
    (destination / "nested_cv_metrics.json").write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="modified"):
        finalizer._snapshot_nested(root, source)


def _minimal_freeze_root(root: Path) -> None:
    root.mkdir()
    (root / "finalization_state.json").write_text(json.dumps({"state": "evidence", "test_read_count": 0, "nested_raw": str(root / "raw")}), encoding="utf-8")
    (root / "raw").mkdir()
    rows = [{"strategy": lane, "seed": seed, "fold": fold, "selection_source": "outer_train_inner_oof_only", "outer_validation_used_for_selection": False, "selected_variant": {"kind": "single", "models": ["a"], "weights": [1.0], "temperatures": [1.0]}} for lane in finalizer.LANES for seed in finalizer.SEEDS for fold in finalizer.FOLDS]
    (root / "raw" / "nested_selection_records.json").write_text(json.dumps({"selections": rows}), encoding="utf-8")
    (root / "raw" / "nested_run_manifest.json").write_text("{}", encoding="utf-8")
    (root / "final_selection.json").write_text(json.dumps({"selections": {"primary": "nested_primary", "robustness_challenger": "nested_robustness", "specialization_challenger": "nested_specialization"}}), encoding="utf-8")
    (root / "candidate_manifest.json").write_text(json.dumps({"candidates": []}), encoding="utf-8")
    (root / "protected_file_hashes_before_after.json").write_text(json.dumps({"before": {"x": "0" * 64}}), encoding="utf-8")
    (root / "fold_assignments.csv").write_text("x\n1\n", encoding="utf-8")
    (root / "class_names.json").write_text(json.dumps([str(i) for i in range(26)]), encoding="utf-8")


def test_freeze_rolls_back_pending_file_on_validation_failure(tmp_path):
    root = tmp_path / "artifacts"; _minimal_freeze_root(root)
    with pytest.raises(Exception):
        finalizer.freeze(root)
    assert not (root / "frozen_manifest.json").exists()
    assert not (root / ".frozen_manifest.pending.json").exists()
    assert json.loads((root / "finalization_state.json").read_text())["state"] == "evidence"


def test_infer_resume_requires_same_freeze(tmp_path):
    root = tmp_path / "artifacts"; root.mkdir()
    frozen = root / "frozen_manifest.json"; frozen.write_text("{}", encoding="utf-8")
    (root / "finalization_state.json").write_text(json.dumps({"state": "inferred", "freeze_hash": "0" * 64}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="freeze mismatch"):
        finalizer.infer(root)


def test_document_requires_exactly_one_test_read(tmp_path):
    root = tmp_path / "artifacts"; root.mkdir()
    (root / "finalization_state.json").write_text(json.dumps({"state": "inferred", "canonical_test_read_count": 2, "fit_after_test": 0, "warm_after_test": 0, "selection_or_tuning_after_test": 0}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="one-read"):
        finalizer.document(root)


def test_document_allows_only_known_collapse_objective_failure(tmp_path, monkeypatch):
    import src.test_007.evaluate_test_007_specialization as evaluator

    root = tmp_path / "artifacts"; root.mkdir()
    state = {
        "state": "inferred", "canonical_test_read_count": 1,
        "fit_after_test": 0, "warm_after_test": 0,
        "selection_or_tuning_after_test": 0,
    }
    selections = {
        "primary": "nested_primary",
        "robustness_challenger": "nested_robustness",
        "specialization_challenger": "nested_specialization",
    }
    (root / "finalization_state.json").write_text(json.dumps(state), encoding="utf-8")
    (root / "frozen_manifest.json").write_text(json.dumps({"selections": selections}), encoding="utf-8")
    (root / "final_selection.json").write_text(json.dumps({
        "selections": selections,
        "selection_evidence": {
            "portfolio_constraints_pass": False,
            "role_status": {
                "primary": "best_available",
                "robustness_challenger": "exploratory_failed_constraints",
                "specialization_challenger": "exploratory_failed_constraints",
            },
        },
    }), encoding="utf-8")

    class Check:
        def __init__(self, name, passed):
            self.name, self.passed = name, passed

    class Result:
        def __init__(self, documented):
            self.checks = [Check("collapse_constraints", False), Check("documentation", documented)]
            self.passed = False

        def report(self, require_complete):
            return {"failed_checks": [check.name for check in self.checks if not check.passed]}

    monkeypatch.setattr(finalizer, "_verify_submission_manifest", lambda *args: None)
    monkeypatch.setattr(
        evaluator, "evaluate",
        lambda artifact_root: Result((Path(artifact_root) / "final_report_ko.md").exists()),
    )

    output = finalizer.document(root)
    recorded = json.loads((root / "finalization_state.json").read_text(encoding="utf-8"))
    report = output.read_text(encoding="utf-8")
    assert recorded["objective_constraints_pass"] is False
    assert recorded["failed_checks"] == ["collapse_constraints"]
    assert recorded["final_evaluation_pass"] is False
    assert "best available 주력" in report
    assert report.count("탐색용 challenger · 제약 미통과") == 2
    assert "완전 PASS로 주장하지 않는다" in report


@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink"])
def test_pretest_guard_blocks_alias_before_reader_bytes(tmp_path, monkeypatch, alias_kind):
    test = tmp_path / "test.csv"; test.write_text('{"secret":true}', encoding="utf-8")
    submission = tmp_path / "submission.csv"; submission.write_text("ID,SUBCLASS\n", encoding="utf-8")
    alias = test
    if alias_kind == "symlink":
        alias = tmp_path / "alias.json"; alias.symlink_to(test)
    elif alias_kind == "hardlink":
        alias = tmp_path / "alias.json"; alias.hardlink_to(test)
    reads = []
    original = Path.read_text

    def tracked(self, *args, **kwargs):
        if self.resolve() == test.resolve(): reads.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked)
    with pytest.raises(ValueError, match="aliases"):
        finalizer._guarded_json(alias, (test, submission), "malicious")
    assert reads == []


def test_evidence_defaults_missing_test_file_to_canonical_without_csv_read(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    test_path = raw / "test.csv"; test_path.write_text("secret", encoding="utf-8")
    submission_path = raw / "sample_submission.csv"; submission_path.write_text("ID,SUBCLASS\n", encoding="utf-8")
    train_path = raw / "train.csv"; train_path.write_text("ID,SUBCLASS\n", encoding="utf-8")
    goal_path = tmp_path / "goal.yaml"; goal_path.write_text("data: {}\n", encoding="utf-8")
    universe_path = tmp_path / "universe.yaml"; universe_path.write_text("search: {}\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    nested_source = tmp_path / "nested"; nested_source.mkdir()
    monkeypatch.setattr(finalizer, "_trusted_forbidden", lambda: (test_path.resolve(), submission_path.resolve()))
    monkeypatch.setattr(
        finalizer,
        "load_config",
        lambda path: {"data": {"raw_dir": str(raw), "train_file": "train.csv"}},
    )
    original_guard = finalizer._guard_not_forbidden
    reached_train = []

    class StopBeforeCsvRead(RuntimeError):
        pass

    def guarded(path, forbidden, role):
        resolved = original_guard(path, forbidden, role)
        if role == "Train":
            reached_train.append(resolved)
            raise StopBeforeCsvRead
        return resolved

    csv_reads = []
    monkeypatch.setattr(finalizer, "_guard_not_forbidden", guarded)
    monkeypatch.setattr(finalizer.pd, "read_csv", lambda *args, **kwargs: csv_reads.append(args[0]))
    with pytest.raises(StopBeforeCsvRead):
        finalizer._build_evidence_impl(artifact_root, nested_source, goal_path, universe_path)
    assert reached_train == [train_path.resolve()]
    assert csv_reads == []


def test_evidence_rejects_explicit_noncanonical_test_before_csv_read(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    test_path = raw / "test.csv"; test_path.write_text("secret", encoding="utf-8")
    submission_path = raw / "sample_submission.csv"; submission_path.write_text("ID,SUBCLASS\n", encoding="utf-8")
    alternate_path = raw / "alternate.csv"; alternate_path.write_text("not canonical", encoding="utf-8")
    goal_path = tmp_path / "goal.yaml"; goal_path.write_text("data: {}\n", encoding="utf-8")
    universe_path = tmp_path / "universe.yaml"; universe_path.write_text("search: {}\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    nested_source = tmp_path / "nested"; nested_source.mkdir()
    monkeypatch.setattr(finalizer, "_trusted_forbidden", lambda: (test_path.resolve(), submission_path.resolve()))
    monkeypatch.setattr(
        finalizer,
        "load_config",
        lambda path: {
            "data": {
                "raw_dir": str(raw),
                "train_file": "train.csv",
                "test_file": alternate_path.name,
            }
        },
    )
    csv_reads = []
    monkeypatch.setattr(finalizer.pd, "read_csv", lambda *args, **kwargs: csv_reads.append(args[0]))
    with pytest.raises(RuntimeError, match="canonical Test/submission paths"):
        finalizer._build_evidence_impl(artifact_root, nested_source, goal_path, universe_path)
    assert csv_reads == []


class _IdentityPreprocessor:
    def transform(self, frame):
        return frame


def _infer_fixture(
    root: Path,
) -> tuple[Path, finalizer.PreflightPlan]:
    root.mkdir(); test_path = root / "test.csv"
    pd.DataFrame({"ID": ["T1", "T2"], "x": [1, 2]}).to_csv(test_path, index=False)
    selection = {"selections": {"primary": "nested_primary", "robustness_challenger": "nested_robustness", "specialization_challenger": "nested_specialization"}}
    (root / "final_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    (root / "candidate_manifest.json").write_text(json.dumps({"candidates": []}), encoding="utf-8")
    variant = {"kind": "single", "models": ["a"], "weights": [1.0], "temperatures": [1.0]}
    raw = root / "nested" / "raw" / "fixture"; raw.mkdir(parents=True)
    nested_sources = {
        "src/pipelines/jyp_preprocessing/pipeline_jyp_f9.py": finalizer._sha256(
            finalizer.REPO_ROOT / "src/pipelines/jyp_preprocessing/pipeline_jyp_f9.py"
        )
    }
    nested_bundle = finalizer._canonical_hash(nested_sources)
    (raw / "nested_run_manifest.json").write_text(json.dumps({
        "trusted_input_hashes": {
            "sources": nested_sources,
            "source_bundle": nested_bundle,
        }
    }), encoding="utf-8")
    raw_hash = finalizer._sha256(raw / "nested_run_manifest.json")
    inference_sources = finalizer._inference_source_inventory(nested_sources, ())
    frozen = {"class_names": [f"C{i}" for i in range(26)], "selections": selection["selections"], "role_to_lane": {role: role for role in finalizer.LANES}, "inference_source_base_files": nested_sources, "inference_source_base_bundle": nested_bundle, "inference_source_files": inference_sources, "inference_source_hash": finalizer._canonical_hash(inference_sources), "selection_hash": finalizer._sha256(root / "final_selection.json"), "candidate_manifest_hash": finalizer._sha256(root / "candidate_manifest.json"), "nested_raw_path": str(raw), "nested_raw_identity": {"hashes": {"nested_run_manifest.json": raw_hash}}, "selected_variants": [{"strategy": lane, "seed": seed, "fold": fold, "variant": variant} for lane in finalizer.LANES for seed in finalizer.SEEDS for fold in finalizer.FOLDS], "candidates": []}
    (root / "frozen_manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
    freeze_hash = finalizer._sha256(root / "frozen_manifest.json")
    (root / "finalization_state.json").write_text(json.dumps({"state": "frozen", "freeze_hash": freeze_hash, "test_read_count": 0}), encoding="utf-8")
    train = pd.DataFrame({"ID": ["R1", "R2"], "SUBCLASS": ["C0", "C1"], "x": [0, 1]})
    train_path = root / "train.csv"; train.to_csv(train_path, index=False)
    specs = []
    preflight_dir = root / "inference_checkpoints" / "preflight"
    for seed in finalizer.SEEDS:
        bundles = [{"fold": fold, "preprocessor": _IdentityPreprocessor(), "model": object(), "probability_mode": "native_proba", "calibrator": None} for fold in finalizer.FOLDS]
        bundle_path = root / "bundles" / f"a_{seed}.pkl"; bundle_path.parent.mkdir(exist_ok=True)
        with bundle_path.open("wb") as handle: pickle.dump(bundles, handle)
        identity = {"fixture": True, "alias": "a", "seed": seed}
        spec = finalizer.RunSpec("a", seed, {}, bundle_path, finalizer._sha256(bundle_path), train_path, "SUBCLASS", "ID", test_path, "native_proba", identity)
        specs.append(spec)
        finalizer._atomic_json(preflight_dir / f"{finalizer._checkpoint_name('a', seed)}.json", finalizer._integrity_envelope(identity))
    return test_path, finalizer.PreflightPlan(tuple(specs), tuple(train.columns), "SUBCLASS", "ID", test_path)


def test_global_infer_reads_test_once_and_averages_fifteen_members(tmp_path, monkeypatch):
    root = tmp_path / "infer"; test_path, plan = _infer_fixture(root)
    submission_path = root / "sample_submission.csv"; submission_path.write_text("ID,SUBCLASS\n", encoding="utf-8")
    monkeypatch.setattr(finalizer, "_trusted_forbidden", lambda: (test_path.resolve(), submission_path.resolve()))
    monkeypatch.setattr(finalizer, "_prepare_all", lambda artifact_root, frozen, **kwargs: plan)
    probability = np.full((2, 26), .001); probability[:, 0] = .975; probability /= probability.sum(1, keepdims=True)
    monkeypatch.setattr(finalizer.bank, "_predict_probability", lambda *args, **kwargs: probability.copy())
    monkeypatch.setattr(finalizer.bank, "_local_class_names", lambda *args, **kwargs: [f"C{i}" for i in range(26)])
    original_release = finalizer._release_run; released = []
    def tracked_release(run=None):
        if run is not None: released.append((id(run), len(run.bundles)))
        return original_release(run)
    monkeypatch.setattr(finalizer, "_release_run", tracked_release)
    original_open = Path.open; test_reads = []
    def tracked_open(path, *args, **kwargs):
        if path.resolve() == test_path.resolve() and (args[0] if args else kwargs.get("mode", "r")) == "rb":
            test_reads.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", tracked_open)
    manifest_path = finalizer.infer(root)
    manifest = json.loads(manifest_path.read_text())
    state = json.loads((root / "finalization_state.json").read_text())
    assert len(test_reads) == 1
    assert all(item["members"] == 15 for item in manifest["submissions"])
    assert state["canonical_test_read_count"] == 1
    assert state["fit_after_test"] == 0
    assert state["warm_after_test"] == 0
    assert state["selection_or_tuning_after_test"] == 0
    assert state["bundle_load_after_test_count"] == 3
    assert len(released) == 3
    assert all(bundle_count == 5 for _, bundle_count in released)
    prediction_files = {path.suffix for path in (root / "inference_checkpoints" / "predictions").iterdir()}
    assert prediction_files == {".json", ".npz"}
    # Even if an attacker forges all three CSVs and refreshes their SHA metadata,
    # the irreversible canonical Test ID-order hash prevents resume acceptance.
    for item in manifest["submissions"]:
        csv_path = root / item["path"]
        forged = pd.read_csv(csv_path); forged["ID"] = ["X1", "X2"]
        forged.to_csv(csv_path, index=False); item["sha256"] = finalizer._sha256(csv_path)
    (root / "final_submissions.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Test contract"):
        finalizer.infer(root)


def test_bad_prepared_bundle_blocks_before_first_test_read(tmp_path, monkeypatch):
    root = tmp_path / "infer"; test_path, _ = _infer_fixture(root)
    submission_path = root / "sample_submission.csv"; submission_path.write_text("ID,SUBCLASS\n", encoding="utf-8")
    monkeypatch.setattr(finalizer, "_trusted_forbidden", lambda: (test_path.resolve(), submission_path.resolve()))
    monkeypatch.setattr(finalizer, "_prepare_all", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad second bundle")))
    original = pd.read_csv; test_reads = []

    def tracked(path, *args, **kwargs):
        if Path(path).resolve() == test_path.resolve(): test_reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(finalizer.pd, "read_csv", tracked)
    with pytest.raises(RuntimeError, match="bad second bundle"):
        finalizer.infer(root)
    assert test_reads == []


def test_transitive_pipeline_source_drift_blocks_before_first_test_read(tmp_path, monkeypatch):
    root = tmp_path / "infer"; test_path, _ = _infer_fixture(root)
    submission_path = root / "sample_submission.csv"; submission_path.write_text("ID,SUBCLASS\n", encoding="utf-8")
    monkeypatch.setattr(finalizer, "_trusted_forbidden", lambda: (test_path.resolve(), submission_path.resolve()))
    nested_source = (finalizer.REPO_ROOT / "src/pipelines/jyp_preprocessing/pipeline_jyp_f9.py").resolve()
    original_sha256 = finalizer._sha256

    def drift_nested_source(path):
        if Path(path).resolve() == nested_source:
            return "0" * 64
        return original_sha256(Path(path))

    monkeypatch.setattr(finalizer, "_sha256", drift_nested_source)
    original_open = Path.open; test_reads = []

    def tracked_open(path, *args, **kwargs):
        if path.resolve() == test_path.resolve():
            test_reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    with pytest.raises(RuntimeError, match="source drifted"):
        finalizer.infer(root)
    assert test_reads == []


@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink"])
def test_inference_inventory_rejects_forbidden_alias_before_hash(
    tmp_path, monkeypatch, alias_kind
):
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
    monkeypatch.setattr(
        finalizer, "_trusted_forbidden", lambda: (forbidden.resolve(), (tmp_path / "submission.csv").resolve())
    )
    monkeypatch.setattr(
        finalizer, "_sha256", lambda path: (_ for _ in ()).throw(AssertionError("hash reader reached"))
    )
    with pytest.raises(ValueError, match="aliases configured"):
        finalizer._inference_source_inventory(
            {str(alias): "0" * 64}, finalizer._trusted_forbidden()
        )


def test_post_open_failure_is_irreversible_and_never_reopens_test(tmp_path, monkeypatch):
    root = tmp_path / "infer"; test_path, plan = _infer_fixture(root)
    submission_path = root / "sample_submission.csv"; submission_path.write_text("ID,SUBCLASS\n", encoding="utf-8")
    monkeypatch.setattr(finalizer, "_trusted_forbidden", lambda: (test_path.resolve(), submission_path.resolve()))
    monkeypatch.setattr(finalizer, "_prepare_all", lambda *args, **kwargs: plan)
    monkeypatch.setattr(finalizer.bank, "_predict_probability", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("prediction failed")))
    original_open = Path.open; reads = []
    def tracked_open(path, *args, **kwargs):
        if path.resolve() == test_path.resolve() and (args[0] if args else kwargs.get("mode", "r")) == "rb":
            reads.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", tracked_open)
    with pytest.raises(RuntimeError, match="prediction failed"):
        finalizer.infer(root)
    assert json.loads((root / "finalization_state.json").read_text())["state"] == "test_opened"
    monkeypatch.setattr(finalizer.bank, "_predict_probability", lambda *args, **kwargs: np.eye(26, dtype=float)[[0, 0]])
    finalizer.infer(root)
    assert len(reads) == 1


def test_corrupt_test_cache_is_hard_stop_without_canonical_reopen(tmp_path, monkeypatch):
    root = tmp_path / "infer"; root.mkdir()
    test_path = tmp_path / "test.csv"; test_path.write_text("ID,x\nT1,1\n", encoding="utf-8")
    freeze_hash = "f" * 64
    original_open = Path.open; reads = []
    def tracked_open(path, *args, **kwargs):
        if path.resolve() == test_path.resolve() and (args[0] if args else kwargs.get("mode", "r")) == "rb":
            reads.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", tracked_open)
    content, opened = finalizer._cached_test_bytes(root, test_path, freeze_hash)
    assert opened is True and content.startswith(b"ID,x") and len(reads) == 1
    (root / "inference_checkpoints" / "test_cache" / "test.csv.bytes").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="hard stop"):
        finalizer._cached_test_bytes(root, test_path, freeze_hash, allow_open=False)
    assert len(reads) == 1


def test_prediction_checkpoint_rejects_corruption_and_stale_identity(tmp_path):
    path = tmp_path / "prediction.npz"; receipt = tmp_path / "prediction.json"
    shape = (2, 26); values = {f"fold_{fold}": np.full(shape, 1 / 26) for fold in finalizer.FOLDS}
    identity = {"freeze_hash": "a" * 64, "test_hash": "b" * 64, "bundle_hash": "c" * 64}
    finalizer._atomic_npz(path, **values)
    finalizer._atomic_json(receipt, finalizer._integrity_envelope({"identity": identity, "npz_hash": finalizer._sha256(path)}))
    assert finalizer._valid_prediction_checkpoint(path, receipt, identity, shape) is True
    assert finalizer._valid_prediction_checkpoint(path, receipt, {**identity, "test_hash": "d" * 64}, shape) is False
    path.write_bytes(b"not-an-npz")
    assert finalizer._valid_prediction_checkpoint(path, receipt, identity, shape) is False


def test_release_run_drops_all_bundle_references():
    run = finalizer.PreparedRun("a", 42, {}, [{"fold": fold} for fold in finalizer.FOLDS])
    finalizer._release_run(run)
    assert run.bundles == []


def test_specialist_reaches_repeated_pair_rescue_and_outside_noninferiority():
    rows, classes = 30, 26
    labels = np.array(([0, 1, 2] * 10), dtype=int)
    assignments = pd.concat([pd.DataFrame({"seed": seed, "row_index": np.arange(rows), "ID": [f"R{i}" for i in range(rows)], "fold": np.arange(rows) % 5}) for seed in finalizer.SEEDS], ignore_index=True)
    backbone = np.full((3, rows, classes), 1e-6); expert = np.full_like(backbone, 1e-6)
    for seed_index in range(3):
        for row, label in enumerate(labels):
            if label == 0:
                backbone[seed_index, row, 1] = .55; backbone[seed_index, row, 0] = .40
                expert[seed_index, row, 0] = .99
            else:
                backbone[seed_index, row, label] = .99; expert[seed_index, row, label] = .99
    backbone /= backbone.sum(2, keepdims=True); expert /= expert.sum(2, keepdims=True)
    variant = {"kind": "collision_gate", "backbone_variant": {"kind": "single", "models": ["b"], "weights": [1.0], "temperatures": [1.0]}, "expert": "e", "ordered_pair": [0, 1], "expert_temperature": 1.0, "margin_threshold": .20, "expert_confidence_threshold": .9, "expert_weight": .2, "global_weight": .8}
    records = {(lane, seed, fold): {"selected_variant": variant, "gate_evidence": {}} for lane in finalizer.LANES for seed in finalizer.SEEDS for fold in finalizer.FOLDS}
    gated = np.stack([nested._apply_variant(variant, {"b": backbone[index], "e": expert[index]}) for index in range(3)])
    qualifies, net, details = finalizer._collision_evidence(records, "specialization_challenger", {"b": backbone, "e": expert}, gated, assignments, labels)
    assert qualifies is True
    assert net > 0
    assert sum(row["rescue"] for row in details["seed_evidence"]) >= 3
    assert sum(row["pair_delta"] >= .05 for row in details["seed_evidence"]) >= 2
    assert details["outside_bootstrap"]["delta"] >= -details["outside_epsilon"]


@pytest.mark.parametrize("failure_point", ["nested_snapshot", "derived_candidate", "generation_commit"])
def test_evidence_failure_removes_new_dirs_and_restores_prior_state_audit(tmp_path, monkeypatch, failure_point):
    root = tmp_path / "artifacts"; root.mkdir()
    old_state = b'{"state":"prior"}'; old_audit = b'{"events":[{"action":"read","path":"train.csv"}]}'
    (root / "finalization_state.json").write_bytes(old_state); (root / "io_audit.json").write_bytes(old_audit)
    def fail(artifact_root, nested_source, goal_config, search_universe):
        created = {
            "nested_snapshot": artifact_root / "nested" / "raw" / "new",
            "derived_candidate": artifact_root / "candidates" / "alias" / "new",
            "generation_commit": artifact_root / "generations" / "new",
        }[failure_point]
        created.mkdir(parents=True); (created / "payload").write_text("new")
        (artifact_root / "finalization_state.json").write_text('{"state":"corrupt"}')
        (artifact_root / "io_audit.json").write_text('{"events":[]}')
        raise RuntimeError(failure_point)
    monkeypatch.setattr(finalizer, "_build_evidence_impl", fail)
    with pytest.raises(RuntimeError, match=failure_point):
        finalizer.build_evidence(root, tmp_path / "nested", tmp_path / "goal.yaml", tmp_path / "universe.yaml")
    assert (root / "finalization_state.json").read_bytes() == old_state
    assert (root / "io_audit.json").read_bytes() == old_audit
    assert finalizer._tracked_directories(root) == set()


def test_metadata_only_materialized_fixture_is_rejected_by_real_evaluator(tmp_path):
    root = tmp_path / "complete"; root.mkdir()
    rows, classes = 6201, [f"C{i:02d}" for i in range(26)]
    alignment = pd.concat([
        pd.DataFrame({"seed": seed, "row_index": np.arange(rows), "ID": [f"I{i:05d}" for i in range(rows)], "fold": np.arange(rows) % 5})
        for seed in finalizer.SEEDS
    ], ignore_index=True)
    alignment.to_csv(root / "fold_assignments.csv", index=False)
    (root / "class_names.json").write_text(json.dumps(classes), encoding="utf-8")
    protected = []
    for index in range(69):
        digest = hashlib.sha256(str(index).encode()).hexdigest()
        protected.append({"path": f"protected_{index}", "before": digest, "after": digest, "unchanged": True})
    (root / "protected_file_hashes_before_after.json").write_text(json.dumps({"unchanged": True, "files": protected}), encoding="utf-8")
    probability = np.full((3, rows, 26), 1 / 26, dtype=np.float64)
    candidates = []
    for name in ("nested_primary", "nested_robustness", "nested_specialization"):
        directory = root / "candidates" / name; directory.mkdir(parents=True)
        np.save(directory / "probability.npy", probability)
        alignment.to_csv(directory / "alignment.csv", index=False)
        candidates.append({"name": name, "class_names": classes, "probability_path": f"candidates/{name}/probability.npy", "alignment_path": f"candidates/{name}/alignment.csv"})
    (root / "candidate_manifest.json").write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
    np.save(root / "nested_oof_probability.npy", probability)
    evaluations = [{"seed": seed, "fold": fold, "macro_f1": .5} for seed in finalizer.SEEDS for fold in finalizer.FOLDS]
    (root / "nested_cv_metrics.json").write_text(json.dumps({"outer_evaluations": evaluations}), encoding="utf-8")
    (root / "class_metrics.json").write_text(json.dumps({"classes": [{"class_name": name, "macro_f1": .5} for name in classes]}), encoding="utf-8")
    (root / "subgroup_metrics.json").write_text(json.dumps({"subgroups": [{"dimension": dimension, "macro_f1": .4} for dimension in ("burden", "novelty")]}), encoding="utf-8")
    comparison = {"candidate": "nested_robustness", "baseline": "nested_primary", "delta": 0, "se_seed": .01, "se_boot": .02, "epsilon": .02, "ci_lower": -.01, "ci_upper": .01}
    (root / "paired_comparisons.json").write_text(json.dumps({"comparisons": [comparison]}), encoding="utf-8")
    constraints = {name: {"pass": True, "repeated_collapse_count": 0} for name in ("seed", "class", "burden", "novelty")}
    (root / "collapse_constraints.json").write_text(json.dumps({"overall_pass": True, "constraints": constraints}), encoding="utf-8")
    (root / "ensemble_ablation.json").write_text(json.dumps({"complete": True, "ablations": [{"name": "fixed", "macro_f1": .5, "delta": 0}]}), encoding="utf-8")
    selection = {"selections": {"primary": "nested_primary", "robustness_challenger": "nested_robustness", "specialization_challenger": "nested_specialization"}, "selection_evidence": {"selection_order": ["mean_nested_oof", "one_se_eligibility", "overfit_gap", "simpler_structure"], "gap_used_only_within_one_se": True, "one_se": {"best_mean": .5, "best_se": .01, "threshold": .49, "eligible_candidates": ["nested_primary", "nested_robustness"]}, "evidence": ["Train-only evidence"]}}
    (root / "final_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    digest = "a" * 64
    (root / "frozen_manifest.json").write_text(json.dumps({"frozen": True, "selection_uses_test": False, "thresholds_frozen": True, "source_hash": digest, "train_hash": digest, "fold_hash": digest, "bundle_hash": digest}), encoding="utf-8")
    submission_entries = []
    for role in ("primary", "robustness_challenger", "specialization_challenger"):
        path = root / f"submission_{role}.csv"; path.write_text("ID,SUBCLASS\nI0,C00\n", encoding="utf-8")
        submission_entries.append({"role": role, "path": path.name, "sha256": finalizer._sha256(path), "row_count": 1, "columns": ["ID", "SUBCLASS"]})
    (root / "final_submissions.json").write_text(json.dumps({"submissions": submission_entries}), encoding="utf-8")
    (root / "io_audit.json").write_text(json.dumps({"events": [], "summary": {"oof_test_reads": 0, "nested_test_reads": 0, "oof_submission_reads": 0, "nested_submission_reads": 0}}), encoding="utf-8")
    (root / "final_report_ko.md").write_text("# 최종 보고서\n\n" + "Train-only 완전 Nested OOF 검증과 동결 후 단일 Test 추론을 확인했습니다. " * 5, encoding="utf-8")
    (root / "leakage_checklist.json").write_text(json.dumps({"complete": True, "all_passed": True, "test_used_for_selection": False, "items": [{"passed": True}]}), encoding="utf-8")
    result = evaluate(root)
    assert not result.passed
    assert {"protected_file_hashes", "io_audit", "candidate_oof_probabilities", "frozen_manifest", "final_submissions"}.intersection(result.report(True)["failed_checks"])


def test_real_nested_runner_source_inventory_snapshots_with_live_contract(tmp_path, request):
    from tests.test_run_test_007_nested import synthetic_nested, _predictor
    train, raw, goal_path, universe_path, fold_path = synthetic_nested.__wrapped__(tmp_path)
    goal = yaml.safe_load(goal_path.read_text())
    universe = yaml.safe_load(universe_path.read_text())
    template = dict(goal["candidates"][0])
    for alias in ("extra1", "extra2", "extra3"):
        goal["candidates"].append({**template, "alias": alias})
        universe["search_universe"]["models"].append(alias)
        universe["search_universe"]["roles"][alias] = ["COLLISION_EXPERT"] if alias != "extra3" else []
    goal["search"]["max_models"] = 5
    universe["search_universe"]["max_models"] = 5
    goal_path.write_text(yaml.safe_dump(goal, sort_keys=False), encoding="utf-8")
    universe_path.write_text(yaml.safe_dump(universe, sort_keys=False), encoding="utf-8")
    output = tmp_path / "nested_output"
    def predictor(candidate, train_x, train_y, valid_x, class_names, model_seed, context):
        truth = valid_x["signal"].to_numpy(int)
        probability = np.full((len(valid_x), len(class_names)), 1e-5)
        probability[np.arange(len(valid_x)), truth] = .98
        if candidate.alias in {"backbone", "robust", "extra3"}:
            mask = truth == 0; probability[mask, 0] = .45; probability[mask, 1] = .54
        probability /= probability.sum(1, keepdims=True)
        return probability
    nested.run_nested(goal_path, universe_path, fold_path, output, predictor=predictor)
    candidate_configs = {item["alias"]: Path(item["config"]) for item in goal["candidates"]}
    snapshot_root = tmp_path / "snapshot"; snapshot_root.mkdir()
    destination, identity = finalizer._snapshot_nested(
        snapshot_root, output, (raw / "test.csv", raw / "sample_submission.csv"),
        {"goal_path": goal_path, "universe_path": universe_path, "train_path": raw / "train.csv", "fold_path": fold_path, "candidate_configs": candidate_configs},
    )
    manifest = json.loads((output / "nested_run_manifest.json").read_text())
    assert identity["verified_from_live_inputs"] is True
    assert len(manifest["trusted_input_hashes"]["sources"]) > 25
    assert destination.is_dir()


def test_inferred_document_and_evaluator_never_reopen_canonical_test(tmp_path, monkeypatch):
    from tests.test_evaluate_test_007_specialization import _make_complete
    import src.test_007.evaluate_test_007_specialization as evaluator
    root = _make_complete(tmp_path / "complete")
    test_path = evaluator.CANONICAL_TEST.resolve(); reads = []
    original = pd.read_csv
    def tracked(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).resolve() == test_path: reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", tracked)
    finalizer.document(root)
    assert evaluator.evaluate(root).passed
    assert reads == []


def test_document_final_state_write_failure_restores_snapshot(tmp_path, monkeypatch):
    import src.test_007.evaluate_test_007_specialization as evaluator

    root = tmp_path / "artifacts"; root.mkdir()
    state_path = root / "finalization_state.json"
    report_path = root / "final_report_ko.md"
    checklist_path = root / "leakage_checklist.json"
    state = {"state": "inferred", "canonical_test_read_count": 1, "fit_after_test": 0,
             "warm_after_test": 0, "selection_or_tuning_after_test": 0}
    selections = {"primary": "nested_primary", "robustness_challenger": "nested_robustness",
                  "specialization_challenger": "nested_specialization"}
    state_path.write_text(json.dumps(state), encoding="utf-8")
    report_path.write_bytes(b"prior report")
    checklist_path.write_bytes(b'{"prior":true}')
    (root / "frozen_manifest.json").write_text(json.dumps({"selections": selections}), encoding="utf-8")
    (root / "final_selection.json").write_text(json.dumps({"selections": selections, "selection_evidence": {}}), encoding="utf-8")
    before = {path: path.read_bytes() for path in (report_path, checklist_path, state_path)}

    class Check:
        def __init__(self, name, passed): self.name, self.passed = name, passed
    class Result:
        checks = [Check("documentation", True)]
        passed = True
        def report(self, require_complete): return {"failed_checks": []}

    monkeypatch.setattr(finalizer, "_verify_submission_manifest", lambda *args: None)
    monkeypatch.setattr(evaluator, "evaluate", lambda artifact_root: Result())
    original_atomic_json = finalizer._atomic_json

    def fail_final_state(path, value):
        if Path(path) == state_path and value.get("final_evaluation_pass") is True:
            raise OSError("injected final state write failure")
        return original_atomic_json(path, value)

    monkeypatch.setattr(finalizer, "_atomic_json", fail_final_state)
    with pytest.raises(OSError, match="injected final state write failure"):
        finalizer.document(root)
    assert {path: path.read_bytes() for path in before} == before
