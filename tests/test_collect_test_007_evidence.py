from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.test_007 import collect_test_007_evidence as collector


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _entry(path: Path) -> dict:
    result = {"sha256": collector._sha256(path), "size_bytes": path.stat().st_size}
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        result.update({"shape": list(array.shape), "dtype": str(array.dtype)})
    return result


def _identity(repo: Path, config_path: Path, config: dict, seed: int, fold_hash: str) -> dict:
    source_paths = collector._expected_sources(config, repo)
    sources = {path.relative_to(repo).as_posix(): collector._sha256(path) for path in source_paths}
    protected = {name: sources[name] for name in collector.PROTECTED_FILES}
    value = {
        "schema_version": 1,
        "phase": "oof",
        "config_path": str(config_path.resolve()),
        "config_hash": collector._sha256(config_path),
        "source_hashes": sources,
        "source_hash": collector._canonical_hash(sources),
        "protected_hashes_before": protected,
        "train_data_hash": collector._sha256(repo / "data/raw/train.csv"),
        "fold_assignments_input_hash": fold_hash,
        "library_versions": {"python": "test"},
        "external_runtime_snapshot": {},
        "seed": seed,
        "folds": 5,
        "test_accessed": False,
    }
    value["manifest_identity_hash"] = collector._canonical_hash(value)
    return value


def _write_run(repo: Path, root: Path, alias: str, experiment: str, config_path: Path, config: dict, assignments: pd.DataFrame, classes: list[str]) -> None:
    config_hash = collector._sha256(config_path)
    fold_hash = collector._sha256(root / "fold_assignments.csv")
    for seed in collector.EXPECTED_SEEDS:
        run = repo / "data/processed/oof_bank" / experiment / config_hash / str(seed)
        run.mkdir(parents=True)
        np.save(run / "oof_probability_by_seed.npy", np.full((1, 6201, 26), 1 / 26, dtype=np.float32), allow_pickle=False)
        np.save(run / "oof_prediction_by_seed.npy", np.zeros((1, 6201), dtype=np.int32), allow_pickle=False)
        assignments.loc[assignments.seed == seed].to_csv(run / "fold_assignments.csv", index=False)
        pd.DataFrame({"fold": range(5), "train_f1": [.5] * 5, "valid_f1": [.4] * 5}).to_csv(run / "fold_metrics.csv", index=False)
        _write_json(run / "class_names.json", classes)
        _write_json(run / "runtime_metrics.json", {"total_seconds": 1.0})
        _write_json(run / "model_passport.json", {"probability_mode": "native_proba"})
        (run / "fold_bundles.pkl").write_bytes(f"bundle-{alias}-{seed}".encode())
        _write_json(
            run / "io_audit.json",
            {
                "phase": "oof",
                "allowed_roles": ["config", "train", "fold_assignments", "checkpoint"],
                "inputs": {
                    "config": {"resolved_path": str(config_path.resolve()), "allowed": True, "accessed": True},
                    "train": {"resolved_path": str((repo / "data/raw/train.csv").resolve()), "allowed": True, "accessed": True},
                    "test": {"resolved_path": str((repo / "data/raw/hidden_eval.csv").resolve()), "allowed": False, "accessed": False},
                    "submission": {"resolved_path": str((repo / "data/raw/hidden_submit.csv").resolve()), "allowed": False, "accessed": False},
                    "fold_assignments": {"resolved_path": str((root / "fold_assignments.csv").resolve()), "allowed": True, "accessed": True},
                },
                "test_accessed": False,
            },
        )
        artifacts = {name: _entry(run / name) for name in collector.REQUIRED_ARTIFACTS}
        manifest = {
            **_identity(repo, config_path, config, seed, fold_hash),
            "score_source": "train_only_outer_oof",
            "train_rows": 6201,
            "probability_mode": "native_proba",
            "protected_hashes_after": _identity(repo, config_path, config, seed, fold_hash)["protected_hashes_before"],
            "class_names": classes,
            "class_names_hash": artifacts["class_names.json"]["sha256"],
            "fold_assignments_hash": artifacts["fold_assignments.csv"]["sha256"],
            "artifact_bundle_hash": artifacts["fold_bundles.pkl"]["sha256"],
            "artifacts": artifacts,
        }
        _write_json(run / "run_manifest.json", manifest)


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(collector, "_live_library_versions", lambda config: {"python": "test"})
    repo = tmp_path / "repo"
    for directory in ("configs", "data/raw", "src/models", "src/pipelines/jyp_preprocessing"):
        (repo / directory).mkdir(parents=True, exist_ok=True)
    (repo / "data/raw/train.csv").write_text("ID,SUBCLASS\nR0,C00\n", encoding="utf-8")
    (repo / "data/raw/hidden_eval.csv").write_text("ID\nT0\n", encoding="utf-8")
    (repo / "data/raw/hidden_submit.csv").write_text("ID,SUBCLASS\n", encoding="utf-8")
    for path in (
        "src/test_007/run_test_007_oof_bank.py",
        *collector.PROTECTED_FILES,
        "src/models/fake_model.py",
    ):
        target = repo / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(path, encoding="utf-8")
    root = repo / "data/processed/train_only_specialization"; root.mkdir(parents=True)
    classes = [f"C{i:02d}" for i in range(26)]; _write_json(root / "class_names.json", classes)
    assignments = pd.concat([
        pd.DataFrame({"seed": seed, "row_index": np.arange(6201), "ID": [f"R{i:04d}" for i in range(6201)], "fold": np.arange(6201) % 5})
        for seed in collector.EXPECTED_SEEDS
    ], ignore_index=True)
    assignments.to_csv(root / "fold_assignments.csv", index=False)
    config = {
        "project": {"experiment_name": "synthetic_candidate"},
        "data": {"raw_dir": "data/raw", "train_file": "train.csv", "test_file": "hidden_eval.csv", "submission_file": "hidden_submit.csv"},
        "preprocessing": {"name": "pipeComb_v3"}, "model": {"name": "fake"},
    }
    candidate = repo / "configs/candidate.yaml"; candidate.write_text(yaml.safe_dump(config), encoding="utf-8")
    goal_value = {
        "data": {"artifact_root": "data/processed/train_only_specialization", "oof_bank_root": "data/processed/oof_bank", "raw_dir": "data/raw", "train_file": "train.csv", "test_file": "hidden_eval.csv", "submission_file": "hidden_submit.csv"},
        "candidates": [{"alias": "candidate", "config": "configs/candidate.yaml"}],
    }
    goal = repo / "configs/goal.yaml"; goal.write_text(yaml.safe_dump(goal_value), encoding="utf-8")
    _write_run(repo, root, "candidate", "synthetic_candidate", candidate, config, assignments, classes)
    return repo, goal, root, assignments, classes


def _run_dir(repo: Path, seed: int) -> Path:
    config = repo / "configs/candidate.yaml"
    return repo / "data/processed/oof_bank/synthetic_candidate" / collector._sha256(config) / str(seed)


def _add_second_candidate(repo: Path, goal: Path, root: Path, assignments: pd.DataFrame, classes: list[str]) -> None:
    model = repo / "src/models/fake2_model.py"; model.write_text("fake2", encoding="utf-8")
    config = {
        "project": {"experiment_name": "synthetic_candidate2"},
        "data": {"raw_dir": "data/raw", "train_file": "train.csv", "test_file": "hidden_eval.csv", "submission_file": "hidden_submit.csv"},
        "preprocessing": {"name": "pipeComb_v3"}, "model": {"name": "fake2"},
    }
    config_path = repo / "configs/candidate2.yaml"; config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    goal_value = yaml.safe_load(goal.read_text(encoding="utf-8"))
    goal_value["candidates"].append({"alias": "candidate2", "config": "configs/candidate2.yaml"})
    goal.write_text(yaml.safe_dump(goal_value), encoding="utf-8")
    _write_run(repo, root, "candidate2", "synthetic_candidate2", config_path, config, assignments, classes)


def _resign_artifact(run: Path, name: str) -> dict:
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][name] = _entry(run / name)
    if name == "fold_assignments.csv": manifest["fold_assignments_hash"] = manifest["artifacts"][name]["sha256"]
    if name == "class_names.json": manifest["class_names_hash"] = manifest["artifacts"][name]["sha256"]
    _write_json(manifest_path, manifest)
    return manifest


def test_collects_versioned_three_seed_bank_and_derived_audit(synthetic):
    repo, goal, root, _, classes = synthetic
    output = collector.collect_evidence(goal, ["candidate"], repo_root=repo)
    manifest = json.loads(output.read_text(encoding="utf-8")); item = manifest["candidates"][0]
    probability = np.load(root / item["probability_path"], allow_pickle=False)
    alignment = pd.read_csv(root / item["alignment_path"])
    audit = json.loads((root / "io_audit.json").read_text(encoding="utf-8"))
    assert probability.shape == (3, 6201, 26) and np.allclose(probability.sum(axis=-1), 1)
    assert len(alignment) == 18603 and item["class_names"] == classes
    assert item["collection_identity_hash"] in item["probability_path"]
    assert audit["selection_uses_test"] is False
    assert all(audit["summary"][key] == 0 for key in ("oof_test_reads", "nested_test_reads", "oof_submission_reads", "nested_submission_reads"))
    assert any(event["evidence_kind"] == "verified_collector_read" for event in audit["events"])
    assert any(event["evidence_kind"] == "verified_upstream_declaration" for event in audit["events"])


def test_empty_or_incomplete_upstream_audit_fails(synthetic):
    repo, goal, _, _, _ = synthetic; run = _run_dir(repo, 777)
    _write_json(run / "io_audit.json", {"phase": "oof", "inputs": {}, "test_accessed": False})
    _resign_artifact(run, "io_audit.json")
    with pytest.raises(ValueError, match="empty upstream OOF audit"):
        collector.collect_evidence(goal, ["candidate"], repo_root=repo)


def test_cross_seed_row_to_id_shift_fails_even_when_artifact_resigned(synthetic):
    repo, goal, _, _, _ = synthetic; run = _run_dir(repo, 2026)
    frame = pd.read_csv(run / "fold_assignments.csv"); frame.loc[0, "ID"], frame.loc[1, "ID"] = frame.loc[1, "ID"], frame.loc[0, "ID"]
    frame.to_csv(run / "fold_assignments.csv", index=False); _resign_artifact(run, "fold_assignments.csv")
    with pytest.raises(ValueError, match="alignment differs from canonical"):
        collector.collect_evidence(goal, ["candidate"], repo_root=repo)


@pytest.mark.parametrize("target_kind", ["direct_train", "source_symlink", "cache_hardlink", "protected_hardlink"])
def test_custom_test_aliases_are_blocked_before_target_reader_call(synthetic, monkeypatch, target_kind):
    repo, goal, root, _, _ = synthetic; test = repo / "data/raw/hidden_eval.csv"
    if target_kind == "cache_hardlink":
        collector.collect_evidence(goal, ["candidate"], repo_root=repo)
        manifest = json.loads((root / "candidate_manifest.json").read_text(encoding="utf-8")); target = root / manifest["candidates"][0]["probability_path"]
        target.unlink()
        try: os.link(test, target)
        except OSError: pytest.skip("hardlinks unavailable")
    elif target_kind == "source_symlink":
        target = repo / "src/models/fake_model.py"; target.unlink()
        try: target.symlink_to(test)
        except OSError: pytest.skip("symlinks unavailable")
    elif target_kind == "direct_train":
        config_path = repo / "configs/candidate.yaml"; config = yaml.safe_load(config_path.read_text(encoding="utf-8")); config["data"]["train_file"] = "hidden_eval.csv"; config_path.write_text(yaml.safe_dump(config), encoding="utf-8"); target = test
    else:
        protected = repo / "src/train.py"; digest = collector._sha256(protected)
        _write_json(root / "protected_file_hashes_before_after.json", {"before": {"src/train.py": {"sha256": digest}}})
        protected.unlink()
        try: os.link(test, protected)
        except OSError: pytest.skip("hardlinks unavailable")
        target = test
    original_open = Path.open; calls = []
    def watched(path, *args, **kwargs):
        try:
            if path.exists() and test.exists() and os.path.samefile(path, test): calls.append(path)
        except OSError: pass
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", watched)
    with pytest.raises((RuntimeError, ValueError)):
        if target_kind == "protected_hardlink": collector.finalize_protected(goal, root, repo_root=repo)
        else: collector.collect_evidence(goal, ["candidate"], repo_root=repo)
    assert calls == []


def test_fake_source_and_train_identity_are_rejected_even_if_manifest_is_cosigned(synthetic):
    repo, goal, root, _, _ = synthetic; run = _run_dir(repo, 42)
    manifest_path = run / "run_manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_hashes"]["src/models/fake_model.py"] = "f" * 64
    manifest["source_hash"] = collector._canonical_hash(manifest["source_hashes"])
    manifest["train_data_hash"] = "a" * 64
    identity_keys = ("schema_version", "phase", "config_path", "config_hash", "source_hashes", "source_hash", "protected_hashes_before", "train_data_hash", "fold_assignments_input_hash", "library_versions", "external_runtime_snapshot", "seed", "folds", "test_accessed")
    manifest["manifest_identity_hash"] = collector._canonical_hash({key: manifest[key] for key in identity_keys})
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="live manifest identity mismatch"):
        collector.collect_evidence(goal, ["candidate"], repo_root=repo)


def test_cache_manifest_and_probability_cotamper_does_not_bypass_source_restack(synthetic):
    repo, goal, root, _, _ = synthetic; collector.collect_evidence(goal, ["candidate"], repo_root=repo)
    manifest_path = root / "candidate_manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); item = manifest["candidates"][0]
    path = root / item["probability_path"]; values = np.load(path, allow_pickle=False); values[:, :, :] = 1 / 26; values[:, 0, :] = 0; values[:, 0, 1] = 1
    np.save(path, values, allow_pickle=False); item["probability_sha256"] = collector._sha256(path); _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="disagrees with revalidated source runs"):
        collector.collect_evidence(goal, ["candidate"], repo_root=repo)


def test_cached_metadata_cotamper_is_rebuilt_from_verified_sources(synthetic):
    repo, goal, root, _, classes = synthetic; collector.collect_evidence(goal, ["candidate"], repo_root=repo)
    manifest_path = root / "candidate_manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); item = manifest["candidates"][0]
    item["class_names"] = list(reversed(classes)); item["seeds"] = [777]
    item["source_runs"] = [{"seed": 777, "train_hash": "f" * 64}]; item["test_accessed"] = True
    _write_json(manifest_path, manifest)
    collector.collect_evidence(goal, ["candidate"], repo_root=repo)
    rebuilt = json.loads(manifest_path.read_text(encoding="utf-8"))["candidates"][0]
    assert rebuilt["class_names"] == classes
    assert rebuilt["seeds"] == [42, 2026, 777]
    assert [run["seed"] for run in rebuilt["source_runs"]] == [42, 2026, 777]
    assert rebuilt["test_accessed"] is False
    assert rebuilt["probability_sha256"] == collector._sha256(root / rebuilt["probability_path"])
    assert rebuilt["alignment_sha256"] == collector._sha256(root / rebuilt["alignment_path"])


def test_invoked_analysis_stage_requires_nonempty_audit(synthetic):
    repo, goal, root, _, _ = synthetic; (root / "nested").mkdir(); (root / "nested/metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError): collector.collect_evidence(goal, ["candidate"], repo_root=repo)
    _write_json(root / "nested/io_audit.json", {"stage": "nested", "events": [], "summary": {"test_reads": 0, "submission_reads": 0}, "selection_uses_test": False})
    with pytest.raises(ValueError, match="events are empty"): collector.collect_evidence(goal, ["candidate"], repo_root=repo)


def test_transaction_failure_keeps_previous_manifest_and_artifacts_valid(synthetic):
    repo, goal, root, _, _ = synthetic; collector.collect_evidence(goal, ["candidate"], repo_root=repo)
    manifest_path = root / "candidate_manifest.json"; before = manifest_path.read_bytes(); manifest = json.loads(before); probability = root / manifest["candidates"][0]["probability_path"]; probability_hash = collector._sha256(probability)
    (_run_dir(repo, 777) / "run_manifest.json").unlink()
    with pytest.raises(FileNotFoundError): collector.collect_evidence(goal, ["candidate"], repo_root=repo)
    assert manifest_path.read_bytes() == before and collector._sha256(probability) == probability_hash


def test_failure_after_version_write_removes_generation_and_restores_empty_state(synthetic, monkeypatch):
    repo, goal, root, _, _ = synthetic
    monkeypatch.setattr(collector, "_analysis_audits", lambda *args: (_ for _ in ()).throw(RuntimeError("injected after version")))
    with pytest.raises(RuntimeError, match="injected after version"):
        collector.collect_evidence(goal, ["candidate"], repo_root=repo)
    assert not (root / "candidate_manifest.json").exists()
    assert not (root / "io_audit.json").exists()
    candidate_dir = root / "candidates/candidate"
    assert not candidate_dir.exists() or list(candidate_dir.iterdir()) == []


def test_final_manifest_replace_failure_rolls_back_audit_and_new_version(synthetic, monkeypatch):
    repo, goal, root, assignments, classes = synthetic
    collector.collect_evidence(goal, ["candidate"], repo_root=repo)
    manifest_path, audit_path = root / "candidate_manifest.json", root / "io_audit.json"
    before_manifest, before_audit = manifest_path.read_bytes(), audit_path.read_bytes()
    _add_second_candidate(repo, goal, root, assignments, classes)
    original = collector._atomic_json
    def injected(path, value):
        if path.resolve() == manifest_path.resolve():
            raise OSError("injected final replace")
        return original(path, value)
    monkeypatch.setattr(collector, "_atomic_json", injected)
    with pytest.raises(OSError, match="injected final replace"):
        collector.collect_evidence(goal, ["candidate2"], repo_root=repo)
    assert manifest_path.read_bytes() == before_manifest
    assert audit_path.read_bytes() == before_audit
    candidate2 = root / "candidates/candidate2"
    assert not candidate2.exists() or list(candidate2.iterdir()) == []


def test_finalize_protected_emits_evaluator_schema(synthetic):
    repo, goal, root, _, _ = synthetic; protected = repo / "src/train.py"; digest = collector._sha256(protected)
    _write_json(root / "protected_file_hashes_before_after.json", {"before": {"src/train.py": {"sha256": digest, "bytes": protected.stat().st_size}}})
    _, unchanged = collector.finalize_protected(goal, root, repo_root=repo)
    result = json.loads((root / "protected_file_hashes_before_after.json").read_text(encoding="utf-8"))
    assert unchanged is True and result["unchanged"] is True and result["files"][0]["before"] == result["files"][0]["after"]
