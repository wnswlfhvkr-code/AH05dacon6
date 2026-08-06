from __future__ import annotations

import hashlib
import importlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from src.test_007 import run_test_007_nested_parallel as parallel

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pid_alive(pid):
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                handle, ctypes.byref(exit_code)
            ):
                return False
            return exit_code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


HELPER_SOURCE = r'''
from contextlib import contextmanager
import os
import random
from pathlib import Path
import time
import numpy as np
from src.test_007.run_test_007_nested_parallel import JobResult, nested_runtime_guard

def runner(task, state):
    mode = task.payload.get("mode", "ok")
    if mode == "error":
        raise RuntimeError("synthetic runner error")
    if mode == "hang":
        time.sleep(float(task.payload.get("seconds", 30)))
    if mode == "exit_once":
        marker = Path(task.payload["marker"])
        if not marker.exists():
            marker.write_text("crashed", encoding="utf-8")
            os._exit(17)
    time.sleep(float(task.payload.get("delay", 0)))
    state["calls"] = state.get("calls", 0) + 1
    value = float(len(task.key.text))
    return JobResult(
        {"worker_call": state["calls"], "observed_stage": task.key.stage},
        {"probability": np.full((2,), value, dtype=np.float64)},
    )

def deterministic_runner(task, state):
    time.sleep(float(task.payload.get("delay", 0)))
    return JobResult(
        {"deterministic": True, "observed_stage": task.key.stage},
        {"probability": np.full((2,), float(len(task.key.text)), dtype=np.float64)},
    )

def random_consuming_runner(task, state):
    time.sleep(float(task.payload.get("delay", 0)))
    values = np.array([random.random(), np.random.random()], dtype=np.float64)
    return JobResult({"random_consumed": True}, {"probability": values})

def overlap_runner(task, state):
    barrier = Path(task.payload["barrier"])
    barrier.mkdir(parents=True, exist_ok=True)
    marker = barrier / task.resource
    other = barrier / ("cpu" if task.resource == "gpu" else "gpu")
    started = time.time()
    marker.write_text(str(os.getpid()), encoding="utf-8")
    deadline = time.monotonic() + 5
    while not other.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    if not other.exists():
        raise RuntimeError("resource overlap barrier timed out")
    time.sleep(0.15)
    finished = time.time()
    return JobResult(
        {"pid": os.getpid(), "started": started, "finished": finished},
        {"probability": np.array([started, finished], dtype=np.float64)},
    )

def guard_probe_runner(task, state):
    blocked = False
    try:
        Path(task.payload["_protected_test_path"]).read_bytes()
    except (PermissionError, RuntimeError) as error:
        blocked = "protected read blocked" in str(error) or isinstance(error, PermissionError)
    if not blocked:
        raise RuntimeError("protected read was not blocked")
    return JobResult(
        {"pid": os.getpid(), "guard_blocked": blocked},
        {"probability": np.ones(2, dtype=np.float64)},
    )

def forged_record_runner(task, state):
    return JobResult({"task_key": "forged"}, {"probability": np.ones(2, dtype=np.float64)})

def missing_array_runner(task, state):
    return JobResult({}, {})

def extra_array_runner(task, state):
    return JobResult({}, {
        "probability": np.ones(2, dtype=np.float64),
        "extra": np.ones(2, dtype=np.float64),
    })

def lazy_torch_runner(task, state):
    import sys
    from types import SimpleNamespace
    sys.modules["torch"] = SimpleNamespace()
    return JobResult({}, {"probability": np.ones(2, dtype=np.float64)})

@contextmanager
def guard(task):
    with nested_runtime_guard(task):
        yield

@contextmanager
def failing_guard(task):
    raise RuntimeError("synthetic guard error")
    yield
'''


@pytest.fixture(scope="session")
def helper_module(tmp_path_factory):
    root = tmp_path_factory.mktemp("parallel_helper")
    module_path = root / "parallel_spawn_helper.py"
    module_path.write_text(HELPER_SOURCE, encoding="utf-8")
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    yield "parallel_spawn_helper", module_path
    sys.modules.pop("parallel_spawn_helper", None)
    sys.path.remove(str(root))


@pytest.fixture
def trusted_inputs(tmp_path):
    paths = {}
    for role in ("goal_config", "search_universe", "fold_assignments", "train"):
        path = tmp_path / f"{role}.synthetic"
        path.write_text(f"synthetic-{role}", encoding="utf-8")
        paths[role] = path
    return paths


def _coverage() -> parallel.PhaseCoverage:
    return parallel.PhaseCoverage((42,), 1, 2, ("alpha", "zeta"), ("alpha", "zeta"))


def _schema():
    return {"probability": {"shape": [2], "dtype": "float64"}}


def _rehash_mutable(manifest):
    manifest["mutable_state_hash"] = parallel._canonical_hash(
        parallel.ParallelCoordinator._mutable_state_projection(manifest)
    )


def _tasks(payloads=None):
    payloads = payloads or {}
    tasks = []
    resources = {"alpha": "cpu", "zeta": "gpu"}
    for alias in ("alpha", "zeta"):
        for inner in range(2):
            key = parallel.TaskKey("inner", alias, 42, 0, inner)
            tasks.append(parallel.ParallelTask(key, resources[alias], _schema(), payloads.get(key.text, {})))
        key = parallel.TaskKey("outer_refit", alias, 42, 0)
        tasks.append(parallel.ParallelTask(key, resources[alias], _schema(), payloads.get(key.text, {})))
    return tuple(tasks)


def _full_protected_contract(test_path, submission_path):
    coverage = parallel.PhaseCoverage(
        parallel.nested.EXPECTED_SEEDS,
        parallel.nested.EXPECTED_OUTER_FOLDS,
        parallel.nested.EXPECTED_INNER_FOLDS,
        ("alpha", "zeta"),
        ("alpha", "zeta"),
    )
    payload = {
        "_protected_test_path": str(test_path),
        "_protected_submission_path": str(submission_path),
    }
    tasks = []
    for alias, resource in (("alpha", "cpu"), ("zeta", "gpu")):
        for seed in coverage.seeds:
            for outer in range(coverage.outer_folds):
                for inner in range(coverage.inner_folds):
                    tasks.append(
                        parallel.ParallelTask(
                            parallel.TaskKey("inner", alias, seed, outer, inner),
                            resource,
                            _schema(),
                            payload,
                        )
                    )
                tasks.append(
                    parallel.ParallelTask(
                        parallel.TaskKey("outer_refit", alias, seed, outer),
                        resource,
                        _schema(),
                        payload,
                    )
                )
    return coverage, tuple(tasks)


def _protected_coordinator(
    tmp_path,
    helper_module,
    trusted_inputs,
    test_path,
    submission_path,
    *,
    runner_spec=None,
    approved_extra=(),
):
    coverage, tasks = _full_protected_contract(test_path, submission_path)
    inputs = dict(trusted_inputs)
    for alias in ("alpha", "zeta"):
        candidate = tmp_path / f"candidate-{alias}.yaml"
        if not candidate.exists():
            candidate.write_text(f"synthetic-{alias}", encoding="utf-8")
        inputs[f"candidate_config_{alias}"] = candidate
    return parallel.ParallelCoordinator(
        tasks=tasks,
        coverage=coverage,
        alias_order=("alpha", "zeta"),
        trusted_input_paths=inputs,
        staging_root=tmp_path / "protected-staging",
        production_artifact_root=tmp_path / "production-artifacts",
        job_runner_spec=runner_spec or f"{helper_module[0]}:runner",
        guard_factory_spec="src.test_007.run_test_007_nested_parallel:nested_runtime_guard",
        approved_code_roots=(
            *approved_extra,
            helper_module[1].parent,
            REPO_ROOT,
        ),
        protected_test_path=test_path,
        protected_submission_path=submission_path,
        task_timeout_seconds=5,
        cleanup_timeout_seconds=1,
    )


def _coordinator(
    tmp_path,
    helper_module,
    trusted_inputs,
    *,
    tasks=None,
    runner="runner",
    guard=None,
    timeout=5.0,
    staging=None,
):
    module, _ = helper_module
    return parallel.ParallelCoordinator(
        tasks=_tasks() if tasks is None else tasks,
        coverage=_coverage(),
        alias_order=("alpha", "zeta"),
        trusted_input_paths=trusted_inputs,
        staging_root=staging or tmp_path / "staging",
        production_artifact_root=tmp_path / "production-artifacts",
        job_runner_spec=f"{module}:{runner}",
        guard_factory_spec=None if guard is None else f"{module}:{guard}",
        approved_code_roots=(
            helper_module[1].parent,
            REPO_ROOT,
        ),
        unsafe_synthetic_no_guard=True,
        task_timeout_seconds=timeout,
        cleanup_timeout_seconds=1.0,
    )


def test_spawn_workers_are_persistent_single_resource_and_merge_is_deterministic(
    tmp_path, helper_module, trusted_inputs
):
    payloads = {}
    for index, task in enumerate(reversed(_tasks())):
        payloads[task.key.text] = {"delay": index * 0.005}
    report = _coordinator(
        tmp_path, helper_module, trusted_inputs, tasks=_tasks(payloads)
    ).run()

    assert report.max_concurrency == {"gpu": 1, "cpu": 1}
    assert report.merged.task_order == (
        "inner__alpha__seed_42__outer_0__inner_0",
        "inner__alpha__seed_42__outer_0__inner_1",
        "outer_refit__alpha__seed_42__outer_0",
        "inner__zeta__seed_42__outer_0__inner_0",
        "inner__zeta__seed_42__outer_0__inner_1",
        "outer_refit__zeta__seed_42__outer_0",
    )
    assert tuple(record["task_key"] for record in report.merged.records) == report.merged.task_order
    calls = {"cpu": [], "gpu": []}
    for record in report.merged.records:
        calls[record["resource"]].append(record["worker_call"])
    assert sorted(calls["cpu"]) == [1, 2, 3]
    assert sorted(calls["gpu"]) == [1, 2, 3]


def test_cpu_gpu_processes_overlap_with_distinct_pids_and_max_one_each(
    tmp_path, helper_module, trusted_inputs
):
    barrier = tmp_path / "overlap-barrier"
    payloads = {task.key.text: {"barrier": str(barrier)} for task in _tasks()}
    report = _coordinator(
        tmp_path,
        helper_module,
        trusted_inputs,
        tasks=_tasks(payloads),
        runner="overlap_runner",
    ).run()
    assert report.max_concurrency == {"gpu": 1, "cpu": 1}
    assert report.worker_pids["gpu"] != report.worker_pids["cpu"]
    assert report.worker_exitcodes == {"gpu": 0, "cpu": 0}
    first = {}
    for record in report.merged.records:
        if record["stage"] == "inner" and record["inner_fold"] == 0:
            first[record["resource"]] = record
    assert set(first) == {"gpu", "cpu"}
    assert first["gpu"]["pid"] == report.worker_pids["gpu"]
    assert first["cpu"]["pid"] == report.worker_pids["cpu"]
    assert max(first["gpu"]["started"], first["cpu"]["started"]) < min(
        first["gpu"]["finished"], first["cpu"]["finished"]
    )


def test_concurrent_child_local_guards_block_reads_without_global_race(
    tmp_path, helper_module, trusted_inputs
):
    protected_test = tmp_path / "synthetic-protected-test.csv"
    protected_submission = tmp_path / "synthetic-protected-submission.csv"
    protected_test.write_bytes(b"must-not-be-read")
    protected_submission.write_bytes(b"must-not-be-read")
    payload = {
        "_protected_test_path": str(protected_test),
        "_protected_submission_path": str(protected_submission),
    }
    report = _coordinator(
        tmp_path,
        helper_module,
        trusted_inputs,
        tasks=_tasks({task.key.text: payload for task in _tasks()}),
        runner="guard_probe_runner",
        guard="guard",
    ).run()
    assert all(record["guard_blocked"] is True for record in report.merged.records)
    assert {
        record["pid"] for record in report.merged.records if record["resource"] == "gpu"
    } == {report.worker_pids["gpu"]}
    assert {
        record["pid"] for record in report.merged.records if record["resource"] == "cpu"
    } == {report.worker_pids["cpu"]}


def test_phase_barrier_commits_all_inner_before_outer(tmp_path, helper_module, trusted_inputs):
    report = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    first_outer = min(
        index for index, key in enumerate(report.executed_keys) if key.startswith("outer_refit__")
    )
    assert all(key.startswith("inner__") for key in report.executed_keys[:first_outer])
    assert first_outer == 4
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert manifest["mode"] == "SHADOW_EXPERIMENTAL"
    assert manifest["production_nested_adapter"] is False
    assert manifest["production_publish_allowed"] is False
    assert manifest["execution_semantics"] == "at_least_once_idempotent_runner_required"
    assert set(manifest["limitations"]) == set(parallel.LIMITATIONS)
    assert set(manifest["completed"]) == {task.key.text for task in _tasks()}
    barrier_key = "inner_complete__seed_42__outer_0"
    assert set(manifest["outer_fold_barriers"]) == {barrier_key}
    barrier = manifest["outer_fold_barriers"][barrier_key]
    assert barrier["sha256"] == parallel._canonical_hash(barrier["payload"])
    assert set(barrier["payload"]["inner_task_keys"]) == {
        task.key.text for task in _tasks() if task.key.stage == "inner"
    }


def test_exact_coverage_rejects_duplicate_missing_and_extra_tasks(
    tmp_path, helper_module, trusted_inputs
):
    tasks = _tasks()
    for invalid in (tasks + (tasks[0],), tasks[:-1]):
        with pytest.raises(ValueError, match="exactly cover|committed-once"):
            _coordinator(tmp_path, helper_module, trusted_inputs, tasks=invalid)
    extra = parallel.ParallelTask(
        parallel.TaskKey("outer_refit", "alpha", 42, 1), "cpu", _schema()
    )
    with pytest.raises(ValueError, match="exactly cover"):
        _coordinator(tmp_path, helper_module, trusted_inputs, tasks=tasks + (extra,))


@pytest.mark.parametrize("alias", ["Bad", "bad.name", "bad:name", "bad/name", "bad\\name", "con", "LPT1"])
def test_alias_rejects_nonportable_windows_or_device_names(alias):
    with pytest.raises(ValueError, match="portable|reserved"):
        parallel.TaskKey("inner", alias, 42, 0, 0)


def test_alias_order_rejects_casefold_collisions(tmp_path, helper_module, trusted_inputs):
    with pytest.raises(ValueError, match="portable|casefold"):
        parallel.ParallelCoordinator(
            tasks=_tasks(),
            coverage=_coverage(),
            alias_order=("alpha", "Alpha", "zeta"),
            trusted_input_paths=trusted_inputs,
            staging_root=tmp_path / "stage",
            production_artifact_root=tmp_path / "artifacts",
            job_runner_spec=f"{helper_module[0]}:runner",
            guard_factory_spec=None,
            approved_code_roots=(
                helper_module[1].parent,
                REPO_ROOT,
            ),
            unsafe_synthetic_no_guard=True,
        )


def test_guard_is_mandatory_by_default_and_unsafe_mode_is_explicit(
    tmp_path, helper_module, trusted_inputs
):
    with pytest.raises(ValueError, match="guard spec is mandatory"):
        parallel.ParallelCoordinator(
            tasks=_tasks(),
            coverage=_coverage(),
            alias_order=("alpha", "zeta"),
            trusted_input_paths=trusted_inputs,
            staging_root=tmp_path / "stage",
            production_artifact_root=tmp_path / "artifacts",
            job_runner_spec=f"{helper_module[0]}:runner",
            guard_factory_spec=None,
            approved_code_roots=(
                helper_module[1].parent,
                REPO_ROOT,
            ),
        )


def test_protected_input_roles_are_rejected_without_opening_path(
    tmp_path, helper_module, trusted_inputs
):
    forbidden = tmp_path / "must-not-open.csv"
    values = dict(trusted_inputs, test=forbidden)
    with pytest.raises(ValueError, match="must never enter"):
        _coordinator(tmp_path, helper_module, values)
    assert not forbidden.exists()


def test_builtin_nested_guard_delegates_without_opening_protected_files(monkeypatch, tmp_path):
    calls = []

    @contextmanager
    def fake_guard(test, submission, audit, stage, alias):
        calls.append((test, submission, audit, stage, alias))
        yield

    monkeypatch.setattr(parallel.nested, "_runtime_read_guard", fake_guard)
    test_path = tmp_path / "never-open-test.csv"
    submission_path = tmp_path / "never-open-submission.csv"
    task = parallel.ParallelTask(
        parallel.TaskKey("inner", "alpha", 42, 0, 0),
        "cpu",
        _schema(),
        {
            "_protected_test_path": str(test_path),
            "_protected_submission_path": str(submission_path),
        },
    )
    with parallel.nested_runtime_guard(task):
        pass
    assert len(calls) == 1
    assert calls[0][0:2] == (test_path.resolve(), submission_path.resolve())
    assert calls[0][3:] == ("inner", "alpha")
    assert not test_path.exists()
    assert not submission_path.exists()


def test_protected_identity_alias_is_rejected_before_any_hash_read(
    tmp_path, helper_module, trusted_inputs, monkeypatch
):
    protected_test = tmp_path / "protected-test.csv"
    protected_submission = tmp_path / "protected-submission.csv"
    protected_test.write_bytes(b"secret-test")
    protected_submission.write_bytes(b"secret-submission")
    trusted_inputs["train"].unlink()
    os.link(protected_test, trusted_inputs["train"])
    hash_calls = []
    monkeypatch.setattr(
        parallel,
        "_hash_regular_input",
        lambda path: hash_calls.append(Path(path)) or hashlib.sha256(b"x").hexdigest(),
    )
    with pytest.raises(ValueError, match="aliases a protected"):
        _protected_coordinator(
            tmp_path,
            helper_module,
            trusted_inputs,
            protected_test,
            protected_submission,
        )
    assert hash_calls == []


def test_protected_reparse_alias_is_rejected_before_hashing(
    tmp_path, helper_module, trusted_inputs, monkeypatch
):
    protected_test = tmp_path / "protected-test.csv"
    protected_submission = tmp_path / "protected-submission.csv"
    protected_test.write_bytes(b"secret-test")
    protected_submission.write_bytes(b"secret-submission")
    trusted_inputs["search_universe"].unlink()
    try:
        trusted_inputs["search_universe"].symlink_to(protected_test)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    hash_calls = []
    monkeypatch.setattr(
        parallel,
        "_hash_regular_input",
        lambda path: hash_calls.append(Path(path)) or hashlib.sha256(b"x").hexdigest(),
    )
    with pytest.raises(ValueError, match="reparse/symlink"):
        _protected_coordinator(
            tmp_path,
            helper_module,
            trusted_inputs,
            protected_test,
            protected_submission,
        )
    assert hash_calls == []


def test_realistic_repo_src_and_data_layout_hashes_only_python_sources(
    tmp_path, helper_module, trusted_inputs, monkeypatch
):
    synthetic_repo = tmp_path / "synthetic-repo"
    (synthetic_repo / "src").mkdir(parents=True)
    (synthetic_repo / "src" / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
    (synthetic_repo / "layout_runner.py").write_text(
        "def runner(task, state):\n    raise RuntimeError('not executed by parent')\n",
        encoding="utf-8",
    )
    raw = synthetic_repo / "data" / "raw"
    raw.mkdir(parents=True)
    protected_test = raw / "test.csv"
    protected_submission = raw / "sample_submission.csv"
    protected_test.write_bytes(b"protected-test-bytes")
    protected_submission.write_bytes(b"protected-submission-bytes")
    protected = {protected_test.resolve(), protected_submission.resolve()}
    original_hash = parallel._hash_regular_input
    hashed = []

    def recording_hash(path):
        resolved = Path(path).resolve()
        assert resolved not in protected
        hashed.append(resolved)
        return original_hash(path)

    monkeypatch.setattr(parallel, "_hash_regular_input", recording_hash)
    coordinator = _protected_coordinator(
        tmp_path,
        helper_module,
        trusted_inputs,
        protected_test,
        protected_submission,
        runner_spec="layout_runner:runner",
        approved_extra=(synthetic_repo,),
    )
    assert synthetic_repo / "src" / "feature.py" in hashed
    assert synthetic_repo / "layout_runner.py" in hashed
    assert coordinator.initial_hashes["sources"]
    assert protected_test.read_bytes() == b"protected-test-bytes"
    assert protected_submission.read_bytes() == b"protected-submission-bytes"


def test_parent_resolves_approved_runner_source_without_executing_module(
    tmp_path, helper_module, trusted_inputs
):
    module_root = tmp_path / "untrusted-module"
    module_root.mkdir()
    marker = tmp_path / "parent-imported.marker"
    module_path = module_root / "parent_import_probe.py"
    module_path.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
        "def runner(task, state):\n    raise RuntimeError('child only')\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(module_root))
    importlib.invalidate_caches()
    try:
        parallel.ParallelCoordinator(
            tasks=_tasks(),
            coverage=_coverage(),
            alias_order=("alpha", "zeta"),
            trusted_input_paths=trusted_inputs,
            staging_root=tmp_path / "stage",
            production_artifact_root=tmp_path / "production",
            job_runner_spec="parent_import_probe:runner",
            guard_factory_spec=None,
            approved_code_roots=(
                module_root,
                REPO_ROOT,
            ),
            unsafe_synthetic_no_guard=True,
        )
        assert not marker.exists()
        assert "parent_import_probe" not in sys.modules
    finally:
        sys.path.remove(str(module_root))


def test_runner_spec_outside_approved_code_roots_is_rejected(
    tmp_path, helper_module, trusted_inputs
):
    with pytest.raises(ValueError, match="approved code roots"):
        parallel.ParallelCoordinator(
            tasks=_tasks(),
            coverage=_coverage(),
            alias_order=("alpha", "zeta"),
            trusted_input_paths=trusted_inputs,
            staging_root=tmp_path / "stage",
            production_artifact_root=tmp_path / "production",
            job_runner_spec="json:loads",
            guard_factory_spec=None,
            approved_code_roots=(REPO_ROOT,),
            unsafe_synthetic_no_guard=True,
        )


def test_child_bootstrap_guard_blocks_protected_read_during_runner_import(
    tmp_path, helper_module, trusted_inputs, monkeypatch
):
    protected_test = tmp_path / "protected-test.csv"
    protected_submission = tmp_path / "protected-submission.csv"
    protected_test.write_bytes(b"secret-test")
    protected_submission.write_bytes(b"secret-submission")
    module_root = tmp_path / "bootstrap-module"
    module_root.mkdir()
    marker = tmp_path / "bootstrap-read.marker"
    module_path = module_root / "bootstrap_probe.py"
    module_path.write_text(
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['PROTECTED_PROBE']).read_bytes()\n"
        f"Path({str(marker)!r}).write_text('read', encoding='utf-8')\n"
        "def runner(task, state):\n    raise RuntimeError('unreachable')\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(module_root))
    monkeypatch.setenv("PROTECTED_PROBE", str(protected_test))
    importlib.invalidate_caches()
    coordinator = _protected_coordinator(
        tmp_path,
        helper_module,
        trusted_inputs,
        protected_test,
        protected_submission,
        runner_spec="bootstrap_probe:runner",
        approved_extra=(module_root,),
    )
    try:
        with pytest.raises(parallel.WorkerExecutionError, match="bootstrap failed"):
            coordinator.run()
    finally:
        sys.path.remove(str(module_root))
    assert not marker.exists()
    assert coordinator.last_worker_survivors == ()
    assert not any(_pid_alive(pid) for pid in coordinator.last_worker_pids.values())


def test_decorated_guard_unwrap_and_transitive_decorator_drift_rejected(
    tmp_path, helper_module, trusted_inputs
):
    module_root = tmp_path / "decorated"
    module_root.mkdir()
    decorator_path = module_root / "guard_decorator_probe.py"
    decorator_path.write_text(
        "from functools import wraps\n"
        "def decorate(fn):\n"
        "    @wraps(fn)\n"
        "    def wrapper(*args, **kwargs):\n"
        "        return fn(*args, **kwargs)\n"
        "    return wrapper\n",
        encoding="utf-8",
    )
    guard_path = module_root / "decorated_guard_probe.py"
    guard_path.write_text(
        "from contextlib import contextmanager\n"
        "from guard_decorator_probe import decorate\n"
        "@decorate\n@contextmanager\n"
        "def guard(task):\n    yield\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(module_root))
    importlib.invalidate_caches()

    def coordinator():
        return parallel.ParallelCoordinator(
            tasks=_tasks(),
            coverage=_coverage(),
            alias_order=("alpha", "zeta"),
            trusted_input_paths=trusted_inputs,
            staging_root=tmp_path / "stage",
            production_artifact_root=tmp_path / "production",
            job_runner_spec=f"{helper_module[0]}:runner",
            guard_factory_spec="decorated_guard_probe:guard",
            approved_code_roots=(
                module_root,
                helper_module[1].parent,
                REPO_ROOT,
            ),
            unsafe_synthetic_no_guard=True,
        )

    try:
        first = coordinator().run()
        guard_module = importlib.import_module("decorated_guard_probe")
        assert parallel._spec_source_path(guard_module.guard) == guard_path.resolve()
        manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
        assert str(decorator_path.resolve()) in manifest["trusted_hashes"]["sources"]
        decorator_path.write_text(decorator_path.read_text(encoding="utf-8") + "\n# drift\n")
        with pytest.raises(ValueError, match="actual current input/source hashes"):
            coordinator().run()
    finally:
        sys.modules.pop("decorated_guard_probe", None)
        sys.modules.pop("guard_decorator_probe", None)
        sys.path.remove(str(module_root))


def test_resume_revalidates_actual_input_and_source_hashes(
    tmp_path, helper_module, trusted_inputs
):
    first = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    second = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    assert not second.executed_keys
    assert set(second.resumed_keys) == {task.key.text for task in _tasks()}

    trusted_inputs["train"].write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError, match="actual current input/source hashes"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()

    trusted_inputs["train"].write_text("synthetic-train", encoding="utf-8")
    helper_module[1].write_text(HELPER_SOURCE + "\n# source drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="actual current input/source hashes"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()


def test_manifest_records_actual_input_runner_and_scheduler_source_hashes(
    tmp_path, helper_module, trusted_inputs
):
    report = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    hashes = manifest["trusted_hashes"]
    assert set(hashes["inputs"]) == set(trusted_inputs)
    for role, path in trusted_inputs.items():
        assert hashes["inputs"][role]["path"] == str(path.resolve())
        assert hashes["inputs"][role]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    scheduler = Path(parallel.__file__).resolve()
    assert hashes["sources"][str(helper_module[1].resolve())] == hashlib.sha256(
        helper_module[1].read_bytes()
    ).hexdigest()
    assert hashes["sources"][str(scheduler)] == hashlib.sha256(
        scheduler.read_bytes()
    ).hexdigest()
    assert hashes["entry_points"]["job_runner"] == "parallel_spawn_helper:runner"
    assert hashes["entry_points"]["guard_factory"] is None
    assert "parallel_spawn_helper" in hashes["entry_points"]["modules"]


def test_resume_rejects_direct_committed_shard_content_corruption(
    tmp_path, helper_module, trusted_inputs
):
    report = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    key = next(iter(manifest["completed"]))
    shard = tmp_path / "staging" / "shards" / f"{key}.npz"
    shard.write_bytes(shard.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="trusted shard hash mismatch"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()


def test_resume_rejects_forged_receipt_phase_and_phase_barrier(
    tmp_path, helper_module, trusted_inputs
):
    report = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    inner_key = next(key for key in manifest["completed"] if key.startswith("inner__"))
    manifest["completed"][inner_key]["phase"] = "outer_refit"
    _rehash_mutable(manifest)
    report.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt schema"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()

    manifest["completed"][inner_key]["phase"] = "inner"
    outer_key = next(key for key in manifest["completed"] if key.startswith("outer_refit__"))
    manifest["completed"] = {outer_key: manifest["completed"][outer_key]}
    _rehash_mutable(manifest)
    report.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="phase barrier"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()


def test_resume_rejects_corrupt_and_duplicate_outer_fold_checkpoints(
    tmp_path, helper_module, trusted_inputs
):
    report = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    barrier = manifest["outer_fold_barriers"]["inner_complete__seed_42__outer_0"]
    barrier["sha256"] = "0" * 64
    _rehash_mutable(manifest)
    report.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="stale or corrupt"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()

    valid_value = {**manifest, "outer_fold_barriers": {
        "inner_complete__seed_42__outer_0": {
            **barrier,
            "sha256": parallel._canonical_hash(barrier["payload"]),
        }
    }}
    _rehash_mutable(valid_value)
    valid = json.dumps(valid_value)
    duplicate = '{"outer_fold_barriers":{},' + valid[1:]
    report.manifest_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()


def test_manifest_authenticates_every_immutable_contract_field(
    tmp_path, helper_module, trusted_inputs
):
    report = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    valid = json.loads(report.manifest_path.read_text(encoding="utf-8"))

    def mutate_limitations(value):
        value["limitations"].append("forged")

    def mutate_task(value):
        value["tasks"][0]["resource"] = "gpu" if value["tasks"][0]["resource"] == "cpu" else "cpu"

    def mutate_input(value):
        value["trusted_hashes"]["inputs"]["train"]["sha256"] = "0" * 64

    def mutate_source(value):
        source = next(iter(value["trusted_hashes"]["sources"]))
        value["trusted_hashes"]["sources"][source] = "0" * 64

    for mutate in (mutate_limitations, mutate_task, mutate_input, mutate_source):
        forged = json.loads(json.dumps(valid))
        mutate(forged)
        projection = {key: forged[key] for key in _coordinator(
            tmp_path, helper_module, trusted_inputs
        ).contract_payload}
        forged["contract_hash"] = parallel._canonical_hash(projection)
        report.manifest_path.write_text(json.dumps(forged), encoding="utf-8")
        with pytest.raises(ValueError, match="actual current input/source hashes or task contract"):
            _coordinator(tmp_path, helper_module, trusted_inputs).run()
        report.manifest_path.write_text(json.dumps(valid), encoding="utf-8")


def test_mutable_manifest_rejects_forged_recovered_checkpoint_entry(
    tmp_path, helper_module, trusted_inputs
):
    report = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    key = next(iter(manifest["completed"]))
    manifest["recovered_atomic_job_checkpoints"].append(
        {"task_key": key, "sha256": manifest["completed"][key]["sha256"]}
    )
    _rehash_mutable(manifest)
    report.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical empty"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()


def test_mutable_manifest_rejects_escaping_discarded_temp_entry(
    tmp_path, helper_module, trusted_inputs
):
    report = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    manifest["discarded_atomic_temps"].append(
        {"parent": "staging", "name": ".valid-looking.tmp"}
    )
    _rehash_mutable(manifest)
    report.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical empty"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()


def test_mutable_manifest_hash_rejects_unrehash_tamper(
    tmp_path, helper_module, trusted_inputs
):
    report = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    manifest["discarded_atomic_temps"].append(
        {"parent": "staging", "name": ".forged.tmp"}
    )
    report.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="mutable-state authentication"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()


def test_multi_seed_multi_fold_barrier_checkpoint_coverage(
    tmp_path, helper_module, trusted_inputs
):
    coverage = parallel.PhaseCoverage((42, 777), 2, 2, ("alpha", "zeta"), ("alpha", "zeta"))
    tasks = []
    for alias, resource in (("alpha", "cpu"), ("zeta", "gpu")):
        for seed in coverage.seeds:
            for outer in range(coverage.outer_folds):
                for inner in range(coverage.inner_folds):
                    tasks.append(parallel.ParallelTask(
                        parallel.TaskKey("inner", alias, seed, outer, inner), resource, _schema()
                    ))
                tasks.append(parallel.ParallelTask(
                    parallel.TaskKey("outer_refit", alias, seed, outer), resource, _schema()
                ))
    coordinator = parallel.ParallelCoordinator(
        tasks=tasks,
        coverage=coverage,
        alias_order=("alpha", "zeta"),
        trusted_input_paths=trusted_inputs,
        staging_root=tmp_path / "stage",
        production_artifact_root=tmp_path / "production",
        job_runner_spec=f"{helper_module[0]}:runner",
        guard_factory_spec=None,
        approved_code_roots=(
            helper_module[1].parent,
            REPO_ROOT,
        ),
        unsafe_synthetic_no_guard=True,
    )
    report = coordinator.run()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    expected = {
        f"inner_complete__seed_{seed}__outer_{outer}"
        for seed in coverage.seeds
        for outer in range(coverage.outer_folds)
    }
    assert set(manifest["outer_fold_barriers"]) == expected
    for receipt in manifest["outer_fold_barriers"].values():
        assert receipt["sha256"] == parallel._canonical_hash(receipt["payload"])


def test_committed_shard_hash_hardlink_and_symlink_are_rejected(
    tmp_path, helper_module, trusted_inputs
):
    report = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    key = next(iter(manifest["completed"]))
    shard = tmp_path / "staging" / "shards" / f"{key}.npz"
    backup = tmp_path / "backup.npz"
    os.link(shard, backup)
    with pytest.raises(ValueError, match="singly linked"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()
    backup.unlink()

    original = shard.read_bytes()
    shard.unlink()
    target = tmp_path / "target.npz"
    target.write_bytes(original)
    try:
        shard.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="regular and singly linked"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()


def test_symlink_shard_root_is_rejected(tmp_path, helper_module, trusted_inputs):
    staging = tmp_path / "staging"
    staging.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    try:
        (staging / "shards").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    with pytest.raises(ValueError, match="reparse/symlink"):
        _coordinator(tmp_path, helper_module, trusted_inputs, staging=staging).run()


def test_preexisting_unowned_staging_root_is_rejected(
    tmp_path, helper_module, trusted_inputs
):
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(ValueError, match="not exclusively initialized"):
        _coordinator(tmp_path, helper_module, trusted_inputs, staging=staging).run()


def test_private_staging_identity_marker_tamper_is_rejected(
    tmp_path, helper_module, trusted_inputs
):
    report = _coordinator(tmp_path, helper_module, trusted_inputs).run()
    marker = tmp_path / "staging" / ".private_staging_identity.json"
    value = json.loads(marker.read_text(encoding="utf-8"))
    value["token"] = "0" * 64
    marker.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="marker"):
        _coordinator(tmp_path, helper_module, trusted_inputs).run()


def test_coordinator_rejects_mocked_final_handle_alias(
    tmp_path, helper_module, trusted_inputs, monkeypatch
):
    coordinator = _coordinator(tmp_path, helper_module, trusted_inputs)

    def alias_projection(_path, **_kwargs):
        proof = parallel._directory_proof(
            coordinator.staging_root, require_no_reparse=True
        )
        return parallel._TargetProjection(
            proof.canonical_path,
            proof.canonical_path,
            proof.identity,
            (),
            (proof.identity,),
        )

    monkeypatch.setattr(parallel, "_physical_target_projection", alias_projection)
    with pytest.raises(ValueError, match="share physical output"):
        coordinator.run()


def test_shadow_publish_rejects_mocked_final_handle_alias(tmp_path, monkeypatch):
    task = _tasks()[0]
    merged = parallel.merge_results(
        (task,),
        {task.key.text: parallel.JobResult({}, {"probability": np.ones(2, dtype=np.float64)})},
        ("alpha", "zeta"),
    )
    shadow = tmp_path / "shadow"

    def alias_projection(_path, **_kwargs):
        proof = parallel._directory_proof(shadow, require_no_reparse=True)
        return parallel._TargetProjection(
            proof.canonical_path,
            proof.canonical_path,
            proof.identity,
            (),
            (proof.identity,),
        )

    monkeypatch.setattr(parallel, "_physical_target_projection", alias_projection)
    with pytest.raises(ValueError, match="share physical output"):
        parallel.publish_shadow_npz(
            merged, shadow, tmp_path / "production", "shadow_result.npz"
        )


def test_physical_projection_retains_alias_ancestor_for_nonexistent_child(
    tmp_path, monkeypatch
):
    staging = tmp_path / "staging"
    staging.mkdir()
    production = tmp_path / "alias-spelling" / "future-child"
    proof = parallel._directory_proof(staging, require_no_reparse=True)

    def projection(path, **_kwargs):
        if Path(path) == staging:
            return parallel._TargetProjection(
                proof.canonical_path,
                proof.canonical_path,
                proof.identity,
                (),
                (proof.identity,),
            )
        return parallel._TargetProjection(
            os.path.join(proof.canonical_path, "future-child"),
            proof.canonical_path,
            proof.identity,
            ("future-child",),
            (proof.identity,),
        )

    monkeypatch.setattr(parallel, "_physical_target_projection", projection)
    with pytest.raises(ValueError, match="share physical output"):
        parallel._assert_physical_output_separation(staging, production)


def test_physical_projection_rejects_existing_alias_descendant_chain(
    tmp_path, monkeypatch
):
    staging = tmp_path / "staging"
    staging.mkdir()
    descendant = tmp_path / "mapped-descendant"
    proof = parallel._directory_proof(staging, require_no_reparse=True)
    child_identity = (proof.identity[0], proof.identity[1] + 1)

    def projection(path, **_kwargs):
        if Path(path) == staging:
            return parallel._TargetProjection(
                proof.canonical_path,
                proof.canonical_path,
                proof.identity,
                (),
                (proof.identity,),
            )
        child_path = os.path.join(proof.canonical_path, "mapped-descendant")
        return parallel._TargetProjection(
            child_path,
            child_path,
            child_identity,
            (),
            (proof.identity, child_identity),
        )

    monkeypatch.setattr(parallel, "_physical_target_projection", projection)
    with pytest.raises(ValueError, match="share physical output"):
        parallel._assert_physical_output_separation(staging, descendant)


@pytest.mark.skipif(os.name != "nt", reason="Windows final-handle alias probe")
def test_windows_final_handle_path_detects_directory_alias_when_available(tmp_path):
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(physical, target_is_directory=True)
    except OSError:
        pytest.skip("Windows directory alias creation is unavailable")
    with pytest.raises(ValueError, match="share physical output"):
        parallel._assert_physical_output_separation(physical, alias)


def test_preexisting_exclusive_temp_hardlink_is_never_followed(tmp_path, monkeypatch):
    root = tmp_path / "secure"
    root.mkdir()
    target = root / "manifest.json"
    victim = root / "victim"
    victim.write_bytes(b"unchanged")
    collision = root / ".manifest.json.fixed.tmp"
    os.link(victim, collision)
    monkeypatch.setattr(parallel.secrets, "token_hex", lambda _size: "fixed")
    with pytest.raises(FileExistsError, match="collision-free"):
        parallel._atomic_bytes(target, b"new", root.resolve())
    assert victim.read_bytes() == b"unchanged"
    assert not target.exists()


def test_preexisting_exclusive_temp_symlink_is_never_followed(tmp_path, monkeypatch):
    root = tmp_path / "secure"
    root.mkdir()
    target = root / "manifest.json"
    victim = root / "victim"
    victim.write_bytes(b"unchanged")
    collision = root / ".manifest.json.fixed.tmp"
    try:
        collision.symlink_to(victim)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    monkeypatch.setattr(parallel.secrets, "token_hex", lambda _size: "fixed")
    with pytest.raises(FileExistsError, match="collision-free"):
        parallel._atomic_bytes(target, b"new", root.resolve())
    assert victim.read_bytes() == b"unchanged"
    assert collision.is_symlink()
    assert not target.exists()


def test_link_swap_target_never_modifies_victim(tmp_path, monkeypatch):
    root = tmp_path / "secure"
    root.mkdir()
    target = root / "manifest.json"
    victim = root / "victim"
    victim.write_bytes(b"victim-unchanged")
    real_replace = os.replace

    def swap_then_replace(source, destination):
        destination = Path(destination)
        if not destination.exists() and not destination.is_symlink():
            try:
                destination.symlink_to(victim)
            except OSError:
                pytest.skip("symlink creation is unavailable")
        return real_replace(source, destination)

    monkeypatch.setattr(parallel.os, "replace", swap_then_replace)
    parallel._atomic_bytes(target, b"committed", root.resolve())
    assert victim.read_bytes() == b"victim-unchanged"
    assert target.read_bytes() == b"committed"
    assert not target.is_symlink()


@pytest.mark.parametrize("failure_point", ["write", "replace"])
def test_atomic_write_errors_leave_no_temporary_files(tmp_path, monkeypatch, failure_point):
    root = tmp_path / "secure"
    root.mkdir()
    target = root / "manifest.json"
    if failure_point == "write":
        monkeypatch.setattr(
            parallel.os, "write", lambda *_args: (_ for _ in ()).throw(OSError("write failed"))
        )
    else:
        monkeypatch.setattr(
            parallel.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace failed"))
        )
    with pytest.raises(OSError, match=f"{failure_point} failed"):
        parallel._atomic_bytes(target, b"payload", root.resolve())
    assert not target.exists()
    assert list(root.iterdir()) == []


def test_crash_left_atomic_temp_is_securely_discarded_on_resume(
    tmp_path, helper_module, trusted_inputs
):
    coordinator = _coordinator(tmp_path, helper_module, trusted_inputs)
    staging = coordinator._prepare_private_staging()
    leftover = staging / ".parallel_manifest.json.crash.tmp"
    leftover.write_bytes(b"partial")
    report = coordinator.run()
    assert not leftover.exists()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert manifest["discarded_atomic_temps"] == []
    assert {"parent": "staging", "name": leftover.name} in report.discarded_temp_entries


@pytest.mark.parametrize(
    "runner,match",
    [
        ("forged_record_runner", "override coordinator metadata"),
        ("missing_array_runner", "exactly match task schema"),
        ("extra_array_runner", "exactly match task schema"),
    ],
)
def test_forged_record_and_array_coverage_are_rejected(
    tmp_path, helper_module, trusted_inputs, runner, match
):
    with pytest.raises(parallel.WorkerExecutionError, match=match):
        _coordinator(tmp_path, helper_module, trusted_inputs, runner=runner).run()


def test_lazy_torch_import_after_task_rng_reset_is_rejected_without_commit(
    tmp_path, helper_module, trusted_inputs
):
    coordinator = _coordinator(
        tmp_path, helper_module, trusted_inputs, runner="lazy_torch_runner"
    )
    with pytest.raises(parallel.WorkerExecutionError, match="lazily loaded torch"):
        coordinator.run()
    manifest = json.loads(coordinator.manifest_path.read_text(encoding="utf-8"))
    assert manifest["completed"] == {}


def test_merge_and_shadow_publish_reject_forged_identity_or_schema(tmp_path):
    task = _tasks()[0]
    valid = parallel.JobResult({}, {"probability": np.ones(2, dtype=np.float64)})
    with pytest.raises(ValueError, match="missing or extra"):
        parallel.merge_results((task,), {}, ("alpha", "zeta"))
    merged = parallel.merge_results((task,), {task.key.text: valid}, ("alpha", "zeta"))
    forged_record = parallel.MergedResult(
        merged.task_order,
        ({**merged.records[0], "task_key": "forged"},),
        merged.arrays,
        merged.array_schema,
        merged.task_metadata,
    )
    with pytest.raises(ValueError, match="immutable task order"):
        parallel.publish_shadow_npz(
            forged_record, tmp_path / "shadow", tmp_path / "production", "shadow_result.npz"
        )
    forged_metadata_record = parallel.MergedResult(
        merged.task_order,
        ({**merged.records[0], "alias": "zeta"},),
        merged.arrays,
        merged.array_schema,
        merged.task_metadata,
    )
    with pytest.raises(ValueError, match="coordinator-owned"):
        parallel.publish_shadow_npz(
            forged_metadata_record,
            tmp_path / "shadow",
            tmp_path / "production",
            "shadow_result.npz",
        )
    forged_schema = parallel.MergedResult(
        merged.task_order, merged.records, merged.arrays, {}, merged.task_metadata
    )
    with pytest.raises(ValueError, match="missing or extra schema"):
        parallel.publish_shadow_npz(
            forged_schema, tmp_path / "shadow", tmp_path / "production", "shadow_result.npz"
        )


def test_shadow_publish_is_disjoint_and_never_uses_production_root(tmp_path):
    task = _tasks()[0]
    merged = parallel.merge_results(
        (task,),
        {task.key.text: parallel.JobResult({}, {"probability": np.ones(2, dtype=np.float64)})},
        ("alpha", "zeta"),
    )
    production = tmp_path / "production"
    with pytest.raises(ValueError, match="disjoint"):
        parallel.publish_shadow_npz(merged, production, production, "shadow_result.npz")
    output = parallel.publish_shadow_npz(
        merged, tmp_path / "shadow", production, "shadow_result.npz"
    )
    assert output.is_file()
    assert not production.exists()


@pytest.mark.parametrize(
    "name",
    ["shadow_bad:name.npz", "shadow_bad name.npz", "shadow_bad..npz", "Shadow_bad.npz", "con.npz"],
)
def test_shadow_output_name_is_strictly_portable(tmp_path, name):
    task = _tasks()[0]
    merged = parallel.merge_results(
        (task,),
        {task.key.text: parallel.JobResult({}, {"probability": np.ones(2, dtype=np.float64)})},
        ("alpha", "zeta"),
    )
    with pytest.raises(ValueError, match="plain shadow"):
        parallel.publish_shadow_npz(
            merged, tmp_path / "shadow", tmp_path / "production", name
        )


def test_runner_guard_and_parent_write_errors_propagate(
    tmp_path, helper_module, trusted_inputs, monkeypatch
):
    payloads = {_tasks()[0].key.text: {"mode": "error"}}
    with pytest.raises(parallel.WorkerExecutionError, match="synthetic runner error"):
        _coordinator(tmp_path, helper_module, trusted_inputs, tasks=_tasks(payloads)).run()

    with pytest.raises(parallel.WorkerExecutionError, match="synthetic guard error"):
        _coordinator(
            tmp_path / "guard", helper_module, trusted_inputs, guard="failing_guard"
        ).run()

    coordinator = _coordinator(tmp_path / "write", helper_module, trusted_inputs)
    monkeypatch.setattr(
        coordinator, "_write_shard", lambda *_args: (_ for _ in ()).throw(OSError("write failed"))
    )
    with pytest.raises(OSError, match="write failed"):
        coordinator.run()


def test_timeout_terminates_hung_spawn_worker_with_bounded_cleanup(
    tmp_path, helper_module, trusted_inputs
):
    payloads = {_tasks()[0].key.text: {"mode": "hang", "seconds": 30}}
    coordinator = _coordinator(
        tmp_path, helper_module, trusted_inputs, tasks=_tasks(payloads), timeout=0.2
    )
    started = time.monotonic()
    with pytest.raises(parallel.WorkerTimeoutError, match="worker timeout"):
        coordinator.run()
    assert time.monotonic() - started < 5
    assert coordinator.last_worker_survivors == ()
    assert coordinator.last_worker_pids
    assert not any(_pid_alive(pid) for pid in coordinator.last_worker_pids.values())


def test_cancellation_terminates_workers(tmp_path, helper_module, trusted_inputs):
    payloads = {_tasks()[0].key.text: {"mode": "hang", "seconds": 30}}
    cancel = threading.Event()
    coordinator = _coordinator(tmp_path, helper_module, trusted_inputs, tasks=_tasks(payloads))
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(parallel.ExecutionCancelled, match="cancelled"):
            coordinator.run(cancel_event=cancel)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 5
    assert coordinator.last_worker_survivors == ()
    assert not any(_pid_alive(pid) for pid in coordinator.last_worker_pids.values())


def test_forced_stop_resume_equals_uninterrupted_deterministic_execution(
    tmp_path, helper_module, trusted_inputs
):
    payloads = {}
    for task in _tasks():
        if task.key.stage == "inner":
            payloads[task.key.text] = {"delay": 0.02 if task.key.inner_fold == 0 else 1.0}
    tasks = _tasks(payloads)
    interrupted_root = tmp_path / "interrupted"
    cancel = threading.Event()
    manifest_path = interrupted_root / "staging" / "parallel_manifest.json"

    def cancel_after_checkpoint():
        deadline = time.monotonic() + 10
        shard_root = interrupted_root / "staging" / "shards"
        while time.monotonic() < deadline and not cancel.is_set():
            if shard_root.exists() and any(shard_root.glob("*.npz")):
                time.sleep(0.2)
                cancel.set()
                return
            time.sleep(0.01)

    watcher = threading.Thread(target=cancel_after_checkpoint, daemon=True)
    watcher.start()
    try:
        with pytest.raises(parallel.ExecutionCancelled):
            _coordinator(
                interrupted_root,
                helper_module,
                trusted_inputs,
                tasks=tasks,
                runner="random_consuming_runner",
            ).run(cancel_event=cancel)
    finally:
        cancel.set()
        watcher.join(timeout=2)
    checkpoint_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    assert 0 < len(checkpoint_manifest["completed"]) < len(tasks)
    checkpoint_keys = set(checkpoint_manifest["completed"])

    resumed = _coordinator(
        interrupted_root,
        helper_module,
        trusted_inputs,
        tasks=tasks,
        runner="random_consuming_runner",
    ).run()
    uninterrupted = _coordinator(
        tmp_path / "uninterrupted",
        helper_module,
        trusted_inputs,
        tasks=tasks,
        runner="random_consuming_runner",
    ).run()
    assert resumed.merged.task_order == uninterrupted.merged.task_order
    assert resumed.merged.records == uninterrupted.merged.records
    assert resumed.merged.array_schema == uninterrupted.merged.array_schema
    assert resumed.merged.task_metadata == uninterrupted.merged.task_metadata
    assert set(resumed.merged.arrays) == set(uninterrupted.merged.arrays)
    assert all(
        np.array_equal(resumed.merged.arrays[key], uninterrupted.merged.arrays[key])
        for key in resumed.merged.arrays
    )
    resumed_manifest = json.loads(resumed.manifest_path.read_text(encoding="utf-8"))
    uninterrupted_manifest = json.loads(
        uninterrupted.manifest_path.read_text(encoding="utf-8")
    )
    assert resumed_manifest["completed"] == uninterrupted_manifest["completed"]
    assert (
        resumed_manifest["outer_fold_barriers"]
        == uninterrupted_manifest["outer_fold_barriers"]
    )
    assert set(resumed.resumed_keys) == checkpoint_keys
    assert not checkpoint_keys.intersection(resumed.executed_keys)
    assert set(resumed.executed_keys) == {task.key.text for task in tasks} - checkpoint_keys


def test_worker_crash_leaves_at_least_once_orphan_that_is_discarded_on_resume(
    tmp_path, helper_module, trusted_inputs
):
    marker = tmp_path / "crash.marker"
    payloads = {_tasks()[0].key.text: {"mode": "exit_once", "marker": str(marker)}}
    coordinator = _coordinator(tmp_path, helper_module, trusted_inputs, tasks=_tasks(payloads))
    with pytest.raises(parallel.WorkerExecutionError, match="exited unexpectedly"):
        coordinator.run()
    assert marker.exists()
    assert coordinator.last_worker_survivors == ()
    assert not any(_pid_alive(pid) for pid in coordinator.last_worker_pids.values())
    report = _coordinator(tmp_path, helper_module, trusted_inputs, tasks=_tasks(payloads)).run()
    assert len(report.merged.task_order) == len(_tasks())


def test_atomic_shard_crash_point_is_hash_validated_and_recovered_without_rerun(
    tmp_path, helper_module, trusted_inputs
):
    coordinator = _coordinator(tmp_path, helper_module, trusted_inputs)
    coordinator._prepare_private_staging()
    coordinator.shard_root.mkdir()
    task = _tasks()[0]
    payload = coordinator._serialise_shard(
        task,
        parallel.JobResult({}, {"probability": np.ones(2, dtype=np.float64)}),
    )
    (coordinator.shard_root / f"{task.key.text}.npz").write_bytes(payload)
    report = coordinator.run()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert manifest["recovered_atomic_job_checkpoints"] == []
    assert task.key.text in report.recovered_checkpoint_keys
    assert task.key.text in report.resumed_keys
    assert task.key.text not in report.executed_keys


def test_stale_extra_job_checkpoint_is_rejected(tmp_path, helper_module, trusted_inputs):
    coordinator = _coordinator(tmp_path, helper_module, trusted_inputs)
    coordinator._prepare_private_staging()
    coordinator.shard_root.mkdir()
    (coordinator.shard_root / "stale_extra.npz").write_bytes(b"not trusted")
    with pytest.raises(ValueError, match="stale or extra"):
        coordinator.run()


def test_exclusive_run_lock_rejects_second_owner(tmp_path, helper_module, trusted_inputs):
    coordinator = _coordinator(tmp_path, helper_module, trusted_inputs)
    coordinator._prepare_private_staging()
    with parallel._RunLock(coordinator.staging_root):
        with pytest.raises(RuntimeError, match="already owned"):
            coordinator.run()


def test_interprocess_run_lock_contention(tmp_path):
    root = tmp_path / "interprocess-lock"
    script = (
        "from pathlib import Path; import time; "
        "from src.test_007.run_test_007_nested_parallel import _RunLock; "
        f"lock=_RunLock(Path({str(root)!r})); lock.__enter__(); "
        "print('READY', flush=True); time.sleep(30)"
    )
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "READY"
        with pytest.raises(RuntimeError, match="already owned"):
            with parallel._RunLock(root):
                pass
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_cli_has_no_publish_option_and_requires_shadow_plan():
    with pytest.raises(SystemExit):
        parallel.parse_args(["--publish-name", "nested_oof_probability.npy"])
