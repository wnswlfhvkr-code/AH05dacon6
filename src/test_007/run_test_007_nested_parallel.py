"""SHADOW/EXPERIMENTAL process scheduler for TEST_007 Nested jobs.

This additive module is not a production Nested OOF adapter.  It executes a static,
exact two-phase plan using one persistent spawned process for GPU jobs and one for
CPU jobs.  Inner jobs are committed before any outer job is dispatched.  Execution
is at-least-once and runners must be idempotent; each immutable task key is committed
once in the trusted manifest.  The audited sequential runner never imports this file.

The shadow scheduler never publishes into the production artifact root.  A real
adapter must still turn inner-OOF selections into the dynamic set of outer refits.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import io
import json
import multiprocessing
import os
from pathlib import Path
import queue
import random
import re
import secrets
import stat
import sys
import threading
import time
import traceback
from typing import Any, Callable, ContextManager, Iterable, Mapping
import zipfile

import numpy as np

from src.test_007 import run_test_007_nested as nested


PARALLEL_SCHEMA_VERSION = 2
SCHEDULER_MODE = "SHADOW_EXPERIMENTAL"
RESOURCES = ("gpu", "cpu")
FORBIDDEN_INPUT_ROLES = frozenset({"test", "submission", "sample_submission"})
REQUIRED_INPUT_ROLES = frozenset({"goal_config", "search_universe", "fold_assignments", "train"})
RESERVED_RECORD_KEYS = frozenset(
    {
        "task_key",
        "stage",
        "alias",
        "seed",
        "outer_fold",
        "inner_fold",
        "resource",
        "task_rng_seed",
    }
)
WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
)
STRICT_SLUG = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
LIMITATIONS = (
    "no_two_phase_nested_selection_adapter",
    "outer_tasks_are_static_plan_not_inner_selected",
    "production_artifact_publish_forbidden",
    "python_cannot_guarantee_privileged_hostile_handle_races",
)
SPECIFICATION = re.compile(
    r"^(?P<module>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*):"
    r"(?P<symbol>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)$"
)


def _json_loads_no_duplicates(payload: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden in trusted state: {key}")
            result[key] = value
        return result

    return json.loads(payload, object_pairs_hook=object_pairs)


class WorkerExecutionError(RuntimeError):
    pass


class WorkerTimeoutError(TimeoutError):
    pass


class ExecutionCancelled(RuntimeError):
    pass


@contextmanager
def nested_runtime_guard(task: "ParallelTask") -> Iterable[None]:
    """Child-local delegation to the audited Nested protected-path guard.

    Protected paths are guard-only task metadata.  They are resolved but never opened
    or hashed by the coordinator.  The child-local audit is intentionally not merged
    into production audit artifacts because this scheduler is shadow-only.
    """

    test_value = task.payload.get("_protected_test_path")
    submission_value = task.payload.get("_protected_submission_path")
    if not isinstance(test_value, str) or not isinstance(submission_value, str):
        raise ValueError("nested runtime guard requires protected Test/submission path metadata")
    audit = nested.Audit()
    with nested._runtime_read_guard(
        Path(test_value).resolve(),
        Path(submission_value).resolve(),
        audit,
        task.key.stage,
        task.key.alias,
    ):
        yield


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(_canonical_bytes(list(values.shape)))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            array_stream = io.BytesIO()
            np.lib.format.write_array(array_stream, np.asarray(arrays[name]), allow_pickle=False)
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, array_stream.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def _is_reparse(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(details, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _existing_components(path: Path) -> list[Path]:
    resolved = path.absolute()
    components: list[Path] = []
    current = resolved
    while True:
        if current.exists() or current.is_symlink():
            components.append(current)
        if current.parent == current:
            break
        current = current.parent
    return list(reversed(components))


def _ensure_secure_directory(path: Path, *, create: bool) -> Path:
    absolute = path.absolute()
    for component in _existing_components(absolute):
        if _is_reparse(component):
            raise ValueError(f"reparse/symlink path component is forbidden: {component}")
    if create:
        absolute.mkdir(parents=True, exist_ok=True)
    if not absolute.is_dir() or _is_reparse(absolute):
        raise ValueError(f"secure directory required: {absolute}")
    resolved = absolute.resolve(strict=True)
    if resolved != absolute.resolve(strict=False):
        raise ValueError(f"directory resolution changed unexpectedly: {absolute}")
    for component in _existing_components(absolute):
        if _is_reparse(component):
            raise ValueError(f"reparse/symlink path component is forbidden: {component}")
    return resolved


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class _DirectoryProof:
    canonical_path: str
    identity: tuple[int, int]


@dataclass(frozen=True)
class _TargetProjection:
    canonical_path: str
    ancestor_canonical_path: str
    ancestor_identity: tuple[int, int]
    unresolved_suffix: tuple[str, ...]
    identity_chain: tuple[tuple[int, int], ...]


def _directory_proof(path: Path, *, require_no_reparse: bool) -> _DirectoryProof:
    absolute = path.absolute()
    if not absolute.is_dir():
        raise ValueError(f"physical directory proof requires an existing directory: {absolute}")
    if require_no_reparse and any(_is_reparse(part) for part in _existing_components(absolute)):
        raise ValueError(f"physical directory proof rejects reparse components: {absolute}")
    if os.name != "nt":
        resolved = absolute.resolve(strict=True)
        details = resolved.stat()
        return _DirectoryProof(str(resolved), (details.st_dev, details.st_ino))

    import ctypes
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(absolute),
        0,
        0x1 | 0x2 | 0x4,
        None,
        3,
        0x02000000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle in (None, invalid_handle):
        raise ValueError(
            f"Windows physical directory handle proof unavailable: {ctypes.get_last_error()}"
        )
    try:
        information = ByHandleFileInformation()
        if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
            raise ValueError("Windows file identity proof unavailable")
        if require_no_reparse and information.dwFileAttributes & 0x400:
            raise ValueError("Windows directory handle resolved to a reparse point")
        buffer = ctypes.create_unicode_buffer(32768)
        length = kernel32.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
        if length == 0 or length >= len(buffer):
            raise ValueError("Windows final handle path proof unavailable")
        canonical = buffer.value
        if canonical.startswith("\\\\?\\UNC\\"):
            canonical = "\\\\" + canonical[8:]
        elif canonical.startswith("\\\\?\\"):
            canonical = canonical[4:]
        file_index = (int(information.nFileIndexHigh) << 32) | int(
            information.nFileIndexLow
        )
        return _DirectoryProof(
            os.path.normcase(os.path.normpath(canonical)),
            (int(information.dwVolumeSerialNumber), file_index),
        )
    finally:
        kernel32.CloseHandle(handle)


def _directory_identity_chain(path: Path) -> tuple[tuple[int, int], ...]:
    absolute = path.absolute()
    if not absolute.is_dir():
        raise ValueError(f"identity chain requires an existing directory: {absolute}")
    directories = list(reversed([absolute, *absolute.parents]))
    identities: list[tuple[int, int]] = []
    for directory in directories:
        if not directory.exists() or not directory.is_dir():
            continue
        proof = _directory_proof(directory, require_no_reparse=False)
        if not identities or identities[-1] != proof.identity:
            identities.append(proof.identity)
    return tuple(identities)


def _physical_target_projection(
    path: Path, *, require_no_reparse_existing: bool = False
) -> _TargetProjection:
    absolute = path.absolute()
    current = absolute
    remainder: list[str] = []
    while not current.exists():
        if current.parent == current:
            raise ValueError(f"no existing ancestor for physical path proof: {absolute}")
        remainder.append(current.name)
        current = current.parent
    if not current.is_dir():
        raise ValueError(f"physical path ancestor is not a directory: {current}")
    proof = _directory_proof(current, require_no_reparse=require_no_reparse_existing)
    suffix = tuple(reversed(remainder))
    projected = proof.canonical_path
    for component in suffix:
        projected = os.path.join(projected, component)
    return _TargetProjection(
        canonical_path=os.path.normcase(os.path.normpath(projected)),
        ancestor_canonical_path=proof.canonical_path,
        ancestor_identity=proof.identity,
        unresolved_suffix=suffix,
        identity_chain=_directory_identity_chain(current),
    )


def _assert_physical_output_separation(existing_root: Path, other_root: Path) -> None:
    try:
        left = _physical_target_projection(
            existing_root, require_no_reparse_existing=True
        )
        if left.unresolved_suffix:
            raise ValueError("existing output root unexpectedly has an unresolved suffix")
        right = _physical_target_projection(other_root)
        left_path = left.canonical_path
        right_path = right.canonical_path
        try:
            common = os.path.commonpath((left_path, right_path))
        except ValueError:
            common = ""
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "share physical output" in str(error):
            raise
        raise ValueError("could not prove physical output-root separation") from error
    if (
        right.ancestor_identity == left.ancestor_identity
        or left.ancestor_identity in right.identity_chain
        or (
            not right.unresolved_suffix
            and right.ancestor_identity in left.identity_chain
        )
        or common == left_path
        or common == right_path
    ):
        raise ValueError("roots share physical output identity or ancestor containment")


def _validate_slug(value: str, role: str) -> str:
    if not STRICT_SLUG.fullmatch(value):
        raise ValueError(f"{role} must be a portable lowercase slug")
    if value.casefold() in WINDOWS_RESERVED_NAMES:
        raise ValueError(f"{role} uses a Windows reserved device name")
    return value


def _secure_file_bytes(path: Path, expected_parent: Path) -> bytes:
    parent = _ensure_secure_directory(path.parent, create=False)
    if parent != expected_parent:
        raise ValueError("file parent escaped the trusted staging directory")
    details = path.lstat()
    if _is_reparse(path) or not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise ValueError(f"trusted staging file must be regular and singly linked: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (details.st_dev, details.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ValueError(f"trusted staging file changed during open: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final_details = path.lstat()
        if (path.parent.resolve(strict=True), final_details.st_dev, final_details.st_ino) != (
            parent,
            opened.st_dev,
            opened.st_ino,
        ):
            raise ValueError(f"trusted staging file changed during read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _exclusive_temp(parent: Path, stem: str) -> tuple[Path, int]:
    parent = _ensure_secure_directory(parent, create=False)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _ in range(16):
        temporary = parent / f".{stem}.{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(temporary, flags, 0o600)
        except FileExistsError:
            continue
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or _is_reparse(temporary):
            os.close(descriptor)
            try:
                temporary.unlink()
            except OSError:
                pass
            raise ValueError("exclusive temporary file is not a singly linked regular file")
        return temporary, descriptor
    raise FileExistsError("could not allocate a collision-free exclusive temporary file")


def _atomic_bytes(path: Path, payload: bytes, expected_parent: Path) -> None:
    parent = _ensure_secure_directory(path.parent, create=False)
    if parent != expected_parent:
        raise ValueError("atomic write parent escaped the trusted staging directory")
    temporary, descriptor = _exclusive_temp(parent, path.name)
    try:
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        temp_details = temporary.lstat()
        if _is_reparse(temporary) or temp_details.st_nlink != 1:
            raise ValueError("temporary file link state changed before commit")
        if path.exists() or path.is_symlink():
            target_details = path.lstat()
            if _is_reparse(path) or not stat.S_ISREG(target_details.st_mode) or target_details.st_nlink != 1:
                raise ValueError("atomic target is not a singly linked regular file")
        if _ensure_secure_directory(path.parent, create=False) != expected_parent:
            raise ValueError("atomic target parent changed before commit")
        for attempt in range(100):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 99:
                    raise
                if _ensure_secure_directory(path.parent, create=False) != expected_parent:
                    raise ValueError("atomic target parent changed during replace retry")
                if path.exists() or path.is_symlink():
                    retry_target = path.lstat()
                    if (
                        _is_reparse(path)
                        or not stat.S_ISREG(retry_target.st_mode)
                        or retry_target.st_nlink != 1
                    ):
                        raise ValueError("atomic target link state changed during replace retry")
                time.sleep(0.01)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _discard_secure_temps(parent: Path) -> tuple[str, ...]:
    """Remove only singly-linked regular atomic leftovers while holding the run lock."""

    trusted_parent = _ensure_secure_directory(parent, create=False)
    discarded: list[str] = []
    for path in trusted_parent.iterdir():
        if not (path.name.startswith(".") and path.name.endswith(".tmp")):
            continue
        _secure_file_bytes(path, trusted_parent)
        path.unlink()
        discarded.append(path.name)
    return tuple(sorted(discarded))


def _atomic_json(path: Path, value: Any, expected_parent: Path) -> None:
    _atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"), expected_parent)


def _hash_regular_input(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"trusted input must be a file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_spec(specification: str) -> tuple[str, str]:
    matched = SPECIFICATION.fullmatch(specification)
    if matched is None:
        raise ValueError("callable must use an approved importable.module:portable_symbol spec")
    return matched.group("module"), matched.group("symbol")


def _load_symbol(specification: str) -> Any:
    module_name, symbol_name = _parse_spec(specification)
    value: Any = importlib.import_module(module_name)
    for component in symbol_name.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError(f"loaded symbol is not callable: {specification}")
    return value


def _spec_source_path(symbol: Any) -> Path:
    """Return the actual implementation file for an already child-loaded callable."""

    implementation = inspect.unwrap(symbol)
    source = inspect.getsourcefile(implementation) or inspect.getfile(implementation)
    path = Path(source).resolve(strict=True)
    if not path.is_file():
        raise ValueError("callable implementation source is not a regular file")
    return path


def _resolve_module_without_import(module_name: str, approved_roots: Iterable[Path]) -> Path:
    relative = Path(*module_name.split("."))
    candidates: list[Path] = []
    for root in approved_roots:
        for candidate in (root / relative.with_suffix(".py"), root / relative / "__init__.py"):
            if candidate.is_file():
                candidates.append(candidate.resolve(strict=True))
    unique = tuple(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise ValueError(f"module must resolve exactly once inside approved code roots: {module_name}")
    return unique[0]


def _try_resolve_module(module_name: str, approved_roots: Iterable[Path]) -> Path | None:
    try:
        return _resolve_module_without_import(module_name, approved_roots)
    except ValueError:
        return None


def _imported_local_modules(
    module_name: str, path: Path, approved_roots: tuple[Path, ...]
) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    package = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                keep = max(0, len(parts) - node.level + 1)
                base = ".".join(parts[:keep])
                target = ".".join(part for part in (base, node.module or "") if part)
            else:
                target = node.module or ""
            if target:
                candidates.append(target)
            candidates.extend(
                ".".join(part for part in (target, alias.name) if part)
                for alias in node.names
                if alias.name != "*"
            )
        for candidate in candidates:
            if candidate and _try_resolve_module(candidate, approved_roots) is not None:
                result.add(candidate)
    return result


def _static_source_closure(
    entry_modules: Iterable[str], approved_roots: tuple[Path, ...]
) -> dict[str, Path]:
    pending = list(dict.fromkeys(entry_modules))
    closure: dict[str, Path] = {}
    while pending:
        module_name = pending.pop()
        if module_name in closure:
            continue
        path = _resolve_module_without_import(module_name, approved_roots)
        closure[module_name] = path
        pending.extend(
            sorted(_imported_local_modules(module_name, path, approved_roots) - set(closure))
        )
    return closure


@dataclass(frozen=True, order=True)
class TaskKey:
    stage: str
    alias: str
    seed: int
    outer_fold: int
    inner_fold: int | None = None

    def __post_init__(self) -> None:
        if self.stage not in {"inner", "outer_refit"}:
            raise ValueError("task stage must be inner or outer_refit")
        _validate_slug(self.alias, "alias")
        if self.outer_fold not in range(nested.EXPECTED_OUTER_FOLDS):
            raise ValueError("outer fold is outside the fixed Nested limit")
        if self.stage == "inner" and self.inner_fold not in range(nested.EXPECTED_INNER_FOLDS):
            raise ValueError("inner task requires an exact inner fold")
        if self.stage == "outer_refit" and self.inner_fold is not None:
            raise ValueError("outer_refit task must not have an inner fold")

    @property
    def text(self) -> str:
        base = f"{self.stage}__{self.alias}__seed_{self.seed}__outer_{self.outer_fold}"
        return base if self.inner_fold is None else f"{base}__inner_{self.inner_fold}"

    def payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "alias": self.alias,
            "seed": self.seed,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "TaskKey":
        return cls(
            str(value["stage"]),
            str(value["alias"]),
            int(value["seed"]),
            int(value["outer_fold"]),
            None if value.get("inner_fold") is None else int(value["inner_fold"]),
        )


def _normalise_array_schema(value: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, specification in value.items():
        _validate_slug(str(name), "array name")
        shape = tuple(int(size) for size in specification["shape"])
        if any(size < 0 for size in shape):
            raise ValueError("array shape cannot contain negative dimensions")
        dtype = str(np.dtype(specification["dtype"]))
        if np.dtype(dtype).hasobject:
            raise ValueError("object array schemas are forbidden")
        result[str(name)] = {"shape": list(shape), "dtype": dtype}
    if not result:
        raise ValueError("each task needs a non-empty exact array schema")
    return result


@dataclass(frozen=True)
class ParallelTask:
    key: TaskKey
    resource: str
    array_schema: Mapping[str, Mapping[str, Any]]
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.resource not in RESOURCES:
            raise ValueError(f"resource must be one of {RESOURCES}")
        schema = _normalise_array_schema(self.array_schema)
        _canonical_bytes(dict(self.payload))
        object.__setattr__(self, "array_schema", schema)
        object.__setattr__(self, "payload", dict(self.payload))

    def contract_payload(self) -> dict[str, Any]:
        return {
            "key": self.key.payload(),
            "resource": self.resource,
            "array_schema": {name: dict(spec) for name, spec in self.array_schema.items()},
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class PhaseCoverage:
    seeds: tuple[int, ...]
    outer_folds: int
    inner_folds: int
    inner_aliases: tuple[str, ...]
    outer_aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("coverage seeds must be non-empty and unique")
        if self.outer_folds not in range(1, nested.EXPECTED_OUTER_FOLDS + 1):
            raise ValueError("invalid outer fold coverage")
        if self.inner_folds not in range(1, nested.EXPECTED_INNER_FOLDS + 1):
            raise ValueError("invalid inner fold coverage")
        for role, aliases in (("inner", self.inner_aliases), ("outer", self.outer_aliases)):
            if not aliases or len({alias.casefold() for alias in aliases}) != len(aliases):
                raise ValueError(f"{role} aliases must be non-empty and casefold-unique")
            for alias in aliases:
                _validate_slug(alias, f"{role} alias")

    def payload(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "outer_folds": self.outer_folds,
            "inner_folds": self.inner_folds,
            "inner_aliases": list(self.inner_aliases),
            "outer_aliases": list(self.outer_aliases),
        }

    def expected_keys(self) -> set[str]:
        inner = {
            TaskKey("inner", alias, seed, outer, inner).text
            for alias in self.inner_aliases
            for seed in self.seeds
            for outer in range(self.outer_folds)
            for inner in range(self.inner_folds)
        }
        outer = {
            TaskKey("outer_refit", alias, seed, fold).text
            for alias in self.outer_aliases
            for seed in self.seeds
            for fold in range(self.outer_folds)
        }
        return inner | outer


@dataclass(frozen=True)
class JobResult:
    records: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class MergedResult:
    task_order: tuple[str, ...]
    records: tuple[dict[str, Any], ...]
    arrays: Mapping[str, np.ndarray]
    array_schema: Mapping[str, Mapping[str, Any]]
    task_metadata: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ExecutionReport:
    merged: MergedResult
    executed_keys: tuple[str, ...]
    resumed_keys: tuple[str, ...]
    manifest_path: Path
    max_concurrency: Mapping[str, int]
    worker_pids: Mapping[str, int]
    worker_exitcodes: Mapping[str, int]
    recovered_checkpoint_keys: tuple[str, ...]
    discarded_temp_entries: tuple[dict[str, str], ...]


def _validate_result(task: ParallelTask, value: Any) -> JobResult:
    if isinstance(value, JobResult):
        records, arrays = dict(value.records), dict(value.arrays)
    elif isinstance(value, Mapping) and set(value) == {"records", "arrays"}:
        records, arrays = dict(value["records"]), dict(value["arrays"])
    else:
        raise TypeError("runner must return JobResult or exactly {'records', 'arrays'}")
    overlap = RESERVED_RECORD_KEYS.intersection(records)
    if overlap:
        raise ValueError(f"runner records attempted to override coordinator metadata: {sorted(overlap)}")
    _canonical_bytes(records)
    if set(arrays) != set(task.array_schema):
        raise ValueError(f"{task.key.text}: result arrays do not exactly match task schema")
    normalised: dict[str, np.ndarray] = {}
    for name, specification in task.array_schema.items():
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        if list(array.shape) != specification["shape"] or str(array.dtype) != specification["dtype"]:
            raise ValueError(f"{task.key.text}:{name}: array shape/dtype differs from task schema")
        normalised[name] = array
    return JobResult(records, normalised)


def _task_rng_seed(task: ParallelTask) -> int:
    digest = hashlib.sha256(_canonical_bytes(task.key.payload())).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _reset_task_rng(task: ParallelTask) -> int:
    seed = _task_rng_seed(task)
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = sys.modules.get("torch")
    if torch is None:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def _verify_loaded_approved_module_sources(
    approved_roots: tuple[Path, ...], trusted_source_hashes: Mapping[str, str]
) -> None:
    for module in tuple(sys.modules.values()):
        source_value = getattr(module, "__file__", None)
        if not source_value:
            continue
        source = Path(source_value)
        if source.suffix in {".pyc", ".pyo"}:
            try:
                source = Path(importlib.util.source_from_cache(str(source)))
            except ValueError:
                continue
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not any(_path_contains(root, resolved) for root in approved_roots):
            continue
        expected = trusted_source_hashes.get(str(resolved))
        if expected is None or _hash_regular_input(resolved) != expected:
            raise ValueError(
                "child loaded an untrusted or drifted local module outside the static source "
                f"closure: {resolved}"
            )


def _resource_worker(
    resource: str,
    task_queue: Any,
    result_queue: Any,
    runner_spec: str,
    guard_spec: str | None,
    unsafe_synthetic_no_guard: bool,
    protected_paths: tuple[str, str] | None,
    approved_code_roots: tuple[str, ...],
    trusted_source_hashes: Mapping[str, str],
) -> None:
    try:
        if unsafe_synthetic_no_guard:
            bootstrap_context = nullcontext()
        else:
            if protected_paths is None:
                raise ValueError("protected child bootstrap paths are missing")
            bootstrap_context = nested._runtime_read_guard(
                Path(protected_paths[0]).resolve(),
                Path(protected_paths[1]).resolve(),
                nested.Audit(),
                "bootstrap_import",
                f"bootstrap_{resource}",
            )
        with bootstrap_context:
            for source_path, expected_hash in trusted_source_hashes.items():
                if _hash_regular_input(Path(source_path)) != expected_hash:
                    raise ValueError("source closure drifted before protected child import")
            runner = _load_symbol(runner_spec)
            guard_factory = None if guard_spec is None else _load_symbol(guard_spec)
        roots = tuple(Path(root).resolve(strict=True) for root in approved_code_roots)
        for symbol in (runner, guard_factory):
            if symbol is None:
                continue
            source = _spec_source_path(symbol)
            if not any(_path_contains(root, source) for root in roots):
                raise ValueError("loaded callable implementation escaped approved code roots")
            if trusted_source_hashes.get(str(source)) != _hash_regular_input(source):
                raise ValueError("loaded callable implementation differs from trusted source closure")
        _verify_loaded_approved_module_sources(roots, trusted_source_hashes)
        for source_path, expected_hash in trusted_source_hashes.items():
            if _hash_regular_input(Path(source_path)) != expected_hash:
                raise ValueError("source closure drifted during protected child import")
    except BaseException:
        result_queue.put(("bootstrap_error", resource, None, traceback.format_exc()))
        return
    state: dict[str, Any] = {}
    while True:
        task = task_queue.get()
        if task is None:
            return
        result_queue.put(("started", resource, task.key.text, None))
        try:
            torch_loaded_before_reset = "torch" in sys.modules
            _reset_task_rng(task)
            context = nullcontext() if guard_factory is None else guard_factory(task)
            with context:
                result = _validate_result(task, runner(task, state))
            if not torch_loaded_before_reset and "torch" in sys.modules:
                raise RuntimeError(
                    "runner lazily loaded torch after deterministic RNG reset; "
                    "import torch at module bootstrap"
                )
            _verify_loaded_approved_module_sources(roots, trusted_source_hashes)
            result_queue.put(("result", resource, task.key.text, result))
        except BaseException:
            result_queue.put(("error", resource, task.key.text, traceback.format_exc()))


def _task_sort_key(task: ParallelTask, aliases: tuple[str, ...]) -> tuple[int, int, int, int, int]:
    return (
        aliases.index(task.key.alias),
        task.key.seed,
        task.key.outer_fold,
        0 if task.key.stage == "inner" else 1,
        -1 if task.key.inner_fold is None else task.key.inner_fold,
    )


def merge_results(
    tasks: Iterable[ParallelTask], results: Mapping[str, JobResult], alias_order: Iterable[str]
) -> MergedResult:
    task_list, aliases = tuple(tasks), tuple(alias_order)
    keys = [task.key.text for task in task_list]
    if len(keys) != len(set(keys)):
        raise ValueError("task keys must have committed-once identity")
    if set(results) != set(keys):
        raise ValueError("results have missing or extra task keys")
    ordered = sorted(task_list, key=lambda task: _task_sort_key(task, aliases))
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    schema: dict[str, dict[str, Any]] = {}
    task_metadata: dict[str, dict[str, Any]] = {}
    for task in ordered:
        result = _validate_result(task, results[task.key.text])
        coordinator_metadata = {
            "task_key": task.key.text,
            "stage": task.key.stage,
            "alias": task.key.alias,
            "seed": task.key.seed,
            "outer_fold": task.key.outer_fold,
            "inner_fold": task.key.inner_fold,
            "resource": task.resource,
            "task_rng_seed": _task_rng_seed(task),
        }
        task_metadata[task.key.text] = coordinator_metadata
        record = {**coordinator_metadata, **dict(result.records)}
        records.append(record)
        for name in sorted(result.arrays):
            merged_name = f"{task.key.text}::{name}"
            arrays[merged_name] = result.arrays[name]
            schema[merged_name] = dict(task.array_schema[name])
    task_order = tuple(task.key.text for task in ordered)
    if tuple(record["task_key"] for record in records) != task_order:
        raise RuntimeError("merged record identity differs from immutable task order")
    if set(arrays) != set(schema):
        raise RuntimeError("merged array schema coverage is incomplete")
    return MergedResult(task_order, tuple(records), arrays, schema, task_metadata)


class _RunLock:
    def __init__(self, staging_root: Path) -> None:
        self.root = staging_root
        self.path = staging_root / ".parallel_run.lock"
        self.descriptor: int | None = None

    def __enter__(self) -> "_RunLock":
        self.root = _ensure_secure_directory(self.root, create=True)
        before = self.path.lstat() if self.path.exists() or self.path.is_symlink() else None
        if before is not None and (
            _is_reparse(self.path)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise ValueError("parallel run lock must be a singly linked regular file")
        descriptor = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            opened = os.fstat(descriptor)
            after = self.path.lstat()
            if (
                _is_reparse(self.path)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
            ):
                raise ValueError("parallel run lock changed during secure open")
            if opened.st_size == 0:
                os.write(descriptor, b"0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                except OSError as error:
                    raise RuntimeError("parallel staging is already owned by a live process") from error
            else:
                import fcntl

                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    raise RuntimeError("parallel staging is already owned by a live process") from error
            self.descriptor = descriptor
            return self
        except BaseException:
            os.close(descriptor)
            raise

    def __exit__(self, *_args: Any) -> None:
        if self.descriptor is None:
            return
        try:
            os.lseek(self.descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)
            self.descriptor = None


class ParallelCoordinator:
    """Static shadow plan coordinator; never a production Nested artifact publisher."""

    def __init__(
        self,
        *,
        tasks: Iterable[ParallelTask],
        coverage: PhaseCoverage,
        alias_order: Iterable[str],
        trusted_input_paths: Mapping[str, Path],
        staging_root: Path,
        production_artifact_root: Path,
        job_runner_spec: str,
        guard_factory_spec: str | None,
        approved_code_roots: Iterable[Path],
        protected_test_path: Path | None = None,
        protected_submission_path: Path | None = None,
        unsafe_synthetic_no_guard: bool = False,
        task_timeout_seconds: float = 3600.0,
        cleanup_timeout_seconds: float = 5.0,
    ) -> None:
        self.tasks = tuple(tasks)
        self.coverage = coverage
        self.alias_order = tuple(alias_order)
        self.trusted_input_paths = {str(role): Path(path) for role, path in trusted_input_paths.items()}
        self.staging_root = Path(staging_root).absolute()
        self.production_artifact_root = Path(production_artifact_root).absolute()
        self.job_runner_spec = job_runner_spec
        self.guard_factory_spec = guard_factory_spec
        self.approved_code_roots = tuple(
            dict.fromkeys(Path(path).resolve(strict=True) for path in approved_code_roots)
        )
        self.protected_test_path = (
            None if protected_test_path is None else Path(protected_test_path).absolute()
        )
        self.protected_submission_path = (
            None
            if protected_submission_path is None
            else Path(protected_submission_path).absolute()
        )
        self.unsafe_synthetic_no_guard = bool(unsafe_synthetic_no_guard)
        self.task_timeout_seconds = float(task_timeout_seconds)
        self.cleanup_timeout_seconds = float(cleanup_timeout_seconds)
        self._validate_contract()
        self.source_paths = self._source_paths()
        self.initial_hashes = self._compute_actual_hashes()
        self.contract_payload = {
            "schema_version": PARALLEL_SCHEMA_VERSION,
            "mode": SCHEDULER_MODE,
            "production_nested_adapter": False,
            "production_publish_allowed": False,
            "limitations": list(LIMITATIONS),
            "execution_semantics": "at_least_once_idempotent_runner_required",
            "commit_semantics": "one_manifest_commit_per_immutable_task_key",
            "phase_barrier": "all_inner_committed_before_any_outer_dispatch",
            "checkpoint_policy": {
                "job": "atomic_shard_then_atomic_hash_receipt_after_every_completed_job",
                "outer_fold_barrier": "atomic_receipt_hash_bundle_after_exact_inner_fold_coverage",
                "resume": "actual_input_source_job_and_barrier_hash_validation",
            },
            "rng_policy": {
                "seed_source": "sha256_of_immutable_task_key",
                "reset_before_every_task": [
                    "python_random",
                    "numpy",
                    "torch_cpu_cuda_if_runner_loaded",
                ],
                "lazy_torch_import_after_reset": "reject_without_commit",
            },
            "coverage": coverage.payload(),
            "alias_order": list(self.alias_order),
            "trusted_hashes": self.initial_hashes,
            "approved_code_roots": list(map(str, self.approved_code_roots)),
            "protected_paths": (
                None
                if self.unsafe_synthetic_no_guard
                else {
                    "test": {
                        "path": str(self._path_identity_without_content(self.protected_test_path)[0]),
                        "identity": list(
                            self._path_identity_without_content(self.protected_test_path)[1]
                        ),
                    },
                    "submission": {
                        "path": str(
                            self._path_identity_without_content(self.protected_submission_path)[0]
                        ),
                        "identity": list(
                            self._path_identity_without_content(self.protected_submission_path)[1]
                        ),
                    },
                }
            ),
            "tasks": [task.contract_payload() for task in self.tasks],
            "unsafe_synthetic_no_guard": self.unsafe_synthetic_no_guard,
            "threat_model": {
                "supported": "unprivileged_pathname_link_reparse_swap_and_stale_checkpoint_defense",
                "excluded": "privileged_hostile_handle_or_kernel_level_mutation",
                "production_safe_claimed": False,
            },
        }
        self.contract_hash = _canonical_hash(self.contract_payload)
        self.manifest_path = self.staging_root / "parallel_manifest.json"
        self.shard_root = self.staging_root / "shards"
        self._staging_identity: tuple[int, int] | None = None
        self._staging_token: str | None = None
        self.last_worker_pids: dict[str, int] = {}
        self.last_worker_survivors: tuple[int, ...] = ()

    def _validate_contract(self) -> None:
        if not self.tasks:
            raise ValueError("at least one task is required")
        if self.task_timeout_seconds <= 0 or self.cleanup_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if not self.unsafe_synthetic_no_guard and not self.guard_factory_spec:
            raise ValueError("protected guard spec is mandatory outside unsafe synthetic/test mode")
        if not self.approved_code_roots:
            raise ValueError("at least one approved code root is required")
        for root in self.approved_code_roots:
            _ensure_secure_directory(root, create=False)
        _parse_spec(self.job_runner_spec)
        if self.guard_factory_spec:
            _parse_spec(self.guard_factory_spec)
        if not self.unsafe_synthetic_no_guard and (
            self.protected_test_path is None or self.protected_submission_path is None
        ):
            raise ValueError("protected Test and submission identities are required")
        if set(self.coverage.expected_keys()) != {task.key.text for task in self.tasks}:
            raise ValueError("plan does not exactly cover declared inner/outer phases")
        if len(self.tasks) != len({task.key.text for task in self.tasks}):
            raise ValueError("task keys must have committed-once identity")
        folded_aliases = [alias.casefold() for alias in self.alias_order]
        if len(folded_aliases) != len(set(folded_aliases)):
            raise ValueError("alias order has a casefold collision")
        for alias in self.alias_order:
            _validate_slug(alias, "alias order entry")
        if set(self.coverage.inner_aliases) | set(self.coverage.outer_aliases) != set(self.alias_order):
            raise ValueError("alias order must exactly cover phase aliases")
        if not self.unsafe_synthetic_no_guard and (
            self.coverage.seeds != nested.EXPECTED_SEEDS
            or self.coverage.outer_folds != nested.EXPECTED_OUTER_FOLDS
            or self.coverage.inner_folds != nested.EXPECTED_INNER_FOLDS
        ):
            raise ValueError("protected shadow plans require the full fixed Nested seed/fold matrix")
        if not self.trusted_input_paths:
            raise ValueError("actual trusted input role->path mappings are required")
        required_inputs = REQUIRED_INPUT_ROLES | {
            f"candidate_config_{alias}" for alias in self.alias_order
        }
        if not self.unsafe_synthetic_no_guard and not required_inputs.issubset(
            self.trusted_input_paths
        ):
            missing = sorted(required_inputs - set(self.trusted_input_paths))
            raise ValueError(
                f"protected shadow plans require all train-only trusted input roles: {missing}"
            )
        for role in self.trusted_input_paths:
            _validate_slug(role, "trusted input role")
            if role.casefold() in FORBIDDEN_INPUT_ROLES:
                raise ValueError("protected Test/submission paths must never enter trusted input hashing")
        staging, artifact = self.staging_root.resolve(strict=False), self.production_artifact_root.resolve(strict=False)
        if _path_contains(staging, artifact) or _path_contains(artifact, staging):
            raise ValueError("staging and production artifact roots must be disjoint")
        if self.staging_root.exists() and self.production_artifact_root.exists():
            try:
                if self.staging_root.samefile(self.production_artifact_root):
                    raise ValueError("staging and production artifact roots share physical identity")
            except OSError as error:
                raise ValueError("could not verify physical output root identities") from error
        self._validate_protected_separation()

    def _source_paths(self) -> dict[str, Path]:
        runner_module, _ = _parse_spec(self.job_runner_spec)
        entry_modules = [runner_module, __name__]
        if self.guard_factory_spec:
            guard_module, _ = _parse_spec(self.guard_factory_spec)
            entry_modules.append(guard_module)
        closure = _static_source_closure(entry_modules, self.approved_code_roots)
        known_paths = set(closure.values())
        for key, resolved in self._enumerate_approved_source_files().items():
            if resolved not in known_paths:
                closure[key] = resolved
                known_paths.add(resolved)
        return closure

    def _enumerate_approved_source_files(self) -> dict[str, Path]:
        """Enumerate physical Python source identities without opening file bytes."""

        result: dict[str, Path] = {}
        identities: dict[tuple[int, int], Path] = {}
        for root_index, root in enumerate(self.approved_code_roots):
            for path in sorted(root.rglob("*.py"), key=str):
                if any(_is_reparse(component) for component in _existing_components(path)):
                    raise ValueError(f"approved source envelope contains a reparse path: {path}")
                resolved = path.resolve(strict=True)
                details = resolved.stat()
                if not stat.S_ISREG(details.st_mode):
                    raise ValueError(f"approved Python source must be a regular file: {resolved}")
                identity = (details.st_dev, details.st_ino)
                previous = identities.get(identity)
                if previous is not None and previous != resolved:
                    raise ValueError("approved Python source envelope contains a hardlink alias")
                identities[identity] = resolved
                key = f"approved_envelope_{root_index}_{resolved.relative_to(root).as_posix()}"
                result[key] = resolved
        return result

    @staticmethod
    def _path_identity_without_content(path: Path) -> tuple[Path, tuple[int, int]]:
        absolute = path.absolute()
        for component in _existing_components(absolute):
            if _is_reparse(component):
                raise ValueError(f"reparse/symlink identity path is forbidden: {component}")
        resolved = absolute.resolve(strict=True)
        details = resolved.stat()
        if not stat.S_ISREG(details.st_mode):
            raise ValueError(f"identity path must be a regular file: {resolved}")
        return resolved, (details.st_dev, details.st_ino)

    def _validate_protected_separation(self) -> None:
        if self.unsafe_synthetic_no_guard:
            return
        test_path, test_identity = self._path_identity_without_content(self.protected_test_path)
        submission_path, submission_identity = self._path_identity_without_content(
            self.protected_submission_path
        )
        if test_identity == submission_identity or test_path == submission_path:
            raise ValueError("protected Test and submission identities must be distinct")
        protected = {test_identity, submission_identity}
        protected_paths = {test_path, submission_path}
        checked = [
            (f"trusted input {role}", path)
            for role, path in self.trusted_input_paths.items()
        ]
        for label, path in checked:
            resolved, identity = self._path_identity_without_content(path)
            if identity in protected or resolved in protected_paths:
                raise ValueError(f"{label} aliases a protected Test/submission identity")
        for source in self._enumerate_approved_source_files().values():
            resolved_source, source_identity = self._path_identity_without_content(source)
            if source_identity in protected or resolved_source in protected_paths:
                raise ValueError(
                    "approved Python source aliases a protected Test/submission identity"
                )
        for task in self.tasks:
            task_test = task.payload.get("_protected_test_path")
            task_submission = task.payload.get("_protected_submission_path")
            if not isinstance(task_test, str) or not isinstance(task_submission, str):
                raise ValueError("every protected task must carry exact guard-only path metadata")
            resolved_test, identity_test = self._path_identity_without_content(Path(task_test))
            resolved_submission, identity_submission = self._path_identity_without_content(
                Path(task_submission)
            )
            if (
                (resolved_test, identity_test) != (test_path, test_identity)
                or (resolved_submission, identity_submission)
                != (submission_path, submission_identity)
            ):
                raise ValueError("task guard metadata differs from coordinator protected identities")

    def _compute_actual_hashes(self) -> dict[str, dict[str, str]]:
        self._validate_protected_separation()
        current_sources = self._source_paths()
        inputs = {
            role: {"path": str(path.resolve(strict=True)), "sha256": _hash_regular_input(path)}
            for role, path in sorted(self.trusted_input_paths.items())
        }
        sources = {
            str(path): _hash_regular_input(path)
            for path in sorted(set(current_sources.values()), key=str)
        }
        entries = {
            "job_runner": self.job_runner_spec,
            "guard_factory": self.guard_factory_spec,
            "modules": {module: str(path) for module, path in sorted(current_sources.items())},
        }
        return {"inputs": inputs, "sources": sources, "entry_points": entries}

    def _prepare_private_staging(self) -> Path:
        parent = _ensure_secure_directory(self.staging_root.parent, create=True)
        created = False
        if not self.staging_root.exists() and not self.staging_root.is_symlink():
            try:
                os.mkdir(self.staging_root, 0o700)
                created = True
            except FileExistsError:
                pass
        root = _ensure_secure_directory(self.staging_root, create=False)
        if root.parent.resolve(strict=True) != parent:
            raise ValueError("private staging root parent identity changed during creation")
        proof = _directory_proof(root, require_no_reparse=True)
        self._staging_identity = proof.identity
        marker_path = root / ".private_staging_identity.json"
        if created:
            self._staging_token = secrets.token_hex(32)
            _atomic_json(
                marker_path,
                {
                    "schema_version": PARALLEL_SCHEMA_VERSION,
                    "identity": list(self._staging_identity),
                    "token": self._staging_token,
                },
                root,
            )
        else:
            if not marker_path.is_file():
                for child in root.iterdir():
                    if _is_reparse(child):
                        raise ValueError("preexisting staging contains a reparse/symlink entry")
                raise ValueError("preexisting staging root was not exclusively initialized")
            marker = _json_loads_no_duplicates(
                _secure_file_bytes(marker_path, root).decode("utf-8")
            )
            if (
                set(marker) != {"schema_version", "identity", "token"}
                or marker["schema_version"] != PARALLEL_SCHEMA_VERSION
                or marker["identity"] != list(self._staging_identity)
                or not isinstance(marker["token"], str)
                or len(marker["token"]) != 64
            ):
                raise ValueError("private staging identity marker is invalid")
            self._staging_token = marker["token"]
        _assert_physical_output_separation(root, self.production_artifact_root)
        return root

    def _assert_staging_identity(self) -> Path:
        if self._staging_identity is None:
            raise RuntimeError("private staging identity is unavailable")
        root = _ensure_secure_directory(self.staging_root, create=False)
        proof = _directory_proof(root, require_no_reparse=True)
        if proof.identity != self._staging_identity:
            raise ValueError("private staging root identity changed during operation")
        _assert_physical_output_separation(root, self.production_artifact_root)
        marker_path = root / ".private_staging_identity.json"
        marker = _json_loads_no_duplicates(_secure_file_bytes(marker_path, root).decode("utf-8"))
        if marker.get("identity") != list(self._staging_identity) or marker.get(
            "token"
        ) != self._staging_token:
            raise ValueError("private staging marker changed during operation")
        return root

    @staticmethod
    def _mutable_state_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "completed": manifest["completed"],
            "outer_fold_barriers": manifest["outer_fold_barriers"],
            "recovered_atomic_job_checkpoints": manifest[
                "recovered_atomic_job_checkpoints"
            ],
            "discarded_atomic_temps": manifest["discarded_atomic_temps"],
        }

    def _validate_mutable_state(self, manifest: Mapping[str, Any]) -> None:
        completed = manifest.get("completed")
        expected = {task.key.text for task in self.tasks}
        if not isinstance(completed, dict) or not set(completed).issubset(expected):
            raise ValueError("resume manifest contains missing/extra/invalid committed task keys")
        task_lookup = {task.key.text: task for task in self.tasks}
        for key, receipt in completed.items():
            expected_phase = task_lookup[key].key.stage
            if (
                not isinstance(receipt, dict)
                or set(receipt) != {"sha256", "phase"}
                or receipt["phase"] != expected_phase
                or not isinstance(receipt["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"])
            ):
                raise ValueError(f"{key}: committed receipt schema differs from immutable task")

        recovered = manifest.get("recovered_atomic_job_checkpoints")
        if recovered != []:
            raise ValueError(
                "persistent recovered checkpoint transcript must remain canonical empty"
            )
        discarded = manifest.get("discarded_atomic_temps")
        if discarded != []:
            raise ValueError(
                "persistent discarded temp transcript must remain canonical empty"
            )

        inner_keys = {task.key.text for task in self.tasks if task.key.stage == "inner"}
        outer_keys = {task.key.text for task in self.tasks if task.key.stage == "outer_refit"}
        if set(completed).intersection(outer_keys) and not inner_keys.issubset(completed):
            raise ValueError("resume manifest violates the inner-before-outer phase barrier")
        self._validate_outer_fold_barriers(manifest)

    def _commit_manifest(self, manifest: dict[str, Any]) -> None:
        self._validate_mutable_state(manifest)
        manifest["mutable_state_hash"] = _canonical_hash(
            self._mutable_state_projection(manifest)
        )
        root = self._assert_staging_identity()
        _atomic_json(self.manifest_path, manifest, root)
        self._assert_staging_identity()

    def _new_manifest(self) -> dict[str, Any]:
        if self._staging_identity is None:
            raise RuntimeError("private staging identity is not established")
        manifest = {
            **self.contract_payload,
            "contract_hash": self.contract_hash,
            "completed": {},
            "outer_fold_barriers": {},
            "recovered_atomic_job_checkpoints": [],
            "discarded_atomic_temps": [],
            "mutable_state_hash": "",
            "staging_identity": {
                "physical": list(self._staging_identity),
                "token": self._staging_token,
            },
        }
        manifest["mutable_state_hash"] = _canonical_hash(
            self._mutable_state_projection(manifest)
        )
        return manifest

    def _load_manifest(self, root: Path) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return self._new_manifest()
        manifest = _json_loads_no_duplicates(
            _secure_file_bytes(self.manifest_path, root).decode("utf-8")
        )
        allowed_keys = set(self.contract_payload) | {
            "contract_hash",
            "completed",
            "outer_fold_barriers",
            "recovered_atomic_job_checkpoints",
            "discarded_atomic_temps",
            "mutable_state_hash",
            "staging_identity",
        }
        if set(manifest) != allowed_keys:
            raise ValueError("resume manifest has missing or extra authenticated fields")
        projection = {key: manifest[key] for key in self.contract_payload}
        if (
            projection != self.contract_payload
            or _canonical_hash(projection) != self.contract_hash
            or manifest.get("contract_hash") != self.contract_hash
        ):
            raise ValueError("resume manifest differs from actual current input/source hashes or task contract")
        if manifest.get("staging_identity") != {
            "physical": list(self._staging_identity or ()),
            "token": self._staging_token,
        }:
            raise ValueError(
                "resume manifest staging identity differs from the live private marker/root"
            )
        expected_mutable_hash = _canonical_hash(self._mutable_state_projection(manifest))
        if manifest.get("mutable_state_hash") != expected_mutable_hash:
            raise ValueError("resume manifest mutable-state authentication failed")
        self._validate_mutable_state(manifest)
        return manifest

    @staticmethod
    def _barrier_key(seed: int, outer_fold: int) -> str:
        return f"inner_complete__seed_{seed}__outer_{outer_fold}"

    def _barrier_payload(
        self, seed: int, outer_fold: int, manifest: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        keys = sorted(
            task.key.text
            for task in self.tasks
            if task.key.stage == "inner"
            and task.key.seed == seed
            and task.key.outer_fold == outer_fold
        )
        completed = manifest["completed"]
        if not set(keys).issubset(completed):
            return None
        return {
            "contract_hash": self.contract_hash,
            "seed": seed,
            "outer_fold": outer_fold,
            "inner_task_keys": keys,
            "inner_shard_hashes": {key: completed[key]["sha256"] for key in keys},
        }

    def _validate_outer_fold_barriers(self, manifest: Mapping[str, Any]) -> None:
        barriers = manifest.get("outer_fold_barriers")
        if not isinstance(barriers, dict):
            raise ValueError("outer-fold barrier checkpoint mapping is missing")
        expected_keys = {
            self._barrier_key(seed, fold)
            for seed in self.coverage.seeds
            for fold in range(self.coverage.outer_folds)
        }
        if not set(barriers).issubset(expected_keys):
            raise ValueError("outer-fold barriers contain stale or extra checkpoint keys")
        for key, receipt in barriers.items():
            if not isinstance(receipt, dict) or set(receipt) != {"payload", "sha256"}:
                raise ValueError(f"{key}: malformed outer-fold barrier checkpoint")
            payload = receipt["payload"]
            if not isinstance(payload, dict) or self._barrier_key(
                int(payload.get("seed", -1)), int(payload.get("outer_fold", -1))
            ) != key:
                raise ValueError(f"{key}: outer-fold barrier identity mismatch")
            actual = self._barrier_payload(
                int(payload["seed"]), int(payload["outer_fold"]), manifest
            )
            if actual is None or payload != actual or receipt["sha256"] != _canonical_hash(actual):
                raise ValueError(f"{key}: stale or corrupt outer-fold barrier checkpoint")
        for task_key in manifest["completed"]:
            task = next(task for task in self.tasks if task.key.text == task_key)
            if task.key.stage == "outer_refit":
                barrier_key = self._barrier_key(task.key.seed, task.key.outer_fold)
                if barrier_key not in barriers:
                    raise ValueError(f"{task_key}: outer checkpoint exists without its inner barrier")

    def _checkpoint_ready_outer_fold_barriers(
        self, manifest: dict[str, Any], root: Path
    ) -> None:
        barriers = manifest["outer_fold_barriers"]
        for seed in self.coverage.seeds:
            for outer_fold in range(self.coverage.outer_folds):
                key = self._barrier_key(seed, outer_fold)
                payload = self._barrier_payload(seed, outer_fold, manifest)
                if payload is not None and key not in barriers:
                    barriers[key] = {"payload": payload, "sha256": _canonical_hash(payload)}
                    self._commit_manifest(manifest)

    def _shard_path(self, task: ParallelTask) -> Path:
        return self.shard_root / f"{task.key.text}.npz"

    def _serialise_shard(self, task: ParallelTask, result: JobResult) -> bytes:
        result = _validate_result(task, result)
        record = {
            "task_key": task.key.text,
            "stage": task.key.stage,
            "alias": task.key.alias,
            "seed": task.key.seed,
            "outer_fold": task.key.outer_fold,
            "inner_fold": task.key.inner_fold,
            "resource": task.resource,
            "task_rng_seed": _task_rng_seed(task),
            **dict(result.records),
        }
        metadata = {
            "schema_version": PARALLEL_SCHEMA_VERSION,
            "contract_hash": self.contract_hash,
            "task_contract": task.contract_payload(),
            "record": record,
            "record_hash": _canonical_hash(record),
            "array_schema": {name: dict(spec) for name, spec in task.array_schema.items()},
            "array_hashes": {name: _array_hash(array) for name, array in result.arrays.items()},
        }
        return _deterministic_npz(
            {
                "__metadata__": np.frombuffer(_canonical_bytes(metadata), dtype=np.uint8),
                **{f"array__{name}": array for name, array in result.arrays.items()},
            }
        )

    def _write_shard(self, task: ParallelTask, result: JobResult, shard_root: Path) -> str:
        self._assert_staging_identity()
        path = self._shard_path(task)
        payload = self._serialise_shard(task, result)
        _atomic_bytes(path, payload, shard_root)
        self._assert_staging_identity()
        return _hash_bytes(payload)

    def _read_shard(self, task: ParallelTask, trusted_hash: str, shard_root: Path) -> JobResult:
        self._assert_staging_identity()
        payload = _secure_file_bytes(self._shard_path(task), shard_root)
        if _hash_bytes(payload) != trusted_hash:
            raise ValueError(f"{task.key.text}: trusted shard hash mismatch")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if "__metadata__" not in archive.files:
                raise ValueError(f"{task.key.text}: shard metadata missing")
            metadata = _json_loads_no_duplicates(
                archive["__metadata__"].tobytes().decode("utf-8")
            )
            record = metadata.get("record")
            expected_coordinator_metadata = {
                "task_key": task.key.text,
                "stage": task.key.stage,
                "alias": task.key.alias,
                "seed": task.key.seed,
                "outer_fold": task.key.outer_fold,
                "inner_fold": task.key.inner_fold,
                "resource": task.resource,
                "task_rng_seed": _task_rng_seed(task),
            }
            if (
                metadata.get("schema_version") != PARALLEL_SCHEMA_VERSION
                or metadata.get("contract_hash") != self.contract_hash
                or metadata.get("task_contract") != task.contract_payload()
                or not isinstance(record, dict)
                or any(
                    record.get(name) != expected
                    for name, expected in expected_coordinator_metadata.items()
                )
                or metadata.get("record_hash") != _canonical_hash(record)
                or metadata.get("array_schema")
                != {name: dict(spec) for name, spec in task.array_schema.items()}
            ):
                raise ValueError(f"{task.key.text}: shard metadata/identity mismatch")
            array_hashes = metadata.get("array_hashes")
            expected_files = {"__metadata__", *(f"array__{name}" for name in task.array_schema)}
            if not isinstance(array_hashes, dict) or set(archive.files) != expected_files:
                raise ValueError(f"{task.key.text}: shard array coverage mismatch")
            arrays = {name: np.array(archive[f"array__{name}"], copy=True) for name in task.array_schema}
        runner_records = {key: value for key, value in record.items() if key not in RESERVED_RECORD_KEYS}
        result = _validate_result(task, JobResult(runner_records, arrays))
        if any(_array_hash(result.arrays[name]) != array_hashes.get(name) for name in task.array_schema):
            raise ValueError(f"{task.key.text}: shard array hash mismatch")
        self._assert_staging_identity()
        return result

    def _recover_atomic_job_checkpoints(
        self, manifest: dict[str, Any], shard_root: Path, root: Path
    ) -> None:
        expected_paths = {self._shard_path(task).name for task in self.tasks}
        observed_paths = {path.name for path in shard_root.iterdir()}
        if not observed_paths.issubset(expected_paths):
            raise ValueError("staging contains stale or extra job checkpoint shards")
        completed = set(manifest["completed"])
        for task in self.tasks:
            path = self._shard_path(task)
            if task.key.text not in completed and (path.exists() or path.is_symlink()):
                payload = _secure_file_bytes(path, shard_root)
                digest = _hash_bytes(payload)
                self._read_shard(task, digest, shard_root)
                manifest["completed"][task.key.text] = {
                    "sha256": digest,
                    "phase": task.key.stage,
                }
                self._recovered_this_run.append(task.key.text)
                self._commit_manifest(manifest)

    def _shutdown_workers(
        self, processes: Mapping[str, Any], queues: Mapping[str, Any], *, graceful: bool
    ) -> dict[str, int]:
        if graceful:
            for resource, process in processes.items():
                if process.is_alive():
                    queues[resource].put(None)
        deadline = time.monotonic() + self.cleanup_timeout_seconds
        for process in processes.values():
            process.join(max(0.0, deadline - time.monotonic()))
        for process in processes.values():
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + self.cleanup_timeout_seconds
        for process in processes.values():
            process.join(max(0.0, deadline - time.monotonic()))
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(self.cleanup_timeout_seconds)
        for work_queue in queues.values():
            work_queue.cancel_join_thread()
            work_queue.close()
        survivors = tuple(
            process.pid for process in processes.values() if process.is_alive() and process.pid
        )
        self.last_worker_survivors = survivors
        if survivors:
            raise RuntimeError(f"spawn workers survived bounded cleanup: {survivors}")
        exitcodes = {
            resource: int(process.exitcode if process.exitcode is not None else -999)
            for resource, process in processes.items()
        }
        for process in processes.values():
            process.close()
        return exitcodes

    def _execute_phase(
        self,
        phase: str,
        phase_tasks: list[ParallelTask],
        manifest: dict[str, Any],
        results: dict[str, JobResult],
        processes: Mapping[str, Any],
        task_queues: Mapping[str, Any],
        result_queue: Any,
        shard_root: Path,
        cancel_event: threading.Event | None,
        executed: list[str],
        maximum: dict[str, int],
    ) -> None:
        pending = {
            resource: [
                task
                for task in phase_tasks
                if task.resource == resource and task.key.text not in results
            ]
            for resource in RESOURCES
        }
        active: dict[str, tuple[ParallelTask, float] | None] = {resource: None for resource in RESOURCES}

        def dispatch(resource: str) -> None:
            if active[resource] is None and pending[resource]:
                task = pending[resource].pop(0)
                active[resource] = (task, time.monotonic())
                maximum[resource] = max(maximum[resource], 1)
                task_queues[resource].put(task)

        for resource in RESOURCES:
            dispatch(resource)
        while any(value is not None for value in active.values()):
            if cancel_event is not None and cancel_event.is_set():
                raise ExecutionCancelled(f"{phase} phase cancelled")
            now = time.monotonic()
            for resource, current in active.items():
                if current and now - current[1] > self.task_timeout_seconds:
                    raise WorkerTimeoutError(f"{current[0].key.text}: worker timeout")
                if current and not processes[resource].is_alive():
                    raise WorkerExecutionError(f"{current[0].key.text}: worker process exited unexpectedly")
            try:
                kind, resource, key, payload = result_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if kind == "bootstrap_error":
                raise WorkerExecutionError(f"{resource} worker bootstrap failed\n{payload}")
            current = active.get(resource)
            if current is None or current[0].key.text != key:
                raise WorkerExecutionError("worker returned an unexpected or duplicate task identity")
            if kind == "started":
                active[resource] = (current[0], time.monotonic())
                continue
            if kind == "error":
                raise WorkerExecutionError(f"{key}: worker failed\n{payload}")
            if kind != "result":
                raise WorkerExecutionError(f"{key}: invalid worker message {kind}")
            task = current[0]
            result = _validate_result(task, payload)
            digest = self._write_shard(task, result, shard_root)
            manifest["completed"][key] = {"sha256": digest, "phase": phase}
            self._commit_manifest(manifest)
            if phase == "inner":
                self._checkpoint_ready_outer_fold_barriers(manifest, self.staging_root)
            results[key] = result
            executed.append(key)
            active[resource] = None
            dispatch(resource)

    def run(self, *, cancel_event: threading.Event | None = None) -> ExecutionReport:
        self._recovered_this_run: list[str] = []
        self._discarded_temps_this_run: list[dict[str, str]] = []
        root = self._prepare_private_staging()
        with _RunLock(root):
            self._assert_staging_identity()
            shard_root = _ensure_secure_directory(self.shard_root, create=True)
            discarded_temps = [
                *(
                    {"parent": "staging", "name": name}
                    for name in _discard_secure_temps(root)
                ),
                *(
                    {"parent": "shards", "name": name}
                    for name in _discard_secure_temps(shard_root)
                ),
            ]
            self._discarded_temps_this_run.extend(discarded_temps)
            if self._compute_actual_hashes() != self.initial_hashes:
                raise ValueError("actual trusted inputs or source modules drifted before execution")
            manifest = self._load_manifest(root)
            self._recover_atomic_job_checkpoints(manifest, shard_root, root)
            self._validate_outer_fold_barriers(manifest)
            self._commit_manifest(manifest)
            task_by_key = {task.key.text: task for task in self.tasks}
            results: dict[str, JobResult] = {}
            resumed: list[str] = []
            for key, receipt in manifest["completed"].items():
                if not isinstance(receipt, dict) or not isinstance(receipt.get("sha256"), str):
                    raise ValueError(f"{key}: malformed committed shard receipt")
                results[key] = self._read_shard(task_by_key[key], receipt["sha256"], shard_root)
                resumed.append(key)
            self._checkpoint_ready_outer_fold_barriers(manifest, root)

            context = multiprocessing.get_context("spawn")
            result_queue = context.Queue()
            task_queues = {resource: context.Queue() for resource in RESOURCES}
            processes = {
                resource: context.Process(
                    target=_resource_worker,
                    args=(
                        resource,
                        task_queues[resource],
                        result_queue,
                        self.job_runner_spec,
                        self.guard_factory_spec,
                        self.unsafe_synthetic_no_guard,
                        (
                            None
                            if self.unsafe_synthetic_no_guard
                            else (
                                str(self.protected_test_path.resolve(strict=True)),
                                str(self.protected_submission_path.resolve(strict=True)),
                            )
                        ),
                        tuple(map(str, self.approved_code_roots)),
                        dict(self.initial_hashes["sources"]),
                    ),
                    name=f"nested-shadow-{resource}",
                )
                for resource in RESOURCES
            }
            for process in processes.values():
                process.start()
            self.last_worker_pids = {
                resource: int(process.pid) for resource, process in processes.items()
            }
            executed: list[str] = []
            maximum = {resource: 0 for resource in RESOURCES}
            graceful = False
            exitcodes: dict[str, int] = {}
            try:
                inner = [task for task in self.tasks if task.key.stage == "inner"]
                outer = [task for task in self.tasks if task.key.stage == "outer_refit"]
                self._execute_phase(
                    "inner", inner, manifest, results, processes, task_queues, result_queue,
                    shard_root, cancel_event, executed, maximum,
                )
                expected_inner = {task.key.text for task in inner}
                if not expected_inner.issubset(manifest["completed"]):
                    raise RuntimeError("inner phase barrier reached without exact committed coverage")
                expected_barriers = {
                    self._barrier_key(seed, fold)
                    for seed in self.coverage.seeds
                    for fold in range(self.coverage.outer_folds)
                }
                if set(manifest["outer_fold_barriers"]) != expected_barriers:
                    raise RuntimeError("outer-fold barrier checkpoint coverage is incomplete")
                self._execute_phase(
                    "outer_refit", outer, manifest, results, processes, task_queues, result_queue,
                    shard_root, cancel_event, executed, maximum,
                )
                graceful = True
            finally:
                exitcodes = self._shutdown_workers(processes, task_queues, graceful=graceful)
                result_queue.cancel_join_thread()
                result_queue.close()
            if self._compute_actual_hashes() != self.initial_hashes:
                raise ValueError("actual trusted inputs or source modules drifted during execution")
            merged = merge_results(self.tasks, results, self.alias_order)
            return ExecutionReport(
                merged=merged,
                executed_keys=tuple(executed),
                resumed_keys=tuple(resumed),
                manifest_path=self.manifest_path,
                max_concurrency=maximum,
                worker_pids=dict(self.last_worker_pids),
                worker_exitcodes=exitcodes,
                recovered_checkpoint_keys=tuple(self._recovered_this_run),
                discarded_temp_entries=tuple(self._discarded_temps_this_run),
            )


def publish_shadow_npz(
    merged: MergedResult,
    shadow_output_root: Path,
    production_artifact_root: Path,
    output_name: str,
) -> Path:
    """Publish a clearly named shadow diagnostic, never a production artifact."""

    stem = output_name[:-4] if output_name.endswith(".npz") else ""
    if (
        not output_name.startswith("shadow_")
        or not output_name.endswith(".npz")
        or Path(output_name).name != output_name
        or not STRICT_SLUG.fullmatch(stem)
        or stem.casefold() in WINDOWS_RESERVED_NAMES
    ):
        raise ValueError("shadow output must be a plain shadow_*.npz filename")
    output_root = Path(shadow_output_root).resolve(strict=False)
    production_root = Path(production_artifact_root).resolve(strict=False)
    if _path_contains(output_root, production_root) or _path_contains(production_root, output_root):
        raise ValueError("shadow output and production artifact roots must be disjoint")
    if tuple(record.get("task_key") for record in merged.records) != merged.task_order:
        raise ValueError("record task keys do not equal immutable task order")
    if set(merged.task_metadata) != set(merged.task_order):
        raise ValueError("task metadata does not exactly cover immutable task order")
    for record, key in zip(merged.records, merged.task_order):
        expected_metadata = merged.task_metadata[key]
        if set(expected_metadata) != RESERVED_RECORD_KEYS or any(
            record.get(name) != expected for name, expected in expected_metadata.items()
        ):
            raise ValueError("coordinator-owned record metadata was changed before publish")
    if set(merged.arrays) != set(merged.array_schema):
        raise ValueError("shadow merged arrays have missing or extra schema keys")
    for name, specification in merged.array_schema.items():
        array = np.asarray(merged.arrays[name])
        if list(array.shape) != specification["shape"] or str(array.dtype) != specification["dtype"]:
            raise ValueError(f"{name}: merged array violates published schema")
    root = _ensure_secure_directory(output_root, create=True)
    _assert_physical_output_separation(root, production_root)
    metadata = {
        "schema_version": PARALLEL_SCHEMA_VERSION,
        "mode": SCHEDULER_MODE,
        "production_nested_adapter": False,
        "production_publish_allowed": False,
        "limitations": list(LIMITATIONS),
        "task_order": list(merged.task_order),
        "records": list(merged.records),
        "task_metadata": {
            key: dict(value) for key, value in merged.task_metadata.items()
        },
        "array_schema": {name: dict(spec) for name, spec in merged.array_schema.items()},
        "array_hashes": {name: _array_hash(array) for name, array in merged.arrays.items()},
    }
    payload = _deterministic_npz(
        {
            "__metadata__": np.frombuffer(_canonical_bytes(metadata), dtype=np.uint8),
            **{f"array__{name}": array for name, array in merged.arrays.items()},
        }
    )
    path = root / output_name
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    _atomic_bytes(path, payload, root)
    return path


def _load_plan(path: Path) -> tuple[
    tuple[ParallelTask, ...], PhaseCoverage, tuple[str, ...], dict[str, Path]
]:
    value = _json_loads_no_duplicates(path.read_text(encoding="utf-8"))
    if value.get("mode") != SCHEDULER_MODE:
        raise ValueError(f"plan mode must explicitly be {SCHEDULER_MODE}")
    coverage_raw = value["coverage"]
    coverage = PhaseCoverage(
        tuple(map(int, coverage_raw["seeds"])),
        int(coverage_raw["outer_folds"]),
        int(coverage_raw["inner_folds"]),
        tuple(map(str, coverage_raw["inner_aliases"])),
        tuple(map(str, coverage_raw["outer_aliases"])),
    )
    tasks = tuple(
        ParallelTask(
            TaskKey.from_payload(item["key"]),
            str(item["resource"]),
            dict(item["array_schema"]),
            dict(item.get("payload", {})),
        )
        for item in value["tasks"]
    )
    return (
        tasks,
        coverage,
        tuple(map(str, value["alias_order"])),
        {str(role): Path(path_value) for role, path_value in value["trusted_input_paths"].items()},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-plan", type=Path, required=True)
    parser.add_argument("--job-runner", required=True, help="importable.module:callable")
    parser.add_argument("--guard-factory", required=True, help="importable.module:callable")
    parser.add_argument(
        "--approved-code-root", type=Path, action="append", required=True
    )
    parser.add_argument("--protected-test-path", type=Path, required=True)
    parser.add_argument("--protected-submission-path", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--production-artifact-root", type=Path, required=True)
    parser.add_argument("--task-timeout-seconds", type=float, default=3600.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tasks, coverage, aliases, trusted_paths = _load_plan(args.shadow_plan)
    report = ParallelCoordinator(
        tasks=tasks,
        coverage=coverage,
        alias_order=aliases,
        trusted_input_paths=trusted_paths,
        staging_root=args.staging_root,
        production_artifact_root=args.production_artifact_root,
        job_runner_spec=args.job_runner,
        guard_factory_spec=args.guard_factory,
        approved_code_roots=args.approved_code_root,
        protected_test_path=args.protected_test_path,
        protected_submission_path=args.protected_submission_path,
        task_timeout_seconds=args.task_timeout_seconds,
    ).run()
    print(
        json.dumps(
            {
                "mode": SCHEDULER_MODE,
                "production_nested_adapter": False,
                "production_publish_allowed": False,
                "limitations": list(LIMITATIONS),
                "executed": len(report.executed_keys),
                "resumed": len(report.resumed_keys),
                "manifest": str(report.manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
