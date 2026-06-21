from __future__ import annotations

import os
import re
import shutil
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from smart_beauty_resize.batch.contracts import BatchExecutionResult
from smart_beauty_resize.contracts import (
    ManifestSerializationError,
    ManifestWriteError,
)
from smart_beauty_resize.provenance.manifest import (
    record_to_json_line,
    summary_to_json,
)

RUNS_DIRECTORY_NAME = "_runs"
MANIFEST_FILENAME = "manifest.jsonl"
SUMMARY_FILENAME = "run_summary.json"

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class BatchArtifactPaths:
    """Absolute paths for one persisted batch execution."""

    run_directory: Path
    manifest_path: Path
    summary_path: Path


def _absolute_without_symlink_resolution(path: Path) -> Path:
    """Return a normalized absolute path without resolving symlinks."""
    return Path(os.path.abspath(os.fspath(path)))


def _validate_run_id(run_id: object) -> str:
    """Validate a run ID that will become one directory component."""
    if type(run_id) is not str or not run_id:
        raise ManifestWriteError("run_id must be a non-empty string")

    if _RUN_ID_PATTERN.fullmatch(run_id) is None or run_id in {".", ".."} or ".." in run_id:
        raise ManifestWriteError(
            "run_id must be a filesystem-safe identifier containing only "
            "letters, digits, dots, underscores, and hyphens"
        )

    return run_id


def _prepare_directory(path: Path, *, name: str) -> Path:
    normalized = _absolute_without_symlink_resolution(path)

    if normalized.is_symlink():
        raise ManifestWriteError(f"{name} must not be a symbolic link")

    if normalized.exists() and not normalized.is_dir():
        raise ManifestWriteError(f"{name} exists but is not a directory")

    try:
        normalized.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise ManifestWriteError(f"unable to create {name}: {normalized}") from exc

    if normalized.is_symlink() or not normalized.is_dir():
        raise ManifestWriteError(f"{name} is not a safe directory")

    return normalized


def _acquire_run_lock(
    runs_root: Path,
    run_id: str,
) -> Path:
    """Create an exclusive lock file for one run ID."""
    lock_path = runs_root / f".{run_id}.lock"

    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise ManifestWriteError(
            f"a manifest writer is already active for run_id: {run_id}"
        ) from exc
    except OSError as exc:
        raise ManifestWriteError(f"unable to acquire manifest lock for run_id: {run_id}") from exc

    try:
        os.write(
            descriptor,
            f"pid={os.getpid()}\n".encode("ascii"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    return lock_path


def _write_manifest_file(
    path: Path,
    result: BatchExecutionResult,
) -> None:
    try:
        with path.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as file_handle:
            for record in result.records:
                file_handle.write(record_to_json_line(record))

            file_handle.flush()
            os.fsync(file_handle.fileno())

    except ManifestSerializationError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ManifestWriteError(f"unable to write manifest file: {path}") from exc


def _write_summary_file(
    path: Path,
    result: BatchExecutionResult,
) -> None:
    try:
        content = summary_to_json(result.summary) + "\n"

        with path.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as file_handle:
            file_handle.write(content)
            file_handle.flush()
            os.fsync(file_handle.fileno())

    except ManifestSerializationError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ManifestWriteError(f"unable to write summary file: {path}") from exc


def write_batch_artifacts(
    result: BatchExecutionResult,
    output_root: Path,
) -> BatchArtifactPaths:
    """Persist one batch execution as an atomic audit bundle.

    Files are fully written and flushed inside a temporary directory located
    under ``output_root/_runs``. The complete directory is then renamed into
    its final run-ID location.

    Existing run directories are never overwritten.
    """
    if not isinstance(result, BatchExecutionResult):
        raise ManifestWriteError("result must be a BatchExecutionResult")

    if not isinstance(output_root, Path):
        raise ManifestWriteError("output_root must be a pathlib.Path instance")

    run_id = _validate_run_id(result.summary.run_id)

    normalized_output_root = _prepare_directory(
        output_root,
        name="output_root",
    )

    runs_root = _prepare_directory(
        normalized_output_root / RUNS_DIRECTORY_NAME,
        name="runs directory",
    )

    lock_path = _acquire_run_lock(
        runs_root,
        run_id,
    )

    final_run_directory = runs_root / run_id
    staging_directory: Path | None = None

    try:
        if final_run_directory.exists() or final_run_directory.is_symlink():
            raise ManifestWriteError(f"manifest artifacts already exist for run_id: {run_id}")

        staging_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{run_id}.staging-",
                dir=runs_root,
            )
        )

        manifest_path = staging_directory / MANIFEST_FILENAME
        summary_path = staging_directory / SUMMARY_FILENAME

        _write_manifest_file(
            manifest_path,
            result,
        )
        _write_summary_file(
            summary_path,
            result,
        )

        if not manifest_path.is_file():
            raise ManifestWriteError("staged manifest file does not exist")

        if not summary_path.is_file():
            raise ManifestWriteError("staged summary file does not exist")

        if summary_path.stat().st_size <= 0:
            raise ManifestWriteError("staged summary file is empty")

        try:
            os.rename(
                staging_directory,
                final_run_directory,
            )
        except OSError as exc:
            raise ManifestWriteError("unable to publish manifest bundle atomically") from exc

        staging_directory = None

    except (
        ManifestSerializationError,
        ManifestWriteError,
    ):
        if staging_directory is not None:
            shutil.rmtree(
                staging_directory,
                ignore_errors=True,
            )
        raise

    except OSError as exc:
        if staging_directory is not None:
            shutil.rmtree(
                staging_directory,
                ignore_errors=True,
            )

        raise ManifestWriteError("unable to stage manifest bundle") from exc

    finally:
        with suppress(OSError):
            lock_path.unlink(missing_ok=True)

    final_manifest_path = final_run_directory / MANIFEST_FILENAME
    final_summary_path = final_run_directory / SUMMARY_FILENAME

    if not final_manifest_path.is_file() or not final_summary_path.is_file():
        raise ManifestWriteError("published manifest bundle is incomplete")

    return BatchArtifactPaths(
        run_directory=final_run_directory,
        manifest_path=final_manifest_path,
        summary_path=final_summary_path,
    )
