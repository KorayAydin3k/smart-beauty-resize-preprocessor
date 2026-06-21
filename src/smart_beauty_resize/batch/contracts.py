from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from smart_beauty_resize.contracts import (
    BatchConfigurationError,
    ResizeConfig,
)


def _absolute_without_symlink_resolution(path: Path) -> Path:
    """Return a normalized absolute path without resolving symbolic links."""
    return Path(os.path.abspath(os.fspath(path)))


def _is_inside(child: Path, parent: Path) -> bool:
    """Return whether child is lexically located inside parent."""
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_non_empty_string(name: str, value: object) -> str:
    if type(value) is not str or not value.strip():
        raise BatchConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_relative_path(name: str, value: object) -> Path:
    if not isinstance(value, Path):
        raise BatchConfigurationError(f"{name} must be a pathlib.Path instance")

    if value.is_absolute():
        raise BatchConfigurationError(f"{name} must be relative")

    if value == Path(".") or not value.parts:
        raise BatchConfigurationError(f"{name} must identify a relative file path")

    if ".." in value.parts:
        raise BatchConfigurationError(f"{name} must not contain parent-directory traversal")

    return value


def _validate_sha256(name: str, value: object) -> str:
    if type(value) is not str:
        raise BatchConfigurationError(f"{name} must be a lowercase SHA-256 hexadecimal string")

    if len(value) != 64:
        raise BatchConfigurationError(f"{name} must contain exactly 64 characters")

    if value != value.lower():
        raise BatchConfigurationError(f"{name} must use lowercase hexadecimal characters")

    if any(character not in "0123456789abcdef" for character in value):
        raise BatchConfigurationError(f"{name} must contain only hexadecimal characters")

    return value


def _validate_optional_sha256(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _validate_sha256(name, value)


def _validate_positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise BatchConfigurationError(f"{name} must be a positive integer")
    return value


def _validate_non_negative_integer(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise BatchConfigurationError(f"{name} must be a non-negative integer")
    return value


def _validate_optional_positive_integer(
    name: str,
    value: object,
) -> int | None:
    if value is None:
        return None
    return _validate_positive_integer(name, value)


def _validate_optional_non_negative_integer(
    name: str,
    value: object,
) -> int | None:
    if value is None:
        return None
    return _validate_non_negative_integer(name, value)


def _validate_processing_time(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise BatchConfigurationError(
            "processing_time_ms must be a finite non-negative real number"
        )

    return float(value)


def _validate_utc_datetime(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise BatchConfigurationError(f"{name} must be a datetime")

    if value.tzinfo is None or value.utcoffset() is None:
        raise BatchConfigurationError(f"{name} must be timezone-aware")

    if value.utcoffset() != timedelta(0):
        raise BatchConfigurationError(f"{name} must use UTC")

    return value.astimezone(UTC)


class ProcessingStatus(StrEnum):
    """Possible outcomes for processing one input image."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class BatchConfig:
    """Immutable configuration for a batch preprocessing run.

    Input and output paths are stored as normalized absolute paths. Symbolic
    links are intentionally not resolved at construction time, allowing the
    configuration to be created before the referenced directories exist.
    """

    input_dir: Path
    output_dir: Path
    resize_config: ResizeConfig
    output_format: str = "png"
    overwrite: bool = False
    fail_fast: bool = False
    preserve_directory_structure: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.input_dir, Path):
            raise BatchConfigurationError("input_dir must be a pathlib.Path instance")

        if not isinstance(self.output_dir, Path):
            raise BatchConfigurationError("output_dir must be a pathlib.Path instance")

        if not isinstance(self.resize_config, ResizeConfig):
            raise BatchConfigurationError("resize_config must be a ResizeConfig instance")

        if type(self.output_format) is not str:
            raise BatchConfigurationError("output_format must be a string")

        normalized_format = self.output_format.strip().lower()
        if normalized_format != "png":
            raise BatchConfigurationError("output_format currently supports only 'png'")

        boolean_fields = (
            ("overwrite", self.overwrite),
            ("fail_fast", self.fail_fast),
            (
                "preserve_directory_structure",
                self.preserve_directory_structure,
            ),
        )

        for field_name, field_value in boolean_fields:
            if type(field_value) is not bool:
                raise BatchConfigurationError(f"{field_name} must be a boolean")

        normalized_input = _absolute_without_symlink_resolution(self.input_dir)
        normalized_output = _absolute_without_symlink_resolution(self.output_dir)

        if normalized_input == normalized_output:
            raise BatchConfigurationError("input_dir and output_dir must be different")

        if _is_inside(normalized_output, normalized_input):
            raise BatchConfigurationError("output_dir must not be located inside input_dir")

        object.__setattr__(self, "input_dir", normalized_input)
        object.__setattr__(self, "output_dir", normalized_output)
        object.__setattr__(self, "output_format", normalized_format)


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageProcessingRecord:
    """Immutable manifest record for one discovered input image."""

    schema_version: str
    source_relative_path: Path
    output_relative_path: Path | None
    status: ProcessingStatus
    source_sha256: str | None
    output_sha256: str | None
    config_sha256: str
    original_width: int | None
    original_height: int | None
    resized_width: int | None
    resized_height: int | None
    target_width: int
    target_height: int
    pad_left: int | None
    pad_top: int | None
    pad_right: int | None
    pad_bottom: int | None
    interpolation: str | None
    processing_time_ms: float
    error_type: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        schema_version = _validate_non_empty_string(
            "schema_version",
            self.schema_version,
        )
        source_relative_path = _validate_relative_path(
            "source_relative_path",
            self.source_relative_path,
        )

        output_relative_path: Path | None = None
        if self.output_relative_path is not None:
            output_relative_path = _validate_relative_path(
                "output_relative_path",
                self.output_relative_path,
            )

        if not isinstance(self.status, ProcessingStatus):
            raise BatchConfigurationError("status must be a ProcessingStatus")

        source_sha256 = _validate_optional_sha256(
            "source_sha256",
            self.source_sha256,
        )
        output_sha256 = _validate_optional_sha256(
            "output_sha256",
            self.output_sha256,
        )
        config_sha256 = _validate_sha256(
            "config_sha256",
            self.config_sha256,
        )

        original_width = _validate_optional_positive_integer(
            "original_width",
            self.original_width,
        )
        original_height = _validate_optional_positive_integer(
            "original_height",
            self.original_height,
        )
        resized_width = _validate_optional_positive_integer(
            "resized_width",
            self.resized_width,
        )
        resized_height = _validate_optional_positive_integer(
            "resized_height",
            self.resized_height,
        )

        target_width = _validate_positive_integer(
            "target_width",
            self.target_width,
        )
        target_height = _validate_positive_integer(
            "target_height",
            self.target_height,
        )

        pad_left = _validate_optional_non_negative_integer(
            "pad_left",
            self.pad_left,
        )
        pad_top = _validate_optional_non_negative_integer(
            "pad_top",
            self.pad_top,
        )
        pad_right = _validate_optional_non_negative_integer(
            "pad_right",
            self.pad_right,
        )
        pad_bottom = _validate_optional_non_negative_integer(
            "pad_bottom",
            self.pad_bottom,
        )

        processing_time_ms = _validate_processing_time(self.processing_time_ms)

        interpolation: str | None = None
        if self.interpolation is not None:
            interpolation = _validate_non_empty_string(
                "interpolation",
                self.interpolation,
            )

        error_type: str | None = None
        if self.error_type is not None:
            error_type = _validate_non_empty_string(
                "error_type",
                self.error_type,
            )

        error_message: str | None = None
        if self.error_message is not None:
            error_message = _validate_non_empty_string(
                "error_message",
                self.error_message,
            )

        if self.status is ProcessingStatus.SUCCESS:
            required_success_values = (
                output_relative_path,
                source_sha256,
                output_sha256,
                original_width,
                original_height,
                resized_width,
                resized_height,
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
                interpolation,
            )

            if any(value is None for value in required_success_values):
                raise BatchConfigurationError(
                    "successful records must include output, hashes, geometry, and interpolation"
                )

            if error_type is not None or error_message is not None:
                raise BatchConfigurationError("successful records must not contain error fields")

            if (
                resized_width is None
                or resized_height is None
                or pad_left is None
                or pad_top is None
                or pad_right is None
                or pad_bottom is None
            ):
                raise BatchConfigurationError("successful records require complete geometry")

            if resized_width + pad_left + pad_right != target_width:
                raise BatchConfigurationError(
                    "successful record horizontal geometry must equal target_width"
                )

            if resized_height + pad_top + pad_bottom != target_height:
                raise BatchConfigurationError(
                    "successful record vertical geometry must equal target_height"
                )

        else:
            if error_type is None or error_message is None:
                raise BatchConfigurationError(
                    "failed and skipped records must contain error_type and error_message"
                )

            if output_sha256 is not None:
                raise BatchConfigurationError(
                    "failed and skipped records must not contain an output_sha256"
                )

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(
            self,
            "source_relative_path",
            source_relative_path,
        )
        object.__setattr__(
            self,
            "output_relative_path",
            output_relative_path,
        )
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "output_sha256", output_sha256)
        object.__setattr__(self, "config_sha256", config_sha256)
        object.__setattr__(self, "original_width", original_width)
        object.__setattr__(self, "original_height", original_height)
        object.__setattr__(self, "resized_width", resized_width)
        object.__setattr__(self, "resized_height", resized_height)
        object.__setattr__(self, "target_width", target_width)
        object.__setattr__(self, "target_height", target_height)
        object.__setattr__(self, "pad_left", pad_left)
        object.__setattr__(self, "pad_top", pad_top)
        object.__setattr__(self, "pad_right", pad_right)
        object.__setattr__(self, "pad_bottom", pad_bottom)
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(
            self,
            "processing_time_ms",
            processing_time_ms,
        )
        object.__setattr__(self, "error_type", error_type)
        object.__setattr__(self, "error_message", error_message)


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchRunSummary:
    """Immutable summary for one complete batch-processing run."""

    schema_version: str
    run_id: str
    started_at_utc: datetime
    finished_at_utc: datetime
    total_discovered: int
    successful: int
    failed: int
    skipped: int
    target_width: int
    target_height: int
    config_sha256: str

    def __post_init__(self) -> None:
        schema_version = _validate_non_empty_string(
            "schema_version",
            self.schema_version,
        )
        run_id = _validate_non_empty_string(
            "run_id",
            self.run_id,
        )

        started_at_utc = _validate_utc_datetime(
            "started_at_utc",
            self.started_at_utc,
        )
        finished_at_utc = _validate_utc_datetime(
            "finished_at_utc",
            self.finished_at_utc,
        )

        total_discovered = _validate_non_negative_integer(
            "total_discovered",
            self.total_discovered,
        )
        successful = _validate_non_negative_integer(
            "successful",
            self.successful,
        )
        failed = _validate_non_negative_integer(
            "failed",
            self.failed,
        )
        skipped = _validate_non_negative_integer(
            "skipped",
            self.skipped,
        )

        target_width = _validate_positive_integer(
            "target_width",
            self.target_width,
        )
        target_height = _validate_positive_integer(
            "target_height",
            self.target_height,
        )
        config_sha256 = _validate_sha256(
            "config_sha256",
            self.config_sha256,
        )

        if successful + failed + skipped != total_discovered:
            raise BatchConfigurationError(
                "successful + failed + skipped must equal total_discovered"
            )

        if finished_at_utc < started_at_utc:
            raise BatchConfigurationError("finished_at_utc must not be before started_at_utc")

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(
            self,
            "started_at_utc",
            started_at_utc,
        )
        object.__setattr__(
            self,
            "finished_at_utc",
            finished_at_utc,
        )
        object.__setattr__(
            self,
            "total_discovered",
            total_discovered,
        )
        object.__setattr__(self, "successful", successful)
        object.__setattr__(self, "failed", failed)
        object.__setattr__(self, "skipped", skipped)
        object.__setattr__(self, "target_width", target_width)
        object.__setattr__(self, "target_height", target_height)
        object.__setattr__(self, "config_sha256", config_sha256)

    @property
    def success_rate(self) -> float:
        """Return the successful-image percentage for the run."""
        if self.total_discovered == 0:
            return 0.0
        return (self.successful / self.total_discovered) * 100.0
