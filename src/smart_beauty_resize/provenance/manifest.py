from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from smart_beauty_resize.batch.contracts import (
    BatchRunSummary,
    ImageProcessingRecord,
)
from smart_beauty_resize.contracts import ManifestSerializationError


def _datetime_to_utc_iso(value: datetime) -> str:
    """Serialize an already validated UTC datetime using a trailing Z."""
    return value.isoformat().replace("+00:00", "Z")


def _path_to_posix(value: Path | None) -> str | None:
    if value is None:
        return None
    return value.as_posix()


def record_to_dict(
    record: ImageProcessingRecord,
) -> dict[str, object]:
    """Convert one image record into a JSON-safe dictionary."""
    if not isinstance(record, ImageProcessingRecord):
        raise ManifestSerializationError("record must be an ImageProcessingRecord")

    return {
        "config_sha256": record.config_sha256,
        "error_message": record.error_message,
        "error_type": record.error_type,
        "interpolation": record.interpolation,
        "original_height": record.original_height,
        "original_width": record.original_width,
        "output_relative_path": _path_to_posix(record.output_relative_path),
        "output_sha256": record.output_sha256,
        "pad_bottom": record.pad_bottom,
        "pad_left": record.pad_left,
        "pad_right": record.pad_right,
        "pad_top": record.pad_top,
        "processing_time_ms": record.processing_time_ms,
        "resized_height": record.resized_height,
        "resized_width": record.resized_width,
        "schema_version": record.schema_version,
        "source_relative_path": record.source_relative_path.as_posix(),
        "source_sha256": record.source_sha256,
        "status": record.status.value,
        "target_height": record.target_height,
        "target_width": record.target_width,
    }


def summary_to_dict(
    summary: BatchRunSummary,
) -> dict[str, object]:
    """Convert a batch summary into a JSON-safe dictionary."""
    if not isinstance(summary, BatchRunSummary):
        raise ManifestSerializationError("summary must be a BatchRunSummary")

    return {
        "config_sha256": summary.config_sha256,
        "failed": summary.failed,
        "finished_at_utc": _datetime_to_utc_iso(summary.finished_at_utc),
        "run_id": summary.run_id,
        "schema_version": summary.schema_version,
        "skipped": summary.skipped,
        "started_at_utc": _datetime_to_utc_iso(summary.started_at_utc),
        "success_rate": summary.success_rate,
        "successful": summary.successful,
        "target_height": summary.target_height,
        "target_width": summary.target_width,
        "total_discovered": summary.total_discovered,
    }


def _serialize_json(payload: dict[str, object]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ManifestSerializationError("unable to serialize manifest payload") from exc


def record_to_json_line(record: ImageProcessingRecord) -> str:
    """Serialize one image record as exactly one JSONL line."""
    return _serialize_json(record_to_dict(record)) + "\n"


def summary_to_json(summary: BatchRunSummary) -> str:
    """Serialize one run summary as deterministic compact JSON."""
    return _serialize_json(summary_to_dict(summary))
