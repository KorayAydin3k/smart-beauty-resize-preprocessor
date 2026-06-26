from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from smart_beauty_resize.audit import (
    DATASET_AUDIT_SCHEMA_VERSION,
    build_dataset_audit_summary,
    dataset_audit_to_dict,
    dataset_audit_to_json,
)
from smart_beauty_resize.batch import (
    BatchExecutionResult,
    BatchRunSummary,
    ImageProcessingRecord,
    ProcessingStatus,
)
from smart_beauty_resize.contracts import (
    ManifestSerializationError,
    ProvenanceError,
)
from smart_beauty_resize.io import ImageDecodeMetadata, InputPolicy, SourceImageLimits

CONFIG_HASH = "a" * 64
SOURCE_HASH = "b" * 64
OUTPUT_HASH = "c" * 64
LIMITS = SourceImageLimits(
    max_width=12000,
    max_height=12000,
    max_pixels=64000000,
)


def _metadata(
    *,
    source_format: str,
    source_mode: str,
    source_width: int,
    source_height: int,
    source_bit_depth: int | None,
    source_channel_count: int,
    alpha_present: bool = False,
    icc_profile_present: bool = False,
    exif_orientation: int | None = None,
    exif_orientation_applied: bool = False,
    rgb_conversion_applied: bool = False,
    bit_depth_conversion_applied: bool = False,
) -> ImageDecodeMetadata:
    return ImageDecodeMetadata(
        source_format=source_format,
        source_mode=source_mode,
        source_width=source_width,
        source_height=source_height,
        decoded_width=source_width,
        decoded_height=source_height,
        source_bit_depth=source_bit_depth,
        source_channel_count=source_channel_count,
        alpha_present=alpha_present,
        icc_profile_present=icc_profile_present,
        exif_orientation=exif_orientation,
        exif_orientation_applied=exif_orientation_applied,
        rgb_conversion_applied=rgb_conversion_applied,
        bit_depth_conversion_applied=bit_depth_conversion_applied,
    )


def _record(
    *,
    source_path: str,
    status: ProcessingStatus,
    metadata: ImageDecodeMetadata | None,
    error_type: str | None = None,
) -> ImageProcessingRecord:
    if status is ProcessingStatus.SUCCESS:
        return ImageProcessingRecord(
            schema_version="1.3",
            source_relative_path=Path(source_path),
            output_relative_path=Path(source_path).with_suffix(".png"),
            status=status,
            source_sha256=SOURCE_HASH,
            output_sha256=OUTPUT_HASH,
            config_sha256=CONFIG_HASH,
            original_width=metadata.source_width if metadata is not None else 100,
            original_height=metadata.source_height if metadata is not None else 100,
            resized_width=32,
            resized_height=32,
            target_width=32,
            target_height=32,
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            interpolation="INTER_AREA",
            processing_time_ms=1.0,
            error_type=None,
            error_message=None,
            decode_metadata=metadata,
            input_policy=InputPolicy.STRICT_RGB8,
            source_limits=LIMITS,
        )

    return ImageProcessingRecord(
        schema_version="1.3",
        source_relative_path=Path(source_path),
        output_relative_path=None,
        status=status,
        source_sha256=SOURCE_HASH,
        output_sha256=None,
        config_sha256=CONFIG_HASH,
        original_width=None,
        original_height=None,
        resized_width=None,
        resized_height=None,
        target_width=32,
        target_height=32,
        pad_left=None,
        pad_top=None,
        pad_right=None,
        pad_bottom=None,
        interpolation=None,
        processing_time_ms=1.0,
        error_type=error_type,
        error_message="expected test failure",
        decode_metadata=metadata,
        input_policy=InputPolicy.STRICT_RGB8,
        source_limits=LIMITS,
    )


def _result() -> BatchExecutionResult:
    records = (
        _record(
            source_path="rgb.jpg",
            status=ProcessingStatus.SUCCESS,
            metadata=_metadata(
                source_format="JPEG",
                source_mode="RGB",
                source_width=100,
                source_height=200,
                source_bit_depth=8,
                source_channel_count=3,
                icc_profile_present=True,
                exif_orientation=6,
                exif_orientation_applied=True,
            ),
        ),
        _record(
            source_path="rgba.png",
            status=ProcessingStatus.FAILED,
            metadata=_metadata(
                source_format="PNG",
                source_mode="RGBA",
                source_width=400,
                source_height=300,
                source_bit_depth=None,
                source_channel_count=4,
                alpha_present=True,
                rgb_conversion_applied=True,
                bit_depth_conversion_applied=True,
            ),
            error_type="InputPolicyViolationError",
        ),
        _record(
            source_path="corrupt.jpg",
            status=ProcessingStatus.FAILED,
            metadata=None,
            error_type="ImageDecodeError",
        ),
        _record(
            source_path="existing.png",
            status=ProcessingStatus.SKIPPED,
            metadata=None,
            error_type="OutputExistsError",
        ),
    )
    summary = BatchRunSummary(
        schema_version="1.3",
        run_id="20260625T160000Z-audit",
        started_at_utc=datetime(2026, 6, 25, 16, 0, tzinfo=UTC),
        finished_at_utc=datetime(2026, 6, 25, 16, 1, tzinfo=UTC),
        total_discovered=4,
        successful=1,
        failed=2,
        skipped=1,
        target_width=32,
        target_height=32,
        config_sha256=CONFIG_HASH,
        input_policy=InputPolicy.STRICT_RGB8,
        source_limits=LIMITS,
    )
    return BatchExecutionResult(records=records, summary=summary)


def _empty_result() -> BatchExecutionResult:
    summary = BatchRunSummary(
        schema_version="1.2",
        run_id="20260625T160000Z-empty",
        started_at_utc=datetime(2026, 6, 25, 16, 0, tzinfo=UTC),
        finished_at_utc=datetime(2026, 6, 25, 16, 0, tzinfo=UTC),
        total_discovered=0,
        successful=0,
        failed=0,
        skipped=0,
        target_width=32,
        target_height=32,
        config_sha256=CONFIG_HASH,
        input_policy=InputPolicy.AUDIT_ONLY,
    )
    return BatchExecutionResult(records=(), summary=summary)


def test_build_dataset_audit_summary_aggregates_metadata_and_errors() -> None:
    summary = build_dataset_audit_summary(_result())
    payload = dataset_audit_to_dict(summary)

    assert summary.schema_version == DATASET_AUDIT_SCHEMA_VERSION
    assert summary.decode_metadata_coverage_percent == 50.0
    assert payload["run_id"] == "20260625T160000Z-audit"
    assert payload["input_policy"] == "strict_rgb8"
    assert payload["source_limits"] == {
        "max_height": 12000,
        "max_pixels": 64000000,
        "max_width": 12000,
    }
    assert payload["total_records"] == 4
    assert payload["records_with_decode_metadata"] == 2
    assert payload["records_without_decode_metadata"] == 2
    assert payload["status_counts"] == {"failed": 2, "skipped": 1, "success": 1}
    assert payload["source_format_counts"] == {"JPEG": 1, "PNG": 1}
    assert payload["source_mode_counts"] == {"RGB": 1, "RGBA": 1}
    assert payload["source_bit_depth_counts"] == {"8": 1, "unknown": 1}
    assert payload["source_channel_count_counts"] == {"3": 1, "4": 1}
    assert payload["error_type_counts"] == {
        "ImageDecodeError": 1,
        "InputPolicyViolationError": 1,
        "OutputExistsError": 1,
    }
    assert payload["alpha_present_count"] == 1
    assert payload["icc_profile_present_count"] == 1
    assert payload["exif_orientation_present_count"] == 1
    assert payload["exif_orientation_applied_count"] == 1
    assert payload["rgb_conversion_applied_count"] == 1
    assert payload["bit_depth_conversion_applied_count"] == 1
    assert payload["source_width_statistics"] == {
        "count": 2,
        "maximum": 400,
        "mean": 250.0,
        "minimum": 100,
        "p50": 100,
        "p95": 400,
        "p99": 400,
    }
    assert payload["source_height_statistics"] == {
        "count": 2,
        "maximum": 300,
        "mean": 250.0,
        "minimum": 200,
        "p50": 200,
        "p95": 300,
        "p99": 300,
    }
    assert payload["source_pixel_count_statistics"] == {
        "count": 2,
        "maximum": 120000,
        "mean": 70000.0,
        "minimum": 20000,
        "p50": 20000,
        "p95": 120000,
        "p99": 120000,
    }


def test_empty_dataset_audit_is_explicit_and_serializable() -> None:
    summary = build_dataset_audit_summary(_empty_result())
    payload = json.loads(dataset_audit_to_json(summary))

    assert payload["total_records"] == 0
    assert payload["decode_metadata_coverage_percent"] == 0.0
    assert payload["status_counts"] == {"failed": 0, "skipped": 0, "success": 0}
    assert payload["source_format_counts"] == {}
    assert payload["error_type_counts"] == {}
    assert payload["source_width_statistics"] is None
    assert payload["source_height_statistics"] is None
    assert payload["source_pixel_count_statistics"] is None


def test_dataset_audit_json_is_deterministic_and_compact() -> None:
    summary = build_dataset_audit_summary(_result())

    first = dataset_audit_to_json(summary)
    second = dataset_audit_to_json(summary)

    assert first == second
    assert "\n" not in first
    assert first == json.dumps(
        json.loads(first),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def test_dataset_audit_rejects_inconsistent_coverage() -> None:
    summary = build_dataset_audit_summary(_result())

    with pytest.raises(ProvenanceError, match="decode metadata counts"):
        replace(summary, records_without_decode_metadata=1)


def test_dataset_audit_functions_reject_invalid_inputs() -> None:
    with pytest.raises(ProvenanceError, match="BatchExecutionResult"):
        build_dataset_audit_summary(object())  # type: ignore[arg-type]

    with pytest.raises(ManifestSerializationError, match="DatasetAuditSummary"):
        dataset_audit_to_dict(object())  # type: ignore[arg-type]
