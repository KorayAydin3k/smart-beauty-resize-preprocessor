from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from smart_beauty_resize import ImageDecodeMetadata
from smart_beauty_resize.batch import (
    BatchRunSummary,
    ImageProcessingRecord,
    ProcessingStatus,
)
from smart_beauty_resize.contracts import (
    BatchConfigurationError,
    ManifestSerializationError,
)
from smart_beauty_resize.provenance import (
    record_to_dict,
    record_to_json_line,
    summary_to_dict,
    summary_to_json,
)

CONFIG_HASH = "a" * 64
SOURCE_HASH = "b" * 64
OUTPUT_HASH = "c" * 64


def _decode_metadata() -> ImageDecodeMetadata:
    return ImageDecodeMetadata(
        source_format="JPEG",
        source_mode="RGB",
        source_width=1920,
        source_height=1080,
        decoded_width=1920,
        decoded_height=1080,
        source_bit_depth=8,
        source_channel_count=3,
        alpha_present=False,
        icc_profile_present=False,
        exif_orientation=None,
        exif_orientation_applied=False,
        rgb_conversion_applied=False,
        bit_depth_conversion_applied=False,
    )


def _success_record() -> ImageProcessingRecord:
    return ImageProcessingRecord(
        schema_version="1.0",
        source_relative_path=Path("person/sample.jpg"),
        output_relative_path=Path("person/sample.png"),
        status=ProcessingStatus.SUCCESS,
        source_sha256=SOURCE_HASH,
        output_sha256=OUTPUT_HASH,
        config_sha256=CONFIG_HASH,
        original_width=1920,
        original_height=1080,
        resized_width=512,
        resized_height=288,
        target_width=512,
        target_height=512,
        pad_left=0,
        pad_top=112,
        pad_right=0,
        pad_bottom=112,
        interpolation="INTER_AREA",
        processing_time_ms=12.5,
        error_type=None,
        error_message=None,
        decode_metadata=_decode_metadata(),
    )


def _failed_record() -> ImageProcessingRecord:
    return ImageProcessingRecord(
        schema_version="1.0",
        source_relative_path=Path("broken.jpg"),
        output_relative_path=None,
        status=ProcessingStatus.FAILED,
        source_sha256=SOURCE_HASH,
        output_sha256=None,
        config_sha256=CONFIG_HASH,
        original_width=None,
        original_height=None,
        resized_width=None,
        resized_height=None,
        target_width=512,
        target_height=512,
        pad_left=None,
        pad_top=None,
        pad_right=None,
        pad_bottom=None,
        interpolation=None,
        processing_time_ms=3.2,
        error_type="ImageDecodeError",
        error_message="Unable to decode image.",
    )


def _skipped_record() -> ImageProcessingRecord:
    return ImageProcessingRecord(
        schema_version="1.0",
        source_relative_path=Path("existing.jpg"),
        output_relative_path=Path("existing.png"),
        status=ProcessingStatus.SKIPPED,
        source_sha256=SOURCE_HASH,
        output_sha256=None,
        config_sha256=CONFIG_HASH,
        original_width=None,
        original_height=None,
        resized_width=None,
        resized_height=None,
        target_width=512,
        target_height=512,
        pad_left=None,
        pad_top=None,
        pad_right=None,
        pad_bottom=None,
        interpolation=None,
        processing_time_ms=0.1,
        error_type="OutputExists",
        error_message="Output already exists.",
    )


def _summary() -> BatchRunSummary:
    return BatchRunSummary(
        schema_version="1.0",
        run_id="20260621T120000Z",
        started_at_utc=datetime(
            2026,
            6,
            21,
            12,
            0,
            tzinfo=UTC,
        ),
        finished_at_utc=datetime(
            2026,
            6,
            21,
            12,
            1,
            tzinfo=UTC,
        ),
        total_discovered=10,
        successful=8,
        failed=1,
        skipped=1,
        target_width=512,
        target_height=512,
        config_sha256=CONFIG_HASH,
    )


def test_valid_success_failed_and_skipped_records() -> None:
    assert _success_record().status is ProcessingStatus.SUCCESS
    assert _failed_record().status is ProcessingStatus.FAILED
    assert _skipped_record().status is ProcessingStatus.SKIPPED


@pytest.mark.parametrize(
    "path",
    [
        Path("/absolute/image.jpg"),
        Path("../outside.jpg"),
        Path("nested/../../outside.jpg"),
        Path("."),
    ],
)
def test_record_rejects_unsafe_source_paths(path: Path) -> None:
    with pytest.raises(BatchConfigurationError):
        replace(
            _success_record(),
            source_relative_path=path,
        )


def test_record_rejects_unsafe_output_path() -> None:
    with pytest.raises(BatchConfigurationError):
        replace(
            _success_record(),
            output_relative_path=Path("../outside.png"),
        )


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
    ],
)
def test_record_rejects_invalid_hashes(digest: str) -> None:
    with pytest.raises(BatchConfigurationError):
        replace(
            _success_record(),
            source_sha256=digest,
        )


@pytest.mark.parametrize(
    "processing_time",
    [
        -1.0,
        math.nan,
        math.inf,
        -math.inf,
        True,
    ],
)
def test_record_rejects_invalid_processing_time(
    processing_time: object,
) -> None:
    with pytest.raises(BatchConfigurationError):
        replace(
            _success_record(),
            processing_time_ms=processing_time,  # type: ignore[arg-type]
        )


def test_record_rejects_invalid_decode_metadata() -> None:
    with pytest.raises(
        BatchConfigurationError,
        match="decode_metadata",
    ):
        replace(
            _success_record(),
            decode_metadata=object(),  # type: ignore[arg-type]
        )


def test_success_record_requires_output_and_hashes() -> None:
    with pytest.raises(
        BatchConfigurationError,
        match="successful records",
    ):
        replace(
            _success_record(),
            output_sha256=None,
        )


def test_success_record_rejects_error_fields() -> None:
    with pytest.raises(
        BatchConfigurationError,
        match="must not contain error",
    ):
        replace(
            _success_record(),
            error_type="UnexpectedError",
            error_message="Unexpected error.",
        )


def test_success_record_requires_exact_canvas_geometry() -> None:
    with pytest.raises(
        BatchConfigurationError,
        match="horizontal geometry",
    ):
        replace(
            _success_record(),
            pad_right=1,
        )


def test_failed_record_requires_error_fields() -> None:
    with pytest.raises(
        BatchConfigurationError,
        match="must contain",
    ):
        replace(
            _failed_record(),
            error_message=None,
        )


def test_failed_record_rejects_output_hash() -> None:
    with pytest.raises(
        BatchConfigurationError,
        match="must not contain",
    ):
        replace(
            _failed_record(),
            output_sha256=OUTPUT_HASH,
        )


def test_summary_is_valid_and_calculates_success_rate() -> None:
    summary = _summary()

    assert summary.success_rate == 80.0
    assert summary.total_discovered == 10


def test_empty_summary_has_zero_success_rate() -> None:
    summary = replace(
        _summary(),
        total_discovered=0,
        successful=0,
        failed=0,
        skipped=0,
    )

    assert summary.success_rate == 0.0


def test_summary_rejects_inconsistent_counts() -> None:
    with pytest.raises(
        BatchConfigurationError,
        match="must equal total_discovered",
    ):
        replace(
            _summary(),
            successful=9,
        )


def test_summary_rejects_naive_timestamp() -> None:
    with pytest.raises(
        BatchConfigurationError,
        match="timezone-aware",
    ):
        replace(
            _summary(),
            started_at_utc=datetime(2026, 6, 21, 12, 0),
        )


def test_summary_rejects_non_utc_timestamp() -> None:
    non_utc = timezone(timedelta(hours=3))

    with pytest.raises(
        BatchConfigurationError,
        match="must use UTC",
    ):
        replace(
            _summary(),
            started_at_utc=datetime(
                2026,
                6,
                21,
                12,
                0,
                tzinfo=non_utc,
            ),
        )


def test_summary_rejects_finished_before_started() -> None:
    with pytest.raises(
        BatchConfigurationError,
        match="must not be before",
    ):
        replace(
            _summary(),
            finished_at_utc=datetime(
                2026,
                6,
                21,
                11,
                59,
                tzinfo=UTC,
            ),
        )


def test_record_to_dict_serializes_enum_and_paths() -> None:
    payload = record_to_dict(_success_record())

    assert payload["status"] == "success"
    assert payload["source_relative_path"] == "person/sample.jpg"
    assert payload["output_relative_path"] == "person/sample.png"
    assert payload["decode_metadata"] == {
        "alpha_present": False,
        "bit_depth_conversion_applied": False,
        "decoded_height": 1080,
        "decoded_width": 1920,
        "exif_orientation": None,
        "exif_orientation_applied": False,
        "icc_profile_present": False,
        "rgb_conversion_applied": False,
        "source_bit_depth": 8,
        "source_channel_count": 3,
        "source_format": "JPEG",
        "source_height": 1080,
        "source_mode": "RGB",
        "source_width": 1920,
    }


def test_record_json_line_is_deterministic_and_has_one_newline() -> None:
    first = record_to_json_line(_success_record())
    second = record_to_json_line(_success_record())

    assert first == second
    assert first.endswith("\n")
    assert not first.endswith("\n\n")

    parsed = json.loads(first)
    assert parsed["status"] == "success"
    assert parsed["target_width"] == 512


def test_failed_record_serializes_null_decode_metadata() -> None:
    payload = record_to_dict(_failed_record())

    assert payload["decode_metadata"] is None


def test_summary_serializes_datetimes_with_z() -> None:
    payload = summary_to_dict(_summary())

    assert payload["started_at_utc"] == "2026-06-21T12:00:00Z"
    assert payload["finished_at_utc"] == "2026-06-21T12:01:00Z"
    assert payload["success_rate"] == 80.0


def test_summary_json_is_deterministic_compact_json() -> None:
    first = summary_to_json(_summary())
    second = summary_to_json(_summary())

    assert first == second
    assert "\n" not in first
    assert ": " not in first

    parsed = json.loads(first)
    assert parsed["run_id"] == "20260621T120000Z"


def test_unicode_error_message_is_preserved() -> None:
    record = replace(
        _failed_record(),
        error_message="Görüntü çözülemedi.",
    )

    serialized = record_to_json_line(record)

    assert "Görüntü çözülemedi." in serialized


def test_serializers_reject_invalid_objects() -> None:
    with pytest.raises(ManifestSerializationError):
        record_to_dict(object())  # type: ignore[arg-type]

    with pytest.raises(ManifestSerializationError):
        summary_to_dict(object())  # type: ignore[arg-type]
