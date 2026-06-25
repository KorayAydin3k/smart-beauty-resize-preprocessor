from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from smart_beauty_resize.backends.opencv_backend import resize_sample
from smart_beauty_resize.batch.contracts import (
    BatchConfig,
    BatchExecutionResult,
    BatchRunSummary,
    ImageProcessingRecord,
    ProcessingStatus,
)
from smart_beauty_resize.batch.discovery import (
    DiscoveredImage,
    build_output_relative_path,
    discover_images,
)
from smart_beauty_resize.contracts import (
    BatchConfigurationError,
    OutputExistsError,
    SmartBeautyResizeError,
)
from smart_beauty_resize.io.contracts import ImageDecodeMetadata
from smart_beauty_resize.io.decoder import decode_image_with_metadata
from smart_beauty_resize.provenance.hashing import (
    sha256_file,
    sha256_resize_config,
)
from smart_beauty_resize.writing.safe_writer import write_png_atomic

SCHEMA_VERSION = "1.1"


def _processing_time_ms(started_ns: int) -> float:
    elapsed_ns = time.perf_counter_ns() - started_ns
    return max(0.0, elapsed_ns / 1_000_000.0)


def _new_run_id(started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
    random_suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}-{random_suffix}"


def _error_message(exception: BaseException) -> str:
    message = str(exception).strip()
    if message:
        return message
    return exception.__class__.__name__


def _process_discovered_image(
    discovered: DiscoveredImage,
    config: BatchConfig,
    config_sha256: str,
) -> ImageProcessingRecord:
    started_ns = time.perf_counter_ns()

    output_relative_path: Path | None = None
    source_sha256: str | None = None
    output_sha256: str | None = None

    original_width: int | None = None
    original_height: int | None = None
    resized_width: int | None = None
    resized_height: int | None = None

    pad_left: int | None = None
    pad_top: int | None = None
    pad_right: int | None = None
    pad_bottom: int | None = None
    interpolation: str | None = None
    decode_metadata: ImageDecodeMetadata | None = None

    try:
        output_relative_path = build_output_relative_path(
            discovered.relative_path,
            preserve_directory_structure=(config.preserve_directory_structure),
            output_format=config.output_format,
        )

        source_sha256 = sha256_file(discovered.source_path)

        decoded = decode_image_with_metadata(discovered.source_path)
        image = decoded.image
        decode_metadata = decoded.metadata
        original_height = int(image.shape[0])
        original_width = int(image.shape[1])

        resize_result = resize_sample(
            image=image,
            config=config.resize_config,
        )

        resized_width = resize_result.plan.resized_width
        resized_height = resize_result.plan.resized_height
        pad_left = resize_result.plan.pad_left
        pad_top = resize_result.plan.pad_top
        pad_right = resize_result.plan.pad_right
        pad_bottom = resize_result.plan.pad_bottom
        interpolation = resize_result.interpolation

        output_path = write_png_atomic(
            image=resize_result.image,
            output_root=config.output_dir,
            relative_path=output_relative_path,
            overwrite=config.overwrite,
        )

        output_sha256 = sha256_file(output_path)

    except OutputExistsError as exc:
        if config.fail_fast:
            raise

        return ImageProcessingRecord(
            schema_version=SCHEMA_VERSION,
            source_relative_path=discovered.relative_path,
            output_relative_path=output_relative_path,
            status=ProcessingStatus.SKIPPED,
            source_sha256=source_sha256,
            output_sha256=None,
            config_sha256=config_sha256,
            original_width=original_width,
            original_height=original_height,
            resized_width=resized_width,
            resized_height=resized_height,
            target_width=config.resize_config.target_width,
            target_height=config.resize_config.target_height,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
            interpolation=interpolation,
            processing_time_ms=_processing_time_ms(started_ns),
            error_type=exc.__class__.__name__,
            error_message=_error_message(exc),
            decode_metadata=decode_metadata,
        )

    except SmartBeautyResizeError as exc:
        if config.fail_fast:
            raise

        return ImageProcessingRecord(
            schema_version=SCHEMA_VERSION,
            source_relative_path=discovered.relative_path,
            output_relative_path=output_relative_path,
            status=ProcessingStatus.FAILED,
            source_sha256=source_sha256,
            output_sha256=None,
            config_sha256=config_sha256,
            original_width=original_width,
            original_height=original_height,
            resized_width=resized_width,
            resized_height=resized_height,
            target_width=config.resize_config.target_width,
            target_height=config.resize_config.target_height,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
            interpolation=interpolation,
            processing_time_ms=_processing_time_ms(started_ns),
            error_type=exc.__class__.__name__,
            error_message=_error_message(exc),
            decode_metadata=decode_metadata,
        )

    return ImageProcessingRecord(
        schema_version=SCHEMA_VERSION,
        source_relative_path=discovered.relative_path,
        output_relative_path=output_relative_path,
        status=ProcessingStatus.SUCCESS,
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        config_sha256=config_sha256,
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        resized_height=resized_height,
        target_width=config.resize_config.target_width,
        target_height=config.resize_config.target_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        interpolation=interpolation,
        processing_time_ms=_processing_time_ms(started_ns),
        error_type=None,
        error_message=None,
        decode_metadata=decode_metadata,
    )


def process_batch(config: BatchConfig) -> BatchExecutionResult:
    """Process all discovered images and return records plus a run summary.

    Expected package-level per-image failures become failed or skipped records
    unless fail-fast mode is enabled. Unexpected programming errors propagate.
    """
    if not isinstance(config, BatchConfig):
        raise BatchConfigurationError("config must be a BatchConfig instance")

    started_at = datetime.now(UTC)
    run_id = _new_run_id(started_at)
    config_sha256 = sha256_resize_config(config.resize_config)

    discovered_images = discover_images(config.input_dir)

    records = tuple(
        _process_discovered_image(
            discovered=discovered,
            config=config,
            config_sha256=config_sha256,
        )
        for discovered in discovered_images
    )

    finished_at = datetime.now(UTC)

    successful = sum(record.status is ProcessingStatus.SUCCESS for record in records)
    failed = sum(record.status is ProcessingStatus.FAILED for record in records)
    skipped = sum(record.status is ProcessingStatus.SKIPPED for record in records)

    summary = BatchRunSummary(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        total_discovered=len(records),
        successful=successful,
        failed=failed,
        skipped=skipped,
        target_width=config.resize_config.target_width,
        target_height=config.resize_config.target_height,
        config_sha256=config_sha256,
    )

    return BatchExecutionResult(
        records=records,
        summary=summary,
    )
