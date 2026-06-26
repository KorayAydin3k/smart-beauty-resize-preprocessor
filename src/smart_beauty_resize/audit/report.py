from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Final

from smart_beauty_resize.batch.contracts import (
    BatchExecutionResult,
    ImageProcessingRecord,
    ProcessingStatus,
)
from smart_beauty_resize.contracts import (
    ManifestSerializationError,
    ProvenanceError,
)
from smart_beauty_resize.io.contracts import InputPolicy, SourceImageLimits

DATASET_AUDIT_SCHEMA_VERSION: Final = "1.1"
_UNKNOWN_VALUE: Final = "unknown"


@dataclass(frozen=True, slots=True)
class AuditCount:
    """One deterministic label/count entry in an audit distribution."""

    label: str
    count: int

    def __post_init__(self) -> None:
        if type(self.label) is not str or not self.label:
            raise ProvenanceError("audit count label must be a non-empty string")
        if type(self.count) is not int or self.count < 0:
            raise ProvenanceError("audit count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class IntegerDistributionSummary:
    """Deterministic descriptive statistics over positive integer values."""

    count: int
    minimum: int
    maximum: int
    mean: float
    p50: int
    p95: int
    p99: int

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count <= 0:
            raise ProvenanceError("distribution count must be a positive integer")

        for field_name in ("minimum", "maximum", "p50", "p95", "p99"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ProvenanceError(f"distribution {field_name} must be a positive integer")

        if isinstance(self.mean, bool) or not isinstance(self.mean, (int, float)):
            raise ProvenanceError("distribution mean must be a finite positive real number")
        if not math.isfinite(float(self.mean)) or float(self.mean) <= 0.0:
            raise ProvenanceError("distribution mean must be a finite positive real number")

        if self.minimum > self.maximum:
            raise ProvenanceError("distribution minimum must not exceed maximum")

        for percentile_name in ("p50", "p95", "p99"):
            percentile = getattr(self, percentile_name)
            if not self.minimum <= percentile <= self.maximum:
                raise ProvenanceError(
                    f"distribution {percentile_name} must be within minimum and maximum"
                )

        if not self.p50 <= self.p95 <= self.p99:
            raise ProvenanceError("distribution percentiles must be monotonically non-decreasing")


@dataclass(frozen=True, slots=True)
class DatasetAuditSummary:
    """Immutable aggregate audit summary for one completed batch run."""

    schema_version: str
    run_id: str
    config_sha256: str
    input_policy: InputPolicy
    source_limits: SourceImageLimits
    total_records: int
    records_with_decode_metadata: int
    records_without_decode_metadata: int
    status_counts: tuple[AuditCount, ...]
    source_format_counts: tuple[AuditCount, ...]
    source_mode_counts: tuple[AuditCount, ...]
    source_bit_depth_counts: tuple[AuditCount, ...]
    source_channel_count_counts: tuple[AuditCount, ...]
    error_type_counts: tuple[AuditCount, ...]
    source_width_statistics: IntegerDistributionSummary | None
    source_height_statistics: IntegerDistributionSummary | None
    source_pixel_count_statistics: IntegerDistributionSummary | None
    alpha_present_count: int
    icc_profile_present_count: int
    exif_orientation_present_count: int
    exif_orientation_applied_count: int
    rgb_conversion_applied_count: int
    bit_depth_conversion_applied_count: int

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_AUDIT_SCHEMA_VERSION:
            raise ProvenanceError(
                f"dataset audit schema_version must be '{DATASET_AUDIT_SCHEMA_VERSION}'"
            )
        if type(self.run_id) is not str or not self.run_id:
            raise ProvenanceError("dataset audit run_id must be a non-empty string")
        if (
            type(self.config_sha256) is not str
            or len(self.config_sha256) != 64
            or self.config_sha256 != self.config_sha256.lower()
            or any(character not in "0123456789abcdef" for character in self.config_sha256)
        ):
            raise ProvenanceError("dataset audit config_sha256 must be lowercase SHA-256 hex")
        if not isinstance(self.input_policy, InputPolicy):
            raise ProvenanceError("dataset audit input_policy must be an InputPolicy")
        if not isinstance(self.source_limits, SourceImageLimits):
            raise ProvenanceError(
                "dataset audit source_limits must be a SourceImageLimits instance"
            )

        for field_name in (
            "total_records",
            "records_with_decode_metadata",
            "records_without_decode_metadata",
            "alpha_present_count",
            "icc_profile_present_count",
            "exif_orientation_present_count",
            "exif_orientation_applied_count",
            "rgb_conversion_applied_count",
            "bit_depth_conversion_applied_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ProvenanceError(f"dataset audit {field_name} must be non-negative")

        if (
            self.records_with_decode_metadata + self.records_without_decode_metadata
            != self.total_records
        ):
            raise ProvenanceError("decode metadata counts must equal total_records")

        count_fields = (
            "status_counts",
            "source_format_counts",
            "source_mode_counts",
            "source_bit_depth_counts",
            "source_channel_count_counts",
            "error_type_counts",
        )
        for field_name in count_fields:
            entries = getattr(self, field_name)
            _validate_count_entries(field_name, entries)

        if _count_total(self.status_counts) != self.total_records:
            raise ProvenanceError("status_counts must sum to total_records")

        metadata_distributions = (
            self.source_format_counts,
            self.source_mode_counts,
            self.source_bit_depth_counts,
            self.source_channel_count_counts,
        )
        if any(
            _count_total(entries) != self.records_with_decode_metadata
            for entries in metadata_distributions
        ):
            raise ProvenanceError(
                "source metadata distributions must sum to records_with_decode_metadata"
            )

        failed_or_skipped = _count_for_label(
            self.status_counts,
            ProcessingStatus.FAILED.value,
        ) + _count_for_label(
            self.status_counts,
            ProcessingStatus.SKIPPED.value,
        )
        if _count_total(self.error_type_counts) != failed_or_skipped:
            raise ProvenanceError("error_type_counts must cover failed and skipped records")

        statistics = (
            self.source_width_statistics,
            self.source_height_statistics,
            self.source_pixel_count_statistics,
        )
        if self.records_with_decode_metadata == 0:
            if any(value is not None for value in statistics):
                raise ProvenanceError("source statistics must be None without decode metadata")
        else:
            if any(value is None for value in statistics):
                raise ProvenanceError("source statistics are required with decode metadata")
            if any(
                value is not None and value.count != self.records_with_decode_metadata
                for value in statistics
            ):
                raise ProvenanceError(
                    "source statistics counts must equal records_with_decode_metadata"
                )

        metadata_flag_counts = (
            self.alpha_present_count,
            self.icc_profile_present_count,
            self.exif_orientation_present_count,
            self.exif_orientation_applied_count,
            self.rgb_conversion_applied_count,
            self.bit_depth_conversion_applied_count,
        )
        if any(count > self.records_with_decode_metadata for count in metadata_flag_counts):
            raise ProvenanceError("metadata flag counts cannot exceed metadata coverage")
        if self.exif_orientation_applied_count > self.exif_orientation_present_count:
            raise ProvenanceError(
                "exif_orientation_applied_count cannot exceed orientation presence"
            )

    @property
    def decode_metadata_coverage_percent(self) -> float:
        """Return the percentage of records with source decode metadata."""
        if self.total_records == 0:
            return 0.0
        return (self.records_with_decode_metadata / self.total_records) * 100.0


def _validate_count_entries(
    field_name: str,
    entries: object,
) -> None:
    if type(entries) is not tuple:
        raise ProvenanceError(f"dataset audit {field_name} must be a tuple")
    if any(not isinstance(entry, AuditCount) for entry in entries):
        raise ProvenanceError(f"dataset audit {field_name} must contain AuditCount values")

    labels = [entry.label for entry in entries]
    if labels != sorted(labels):
        raise ProvenanceError(f"dataset audit {field_name} must be sorted by label")
    if len(labels) != len(set(labels)):
        raise ProvenanceError(f"dataset audit {field_name} labels must be unique")


def _count_total(entries: tuple[AuditCount, ...]) -> int:
    return sum(entry.count for entry in entries)


def _count_for_label(
    entries: tuple[AuditCount, ...],
    label: str,
) -> int:
    return next((entry.count for entry in entries if entry.label == label), 0)


def _counter_entries(counter: Counter[str]) -> tuple[AuditCount, ...]:
    return tuple(
        AuditCount(label=label, count=count)
        for label, count in sorted(counter.items())
        if count > 0
    )


def _status_entries(records: tuple[ImageProcessingRecord, ...]) -> tuple[AuditCount, ...]:
    counter = Counter(record.status.value for record in records)
    return tuple(
        AuditCount(label=status.value, count=counter[status.value])
        for status in sorted(ProcessingStatus, key=lambda value: value.value)
    )


def _nearest_rank(sorted_values: tuple[int, ...], percentile: int) -> int:
    rank = math.ceil((percentile / 100.0) * len(sorted_values))
    return sorted_values[max(0, rank - 1)]


def _integer_statistics(values: list[int]) -> IntegerDistributionSummary | None:
    if not values:
        return None

    ordered = tuple(sorted(values))
    return IntegerDistributionSummary(
        count=len(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=sum(ordered) / len(ordered),
        p50=_nearest_rank(ordered, 50),
        p95=_nearest_rank(ordered, 95),
        p99=_nearest_rank(ordered, 99),
    )


def build_dataset_audit_summary(
    result: BatchExecutionResult,
) -> DatasetAuditSummary:
    """Aggregate per-image records into one deterministic dataset audit summary."""
    if not isinstance(result, BatchExecutionResult):
        raise ProvenanceError("result must be a BatchExecutionResult")

    records = result.records
    records_with_metadata = tuple(
        record for record in records if record.decode_metadata is not None
    )
    metadata_values = tuple(
        record.decode_metadata
        for record in records_with_metadata
        if record.decode_metadata is not None
    )

    source_format_counts = Counter(metadata.source_format for metadata in metadata_values)
    source_mode_counts = Counter(metadata.source_mode for metadata in metadata_values)
    source_bit_depth_counts = Counter(
        _UNKNOWN_VALUE if metadata.source_bit_depth is None else str(metadata.source_bit_depth)
        for metadata in metadata_values
    )
    source_channel_count_counts = Counter(
        str(metadata.source_channel_count) for metadata in metadata_values
    )
    error_type_counts = Counter(
        record.error_type
        for record in records
        if record.error_type is not None
    )

    source_widths = [metadata.source_width for metadata in metadata_values]
    source_heights = [metadata.source_height for metadata in metadata_values]
    source_pixel_counts = [
        metadata.source_width * metadata.source_height for metadata in metadata_values
    ]

    return DatasetAuditSummary(
        schema_version=DATASET_AUDIT_SCHEMA_VERSION,
        run_id=result.summary.run_id,
        config_sha256=result.summary.config_sha256,
        input_policy=result.summary.input_policy,
        source_limits=result.summary.source_limits,
        total_records=len(records),
        records_with_decode_metadata=len(metadata_values),
        records_without_decode_metadata=len(records) - len(metadata_values),
        status_counts=_status_entries(records),
        source_format_counts=_counter_entries(source_format_counts),
        source_mode_counts=_counter_entries(source_mode_counts),
        source_bit_depth_counts=_counter_entries(source_bit_depth_counts),
        source_channel_count_counts=_counter_entries(source_channel_count_counts),
        error_type_counts=_counter_entries(error_type_counts),
        source_width_statistics=_integer_statistics(source_widths),
        source_height_statistics=_integer_statistics(source_heights),
        source_pixel_count_statistics=_integer_statistics(source_pixel_counts),
        alpha_present_count=sum(metadata.alpha_present for metadata in metadata_values),
        icc_profile_present_count=sum(
            metadata.icc_profile_present for metadata in metadata_values
        ),
        exif_orientation_present_count=sum(
            metadata.exif_orientation is not None for metadata in metadata_values
        ),
        exif_orientation_applied_count=sum(
            metadata.exif_orientation_applied for metadata in metadata_values
        ),
        rgb_conversion_applied_count=sum(
            metadata.rgb_conversion_applied for metadata in metadata_values
        ),
        bit_depth_conversion_applied_count=sum(
            metadata.bit_depth_conversion_applied for metadata in metadata_values
        ),
    )


def _counts_to_dict(entries: tuple[AuditCount, ...]) -> dict[str, int]:
    return {entry.label: entry.count for entry in entries}


def _statistics_to_dict(
    statistics: IntegerDistributionSummary | None,
) -> dict[str, int | float] | None:
    if statistics is None:
        return None
    return {
        "count": statistics.count,
        "maximum": statistics.maximum,
        "mean": statistics.mean,
        "minimum": statistics.minimum,
        "p50": statistics.p50,
        "p95": statistics.p95,
        "p99": statistics.p99,
    }


def _source_limits_to_dict(
    limits: SourceImageLimits,
) -> dict[str, int | None]:
    return {
        "max_height": limits.max_height,
        "max_pixels": limits.max_pixels,
        "max_width": limits.max_width,
    }


def dataset_audit_to_dict(
    summary: DatasetAuditSummary,
) -> dict[str, object]:
    """Convert one dataset audit summary into a JSON-safe dictionary."""
    if not isinstance(summary, DatasetAuditSummary):
        raise ManifestSerializationError("summary must be a DatasetAuditSummary")

    return {
        "alpha_present_count": summary.alpha_present_count,
        "bit_depth_conversion_applied_count": summary.bit_depth_conversion_applied_count,
        "config_sha256": summary.config_sha256,
        "decode_metadata_coverage_percent": summary.decode_metadata_coverage_percent,
        "error_type_counts": _counts_to_dict(summary.error_type_counts),
        "exif_orientation_applied_count": summary.exif_orientation_applied_count,
        "exif_orientation_present_count": summary.exif_orientation_present_count,
        "icc_profile_present_count": summary.icc_profile_present_count,
        "input_policy": summary.input_policy.value,
        "records_with_decode_metadata": summary.records_with_decode_metadata,
        "records_without_decode_metadata": summary.records_without_decode_metadata,
        "rgb_conversion_applied_count": summary.rgb_conversion_applied_count,
        "run_id": summary.run_id,
        "schema_version": summary.schema_version,
        "source_limits": _source_limits_to_dict(summary.source_limits),
        "source_bit_depth_counts": _counts_to_dict(summary.source_bit_depth_counts),
        "source_channel_count_counts": _counts_to_dict(
            summary.source_channel_count_counts
        ),
        "source_format_counts": _counts_to_dict(summary.source_format_counts),
        "source_height_statistics": _statistics_to_dict(
            summary.source_height_statistics
        ),
        "source_mode_counts": _counts_to_dict(summary.source_mode_counts),
        "source_pixel_count_statistics": _statistics_to_dict(
            summary.source_pixel_count_statistics
        ),
        "source_width_statistics": _statistics_to_dict(summary.source_width_statistics),
        "status_counts": _counts_to_dict(summary.status_counts),
        "total_records": summary.total_records,
    }


def dataset_audit_to_json(summary: DatasetAuditSummary) -> str:
    """Serialize one dataset audit summary as deterministic compact JSON."""
    try:
        return json.dumps(
            dataset_audit_to_dict(summary),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ManifestSerializationError("unable to serialize dataset audit payload") from exc
