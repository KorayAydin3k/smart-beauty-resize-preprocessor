"""Batch-processing contracts, discovery, and execution."""

from smart_beauty_resize.batch.contracts import (
    BatchConfig,
    BatchExecutionResult,
    BatchRunSummary,
    ImageProcessingRecord,
    ProcessingStatus,
)
from smart_beauty_resize.batch.discovery import (
    SUPPORTED_IMAGE_EXTENSIONS,
    DiscoveredImage,
    build_output_relative_path,
    discover_images,
    is_supported_image_path,
    validate_unique_output_paths,
)
from smart_beauty_resize.batch.processor import (
    SCHEMA_VERSION,
    process_batch,
)

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "BatchConfig",
    "BatchExecutionResult",
    "BatchRunSummary",
    "DiscoveredImage",
    "ImageProcessingRecord",
    "ProcessingStatus",
    "build_output_relative_path",
    "discover_images",
    "is_supported_image_path",
    "validate_unique_output_paths",
    "process_batch",
]
