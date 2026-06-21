"""Batch-processing contracts and deterministic discovery."""

from smart_beauty_resize.batch.contracts import (
    BatchConfig,
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
)

__all__ = [
    "SUPPORTED_IMAGE_EXTENSIONS",
    "BatchConfig",
    "BatchRunSummary",
    "DiscoveredImage",
    "ImageProcessingRecord",
    "ProcessingStatus",
    "build_output_relative_path",
    "discover_images",
    "is_supported_image_path",
]
