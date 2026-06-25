"""Public package interface for smart-beauty-resize-preprocessor."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from smart_beauty_resize.backends.opencv_backend import resize_sample
from smart_beauty_resize.config import (
    PreprocessingProfile,
    load_preprocessing_profile,
    profile_from_mapping,
)
from smart_beauty_resize.contracts import (
    ExcessiveUpscaleError,
    ImageDecodeError,
    InvalidImageDimensionsError,
    InvalidImageError,
    InvalidMaskError,
    ProfileConfigurationError,
    ResizeConfig,
    ResizeConfigurationError,
    ResizePlan,
    ResizeResult,
    SmartBeautyResizeError,
)
from smart_beauty_resize.geometry import (
    apply_matrix_to_point,
    calculate_letterbox_plan,
    round_half_up_positive,
)
from smart_beauty_resize.io import (
    DecodedImage,
    ImageDecodeMetadata,
    decode_image,
    decode_image_with_metadata,
)

__all__ = [
    "__version__",
    "ResizeConfig",
    "PreprocessingProfile",
    "ResizePlan",
    "ResizeResult",
    "SmartBeautyResizeError",
    "ResizeConfigurationError",
    "InvalidImageDimensionsError",
    "ExcessiveUpscaleError",
    "InvalidImageError",
    "InvalidMaskError",
    "ImageDecodeError",
    "ProfileConfigurationError",
    "round_half_up_positive",
    "calculate_letterbox_plan",
    "apply_matrix_to_point",
    "ImageDecodeMetadata",
    "DecodedImage",
    "decode_image",
    "decode_image_with_metadata",
    "resize_sample",
    "profile_from_mapping",
    "load_preprocessing_profile",
]

try:
    __version__ = version("smart-beauty-resize-preprocessor")
except PackageNotFoundError:
    __version__ = "0+unknown"
