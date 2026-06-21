from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class SmartBeautyResizeError(Exception):
    """Base exception for all package-specific errors."""


class ResizeConfigurationError(SmartBeautyResizeError):
    """Base error for invalid resize configuration."""


class InvalidImageDimensionsError(ResizeConfigurationError):
    """Raised when image dimensions are invalid."""


class ExcessiveUpscaleError(ResizeConfigurationError):
    """Raised when an upscale exceeds the configured limit."""


class InvalidImageError(SmartBeautyResizeError):
    """Raised when a direct image payload is invalid."""


class InvalidMaskError(SmartBeautyResizeError):
    """Raised when a mask payload is invalid."""


class ImageDecodeError(SmartBeautyResizeError):
    """Raised when an image file cannot be decoded into a valid RGB array."""


class BatchConfigurationError(SmartBeautyResizeError):
    """Raised when batch-processing configuration is invalid."""


class ProvenanceError(SmartBeautyResizeError):
    """Raised when provenance or hashing operations fail."""


class ManifestSerializationError(SmartBeautyResizeError):
    """Raised when manifest data cannot be serialized safely."""


class DiscoveryError(SmartBeautyResizeError):
    """Raised when deterministic input-file discovery fails."""


class OutputWriteError(SmartBeautyResizeError):
    """Raised when a processed output cannot be written safely."""


class OutputExistsError(OutputWriteError):
    """Raised when an output exists and overwrite is disabled."""


class ManifestWriteError(SmartBeautyResizeError):
    """Raised when batch manifest artifacts cannot be written safely."""


Matrix = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _is_int(value: object) -> bool:
    return type(value) is int


def _is_finite_real(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False

    numeric_value = float(value)
    return math.isfinite(numeric_value)


@dataclass(frozen=True, slots=True)
class ResizeConfig:
    """Immutable configuration for deterministic letterbox resize planning."""

    target_width: int
    target_height: int
    allow_upscale: bool = True
    max_upscale_factor: float = 1.5
    padding_value: tuple[int, int, int] = (127, 127, 127)

    def __post_init__(self) -> None:
        if not _is_int(self.target_width) or self.target_width <= 0:
            raise InvalidImageDimensionsError("target_width must be a positive integer")
        if not _is_int(self.target_height) or self.target_height <= 0:
            raise InvalidImageDimensionsError("target_height must be a positive integer")
        if type(self.allow_upscale) is not bool:
            raise ResizeConfigurationError("allow_upscale must be a boolean")
        if not _is_finite_real(self.max_upscale_factor):
            raise ResizeConfigurationError("max_upscale_factor must be a finite real number")
        if float(self.max_upscale_factor) < 1.0:
            raise ResizeConfigurationError("max_upscale_factor must be at least 1.0")
        if type(self.padding_value) is not tuple or len(self.padding_value) != 3:
            raise ResizeConfigurationError(
                "padding_value must be a tuple of exactly three integers"
            )
        if any(type(channel) is not int for channel in self.padding_value):
            raise ResizeConfigurationError("padding_value entries must be integers")
        if any(channel < 0 or channel > 255 for channel in self.padding_value):
            raise ResizeConfigurationError("padding_value entries must be in the range [0, 255]")


@dataclass(frozen=True, slots=True)
class ResizePlan:
    """Immutable geometry plan for a deterministic letterbox transform."""

    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    target_width: int
    target_height: int
    nominal_scale: float
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    padding_fraction: float
    was_upscaled: bool
    forward_matrix: Matrix
    inverse_matrix: Matrix


@dataclass(slots=True)
class ResizeResult:
    """Container for a deterministic pixel resize result.

    The arrays are returned in contiguous layout and are intended to be treated
    as read-only by callers; the object itself is not frozen so the contract
    can remain explicit about the mutable NumPy payloads.
    """

    image: NDArray[np.uint8]
    plan: ResizePlan
    valid_region_mask: NDArray[np.uint8]
    mask: np.ndarray | None
    interpolation: str
