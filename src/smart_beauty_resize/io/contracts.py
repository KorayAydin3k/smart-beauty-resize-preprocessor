from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from smart_beauty_resize.contracts import ImageDecodeError


class InputPolicy(StrEnum):
    """Supported source-image acceptance policies."""

    AUDIT_ONLY = "audit_only"
    STRICT_RGB8 = "strict_rgb8"


@dataclass(frozen=True, slots=True)
class SourceImageLimits:
    """Optional pre-decode limits for source image dimensions.

    ``None`` disables one limit. The object is immutable so it can be shared
    safely across batch records, summaries, and decoder calls.
    """

    max_width: int | None = None
    max_height: int | None = None
    max_pixels: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("max_width", "max_height", "max_pixels"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ImageDecodeError(f"{field_name} must be a positive integer or None")

    @property
    def enabled(self) -> bool:
        """Return whether at least one source limit is active."""
        return any(
            value is not None
            for value in (self.max_width, self.max_height, self.max_pixels)
        )


@dataclass(frozen=True, slots=True)
class ImageDecodeMetadata:
    """Observable source and canonicalization metadata for one decoded image.

    The metadata records what arrived at the decoder and which normalization
    steps were required before downstream resize processing. It is intentionally
    informational in this phase: no new rejection policy is introduced.
    """

    source_format: str
    source_mode: str
    source_width: int
    source_height: int
    decoded_width: int
    decoded_height: int
    source_bit_depth: int | None
    source_channel_count: int
    alpha_present: bool
    icc_profile_present: bool
    exif_orientation: int | None
    exif_orientation_applied: bool
    rgb_conversion_applied: bool
    bit_depth_conversion_applied: bool

    def __post_init__(self) -> None:
        for field_name in ("source_format", "source_mode"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ImageDecodeError(f"{field_name} must be a non-empty string")

        for field_name in (
            "source_width",
            "source_height",
            "decoded_width",
            "decoded_height",
            "source_channel_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ImageDecodeError(f"{field_name} must be a positive integer")

        if (
            self.source_bit_depth is not None
            and (type(self.source_bit_depth) is not int or self.source_bit_depth <= 0)
        ):
            raise ImageDecodeError("source_bit_depth must be a positive integer or None")

        for field_name in (
            "alpha_present",
            "icc_profile_present",
            "exif_orientation_applied",
            "rgb_conversion_applied",
            "bit_depth_conversion_applied",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ImageDecodeError(f"{field_name} must be a boolean")

        if (
            self.exif_orientation is not None
            and (
                type(self.exif_orientation) is not int
                or not 1 <= self.exif_orientation <= 8
            )
        ):
            raise ImageDecodeError("exif_orientation must be an integer from 1 to 8 or None")

        if self.exif_orientation_applied and self.exif_orientation in (None, 1):
            raise ImageDecodeError(
                "exif_orientation_applied requires a transform orientation from 2 to 8"
            )


@dataclass(frozen=True, slots=True)
class DecodedImage:
    """Canonical RGB pixels paired with source decode metadata."""

    image: np.ndarray
    metadata: ImageDecodeMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray):
            raise ImageDecodeError("decoded image must be a NumPy array")
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ImageDecodeError(
                f"decoded image must have shape (height, width, 3); got {self.image.shape}"
            )
        if self.image.shape[0] <= 0 or self.image.shape[1] <= 0:
            raise ImageDecodeError(
                f"decoded image dimensions must be positive; got {self.image.shape[:2]}"
            )
        if self.image.dtype != np.uint8:
            raise ImageDecodeError(f"decoded image must have dtype uint8; got {self.image.dtype}")
        if not self.image.flags.c_contiguous:
            raise ImageDecodeError("decoded image must be contiguous")
        if not isinstance(self.metadata, ImageDecodeMetadata):
            raise ImageDecodeError("metadata must be an ImageDecodeMetadata instance")
        if self.image.shape[1] != self.metadata.decoded_width:
            raise ImageDecodeError("decoded image width must match metadata")
        if self.image.shape[0] != self.metadata.decoded_height:
            raise ImageDecodeError("decoded image height must match metadata")
