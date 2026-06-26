from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageMode, ImageOps, UnidentifiedImageError

from smart_beauty_resize.contracts import ImageDecodeError
from smart_beauty_resize.io.contracts import (
    DecodedImage,
    ImageDecodeMetadata,
    SourceImageLimits,
)
from smart_beauty_resize.io.limits import enforce_source_image_limits

_EXIF_ORIENTATION_TAG = 274
_TRANSFORMING_EXIF_ORIENTATIONS = frozenset({2, 3, 4, 5, 6, 7, 8})
_RAW_MODE_DEPTH_PATTERN = re.compile(r";(\d+)")


def _source_bit_depth(image: Image.Image) -> int | None:
    """Return source sample depth using raw decoder metadata when available."""
    if image.mode == "1":
        return 1

    for tile in getattr(image, "tile", ()):
        raw_mode = getattr(tile, "args", None)
        if raw_mode is None:
            try:
                raw_mode = tile[3]
            except (IndexError, TypeError):
                continue

        if isinstance(raw_mode, str):
            match = _RAW_MODE_DEPTH_PATTERN.search(raw_mode)
            if match is not None:
                return int(match.group(1))

    try:
        descriptor = ImageMode.getmode(image.mode)
        return int(np.dtype(descriptor.typestr).itemsize * 8)
    except (KeyError, TypeError, ValueError):
        return None


def _read_exif_orientation(image: Image.Image) -> int | None:
    orientation = image.getexif().get(_EXIF_ORIENTATION_TAG)
    if type(orientation) is int and 1 <= orientation <= 8:
        return orientation
    return None


def _decode_canonical_rgb(
    path: Path,
    source_limits: SourceImageLimits,
) -> DecodedImage:
    try:
        with Image.open(path) as image_file:
            source_format = (image_file.format or "UNKNOWN").upper()
            source_mode = image_file.mode
            source_width, source_height = image_file.size
            enforce_source_image_limits(
                width=source_width,
                height=source_height,
                limits=source_limits,
            )
            source_channel_count = len(image_file.getbands())
            source_bit_depth = _source_bit_depth(image_file)
            alpha_present = (
                "A" in image_file.getbands() or "transparency" in image_file.info
            )
            icc_profile_present = bool(image_file.info.get("icc_profile"))
            exif_orientation = _read_exif_orientation(image_file)

            oriented_image = ImageOps.exif_transpose(image_file)
            rgb_conversion_applied = oriented_image.mode != "RGB"
            rgb_image = oriented_image.convert("RGB")
            rgb_image.load()
            decoded = np.array(
                rgb_image,
                dtype=np.uint8,
                copy=True,
                order="C",
            )

            metadata = ImageDecodeMetadata(
                source_format=source_format,
                source_mode=source_mode,
                source_width=source_width,
                source_height=source_height,
                decoded_width=rgb_image.width,
                decoded_height=rgb_image.height,
                source_bit_depth=source_bit_depth,
                source_channel_count=source_channel_count,
                alpha_present=alpha_present,
                icc_profile_present=icc_profile_present,
                exif_orientation=exif_orientation,
                exif_orientation_applied=(
                    exif_orientation in _TRANSFORMING_EXIF_ORIENTATIONS
                ),
                rgb_conversion_applied=rgb_conversion_applied,
                bit_depth_conversion_applied=(source_bit_depth not in (None, 8)),
            )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ImageDecodeError(f"unable to decode image file: {path}") from exc

    decoded = np.ascontiguousarray(decoded)
    return DecodedImage(image=decoded, metadata=metadata)


def decode_image_with_metadata(
    path: str | Path,
    *,
    source_limits: SourceImageLimits | None = None,
) -> DecodedImage:
    """Decode an image into canonical RGB pixels plus source audit metadata."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ImageDecodeError(f"image file does not exist: {file_path}")

    resolved_limits = SourceImageLimits() if source_limits is None else source_limits
    if not isinstance(resolved_limits, SourceImageLimits):
        raise ImageDecodeError("source_limits must be a SourceImageLimits instance or None")

    return _decode_canonical_rgb(file_path, resolved_limits)


def decode_image(path: str | Path) -> np.ndarray:
    """Decode an image file into a contiguous RGB uint8 NumPy array.

    The decoder applies EXIF orientation correction before converting to RGB.
    The output is always shaped as ``(height, width, 3)`` with ``uint8`` dtype.
    Existing callers retain the same return type; use
    :func:`decode_image_with_metadata` when source audit fields are required.
    """
    return decode_image_with_metadata(path).image
