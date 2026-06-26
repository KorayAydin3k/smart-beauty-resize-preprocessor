from __future__ import annotations

from smart_beauty_resize.contracts import ImageDecodeError, SourceImageLimitError
from smart_beauty_resize.io.contracts import SourceImageLimits


def enforce_source_image_limits(
    *,
    width: int,
    height: int,
    limits: SourceImageLimits,
) -> None:
    """Reject source dimensions that exceed configured pre-decode limits."""
    if type(width) is not int or width <= 0:
        raise ImageDecodeError("source width must be a positive integer")
    if type(height) is not int or height <= 0:
        raise ImageDecodeError("source height must be a positive integer")
    if not isinstance(limits, SourceImageLimits):
        raise ImageDecodeError("source limits must be a SourceImageLimits instance")

    pixel_count = width * height
    violations: list[str] = []

    if limits.max_width is not None and width > limits.max_width:
        violations.append(f"width {width} > max_width {limits.max_width}")
    if limits.max_height is not None and height > limits.max_height:
        violations.append(f"height {height} > max_height {limits.max_height}")
    if limits.max_pixels is not None and pixel_count > limits.max_pixels:
        violations.append(f"pixels {pixel_count} > max_pixels {limits.max_pixels}")

    if violations:
        details = "; ".join(violations)
        raise SourceImageLimitError(f"source image exceeds configured limits: {details}")
