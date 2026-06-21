from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from smart_beauty_resize.contracts import ImageDecodeError


def decode_image(path: str | Path) -> np.ndarray:
    """Decode an image file into a contiguous RGB uint8 NumPy array.

    The decoder applies EXIF orientation correction before converting to RGB.
    The output is always shaped as ``(height, width, 3)`` with ``uint8`` dtype.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ImageDecodeError(f"image file does not exist: {file_path}")

    try:
        with Image.open(file_path) as image_file:
            image = ImageOps.exif_transpose(image_file)
            rgb_image = image.convert("RGB")
            rgb_image.load()
            decoded = np.array(
                rgb_image,
                dtype=np.uint8,
                copy=True,
                order="C",
            )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ImageDecodeError(f"unable to decode image file: {file_path}") from exc

    if decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ImageDecodeError(
            f"decoded image must have shape (height, width, 3); got {decoded.shape}"
        )
    if decoded.shape[0] <= 0 or decoded.shape[1] <= 0:
        raise ImageDecodeError(
            f"decoded image dimensions must be positive; got {decoded.shape[:2]}"
        )
    decoded = np.ascontiguousarray(decoded)
    if decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ImageDecodeError(
            f"decoded image must have shape (height, width, 3); got {decoded.shape}"
        )
    if decoded.shape[0] <= 0 or decoded.shape[1] <= 0:
        raise ImageDecodeError(
            f"decoded image dimensions must be positive; got {decoded.shape[:2]}"
        )
    if decoded.dtype != np.uint8:
        raise ImageDecodeError(f"decoded image must have dtype uint8; got {decoded.dtype}")
    if not decoded.flags.c_contiguous:
        raise ImageDecodeError("decoded image must be contiguous")

    return decoded
