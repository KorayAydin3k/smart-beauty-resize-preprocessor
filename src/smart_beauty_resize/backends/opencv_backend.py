from __future__ import annotations

import cv2
import numpy as np

from smart_beauty_resize.contracts import (
    InvalidImageError,
    InvalidMaskError,
    ResizeConfig,
    ResizeConfigurationError,
    ResizeResult,
    SmartBeautyResizeError,
)
from smart_beauty_resize.geometry import calculate_letterbox_plan

# OpenCV geometric transforms support these integer depths safely.
# CV_8S (int8) and CV_32S (int32) are intentionally excluded.
_SUPPORTED_MASK_DTYPES = frozenset(
    (
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.int16),
    )
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmartBeautyResizeError(message)


def _validate_rgb_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise InvalidImageError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError("image must have shape (height, width, 3) with RGB channels")
    if image.dtype != np.uint8:
        raise InvalidImageError("image must have dtype uint8")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise InvalidImageError("image dimensions must be positive")
    if not np.isfinite(image).all():
        raise InvalidImageError("image must be finite")


def _validate_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> None:
    if not isinstance(mask, np.ndarray):
        raise InvalidMaskError("mask must be a NumPy array")
    if mask.ndim != 2:
        raise InvalidMaskError("mask must be a 2D array")
    if mask.dtype.kind == "b":
        raise InvalidMaskError("mask must not be boolean")
    if mask.dtype.kind not in ("i", "u"):
        raise InvalidMaskError("mask must have a supported integer dtype: uint8, uint16, int16")
    if mask.dtype not in _SUPPORTED_MASK_DTYPES:
        raise InvalidMaskError("mask dtype must be one of uint8, uint16, int16")
    if mask.shape != image_shape:
        raise InvalidMaskError("mask shape must match the image shape exactly")
    if mask.shape[0] <= 0 or mask.shape[1] <= 0:
        raise InvalidMaskError("mask dimensions must be positive")


def resize_sample(
    image: np.ndarray,
    config: ResizeConfig,
    mask: np.ndarray | None = None,
) -> ResizeResult:
    """Resize an RGB image to a deterministic letterbox canvas.

    The backend uses OpenCV for the pixel operations but keeps the geometry
    contract in the shared pure-Python planning code.
    """
    _validate_rgb_image(image)
    if not isinstance(config, ResizeConfig):
        raise ResizeConfigurationError("config must be a ResizeConfig instance")

    if mask is not None:
        _validate_mask(mask, (image.shape[0], image.shape[1]))

    plan = calculate_letterbox_plan(
        original_width=image.shape[1],
        original_height=image.shape[0],
        config=config,
    )

    output = np.empty(
        (config.target_height, config.target_width, 3),
        dtype=np.uint8,
    )
    output[:] = np.asarray(config.padding_value, dtype=np.uint8)

    if plan.resized_width == image.shape[1] and plan.resized_height == image.shape[0]:
        resized_image = np.array(image, copy=True, order="C")
        interpolation_name = "IDENTITY"
    else:
        if plan.resized_width < image.shape[1] or plan.resized_height < image.shape[0]:
            interpolation = cv2.INTER_AREA
            interpolation_name = "INTER_AREA"
        else:
            interpolation = cv2.INTER_CUBIC
            interpolation_name = "INTER_CUBIC"
        resized_image = cv2.resize(
            image,
            (plan.resized_width, plan.resized_height),
            interpolation=interpolation,
        )

    output[
        plan.pad_top : plan.pad_top + plan.resized_height,
        plan.pad_left : plan.pad_left + plan.resized_width,
    ] = resized_image

    valid_region_mask = np.zeros(
        (config.target_height, config.target_width),
        dtype=np.uint8,
    )
    valid_region_mask[
        plan.pad_top : plan.pad_top + plan.resized_height,
        plan.pad_left : plan.pad_left + plan.resized_width,
    ] = 1

    result_mask: np.ndarray | None = None
    if mask is not None:
        resized_mask = cv2.resize(
            mask,
            (plan.resized_width, plan.resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        output_mask = np.zeros(
            (config.target_height, config.target_width),
            dtype=mask.dtype,
        )
        output_mask[
            plan.pad_top : plan.pad_top + plan.resized_height,
            plan.pad_left : plan.pad_left + plan.resized_width,
        ] = resized_mask

        input_labels = set(np.unique(mask).tolist())
        output_labels = set(np.unique(output_mask).tolist())
        _require(
            output_labels.issubset(input_labels | {0}),
            "transformed mask labels must be a subset of the input labels plus zero",
        )
        result_mask = np.ascontiguousarray(output_mask)

    _require(
        output.shape == (config.target_height, config.target_width, 3),
        "output image shape invariant failed",
    )
    _require(output.dtype == np.uint8, "output image dtype invariant failed")
    _require(
        valid_region_mask.shape == (config.target_height, config.target_width),
        "valid-region shape invariant failed",
    )
    _require(valid_region_mask.dtype == np.uint8, "valid-region dtype invariant failed")
    _require(
        bool(np.all((valid_region_mask == 0) | (valid_region_mask == 1))),
        "valid-region values must be limited to {0, 1}",
    )
    _require(
        int(np.sum(valid_region_mask)) == plan.resized_width * plan.resized_height,
        "valid-region sum invariant failed",
    )
    _require(output.flags.c_contiguous, "output image must be contiguous")
    _require(valid_region_mask.flags.c_contiguous, "valid-region mask must be contiguous")
    if result_mask is not None:
        if mask is None:
            raise SmartBeautyResizeError(
                "internal invariant failed: transformed mask exists without input mask"
            )
        _require(
            result_mask.shape == (config.target_height, config.target_width),
            "transformed mask shape invariant failed",
        )
        _require(
            result_mask.dtype == mask.dtype,
            "transformed mask dtype invariant failed",
        )
        _require(
            result_mask.flags.c_contiguous,
            "transformed mask must be contiguous",
        )

    return ResizeResult(
        image=np.ascontiguousarray(output),
        plan=plan,
        valid_region_mask=valid_region_mask,
        mask=result_mask,
        interpolation=interpolation_name,
    )
