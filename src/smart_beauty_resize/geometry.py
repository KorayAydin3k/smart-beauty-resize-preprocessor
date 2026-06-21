from __future__ import annotations

import math

from smart_beauty_resize.contracts import (
    ExcessiveUpscaleError,
    InvalidImageDimensionsError,
    Matrix,
    ResizeConfig,
    ResizePlan,
)

# Absolute tolerance for floating-point noise when comparing the requested
# upscale against the configured limit.
UPSCALE_TOLERANCE = 1e-12


def round_half_up_positive(value: float) -> int:
    """Round a finite non-negative value using explicit half-up semantics."""
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError("value must be a finite non-negative real number")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("value must be a finite non-negative real number")
    if numeric_value < 0.0:
        raise ValueError("value must be a finite non-negative real number")

    return int(math.floor(numeric_value + 0.5))


def _validate_positive_int(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise InvalidImageDimensionsError(f"{name} must be a positive integer")


def calculate_letterbox_plan(
    original_width: int,
    original_height: int,
    config: ResizeConfig,
) -> ResizePlan:
    """Compute deterministic letterbox geometry for original dimensions."""
    _validate_positive_int(original_width, "original_width")
    _validate_positive_int(original_height, "original_height")

    if not isinstance(config, ResizeConfig):
        raise TypeError("config must be a ResizeConfig instance")

    nominal_scale = min(
        config.target_width / original_width,
        config.target_height / original_height,
    )
    if not config.allow_upscale:
        nominal_scale = min(nominal_scale, 1.0)
    elif nominal_scale > config.max_upscale_factor + UPSCALE_TOLERANCE:
        raise ExcessiveUpscaleError("required upscale exceeds configured max_upscale_factor")

    resized_width = round_half_up_positive(original_width * nominal_scale)
    resized_height = round_half_up_positive(original_height * nominal_scale)
    resized_width = min(config.target_width, max(1, resized_width))
    resized_height = min(config.target_height, max(1, resized_height))

    extra_width = config.target_width - resized_width
    extra_height = config.target_height - resized_height

    pad_left = extra_width // 2
    pad_right = extra_width - pad_left
    pad_top = extra_height // 2
    pad_bottom = extra_height - pad_top

    assert pad_left >= 0 and pad_right >= 0
    assert pad_top >= 0 and pad_bottom >= 0
    assert resized_width + pad_left + pad_right == config.target_width
    assert resized_height + pad_top + pad_bottom == config.target_height

    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    if not (math.isfinite(scale_x) and math.isfinite(scale_y)):
        raise ValueError("computed scales are not finite")

    forward_matrix: Matrix = (
        (scale_x, 0.0, float(pad_left)),
        (0.0, scale_y, float(pad_top)),
        (0.0, 0.0, 1.0),
    )
    inverse_matrix: Matrix = (
        (1.0 / scale_x, 0.0, -float(pad_left) / scale_x),
        (0.0, 1.0 / scale_y, -float(pad_top) / scale_y),
        (0.0, 0.0, 1.0),
    )

    if not all(math.isfinite(value) for row in forward_matrix for value in row):
        raise ValueError("forward matrix contains non-finite values")
    if not all(math.isfinite(value) for row in inverse_matrix for value in row):
        raise ValueError("inverse matrix contains non-finite values")

    target_area = config.target_width * config.target_height
    resized_area = resized_width * resized_height
    padding_fraction = 1.0 - (resized_area / target_area)

    if not math.isfinite(padding_fraction):
        raise ValueError("padding_fraction is not finite")
    if padding_fraction < 0.0 or padding_fraction >= 1.0:
        raise ValueError("padding_fraction must be in [0, 1)")

    return ResizePlan(
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        resized_height=resized_height,
        target_width=config.target_width,
        target_height=config.target_height,
        nominal_scale=nominal_scale,
        scale_x=scale_x,
        scale_y=scale_y,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        padding_fraction=padding_fraction,
        was_upscaled=(resized_width > original_width or resized_height > original_height),
        forward_matrix=forward_matrix,
        inverse_matrix=inverse_matrix,
    )


def apply_matrix_to_point(
    matrix: Matrix,
    x: float,
    y: float,
) -> tuple[float, float]:
    """Apply an affine matrix to a point in 2D space."""
    row0, row1, _ = matrix
    transformed_x = row0[0] * x + row0[1] * y + row0[2]
    transformed_y = row1[0] * x + row1[1] * y + row1[2]
    return transformed_x, transformed_y
