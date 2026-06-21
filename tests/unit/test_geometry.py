from __future__ import annotations

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from smart_beauty_resize import (
    ExcessiveUpscaleError,
    InvalidImageDimensionsError,
    ResizeConfig,
    ResizeConfigurationError,
    apply_matrix_to_point,
    calculate_letterbox_plan,
    round_half_up_positive,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, 0),
        (0.49, 0),
        (0.50, 1),
        (1.49, 1),
        (1.50, 2),
        (2.51, 3),
    ],
)
def test_round_half_up_positive_examples(value: float, expected: int) -> None:
    assert round_half_up_positive(value) == expected


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_round_half_up_positive_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError):
        round_half_up_positive(value)


@pytest.mark.parametrize(
    ("original_width", "original_height", "target_width", "target_height", "expected"),
    [
        (1920, 1080, 640, 360, (640, 360)),
        (1080, 1920, 360, 640, (360, 640)),
        (1000, 1000, 500, 500, (500, 500)),
        (320, 240, 320, 240, (320, 240)),
    ],
)
def test_letterbox_plan_examples(
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
    expected: tuple[int, int],
) -> None:
    config = ResizeConfig(target_width=target_width, target_height=target_height)
    plan = calculate_letterbox_plan(
        original_width=original_width,
        original_height=original_height,
        config=config,
    )

    assert (plan.resized_width, plan.resized_height) == expected
    assert plan.pad_left + plan.resized_width + plan.pad_right == target_width
    assert plan.pad_top + plan.resized_height + plan.pad_bottom == target_height


def test_upscaling_disabled_caps_scale_at_one() -> None:
    config = ResizeConfig(
        target_width=300,
        target_height=150,
        allow_upscale=False,
    )
    plan = calculate_letterbox_plan(200, 100, config)

    assert plan.nominal_scale == 1.0
    assert plan.resized_width == 200
    assert plan.resized_height == 100
    assert plan.pad_left == 50
    assert plan.pad_top == 25


def test_upscaling_enabled_allows_larger_output() -> None:
    config = ResizeConfig(target_width=300, target_height=150, allow_upscale=True)
    plan = calculate_letterbox_plan(200, 100, config)

    assert plan.nominal_scale == 1.5
    assert plan.resized_width == 300
    assert plan.resized_height == 150
    assert plan.pad_left == 0
    assert plan.pad_top == 0


def test_excessive_upscale_rejected() -> None:
    config = ResizeConfig(
        target_width=400,
        target_height=400,
        allow_upscale=True,
        max_upscale_factor=1.5,
    )

    with pytest.raises(ExcessiveUpscaleError):
        calculate_letterbox_plan(100, 100, config)


@pytest.mark.parametrize(
    ("original_width", "original_height"),
    [
        (0, 100),
        (100, 0),
        (-1, 100),
        (100, -1),
    ],
)
def test_invalid_original_dimensions_raise(
    original_width: int,
    original_height: int,
) -> None:
    config = ResizeConfig(target_width=64, target_height=64)

    with pytest.raises(InvalidImageDimensionsError):
        calculate_letterbox_plan(original_width, original_height, config)


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, InvalidImageDimensionsError),
        (False, InvalidImageDimensionsError),
        (0, InvalidImageDimensionsError),
        (-1, InvalidImageDimensionsError),
        (1.5, InvalidImageDimensionsError),
        ("64", InvalidImageDimensionsError),
    ],
)
def test_invalid_target_dimensions_raise_specific_error(
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        ResizeConfig(target_width=value, target_height=64)

    with pytest.raises(error_type):
        ResizeConfig(target_width=64, target_height=value)


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, ResizeConfigurationError),
        (False, ResizeConfigurationError),
        (0.0, ResizeConfigurationError),
        (0.5, ResizeConfigurationError),
        (float("nan"), ResizeConfigurationError),
        (float("inf"), ResizeConfigurationError),
        (float("-inf"), ResizeConfigurationError),
    ],
)
def test_invalid_max_upscale_factor_raises_specific_error(
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        ResizeConfig(
            target_width=64,
            target_height=64,
            max_upscale_factor=value,
        )


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        ((127, 127, 127), None),
        ((True, 127, 127), ResizeConfigurationError),
        ((127, False, 127), ResizeConfigurationError),
        ((127, 127, True), ResizeConfigurationError),
        ((127, 127, 300), ResizeConfigurationError),
        ((127, 127, -1), ResizeConfigurationError),
        ([127, 127, 127], ResizeConfigurationError),
    ],
)
def test_invalid_padding_values_raise_specific_error(
    value: object,
    error_type: type[Exception] | None,
) -> None:
    if error_type is None:
        ResizeConfig(
            target_width=64,
            target_height=64,
            padding_value=value,
        )
        return

    with pytest.raises(error_type):
        ResizeConfig(
            target_width=64,
            target_height=64,
            padding_value=value,
        )


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, InvalidImageDimensionsError),
        (False, InvalidImageDimensionsError),
        (0, InvalidImageDimensionsError),
        (-1, InvalidImageDimensionsError),
        (1.5, InvalidImageDimensionsError),
        ("64", InvalidImageDimensionsError),
    ],
)
def test_invalid_original_dimensions_raise_specific_error(
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        calculate_letterbox_plan(value, 64, ResizeConfig(target_width=64, target_height=64))

    with pytest.raises(error_type):
        calculate_letterbox_plan(64, value, ResizeConfig(target_width=64, target_height=64))


def test_odd_padding_goes_to_right_and_bottom() -> None:
    config = ResizeConfig(
        target_width=10,
        target_height=7,
        max_upscale_factor=2.0,
    )
    plan = calculate_letterbox_plan(5, 4, config)

    assert plan.resized_width == 9
    assert plan.resized_height == 7
    assert plan.pad_left == 0
    assert plan.pad_right == 1
    assert plan.pad_top == 0
    assert plan.pad_bottom == 0


def test_padding_fraction_is_non_negative_and_less_than_one() -> None:
    config = ResizeConfig(
        target_width=8,
        target_height=7,
        max_upscale_factor=2.0,
    )
    plan = calculate_letterbox_plan(5, 4, config)

    assert 0.0 <= plan.padding_fraction < 1.0


def test_forward_matrix_maps_points() -> None:
    config = ResizeConfig(
        target_width=10,
        target_height=8,
        max_upscale_factor=4.0,
    )
    plan = calculate_letterbox_plan(4, 2, config)

    x, y = apply_matrix_to_point(plan.forward_matrix, 2.0, 1.0)

    assert math.isclose(x, plan.scale_x * 2.0 + plan.pad_left)
    assert math.isclose(y, plan.scale_y * 1.0 + plan.pad_top)


def test_inverse_matrix_maps_points_back() -> None:
    config = ResizeConfig(
        target_width=10,
        target_height=8,
        max_upscale_factor=4.0,
    )
    plan = calculate_letterbox_plan(4, 2, config)

    x, y = apply_matrix_to_point(plan.inverse_matrix, 6.0, 4.0)

    assert math.isclose(x, (6.0 - plan.pad_left) / plan.scale_x)
    assert math.isclose(y, (4.0 - plan.pad_top) / plan.scale_y)


def test_forward_then_inverse_round_trip() -> None:
    config = ResizeConfig(
        target_width=10,
        target_height=8,
        max_upscale_factor=4.0,
    )
    plan = calculate_letterbox_plan(4, 2, config)

    x, y = apply_matrix_to_point(plan.forward_matrix, 2.5, 1.5)
    x2, y2 = apply_matrix_to_point(plan.inverse_matrix, x, y)

    assert math.isclose(x2, 2.5, abs_tol=1e-12)
    assert math.isclose(y2, 1.5, abs_tol=1e-12)


def test_repeated_calculation_is_deterministic() -> None:
    config = ResizeConfig(
        target_width=10,
        target_height=8,
        max_upscale_factor=2.0,
    )
    first = calculate_letterbox_plan(5, 4, config)
    second = calculate_letterbox_plan(5, 4, config)

    assert first == second


@given(
    original_width=st.integers(min_value=1, max_value=1024),
    original_height=st.integers(min_value=1, max_value=1024),
    target_width=st.integers(min_value=1, max_value=1024),
    target_height=st.integers(min_value=1, max_value=1024),
    allow_upscale=st.booleans(),
)
def test_property_letterbox_plan_is_valid(
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
    allow_upscale: bool,
) -> None:
    assume(original_width > 0 and original_height > 0)
    assume(target_width > 0 and target_height > 0)

    max_scale = max(
        target_width / original_width,
        target_height / original_height,
    )
    config = ResizeConfig(
        target_width=target_width,
        target_height=target_height,
        allow_upscale=allow_upscale,
        max_upscale_factor=max(1.0, max_scale * 1.1),
    )

    plan = calculate_letterbox_plan(original_width, original_height, config)

    assert plan.resized_width > 0
    assert plan.resized_height > 0
    assert plan.pad_left >= 0
    assert plan.pad_top >= 0
    assert plan.pad_right >= 0
    assert plan.pad_bottom >= 0
    assert plan.resized_width <= target_width
    assert plan.resized_height <= target_height
    assert plan.resized_width + plan.pad_left + plan.pad_right == target_width
    assert plan.resized_height + plan.pad_top + plan.pad_bottom == target_height
    assert 0.0 <= plan.padding_fraction < 1.0
    assert math.isfinite(plan.nominal_scale)
    assert math.isfinite(plan.scale_x)
    assert math.isfinite(plan.scale_y)
    assert all(math.isfinite(value) for row in plan.forward_matrix for value in row)
    assert all(math.isfinite(value) for row in plan.inverse_matrix for value in row)
    assert math.isfinite(plan.padding_fraction)

    repeated = calculate_letterbox_plan(original_width, original_height, config)
    assert repeated == plan

    x, y = apply_matrix_to_point(plan.forward_matrix, 0.0, 0.0)
    x2, y2 = apply_matrix_to_point(plan.inverse_matrix, x, y)
    assert math.isclose(x2, 0.0, abs_tol=1e-9)
    assert math.isclose(y2, 0.0, abs_tol=1e-9)

    ratio_bound = 1.0 / min(original_width, original_height)
    assert abs(plan.scale_x - plan.nominal_scale) <= ratio_bound
    assert abs(plan.scale_y - plan.nominal_scale) <= ratio_bound
