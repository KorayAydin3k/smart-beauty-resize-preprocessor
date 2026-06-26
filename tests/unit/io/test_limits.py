from __future__ import annotations

import pytest

from smart_beauty_resize import (
    ImageDecodeError,
    SourceImageLimitError,
    SourceImageLimits,
    enforce_source_image_limits,
)


@pytest.mark.parametrize("field", ["max_width", "max_height", "max_pixels"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "100"])
def test_source_image_limits_reject_invalid_values(
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}

    with pytest.raises(ImageDecodeError, match=field):
        SourceImageLimits(**kwargs)  # type: ignore[arg-type]


def test_unlimited_source_image_limits_accept_any_positive_dimensions() -> None:
    limits = SourceImageLimits()

    enforce_source_image_limits(
        width=50000,
        height=40000,
        limits=limits,
    )

    assert limits.enabled is False


def test_exact_source_image_limit_boundaries_are_accepted() -> None:
    limits = SourceImageLimits(
        max_width=20,
        max_height=10,
        max_pixels=200,
    )

    enforce_source_image_limits(
        width=20,
        height=10,
        limits=limits,
    )

    assert limits.enabled is True


@pytest.mark.parametrize(
    ("width", "height", "limits", "message"),
    [
        (21, 10, SourceImageLimits(max_width=20), "width 21 > max_width 20"),
        (20, 11, SourceImageLimits(max_height=10), "height 11 > max_height 10"),
        (20, 11, SourceImageLimits(max_pixels=200), "pixels 220 > max_pixels 200"),
    ],
)
def test_source_image_limit_violations_are_structured(
    width: int,
    height: int,
    limits: SourceImageLimits,
    message: str,
) -> None:
    with pytest.raises(SourceImageLimitError, match=message):
        enforce_source_image_limits(
            width=width,
            height=height,
            limits=limits,
        )


def test_multiple_source_image_limit_violations_use_deterministic_order() -> None:
    limits = SourceImageLimits(
        max_width=10,
        max_height=10,
        max_pixels=100,
    )

    with pytest.raises(SourceImageLimitError) as captured:
        enforce_source_image_limits(
            width=20,
            height=30,
            limits=limits,
        )

    assert str(captured.value) == (
        "source image exceeds configured limits: "
        "width 20 > max_width 10; "
        "height 30 > max_height 10; "
        "pixels 600 > max_pixels 100"
    )
