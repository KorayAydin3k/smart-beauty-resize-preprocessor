from __future__ import annotations

import pytest

from smart_beauty_resize import (
    ImageDecodeError,
    ImageDecodeMetadata,
    InputPolicy,
    InputPolicyViolationError,
    enforce_input_policy,
)


def _metadata(
    *,
    source_mode: str = "RGB",
    source_bit_depth: int | None = 8,
    source_channel_count: int = 3,
    alpha_present: bool = False,
) -> ImageDecodeMetadata:
    return ImageDecodeMetadata(
        source_format="PNG",
        source_mode=source_mode,
        source_width=64,
        source_height=64,
        decoded_width=64,
        decoded_height=64,
        source_bit_depth=source_bit_depth,
        source_channel_count=source_channel_count,
        alpha_present=alpha_present,
        icc_profile_present=False,
        exif_orientation=None,
        exif_orientation_applied=False,
        rgb_conversion_applied=source_mode != "RGB",
        bit_depth_conversion_applied=source_bit_depth not in (None, 8),
    )


def test_input_policy_values_are_stable() -> None:
    assert InputPolicy.AUDIT_ONLY.value == "audit_only"
    assert InputPolicy.STRICT_RGB8.value == "strict_rgb8"


def test_audit_only_accepts_convertible_rgba_source() -> None:
    enforce_input_policy(
        _metadata(
            source_mode="RGBA",
            source_channel_count=4,
            alpha_present=True,
        ),
        InputPolicy.AUDIT_ONLY,
    )


def test_strict_rgb8_accepts_native_rgb8_source() -> None:
    enforce_input_policy(
        _metadata(),
        InputPolicy.STRICT_RGB8,
    )


def test_strict_rgb8_rejects_rgba_source_with_structured_reason() -> None:
    with pytest.raises(
        InputPolicyViolationError,
        match=(
            r"source_mode=RGBA.*source_channel_count=4.*"
            r"alpha_present=true"
        ),
    ):
        enforce_input_policy(
            _metadata(
                source_mode="RGBA",
                source_channel_count=4,
                alpha_present=True,
            ),
            InputPolicy.STRICT_RGB8,
        )


def test_strict_rgb8_rejects_unknown_bit_depth() -> None:
    with pytest.raises(
        InputPolicyViolationError,
        match=r"source_bit_depth=None",
    ):
        enforce_input_policy(
            _metadata(source_bit_depth=None),
            InputPolicy.STRICT_RGB8,
        )


def test_policy_api_rejects_invalid_contract_inputs() -> None:
    with pytest.raises(ImageDecodeError, match="metadata"):
        enforce_input_policy(  # type: ignore[arg-type]
            object(),
            InputPolicy.AUDIT_ONLY,
        )

    with pytest.raises(ImageDecodeError, match="policy"):
        enforce_input_policy(  # type: ignore[arg-type]
            _metadata(),
            "audit_only",
        )
