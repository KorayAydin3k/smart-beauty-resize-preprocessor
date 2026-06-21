from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from smart_beauty_resize import (
    ImageDecodeError,
    InvalidImageDimensionsError,
    InvalidImageError,
    InvalidMaskError,
    ResizeConfig,
    ResizeConfigurationError,
    SmartBeautyResizeError,
    decode_image,
    resize_sample,
)


@pytest.fixture()
def sample_rgb_image() -> np.ndarray:
    return np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )


def test_decode_image_rgb_png(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)).save(image_path)

    decoded = decode_image(image_path)

    assert decoded.shape == (1, 2, 3)
    assert decoded.dtype == np.uint8
    assert decoded.flags.c_contiguous


def test_decode_image_converts_grayscale_and_rgba(tmp_path: Path) -> None:
    gray_path = tmp_path / "gray.png"
    Image.fromarray(np.array([[0, 255], [128, 64]], dtype=np.uint8), mode="L").save(gray_path)
    gray = decode_image(gray_path)
    assert gray.shape == (2, 2, 3)
    assert gray.dtype == np.uint8

    rgba_path = tmp_path / "rgba.png"
    Image.fromarray(
        np.array(
            [
                [[255, 0, 0, 255], [0, 255, 0, 255]],
                [[0, 0, 255, 255], [255, 255, 255, 255]],
            ],
            dtype=np.uint8,
        ),
        mode="RGBA",
    ).save(rgba_path)
    rgba = decode_image(rgba_path)
    assert rgba.shape == (2, 2, 3)
    assert rgba.dtype == np.uint8


def test_decode_image_applies_exif_orientation(tmp_path: Path) -> None:
    image_path = tmp_path / "orientation.jpg"
    source = Image.new("RGB", (2, 1))
    source.putpixel((0, 0), (255, 0, 0))
    source.putpixel((1, 0), (0, 255, 0))

    exif = Image.Exif()
    exif[274] = 6
    source.save(image_path, exif=exif, format="JPEG")

    decoded = decode_image(image_path)

    assert decoded.shape == (2, 1, 3)
    assert decoded[0, 0].tolist() != decoded[1, 0].tolist()


def test_decode_image_rejects_missing_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(ImageDecodeError):
        decode_image(tmp_path / "missing.png")

    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"not-a-real-image")
    with pytest.raises(ImageDecodeError):
        decode_image(corrupt_path)


def test_resize_sample_exact_target_shape_and_padding(sample_rgb_image: np.ndarray) -> None:
    config = ResizeConfig(target_width=4, target_height=3, padding_value=(9, 11, 13))
    result = resize_sample(sample_rgb_image, config)

    assert result.image.shape == (3, 4, 3)
    assert result.image.dtype == np.uint8
    assert result.image.flags.c_contiguous
    assert result.valid_region_mask.shape == (3, 4)
    assert set(np.unique(result.valid_region_mask.tolist())).issubset({0, 1})
    assert result.plan.resized_width + result.plan.pad_left + result.plan.pad_right == 4
    assert result.plan.resized_height + result.plan.pad_top + result.plan.pad_bottom == 3

    assert np.all(result.image[:, -1, :] == np.array([9, 11, 13], dtype=np.uint8))
    assert np.all(result.valid_region_mask[:, -1] == 0)
    assert np.all(result.valid_region_mask[:, :3] == 1)


def test_resize_sample_uses_plan_coordinates(sample_rgb_image: np.ndarray) -> None:
    config = ResizeConfig(target_width=2, target_height=2, max_upscale_factor=2.0)
    result = resize_sample(sample_rgb_image, config)

    assert result.plan.resized_width == 2
    assert result.plan.resized_height == 2
    assert result.plan.pad_left == 0
    assert result.plan.pad_top == 0
    assert np.array_equal(result.image, sample_rgb_image)


def test_resize_sample_reports_identity_when_no_resize_needed(
    sample_rgb_image: np.ndarray,
) -> None:
    config = ResizeConfig(target_width=2, target_height=2)
    result = resize_sample(sample_rgb_image, config)

    assert result.interpolation == "IDENTITY"
    assert result.image.shape == (2, 2, 3)
    assert np.array_equal(result.image, sample_rgb_image)


def test_resize_sample_preserves_input_image(sample_rgb_image: np.ndarray) -> None:
    original = sample_rgb_image.copy()
    config = ResizeConfig(target_width=5, target_height=5, max_upscale_factor=4.0)
    resize_sample(sample_rgb_image, config)

    assert np.array_equal(sample_rgb_image, original)


def test_resize_sample_rejects_invalid_direct_inputs() -> None:
    image = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        resize_sample(image, ResizeConfig(target_width=4, target_height=4))

    image_rgb = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(InvalidImageError):
        resize_sample(image_rgb, ResizeConfig(target_width=4, target_height=4))

    image_rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        resize_sample(image_rgba, ResizeConfig(target_width=4, target_height=4))


def test_resize_sample_rejects_invalid_mask_shape_and_dtype(
    sample_rgb_image: np.ndarray,
) -> None:
    config = ResizeConfig(target_width=4, target_height=4, max_upscale_factor=4.0)

    bad_shape = np.zeros((2, 3), dtype=np.uint8)
    with pytest.raises(InvalidMaskError):
        resize_sample(sample_rgb_image, config, mask=bad_shape)

    bad_dtype = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(InvalidMaskError):
        resize_sample(sample_rgb_image, config, mask=bad_dtype)


def test_resize_sample_rejects_bool_and_unsupported_mask_dtypes(
    sample_rgb_image: np.ndarray,
) -> None:
    config = ResizeConfig(target_width=4, target_height=4, max_upscale_factor=4.0)

    bool_mask = np.array([[True, False], [False, True]], dtype=bool)
    with pytest.raises(InvalidMaskError):
        resize_sample(sample_rgb_image, config, mask=bool_mask)

    unsupported = np.zeros((2, 2), dtype=np.uint64)
    with pytest.raises(InvalidMaskError):
        resize_sample(sample_rgb_image, config, mask=unsupported)


def test_resize_sample_with_mask_uses_identical_geometry(sample_rgb_image: np.ndarray) -> None:
    config = ResizeConfig(
        target_width=6,
        target_height=4,
        max_upscale_factor=3.0,
    )
    mask = np.arange(4, dtype=np.uint8).reshape(2, 2)
    result = resize_sample(sample_rgb_image, config, mask=mask)

    assert result.mask is not None
    assert result.mask.shape == (4, 6)
    assert result.mask.dtype == np.uint8
    assert result.plan.pad_left == 1
    assert result.plan.pad_right == 1

    non_padding = result.valid_region_mask == 1
    nonzero_mask = result.mask != 0
    assert np.all(non_padding[nonzero_mask])
    assert np.all(result.mask[:, 0] == 0)
    assert np.all(result.mask[:, -1] == 0)


@settings(deadline=None, max_examples=25)
@given(
    original_width=st.integers(min_value=1, max_value=8),
    original_height=st.integers(min_value=1, max_value=8),
    target_width=st.integers(min_value=1, max_value=8),
    target_height=st.integers(min_value=1, max_value=8),
)
def test_property_resize_sample_shape_and_geometry(
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
) -> None:
    image = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    config = ResizeConfig(
        target_width=target_width,
        target_height=target_height,
        max_upscale_factor=max(
            4.0, max(target_width / original_width, target_height / original_height) * 2
        ),
    )
    result = resize_sample(image, config)

    assert result.image.shape == (target_height, target_width, 3)
    assert result.valid_region_mask.shape == (target_height, target_width)
    assert result.image.dtype == np.uint8
    assert result.valid_region_mask.dtype == np.uint8
    assert result.image.flags.c_contiguous
    assert result.valid_region_mask.flags.c_contiguous
    assert result.plan.resized_width + result.plan.pad_left + result.plan.pad_right == target_width
    assert (
        result.plan.resized_height + result.plan.pad_top + result.plan.pad_bottom == target_height
    )
    assert (
        np.sum(result.valid_region_mask) == result.plan.resized_width * result.plan.resized_height
    )
    assert math.isfinite(result.plan.nominal_scale)
    assert 0.0 <= result.plan.padding_fraction < 1.0


def test_exception_hierarchy_is_semantically_correct() -> None:
    assert issubclass(ResizeConfigurationError, SmartBeautyResizeError)
    assert issubclass(
        InvalidImageDimensionsError,
        ResizeConfigurationError,
    )
    assert issubclass(InvalidImageError, SmartBeautyResizeError)
    assert issubclass(InvalidMaskError, SmartBeautyResizeError)
    assert issubclass(ImageDecodeError, SmartBeautyResizeError)
    assert not issubclass(ImageDecodeError, ResizeConfigurationError)


def test_resize_sample_reports_expected_interpolation_modes() -> None:
    downscale_image = np.zeros((8, 8, 3), dtype=np.uint8)
    downscaled = resize_sample(
        downscale_image,
        ResizeConfig(
            target_width=4,
            target_height=4,
        ),
    )
    assert downscaled.interpolation == "INTER_AREA"

    upscale_image = np.zeros((2, 2, 3), dtype=np.uint8)
    upscaled = resize_sample(
        upscale_image,
        ResizeConfig(
            target_width=4,
            target_height=4,
            max_upscale_factor=2.0,
        ),
    )
    assert upscaled.interpolation == "INTER_CUBIC"

    identity = resize_sample(
        upscale_image,
        ResizeConfig(
            target_width=2,
            target_height=2,
        ),
    )
    assert identity.interpolation == "IDENTITY"


@pytest.mark.parametrize(
    "dtype_name",
    ["uint8", "uint16", "int16"],
)
def test_supported_mask_dtypes_preserve_dtype_and_labels(
    sample_rgb_image: np.ndarray,
    dtype_name: str,
) -> None:
    dtype = np.dtype(dtype_name)
    mask = np.array(
        [
            [0, 1],
            [2, 3],
        ],
        dtype=dtype,
    )

    result = resize_sample(
        sample_rgb_image,
        ResizeConfig(
            target_width=6,
            target_height=4,
            max_upscale_factor=3.0,
        ),
        mask=mask,
    )

    assert result.mask is not None
    assert result.mask.dtype == dtype
    assert set(np.unique(result.mask).tolist()).issubset({0, 1, 2, 3})


@pytest.mark.parametrize(
    "dtype_name",
    ["int8", "int32", "uint32", "uint64", "float32"],
)
def test_unsupported_mask_dtypes_are_rejected(
    sample_rgb_image: np.ndarray,
    dtype_name: str,
) -> None:
    mask = np.zeros(
        (2, 2),
        dtype=np.dtype(dtype_name),
    )

    with pytest.raises(InvalidMaskError):
        resize_sample(
            sample_rgb_image,
            ResizeConfig(
                target_width=4,
                target_height=4,
                max_upscale_factor=2.0,
            ),
            mask=mask,
        )


def test_identity_result_does_not_share_memory_with_inputs() -> None:
    image = np.full(
        (4, 4, 3),
        90,
        dtype=np.uint8,
    )
    mask = np.full(
        (4, 4),
        3,
        dtype=np.uint8,
    )

    image_before = image.copy()
    mask_before = mask.copy()

    result = resize_sample(
        image=image,
        mask=mask,
        config=ResizeConfig(
            target_width=4,
            target_height=4,
            allow_upscale=False,
        ),
    )

    assert result.mask is not None
    assert result.interpolation == "IDENTITY"
    assert not np.shares_memory(result.image, image)
    assert not np.shares_memory(result.mask, mask)

    result.image[0, 0] = (1, 2, 3)
    result.mask[0, 0] = 7

    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)


def test_valid_region_sum_equals_resized_area(
    sample_rgb_image: np.ndarray,
) -> None:
    result = resize_sample(
        sample_rgb_image,
        ResizeConfig(
            target_width=7,
            target_height=5,
            max_upscale_factor=4.0,
        ),
    )

    assert int(result.valid_region_mask.sum()) == (
        result.plan.resized_width * result.plan.resized_height
    )
