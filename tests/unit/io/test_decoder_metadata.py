from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from smart_beauty_resize import (
    DecodedImage,
    ImageDecodeError,
    ImageDecodeMetadata,
    decode_image,
    decode_image_with_metadata,
)


def test_rgb_png_reports_source_metadata_without_conversion(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    source = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(source, mode="RGB").save(image_path)

    result = decode_image_with_metadata(image_path)

    assert isinstance(result, DecodedImage)
    assert np.array_equal(result.image, source)
    assert result.metadata == ImageDecodeMetadata(
        source_format="PNG",
        source_mode="RGB",
        source_width=3,
        source_height=2,
        decoded_width=3,
        decoded_height=2,
        source_bit_depth=8,
        source_channel_count=3,
        alpha_present=False,
        icc_profile_present=False,
        exif_orientation=None,
        exif_orientation_applied=False,
        rgb_conversion_applied=False,
        bit_depth_conversion_applied=False,
    )


def test_grayscale_and_rgba_conversions_are_observable(tmp_path: Path) -> None:
    gray_path = tmp_path / "gray.png"
    Image.fromarray(np.array([[0, 255]], dtype=np.uint8), mode="L").save(gray_path)

    gray = decode_image_with_metadata(gray_path)

    assert gray.image.shape == (1, 2, 3)
    assert gray.metadata.source_mode == "L"
    assert gray.metadata.source_channel_count == 1
    assert gray.metadata.source_bit_depth == 8
    assert gray.metadata.alpha_present is False
    assert gray.metadata.rgb_conversion_applied is True
    assert gray.metadata.bit_depth_conversion_applied is False

    rgba_path = tmp_path / "rgba.png"
    Image.fromarray(
        np.array([[[255, 0, 0, 0], [0, 255, 0, 255]]], dtype=np.uint8),
        mode="RGBA",
    ).save(rgba_path)

    rgba = decode_image_with_metadata(rgba_path)

    assert rgba.image.shape == (1, 2, 3)
    assert rgba.metadata.source_mode == "RGBA"
    assert rgba.metadata.source_channel_count == 4
    assert rgba.metadata.alpha_present is True
    assert rgba.metadata.rgb_conversion_applied is True
    assert rgba.metadata.bit_depth_conversion_applied is False


def test_palette_transparency_is_reported_as_alpha(tmp_path: Path) -> None:
    image_path = tmp_path / "palette.png"
    image = Image.new("P", (2, 1))
    image.putpalette([0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254)
    image.putdata([0, 1])
    image.info["transparency"] = 0
    image.save(image_path)

    result = decode_image_with_metadata(image_path)

    assert result.metadata.source_mode == "P"
    assert result.metadata.alpha_present is True
    assert result.metadata.rgb_conversion_applied is True


def test_exif_orientation_records_source_and_decoded_geometry(tmp_path: Path) -> None:
    image_path = tmp_path / "orientation.jpg"
    source = Image.new("RGB", (2, 1))
    source.putpixel((0, 0), (255, 0, 0))
    source.putpixel((1, 0), (0, 255, 0))

    exif = Image.Exif()
    exif[274] = 6
    source.save(image_path, exif=exif, format="JPEG")

    result = decode_image_with_metadata(image_path)

    assert result.metadata.source_width == 2
    assert result.metadata.source_height == 1
    assert result.metadata.decoded_width == 1
    assert result.metadata.decoded_height == 2
    assert result.metadata.exif_orientation == 6
    assert result.metadata.exif_orientation_applied is True
    assert result.image.shape == (2, 1, 3)


def test_16_bit_rgb_source_depth_is_auditable_before_uint8_conversion(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "depth16-rgb.png"
    source = np.array(
        [[[0, 32768, 65535], [65535, 1, 2]]],
        dtype=np.uint16,
    )
    assert cv2.imwrite(str(image_path), source)

    result = decode_image_with_metadata(image_path)

    assert result.metadata.source_mode == "RGB"
    assert result.metadata.source_bit_depth == 16
    assert result.metadata.source_channel_count == 3
    assert result.metadata.rgb_conversion_applied is False
    assert result.metadata.bit_depth_conversion_applied is True
    assert result.image.dtype == np.uint8
    assert result.image.shape == (1, 2, 3)


def test_icc_profile_presence_is_reported(tmp_path: Path) -> None:
    image_path = tmp_path / "icc.png"
    Image.new("RGB", (1, 1), (1, 2, 3)).save(
        image_path,
        icc_profile=b"test-profile",
    )

    result = decode_image_with_metadata(image_path)

    assert result.metadata.icc_profile_present is True


def test_legacy_decode_image_matches_detailed_decode_pixels(tmp_path: Path) -> None:
    image_path = tmp_path / "compatibility.png"
    Image.new("RGB", (2, 2), (11, 22, 33)).save(image_path)

    legacy = decode_image(image_path)
    detailed = decode_image_with_metadata(image_path)

    assert isinstance(legacy, np.ndarray)
    assert np.array_equal(legacy, detailed.image)
    assert legacy.dtype == np.uint8
    assert legacy.flags.c_contiguous


def test_decode_metadata_contract_rejects_inconsistent_geometry() -> None:
    metadata = ImageDecodeMetadata(
        source_format="PNG",
        source_mode="RGB",
        source_width=2,
        source_height=2,
        decoded_width=3,
        decoded_height=2,
        source_bit_depth=8,
        source_channel_count=3,
        alpha_present=False,
        icc_profile_present=False,
        exif_orientation=None,
        exif_orientation_applied=False,
        rgb_conversion_applied=False,
        bit_depth_conversion_applied=False,
    )

    with pytest.raises(ImageDecodeError, match="width must match metadata"):
        DecodedImage(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            metadata=metadata,
        )
