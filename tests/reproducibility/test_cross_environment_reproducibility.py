from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np
import pytest
from PIL import Image

from smart_beauty_resize.backends.opencv_backend import resize_sample
from smart_beauty_resize.batch import (
    BatchConfig,
    ProcessingStatus,
    process_batch,
)
from smart_beauty_resize.contracts import ResizeConfig
from smart_beauty_resize.io.decoder import decode_image
from smart_beauty_resize.provenance.hashing import sha256_bytes


class GoldenConfig(TypedDict):
    target_width: int
    target_height: int
    allow_upscale: bool
    max_upscale_factor: float
    padding_value: list[int]


class GoldenPlan(TypedDict):
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


class GoldenCase(TypedDict):
    source_filename: str
    expected_filename: str
    source_sha256: str
    expected_sha256: str
    config: GoldenConfig
    plan: GoldenPlan
    interpolation: str
    output_shape: list[int]
    output_dtype: str


GOLDEN_ROOT = Path(__file__).resolve().parents[1] / "golden"

METADATA_PATH = GOLDEN_ROOT / "metadata.json"

METADATA = cast(
    dict[str, GoldenCase],
    json.loads(METADATA_PATH.read_text(encoding="utf-8")),
)


def _interpolation_name(value: object) -> str:
    enum_value = getattr(value, "value", value)
    return str(enum_value)


@pytest.mark.parametrize(
    "case_name",
    sorted(METADATA),
)
def test_golden_pixels_are_reproducible(
    case_name: str,
) -> None:
    """Verify exact pixel output independently of PNG encoding bytes."""
    cv2.setNumThreads(1)

    case = METADATA[case_name]

    source_path = GOLDEN_ROOT / "input" / case["source_filename"]
    expected_path = GOLDEN_ROOT / "expected" / case["expected_filename"]

    source = decode_image(source_path)
    expected = decode_image(expected_path)
    source_before = source.copy()

    config_values = case["config"]

    config = ResizeConfig(
        target_width=config_values["target_width"],
        target_height=config_values["target_height"],
        allow_upscale=config_values["allow_upscale"],
        max_upscale_factor=(config_values["max_upscale_factor"]),
        padding_value=tuple(config_values["padding_value"]),
    )

    output_hashes: set[str] = set()

    for _ in range(3):
        result = resize_sample(
            image=source,
            config=config,
        )

        assert np.array_equal(
            result.image,
            expected,
        )

        output_hashes.add(sha256_bytes(result.image.tobytes()))

        plan = case["plan"]

        assert result.plan.resized_width == plan["resized_width"]
        assert result.plan.resized_height == plan["resized_height"]
        assert result.plan.pad_left == plan["pad_left"]
        assert result.plan.pad_top == plan["pad_top"]
        assert result.plan.pad_right == plan["pad_right"]
        assert result.plan.pad_bottom == plan["pad_bottom"]

        assert _interpolation_name(result.interpolation) == case["interpolation"]

    assert len(output_hashes) == 1

    assert np.array_equal(
        source,
        source_before,
    )


def _write_deterministic_image(
    path: Path,
    *,
    width: int,
    height: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    y_coordinates, x_coordinates = np.indices(
        (height, width),
        dtype=np.int32,
    )

    image = np.stack(
        (
            (x_coordinates * 17 + seed) % 256,
            (y_coordinates * 31 + seed * 3) % 256,
            (x_coordinates * 7 + y_coordinates * 13 + seed * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)

    Image.fromarray(image).save(path)


def _batch_signature(result: object) -> list[tuple[object, ...]]:
    records = result.records

    return [
        (
            record.source_relative_path.as_posix(),
            record.status,
            record.source_sha256,
            record.output_sha256,
            record.config_sha256,
            record.original_width,
            record.original_height,
            record.resized_width,
            record.resized_height,
            record.pad_left,
            record.pad_top,
            record.pad_right,
            record.pad_bottom,
            record.interpolation,
        )
        for record in records
    ]


def test_batch_outputs_are_reproducible_across_runs(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "input"

    _write_deterministic_image(
        input_directory / "a.png",
        width=53,
        height=31,
        seed=11,
    )
    _write_deterministic_image(
        input_directory / "nested" / "b.png",
        width=31,
        height=53,
        seed=23,
    )
    _write_deterministic_image(
        input_directory / "nested" / "c.png",
        width=96,
        height=64,
        seed=37,
    )

    resize_config = ResizeConfig(
        target_width=128,
        target_height=128,
        allow_upscale=True,
        max_upscale_factor=8.0,
        padding_value=(127, 127, 127),
    )

    first = process_batch(
        BatchConfig(
            input_dir=input_directory,
            output_dir=tmp_path / "output-first",
            resize_config=resize_config,
        )
    )

    second = process_batch(
        BatchConfig(
            input_dir=input_directory,
            output_dir=tmp_path / "output-second",
            resize_config=resize_config,
        )
    )

    assert first.summary.successful == 3
    assert second.summary.successful == 3

    assert all(record.status is ProcessingStatus.SUCCESS for record in first.records)
    assert all(record.status is ProcessingStatus.SUCCESS for record in second.records)

    assert _batch_signature(first) == _batch_signature(second)
