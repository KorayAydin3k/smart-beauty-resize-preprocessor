from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from smart_beauty_resize.backends.opencv_backend import resize_sample
from smart_beauty_resize.contracts import ResizeConfig
from smart_beauty_resize.io.decoder import decode_image
from smart_beauty_resize.provenance.hashing import sha256_file
from smart_beauty_resize.writing.safe_writer import write_png_atomic

GOLDEN_ROOT = Path(__file__).resolve().parents[1] / "golden"

INPUT_DIRECTORY = GOLDEN_ROOT / "input"
EXPECTED_DIRECTORY = GOLDEN_ROOT / "expected"
METADATA_PATH = GOLDEN_ROOT / "metadata.json"

METADATA: dict[str, dict[str, Any]] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _interpolation_name(value: object) -> str:
    enum_value = getattr(value, "value", value)
    return str(enum_value)


@pytest.mark.parametrize(
    "case_name",
    sorted(METADATA),
)
def test_resize_matches_committed_golden_output(
    case_name: str,
    tmp_path: Path,
) -> None:
    case = METADATA[case_name]

    source_path = INPUT_DIRECTORY / case["source_filename"]
    expected_path = EXPECTED_DIRECTORY / case["expected_filename"]

    assert sha256_file(source_path) == case["source_sha256"]
    assert sha256_file(expected_path) == case["expected_sha256"]

    config_values = dict(case["config"])
    config_values["padding_value"] = tuple(config_values["padding_value"])

    resize_config = ResizeConfig(
        **config_values,
    )

    source_image = decode_image(source_path)

    source_before = source_image.copy()

    resize_result = resize_sample(
        image=source_image,
        config=resize_config,
    )

    assert np.array_equal(
        source_image,
        source_before,
    )

    actual_path = write_png_atomic(
        image=resize_result.image,
        output_root=tmp_path,
        relative_path=Path(f"{case_name}.png"),
        overwrite=False,
    )

    expected_image = decode_image(expected_path)
    actual_image = decode_image(actual_path)

    # Compressed PNG bytes may differ across operating systems even when the
    # decoded RGB pixels are identical.
    assert np.array_equal(actual_image, expected_image)

    repeated_path = write_png_atomic(
        image=resize_result.image,
        output_root=tmp_path / "repeat",
        relative_path=Path(f"{case_name}.png"),
        overwrite=False,
    )

    # Repeated writes within the same runtime must remain byte-deterministic.
    assert sha256_file(repeated_path) == sha256_file(actual_path)

    assert sha256_file(actual_path) == case["expected_sha256"]

    expected_plan = case["plan"]

    assert resize_result.plan.resized_width == expected_plan["resized_width"]
    assert resize_result.plan.resized_height == expected_plan["resized_height"]
    assert resize_result.plan.pad_left == expected_plan["pad_left"]
    assert resize_result.plan.pad_top == expected_plan["pad_top"]
    assert resize_result.plan.pad_right == expected_plan["pad_right"]
    assert resize_result.plan.pad_bottom == expected_plan["pad_bottom"]

    assert _interpolation_name(resize_result.interpolation) == case["interpolation"]

    assert list(resize_result.image.shape) == case["output_shape"]
    assert str(resize_result.image.dtype) == case["output_dtype"]
