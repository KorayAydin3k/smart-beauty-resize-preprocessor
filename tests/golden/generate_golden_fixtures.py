from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from smart_beauty_resize.backends.opencv_backend import resize_sample
from smart_beauty_resize.contracts import ResizeConfig
from smart_beauty_resize.io.decoder import decode_image
from smart_beauty_resize.provenance.hashing import sha256_file
from smart_beauty_resize.writing.safe_writer import write_png_atomic

ROOT = Path(__file__).resolve().parent
INPUT_DIRECTORY = ROOT / "input"
EXPECTED_DIRECTORY = ROOT / "expected"
METADATA_PATH = ROOT / "metadata.json"


CASES: dict[str, dict[str, Any]] = {
    "landscape_upscale": {
        "source_width": 7,
        "source_height": 3,
        "seed": 11,
        "config": {
            "target_width": 8,
            "target_height": 8,
            "allow_upscale": True,
            "max_upscale_factor": 2.0,
            "padding_value": (10, 20, 30),
        },
    },
    "portrait_upscale": {
        "source_width": 3,
        "source_height": 7,
        "seed": 23,
        "config": {
            "target_width": 8,
            "target_height": 8,
            "allow_upscale": True,
            "max_upscale_factor": 2.0,
            "padding_value": (30, 20, 10),
        },
    },
    "landscape_downscale": {
        "source_width": 13,
        "source_height": 9,
        "seed": 37,
        "config": {
            "target_width": 8,
            "target_height": 8,
            "allow_upscale": True,
            "max_upscale_factor": 2.0,
            "padding_value": (0, 0, 0),
        },
    },
    "odd_dimension_rounding": {
        "source_width": 5,
        "source_height": 3,
        "seed": 41,
        "config": {
            "target_width": 9,
            "target_height": 7,
            "allow_upscale": True,
            "max_upscale_factor": 2.0,
            "padding_value": (127, 127, 127),
        },
    },
    "identity": {
        "source_width": 6,
        "source_height": 4,
        "seed": 53,
        "config": {
            "target_width": 6,
            "target_height": 4,
            "allow_upscale": True,
            "max_upscale_factor": 1.0,
            "padding_value": (5, 5, 5),
        },
    },
    "upscale_disabled": {
        "source_width": 4,
        "source_height": 4,
        "seed": 67,
        "config": {
            "target_width": 8,
            "target_height": 8,
            "allow_upscale": False,
            "max_upscale_factor": 2.0,
            "padding_value": (200, 201, 202),
        },
    },
}


def _pattern(
    *,
    width: int,
    height: int,
    seed: int,
) -> np.ndarray:
    """Create a deterministic non-uniform RGB test image."""
    y_coordinates, x_coordinates = np.indices(
        (height, width),
        dtype=np.int32,
    )

    red = (x_coordinates * 37 + y_coordinates * 11 + seed) % 256

    green = (x_coordinates * 17 + y_coordinates * 43 + seed * 3) % 256

    blue = (x_coordinates * 29 + y_coordinates * 7 + seed * 5) % 256

    return np.stack(
        (red, green, blue),
        axis=-1,
    ).astype(np.uint8)


def _interpolation_name(value: object) -> str:
    enum_value = getattr(value, "value", value)
    return str(enum_value)


def main() -> None:
    """Regenerate committed golden inputs, outputs, and metadata."""
    shutil.rmtree(
        INPUT_DIRECTORY,
        ignore_errors=True,
    )
    shutil.rmtree(
        EXPECTED_DIRECTORY,
        ignore_errors=True,
    )

    INPUT_DIRECTORY.mkdir(parents=True)
    EXPECTED_DIRECTORY.mkdir(parents=True)

    metadata: dict[str, dict[str, Any]] = {}

    for case_name, case in sorted(CASES.items()):
        source_width = int(case["source_width"])
        source_height = int(case["source_height"])
        seed = int(case["seed"])
        config_values = dict(case["config"])

        source_array = _pattern(
            width=source_width,
            height=source_height,
            seed=seed,
        )

        source_path = INPUT_DIRECTORY / f"{case_name}.png"

        Image.fromarray(
            source_array,
            mode="RGB",
        ).save(source_path)

        resize_config = ResizeConfig(
            **config_values,
        )

        decoded = decode_image(source_path)

        resize_result = resize_sample(
            image=decoded,
            config=resize_config,
        )

        expected_path = write_png_atomic(
            image=resize_result.image,
            output_root=EXPECTED_DIRECTORY,
            relative_path=Path(f"{case_name}.png"),
            overwrite=False,
        )

        metadata[case_name] = {
            "source_filename": source_path.name,
            "expected_filename": expected_path.name,
            "source_sha256": sha256_file(source_path),
            "expected_sha256": sha256_file(expected_path),
            "config": {
                "target_width": resize_config.target_width,
                "target_height": resize_config.target_height,
                "allow_upscale": resize_config.allow_upscale,
                "max_upscale_factor": (resize_config.max_upscale_factor),
                "padding_value": list(resize_config.padding_value),
            },
            "plan": {
                "resized_width": (resize_result.plan.resized_width),
                "resized_height": (resize_result.plan.resized_height),
                "pad_left": resize_result.plan.pad_left,
                "pad_top": resize_result.plan.pad_top,
                "pad_right": resize_result.plan.pad_right,
                "pad_bottom": resize_result.plan.pad_bottom,
            },
            "interpolation": _interpolation_name(resize_result.interpolation),
            "output_shape": list(resize_result.image.shape),
            "output_dtype": str(resize_result.image.dtype),
        }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Generated {len(metadata)} golden cases.")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
