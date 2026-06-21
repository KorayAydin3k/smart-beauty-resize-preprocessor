from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
import PIL

from smart_beauty_resize.backends.opencv_backend import resize_sample
from smart_beauty_resize.contracts import ResizeConfig
from smart_beauty_resize.provenance.hashing import sha256_bytes


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One deterministic resize benchmark scenario."""

    name: str
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    allow_upscale: bool
    max_upscale_factor: float
    padding_value: tuple[int, int, int]
    seed: int
    warmup_iterations: int
    measured_iterations: int


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="small_upscale",
        source_width=320,
        source_height=240,
        target_width=512,
        target_height=512,
        allow_upscale=True,
        max_upscale_factor=4.0,
        padding_value=(127, 127, 127),
        seed=11,
        warmup_iterations=3,
        measured_iterations=20,
    ),
    BenchmarkCase(
        name="medium_downscale",
        source_width=1920,
        source_height=1080,
        target_width=512,
        target_height=512,
        allow_upscale=True,
        max_upscale_factor=4.0,
        padding_value=(127, 127, 127),
        seed=23,
        warmup_iterations=3,
        measured_iterations=15,
    ),
    BenchmarkCase(
        name="large_downscale",
        source_width=4032,
        source_height=3024,
        target_width=1024,
        target_height=1024,
        allow_upscale=True,
        max_upscale_factor=4.0,
        padding_value=(127, 127, 127),
        seed=37,
        warmup_iterations=2,
        measured_iterations=8,
    ),
    BenchmarkCase(
        name="identity",
        source_width=512,
        source_height=512,
        target_width=512,
        target_height=512,
        allow_upscale=True,
        max_upscale_factor=1.0,
        padding_value=(0, 0, 0),
        seed=41,
        warmup_iterations=3,
        measured_iterations=20,
    ),
)


def _pattern(
    *,
    width: int,
    height: int,
    seed: int,
) -> np.ndarray:
    """Create a deterministic, non-uniform RGB image."""
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


def _percentile(
    values: list[float],
    percentile: float,
) -> float:
    """Calculate a linearly interpolated percentile."""
    if not values:
        raise ValueError("values must not be empty")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )
    fraction = position - lower_index

    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def _interpolation_name(value: object) -> str:
    enum_value = getattr(value, "value", value)
    return str(enum_value)


def run_case(
    case: BenchmarkCase,
    *,
    warmup_iterations: int | None = None,
    measured_iterations: int | None = None,
) -> dict[str, Any]:
    """Run one benchmark case and return structured measurements."""
    warmups = case.warmup_iterations if warmup_iterations is None else warmup_iterations
    iterations = case.measured_iterations if measured_iterations is None else measured_iterations

    if warmups < 0:
        raise ValueError("warmup_iterations must be non-negative")

    if iterations <= 0:
        raise ValueError("measured_iterations must be positive")

    image = _pattern(
        width=case.source_width,
        height=case.source_height,
        seed=case.seed,
    )

    input_sha256 = sha256_bytes(image.tobytes())

    config = ResizeConfig(
        target_width=case.target_width,
        target_height=case.target_height,
        allow_upscale=case.allow_upscale,
        max_upscale_factor=case.max_upscale_factor,
        padding_value=case.padding_value,
    )

    for _ in range(warmups):
        resize_sample(
            image=image,
            config=config,
        )

    durations_ms: list[float] = []
    output_hashes: set[str] = set()

    last_result = None

    for _ in range(iterations):
        started_ns = time.perf_counter_ns()

        result = resize_sample(
            image=image,
            config=config,
        )

        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0

        durations_ms.append(elapsed_ms)
        output_hashes.add(sha256_bytes(result.image.tobytes()))
        last_result = result

    if last_result is None:
        raise RuntimeError("benchmark produced no resize result")

    if len(output_hashes) != 1:
        raise RuntimeError(f"non-deterministic output detected for {case.name}")

    if sha256_bytes(image.tobytes()) != input_sha256:
        raise RuntimeError(f"input mutation detected for {case.name}")

    median_ms = _percentile(
        durations_ms,
        0.50,
    )
    p95_ms = _percentile(
        durations_ms,
        0.95,
    )

    return {
        "name": case.name,
        "source_width": case.source_width,
        "source_height": case.source_height,
        "target_width": case.target_width,
        "target_height": case.target_height,
        "source_megapixels": (case.source_width * case.source_height / 1_000_000.0),
        "warmup_iterations": warmups,
        "measured_iterations": iterations,
        "latency_ms": {
            "minimum": min(durations_ms),
            "mean": mean(durations_ms),
            "median": median_ms,
            "p95": p95_ms,
            "maximum": max(durations_ms),
        },
        "throughput_images_per_second": (1000.0 / median_ms if median_ms > 0 else None),
        "input_sha256": input_sha256,
        "output_pixel_sha256": next(iter(output_hashes)),
        "interpolation": _interpolation_name(last_result.interpolation),
        "plan": {
            "resized_width": (last_result.plan.resized_width),
            "resized_height": (last_result.plan.resized_height),
            "pad_left": last_result.plan.pad_left,
            "pad_top": last_result.plan.pad_top,
            "pad_right": last_result.plan.pad_right,
            "pad_bottom": last_result.plan.pad_bottom,
        },
    }


def _environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Benchmark deterministic Smart Beauty resize latency and throughput.")
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one warmup and three measured iterations.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in CASES],
        help="Run only the selected case. May be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/resize_benchmark.json"),
        help="JSON report destination.",
    )

    arguments = parser.parse_args()

    selected_names = set(arguments.case) if arguments.case else None

    selected_cases = [
        case for case in CASES if (selected_names is None or case.name in selected_names)
    ]

    results = [
        run_case(
            case,
            warmup_iterations=(1 if arguments.quick else None),
            measured_iterations=(3 if arguments.quick else None),
        )
        for case in selected_cases
    ]

    report = {
        "schema_version": "1.0",
        "generated_at_utc": (datetime.now(UTC).isoformat().replace("+00:00", "Z")),
        "environment": _environment(),
        "cases": results,
    }

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    arguments.output.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for result in results:
        latency = result["latency_ms"]

        print(
            f"{result['name']}: "
            f"median={latency['median']:.3f} ms, "
            f"p95={latency['p95']:.3f} ms, "
            f"throughput="
            f"{result['throughput_images_per_second']:.2f} img/s"
        )

    print(f"Report: {arguments.output}")


if __name__ == "__main__":
    main()
