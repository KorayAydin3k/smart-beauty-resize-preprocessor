from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np
import PIL
from PIL import Image

from smart_beauty_resize.batch import (
    BatchConfig,
    process_batch,
)
from smart_beauty_resize.contracts import ResizeConfig
from smart_beauty_resize.provenance import (
    sha256_file,
    write_batch_artifacts,
)


@dataclass(frozen=True, slots=True)
class BatchBenchmarkCase:
    """One deterministic end-to-end batch benchmark scenario."""

    name: str
    image_count: int
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    seed: int
    warmup_iterations: int
    measured_iterations: int


CASES: tuple[BatchBenchmarkCase, ...] = (
    BatchBenchmarkCase(
        name="small_batch",
        image_count=32,
        source_width=320,
        source_height=240,
        target_width=512,
        target_height=512,
        seed=11,
        warmup_iterations=1,
        measured_iterations=3,
    ),
    BatchBenchmarkCase(
        name="medium_batch",
        image_count=12,
        source_width=1920,
        source_height=1080,
        target_width=512,
        target_height=512,
        seed=23,
        warmup_iterations=1,
        measured_iterations=3,
    ),
    BatchBenchmarkCase(
        name="large_batch",
        image_count=3,
        source_width=4032,
        source_height=3024,
        target_width=1024,
        target_height=1024,
        seed=37,
        warmup_iterations=1,
        measured_iterations=2,
    ),
)


def _pattern(
    *,
    width: int,
    height: int,
    seed: int,
) -> np.ndarray:
    """Create a deterministic non-uniform RGB image."""
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


def _create_dataset(
    input_directory: Path,
    *,
    case: BatchBenchmarkCase,
    image_count: int,
) -> list[Path]:
    """Create deterministic JPEG source files outside timed execution."""
    relative_paths: list[Path] = []

    for index in range(image_count):
        relative_path = Path(f"group-{index % 4:02d}") / f"image-{index:04d}.jpg"

        source_path = input_directory / relative_path
        source_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image = _pattern(
            width=case.source_width,
            height=case.source_height,
            seed=case.seed + index,
        )

        Image.fromarray(image).save(
            source_path,
            format="JPEG",
            quality=92,
            optimize=False,
            progressive=False,
        )

        relative_paths.append(relative_path)

    return relative_paths


def _percentile(
    values: list[float],
    percentile: float,
) -> float:
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


def _aggregate_file_hash(
    root: Path,
    paths: list[Path],
) -> str:
    """Hash ordered relative paths together with their file hashes."""
    digest = hashlib.sha256()

    for path in sorted(
        paths,
        key=lambda value: value.as_posix(),
    ):
        absolute_path = root / path

        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(absolute_path)))

    return digest.hexdigest()


def _output_relative_paths(
    output_directory: Path,
) -> list[Path]:
    return sorted(
        (
            path.relative_to(output_directory)
            for path in output_directory.rglob("*.png")
            if "_runs" not in path.parts
        ),
        key=lambda path: path.as_posix(),
    )


def _execute_once(
    *,
    case: BatchBenchmarkCase,
    input_directory: Path,
    output_directory: Path,
    image_count: int,
) -> tuple[float, str, str]:
    config = BatchConfig(
        input_dir=input_directory,
        output_dir=output_directory,
        resize_config=ResizeConfig(
            target_width=case.target_width,
            target_height=case.target_height,
            allow_upscale=True,
            max_upscale_factor=8.0,
        ),
        overwrite=False,
        fail_fast=False,
        preserve_directory_structure=True,
    )

    started_ns = time.perf_counter_ns()

    result = process_batch(config)
    artifacts = write_batch_artifacts(
        result,
        output_directory,
    )

    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0

    if result.summary.total_discovered != image_count:
        raise RuntimeError("unexpected discovered-image count")

    if result.summary.successful != image_count:
        raise RuntimeError("benchmark batch did not process every image")

    if result.summary.failed != 0:
        raise RuntimeError("benchmark batch produced failed records")

    if result.summary.skipped != 0:
        raise RuntimeError("benchmark batch produced skipped records")

    manifest_lines = artifacts.manifest_path.read_text(encoding="utf-8").splitlines()

    if len(manifest_lines) != image_count:
        raise RuntimeError("manifest record count does not match image count")

    if not artifacts.summary_path.is_file():
        raise RuntimeError("run summary was not written")

    if not artifacts.dataset_audit_path.is_file():
        raise RuntimeError("dataset audit was not written")

    output_paths = _output_relative_paths(output_directory)

    if len(output_paths) != image_count:
        raise RuntimeError("output image count does not match image count")

    output_hash = _aggregate_file_hash(
        output_directory,
        output_paths,
    )

    return (
        elapsed_ms,
        output_hash,
        result.summary.config_sha256,
    )


def run_case(
    case: BatchBenchmarkCase,
    *,
    image_count: int | None = None,
    warmup_iterations: int | None = None,
    measured_iterations: int | None = None,
) -> dict[str, Any]:
    """Run one end-to-end benchmark and validate determinism."""
    effective_image_count = case.image_count if image_count is None else image_count
    warmups = case.warmup_iterations if warmup_iterations is None else warmup_iterations
    iterations = case.measured_iterations if measured_iterations is None else measured_iterations

    if effective_image_count <= 0:
        raise ValueError("image_count must be positive")

    if warmups < 0:
        raise ValueError("warmup_iterations must be non-negative")

    if iterations <= 0:
        raise ValueError("measured_iterations must be positive")

    with TemporaryDirectory() as temporary_directory:
        workspace = Path(temporary_directory)
        input_directory = workspace / "input"
        input_directory.mkdir()

        source_relative_paths = _create_dataset(
            input_directory,
            case=case,
            image_count=effective_image_count,
        )

        source_hash_before = _aggregate_file_hash(
            input_directory,
            source_relative_paths,
        )

        for warmup_index in range(warmups):
            warmup_output = workspace / f"warmup-output-{warmup_index:02d}"

            _execute_once(
                case=case,
                input_directory=input_directory,
                output_directory=warmup_output,
                image_count=effective_image_count,
            )

            shutil.rmtree(warmup_output)

        durations_ms: list[float] = []
        output_hashes: set[str] = set()
        configuration_hashes: set[str] = set()

        for iteration in range(iterations):
            output_directory = workspace / f"measured-output-{iteration:02d}"

            (
                elapsed_ms,
                output_hash,
                configuration_hash,
            ) = _execute_once(
                case=case,
                input_directory=input_directory,
                output_directory=output_directory,
                image_count=effective_image_count,
            )

            durations_ms.append(elapsed_ms)
            output_hashes.add(output_hash)
            configuration_hashes.add(configuration_hash)

            shutil.rmtree(output_directory)

        source_hash_after = _aggregate_file_hash(
            input_directory,
            source_relative_paths,
        )

    if source_hash_before != source_hash_after:
        raise RuntimeError(f"source mutation detected for {case.name}")

    if len(output_hashes) != 1:
        raise RuntimeError(f"non-deterministic outputs detected for {case.name}")

    if len(configuration_hashes) != 1:
        raise RuntimeError(f"non-deterministic configuration hash for {case.name}")

    median_ms = _percentile(
        durations_ms,
        0.50,
    )
    p95_ms = _percentile(
        durations_ms,
        0.95,
    )

    median_seconds = median_ms / 1000.0
    total_source_megapixels = (
        effective_image_count * case.source_width * case.source_height / 1_000_000.0
    )

    return {
        "name": case.name,
        "image_count": effective_image_count,
        "source_width": case.source_width,
        "source_height": case.source_height,
        "target_width": case.target_width,
        "target_height": case.target_height,
        "warmup_iterations": warmups,
        "measured_iterations": iterations,
        "latency_ms": {
            "minimum": min(durations_ms),
            "mean": mean(durations_ms),
            "median": median_ms,
            "p95": p95_ms,
            "maximum": max(durations_ms),
        },
        "throughput_images_per_second": (effective_image_count / median_seconds),
        "throughput_source_megapixels_per_second": (total_source_megapixels / median_seconds),
        "source_aggregate_sha256": source_hash_before,
        "output_aggregate_sha256": next(iter(output_hashes)),
        "config_sha256": next(iter(configuration_hashes)),
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
        description=("Benchmark the complete Smart Beauty batch pipeline.")
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=("Use at most four images, no warmup, and two measured iterations."),
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in CASES],
        help="Run only selected cases. May be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/batch_benchmark.json"),
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
            image_count=(min(case.image_count, 4) if arguments.quick else None),
            warmup_iterations=(0 if arguments.quick else None),
            measured_iterations=(2 if arguments.quick else None),
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
            f"images={result['image_count']}, "
            f"median={latency['median']:.3f} ms, "
            f"p95={latency['p95']:.3f} ms, "
            f"throughput="
            f"{result['throughput_images_per_second']:.2f} img/s"
        )

    print(f"Report: {arguments.output}")


if __name__ == "__main__":
    main()
