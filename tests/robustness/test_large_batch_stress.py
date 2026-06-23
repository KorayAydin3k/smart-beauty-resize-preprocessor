from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from smart_beauty_resize import ResizeConfig
from smart_beauty_resize.batch import (
    BatchConfig,
    ProcessingStatus,
    process_batch,
)
from smart_beauty_resize.contracts import OutputWriteError
from smart_beauty_resize.provenance.hashing import sha256_file


def _write_image(
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
            (x_coordinates * 13 + seed) % 256,
            (y_coordinates * 29 + seed * 3) % 256,
            (x_coordinates * 7 + y_coordinates * 11 + seed * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)

    Image.fromarray(image, mode="RGB").save(path)


def _build_dataset(
    input_dir: Path,
    *,
    count: int,
) -> list[Path]:
    relative_paths: list[Path] = []

    for index in range(count):
        relative_path = (
            Path(f"group-{index % 8:02d}") / f"subject-{index % 16:02d}" / f"image-{index:04d}.png"
        )

        width = 32 + index % 37
        height = 24 + index % 29

        _write_image(
            input_dir / relative_path,
            width=width,
            height=height,
            seed=index + 1,
        )

        relative_paths.append(relative_path)

    return relative_paths


def _config(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> BatchConfig:
    return BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        resize_config=ResizeConfig(
            target_width=96,
            target_height=96,
            allow_upscale=True,
            max_upscale_factor=8.0,
        ),
        overwrite=overwrite,
        fail_fast=False,
        preserve_directory_structure=True,
    )


def _expected_order(
    relative_paths: list[Path],
) -> list[Path]:
    return sorted(
        relative_paths,
        key=lambda path: (
            path.as_posix().casefold(),
            path.as_posix(),
        ),
    )


def test_large_batch_is_complete_deterministic_and_source_safe(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    relative_paths = _build_dataset(
        input_dir,
        count=96,
    )

    source_hashes = {
        relative_path: sha256_file(input_dir / relative_path) for relative_path in relative_paths
    }

    result = process_batch(_config(input_dir, output_dir))

    assert result.summary.total_discovered == 96
    assert result.summary.successful == 96
    assert result.summary.failed == 0
    assert result.summary.skipped == 0

    assert all(record.status is ProcessingStatus.SUCCESS for record in result.records)

    assert [record.source_relative_path for record in result.records] == _expected_order(
        relative_paths
    )

    output_files = sorted(path for path in output_dir.rglob("*.png") if path.is_file())

    assert len(output_files) == 96

    for relative_path in relative_paths:
        assert sha256_file(input_dir / relative_path) == source_hashes[relative_path]

        output_path = output_dir / relative_path.with_suffix(".png")

        assert output_path.is_file()

    for output_path in output_files[::12]:
        with Image.open(output_path) as image:
            assert image.mode == "RGB"
            assert image.size == (96, 96)

    second_result = process_batch(_config(input_dir, output_dir))

    assert second_result.summary.successful == 0
    assert second_result.summary.failed == 0
    assert second_result.summary.skipped == 96

    assert [record.source_relative_path for record in second_result.records] == _expected_order(
        relative_paths
    )


def test_single_write_failure_does_not_block_remaining_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    relative_paths = _build_dataset(
        input_dir,
        count=24,
    )

    from smart_beauty_resize.batch import processor

    original_writer = processor.write_png_atomic

    def selective_failure(
        *args: object,
        **kwargs: object,
    ) -> Path:
        relative_path_value = kwargs.get("relative_path")

        if not isinstance(relative_path_value, Path):
            raise AssertionError("relative_path must be provided as a pathlib.Path")

        relative_path = relative_path_value

        if relative_path.name == "image-0011.png":
            raise OutputWriteError("simulated selective disk write failure")

        return original_writer(*args, **kwargs)

    monkeypatch.setattr(
        processor,
        "write_png_atomic",
        selective_failure,
    )

    result = process_batch(_config(input_dir, output_dir))

    assert result.summary.total_discovered == 24
    assert result.summary.successful == 23
    assert result.summary.failed == 1
    assert result.summary.skipped == 0

    failed_records = [
        record for record in result.records if record.status is ProcessingStatus.FAILED
    ]

    assert len(failed_records) == 1
    assert failed_records[0].source_relative_path.name == "image-0011.png"
    assert failed_records[0].error_type == "OutputWriteError"

    failed_source = next(path for path in relative_paths if path.name == "image-0011.png")

    assert not (output_dir / failed_source).exists()

    assert len(list(output_dir.rglob("*.png"))) == 23


def _open_file_descriptor_count() -> int:
    descriptor_directory = Path("/proc/self/fd")

    if not descriptor_directory.is_dir():
        pytest.skip("/proc/self/fd is unavailable on this platform")

    return sum(1 for _ in descriptor_directory.iterdir())


def test_repeated_batch_runs_do_not_leak_file_descriptors(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _build_dataset(
        input_dir,
        count=16,
    )

    config = _config(
        input_dir,
        output_dir,
        overwrite=True,
    )

    process_batch(config)
    gc.collect()

    baseline_count = _open_file_descriptor_count()

    observed_counts: list[int] = []

    for _ in range(8):
        result = process_batch(config)

        assert result.summary.successful == 16
        assert result.summary.failed == 0

        gc.collect()
        observed_counts.append(_open_file_descriptor_count())

    final_count = observed_counts[-1]

    assert final_count <= baseline_count + 2

    assert max(observed_counts) - min(observed_counts) <= 2
