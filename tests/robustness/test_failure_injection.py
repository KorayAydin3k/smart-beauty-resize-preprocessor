from __future__ import annotations

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
from smart_beauty_resize.contracts import (
    OutputWriteError,
)
from smart_beauty_resize.writing.safe_writer import (
    write_png_atomic,
)


def _write_image(
    path: Path,
    *,
    value: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.full(
        (20, 30, 3),
        value,
        dtype=np.uint8,
    )

    Image.fromarray(image).save(path)


def _batch_config(
    input_dir: Path,
    output_dir: Path,
    *,
    fail_fast: bool = False,
) -> BatchConfig:
    return BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        resize_config=ResizeConfig(
            target_width=64,
            target_height=64,
            allow_upscale=True,
            max_upscale_factor=4.0,
        ),
        fail_fast=fail_fast,
    )


def test_corrupt_file_does_not_block_other_images(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_image(
        input_dir / "01-valid.png",
        value=50,
    )
    (input_dir / "02-corrupt.jpg").write_bytes(b"not-an-image")
    _write_image(
        input_dir / "03-valid.png",
        value=150,
    )

    result = process_batch(_batch_config(input_dir, output_dir))

    assert result.summary.total_discovered == 3
    assert result.summary.successful == 2
    assert result.summary.failed == 1
    assert result.summary.skipped == 0

    assert [record.status for record in result.records] == [
        ProcessingStatus.SUCCESS,
        ProcessingStatus.FAILED,
        ProcessingStatus.SUCCESS,
    ]

    assert (output_dir / "01-valid.png").is_file()
    assert (output_dir / "03-valid.png").is_file()


def test_batch_preserves_all_source_bytes(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    sources = [
        input_dir / "a.png",
        input_dir / "nested" / "b.jpg",
        input_dir / "nested" / "c.webp",
    ]

    for index, source in enumerate(sources):
        _write_image(
            source,
            value=30 + index * 60,
        )

    original_bytes = {source: source.read_bytes() for source in sources}

    process_batch(_batch_config(input_dir, output_dir))

    for source in sources:
        assert source.read_bytes() == original_bytes[source]


def test_atomic_writer_cleans_temporary_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "output"

    image = np.full(
        (32, 32, 3),
        100,
        dtype=np.uint8,
    )

    def fail_replace(
        source: object,
        destination: object,
    ) -> None:
        del source, destination
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        "smart_beauty_resize.writing.safe_writer.os.replace",
        fail_replace,
    )

    with pytest.raises(OutputWriteError):
        write_png_atomic(
            image=image,
            output_root=output_root,
            relative_path=Path("sample.png"),
            overwrite=True,
        )

    assert not (output_root / "sample.png").exists()

    temporary_files = [path for path in output_root.rglob("*") if path.is_file()]

    assert temporary_files == []


def test_unexpected_programming_error_is_not_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_image(
        input_dir / "sample.png",
        value=80,
    )

    def fail_unexpectedly(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("simulated programming error")

    monkeypatch.setattr(
        "smart_beauty_resize.batch.processor.resize_sample",
        fail_unexpectedly,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated programming error",
    ):
        process_batch(_batch_config(input_dir, output_dir))


def test_fail_fast_propagates_expected_processing_failure(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    (input_dir / "corrupt.png").write_bytes(b"not-an-image")

    with pytest.raises(Exception) as exception_info:
        process_batch(
            _batch_config(
                input_dir,
                output_dir,
                fail_fast=True,
            )
        )

    assert exception_info.value.__class__.__name__ == "ImageDecodeError"
