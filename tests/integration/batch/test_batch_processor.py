from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from smart_beauty_resize import (
    ImageDecodeError,
    InputPolicy,
    InputPolicyViolationError,
    OutputPathCollisionError,
    ResizeConfig,
    SourceImageLimitError,
    SourceImageLimits,
    decode_image,
)
from smart_beauty_resize.batch import (
    BatchConfig,
    ProcessingStatus,
    process_batch,
)
from smart_beauty_resize.provenance import (
    sha256_file,
    sha256_resize_config,
)


def _write_rgb_image(
    path: Path,
    *,
    width: int,
    height: int,
    value: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.full(
        (height, width, 3),
        value,
        dtype=np.uint8,
    )

    Image.fromarray(image).save(path)


def _config(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    fail_fast: bool = False,
    preserve_directory_structure: bool = True,
    input_policy: InputPolicy = InputPolicy.AUDIT_ONLY,
    source_limits: SourceImageLimits | None = None,
) -> BatchConfig:
    return BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        resize_config=ResizeConfig(
            target_width=32,
            target_height=32,
            allow_upscale=True,
            max_upscale_factor=8.0,
        ),
        overwrite=overwrite,
        fail_fast=fail_fast,
        preserve_directory_structure=(preserve_directory_structure),
        input_policy=input_policy,
        source_limits=SourceImageLimits() if source_limits is None else source_limits,
    )


def test_process_batch_records_success_and_failure(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_rgb_image(
        input_dir / "b.jpg",
        width=20,
        height=10,
        value=60,
    )
    _write_rgb_image(
        input_dir / "nested" / "c.png",
        width=10,
        height=20,
        value=120,
    )

    corrupt = input_dir / "a_corrupt.jpg"
    corrupt.write_bytes(b"not-an-image")

    result = process_batch(_config(input_dir, output_dir))

    assert result.summary.schema_version == "1.3"
    assert result.summary.total_discovered == 3
    assert result.summary.successful == 2
    assert result.summary.failed == 1
    assert result.summary.skipped == 0

    assert [record.source_relative_path.as_posix() for record in result.records] == [
        "a_corrupt.jpg",
        "b.jpg",
        "nested/c.png",
    ]

    failed_record = result.records[0]
    assert failed_record.status is ProcessingStatus.FAILED
    assert failed_record.error_type == "ImageDecodeError"
    assert failed_record.output_sha256 is None
    assert failed_record.source_sha256 == sha256_file(corrupt)
    assert failed_record.decode_metadata is None

    for record in result.records[1:]:
        assert record.status is ProcessingStatus.SUCCESS
        assert record.output_relative_path is not None
        assert record.source_sha256 is not None
        assert record.output_sha256 is not None
        assert record.schema_version == "1.3"
        assert record.decode_metadata is not None
        assert record.decode_metadata.source_format in {"JPEG", "PNG"}
        assert record.decode_metadata.source_mode == "RGB"
        assert record.decode_metadata.source_bit_depth == 8

        output_path = output_dir / record.output_relative_path

        assert output_path.is_file()
        assert record.output_sha256 == sha256_file(output_path)
        assert decode_image(output_path).shape == (32, 32, 3)

    expected_config_hash = sha256_resize_config(_config(input_dir, output_dir).resize_config)

    assert result.summary.config_sha256 == expected_config_hash
    assert all(record.config_sha256 == expected_config_hash for record in result.records)
    assert result.summary.source_limits == SourceImageLimits()
    assert all(record.source_limits == SourceImageLimits() for record in result.records)


def test_source_limit_violation_becomes_failed_record_before_decode(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_rgb_image(
        input_dir / "oversized.png",
        width=20,
        height=10,
        value=60,
    )

    limits = SourceImageLimits(
        max_width=19,
        max_height=100,
        max_pixels=1000,
    )
    result = process_batch(
        _config(
            input_dir,
            output_dir,
            source_limits=limits,
        )
    )

    assert result.summary.successful == 0
    assert result.summary.failed == 1
    assert result.summary.source_limits == limits

    record = result.records[0]
    assert record.status is ProcessingStatus.FAILED
    assert record.error_type == "SourceImageLimitError"
    assert record.decode_metadata is None
    assert record.source_limits == limits
    assert not (output_dir / "oversized.png").exists()


def test_fail_fast_propagates_source_limit_error(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_rgb_image(
        input_dir / "oversized.png",
        width=20,
        height=10,
        value=60,
    )

    with pytest.raises(SourceImageLimitError):
        process_batch(
            _config(
                input_dir,
                output_dir,
                fail_fast=True,
                source_limits=SourceImageLimits(max_width=19),
            )
        )


def test_second_run_skips_existing_outputs(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_rgb_image(
        input_dir / "first.jpg",
        width=12,
        height=8,
        value=50,
    )
    _write_rgb_image(
        input_dir / "second.png",
        width=8,
        height=12,
        value=100,
    )

    config = _config(input_dir, output_dir)

    first = process_batch(config)
    second = process_batch(config)

    assert first.summary.successful == 2
    assert second.summary.successful == 0
    assert second.summary.failed == 0
    assert second.summary.skipped == 2

    assert all(record.status is ProcessingStatus.SKIPPED for record in second.records)
    assert all(record.decode_metadata is not None for record in second.records)


def test_overwrite_reprocesses_existing_output(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source = input_dir / "sample.png"

    _write_rgb_image(
        source,
        width=10,
        height=10,
        value=40,
    )

    process_batch(_config(input_dir, output_dir))

    _write_rgb_image(
        source,
        width=10,
        height=10,
        value=200,
    )

    result = process_batch(
        _config(
            input_dir,
            output_dir,
            overwrite=True,
        )
    )

    assert result.summary.successful == 1
    assert result.summary.failed == 0
    assert result.summary.skipped == 0

    output = output_dir / "sample.png"
    decoded = decode_image(output)

    assert int(decoded[16, 16, 0]) == 200


def test_fail_fast_propagates_decode_error(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    (input_dir / "corrupt.jpg").write_bytes(b"not-an-image")

    with pytest.raises(ImageDecodeError):
        process_batch(
            _config(
                input_dir,
                output_dir,
                fail_fast=True,
            )
        )


def test_empty_input_directory_returns_empty_summary(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    result = process_batch(_config(input_dir, output_dir))

    assert result.records == ()
    assert result.summary.total_discovered == 0
    assert result.summary.successful == 0
    assert result.summary.failed == 0
    assert result.summary.skipped == 0
    assert result.summary.success_rate == 0.0


def test_batch_does_not_modify_source_file(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source = input_dir / "sample.png"

    _write_rgb_image(
        source,
        width=9,
        height=6,
        value=110,
    )

    before_bytes = source.read_bytes()
    before_hash = sha256_file(source)

    process_batch(_config(input_dir, output_dir))

    assert source.read_bytes() == before_bytes
    assert sha256_file(source) == before_hash


def test_audit_only_preserves_rgba_conversion_behavior(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    rgba = np.zeros((8, 12, 4), dtype=np.uint8)
    rgba[:, :, :3] = 100
    rgba[:, :, 3] = 128
    Image.fromarray(rgba, mode="RGBA").save(input_dir / "rgba.png")

    result = process_batch(_config(input_dir, output_dir))

    assert result.summary.successful == 1
    assert result.summary.failed == 0
    assert result.summary.input_policy is InputPolicy.AUDIT_ONLY
    record = result.records[0]
    assert record.input_policy is InputPolicy.AUDIT_ONLY
    assert record.decode_metadata is not None
    assert record.decode_metadata.source_mode == "RGBA"
    assert record.decode_metadata.alpha_present is True
    assert record.decode_metadata.rgb_conversion_applied is True
    assert (output_dir / "rgba.png").is_file()


def test_strict_rgb8_rejects_rgba_and_preserves_audit_metadata(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    rgba = np.zeros((8, 12, 4), dtype=np.uint8)
    rgba[:, :, :3] = 100
    rgba[:, :, 3] = 128
    Image.fromarray(rgba, mode="RGBA").save(input_dir / "rgba.png")

    result = process_batch(
        _config(
            input_dir,
            output_dir,
            input_policy=InputPolicy.STRICT_RGB8,
        )
    )

    assert result.summary.successful == 0
    assert result.summary.failed == 1
    assert result.summary.input_policy is InputPolicy.STRICT_RGB8
    record = result.records[0]
    assert record.status is ProcessingStatus.FAILED
    assert record.error_type == "InputPolicyViolationError"
    assert record.input_policy is InputPolicy.STRICT_RGB8
    assert record.decode_metadata is not None
    assert record.decode_metadata.source_mode == "RGBA"
    assert record.original_width == 12
    assert record.original_height == 8
    assert not (output_dir / "rgba.png").exists()


def test_strict_rgb8_fail_fast_propagates_policy_violation(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    grayscale = np.full((8, 12), 100, dtype=np.uint8)
    Image.fromarray(grayscale, mode="L").save(input_dir / "gray.png")

    with pytest.raises(InputPolicyViolationError):
        process_batch(
            _config(
                input_dir,
                output_dir,
                input_policy=InputPolicy.STRICT_RGB8,
                fail_fast=True,
            )
        )


@pytest.mark.parametrize("overwrite", [False, True])
def test_output_collision_fails_before_hash_decode_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_rgb_image(
        input_dir / "sample.jpg",
        width=12,
        height=8,
        value=50,
    )
    _write_rgb_image(
        input_dir / "sample.png",
        width=8,
        height=12,
        value=100,
    )

    def unexpected_hash(_: Path) -> str:
        raise AssertionError("source hashing must not start before collision preflight")

    def unexpected_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("decoding must not start before collision preflight")

    monkeypatch.setattr(
        "smart_beauty_resize.batch.processor.sha256_file",
        unexpected_hash,
    )
    monkeypatch.setattr(
        "smart_beauty_resize.batch.processor.decode_image_with_metadata",
        unexpected_decode,
    )

    with pytest.raises(
        OutputPathCollisionError,
        match="multiple source images resolve to the same output path",
    ):
        process_batch(
            _config(
                input_dir,
                output_dir,
                overwrite=overwrite,
            )
        )

    assert not output_dir.exists()
