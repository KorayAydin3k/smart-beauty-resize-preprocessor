from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from smart_beauty_resize import ResizeConfig
from smart_beauty_resize.batch import (
    BatchConfig,
    BatchExecutionResult,
    process_batch,
)
from smart_beauty_resize.contracts import (
    ManifestSerializationError,
    ManifestWriteError,
    ProvenanceError,
    SmartBeautyResizeError,
)
from smart_beauty_resize.provenance import (
    DATASET_AUDIT_FILENAME,
    MANIFEST_FILENAME,
    RUNS_DIRECTORY_NAME,
    SUMMARY_FILENAME,
    write_batch_artifacts,
)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.full(
        (10, 20, 3),
        value,
        dtype=np.uint8,
    )
    Image.fromarray(image).save(path)


def _batch_result(
    tmp_path: Path,
    *,
    include_images: bool = True,
) -> tuple[BatchExecutionResult, Path]:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)

    if include_images:
        _write_image(input_dir / "valid.jpg", 90)
        (input_dir / "corrupt.png").write_bytes(b"not-an-image")

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        resize_config=ResizeConfig(
            target_width=32,
            target_height=32,
            max_upscale_factor=4.0,
        ),
    )

    return process_batch(config), output_dir


def test_write_batch_artifacts_persists_records_and_summary(
    tmp_path: Path,
) -> None:
    result, output_dir = _batch_result(tmp_path)

    artifacts = write_batch_artifacts(
        result,
        output_dir,
    )

    expected_directory = output_dir / RUNS_DIRECTORY_NAME / result.summary.run_id

    assert artifacts.run_directory == expected_directory
    assert artifacts.manifest_path.name == MANIFEST_FILENAME
    assert artifacts.summary_path.name == SUMMARY_FILENAME
    assert artifacts.dataset_audit_path.name == DATASET_AUDIT_FILENAME

    manifest_lines = artifacts.manifest_path.read_text(encoding="utf-8").splitlines()

    assert len(manifest_lines) == len(result.records)

    parsed_records = [json.loads(line) for line in manifest_lines]

    assert [record["source_relative_path"] for record in parsed_records] == [
        record.source_relative_path.as_posix() for record in result.records
    ]

    by_source = {record["source_relative_path"]: record for record in parsed_records}
    assert by_source["valid.jpg"]["decode_metadata"]["source_format"] == "JPEG"
    assert by_source["valid.jpg"]["decode_metadata"]["source_mode"] == "RGB"
    assert by_source["valid.jpg"]["input_policy"] == "audit_only"
    assert by_source["corrupt.png"]["decode_metadata"] is None

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))

    assert summary["schema_version"] == "1.2"
    assert summary["run_id"] == result.summary.run_id
    assert summary["input_policy"] == "audit_only"
    assert summary["total_discovered"] == result.summary.total_discovered
    assert summary["successful"] == result.summary.successful
    assert summary["failed"] == result.summary.failed
    assert summary["skipped"] == result.summary.skipped

    audit = json.loads(artifacts.dataset_audit_path.read_text(encoding="utf-8"))

    assert audit["schema_version"] == "1.0"
    assert audit["run_id"] == result.summary.run_id
    assert audit["config_sha256"] == result.summary.config_sha256
    assert audit["input_policy"] == "audit_only"
    assert audit["total_records"] == 2
    assert audit["records_with_decode_metadata"] == 1
    assert audit["records_without_decode_metadata"] == 1
    assert audit["decode_metadata_coverage_percent"] == 50.0
    assert audit["status_counts"] == {"failed": 1, "skipped": 0, "success": 1}
    assert audit["source_format_counts"] == {"JPEG": 1}
    assert audit["source_mode_counts"] == {"RGB": 1}
    assert audit["source_bit_depth_counts"] == {"8": 1}
    assert audit["source_channel_count_counts"] == {"3": 1}
    assert audit["error_type_counts"] == {"ImageDecodeError": 1}
    assert audit["source_width_statistics"]["maximum"] == 20
    assert audit["source_height_statistics"]["maximum"] == 10
    assert audit["source_pixel_count_statistics"]["maximum"] == 200


def test_empty_batch_writes_empty_manifest(
    tmp_path: Path,
) -> None:
    result, output_dir = _batch_result(
        tmp_path,
        include_images=False,
    )

    artifacts = write_batch_artifacts(
        result,
        output_dir,
    )

    assert artifacts.manifest_path.read_bytes() == b""

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["total_discovered"] == 0

    audit = json.loads(artifacts.dataset_audit_path.read_text(encoding="utf-8"))
    assert audit["total_records"] == 0
    assert audit["records_with_decode_metadata"] == 0
    assert audit["records_without_decode_metadata"] == 0
    assert audit["source_width_statistics"] is None


def test_existing_run_artifacts_are_not_overwritten(
    tmp_path: Path,
) -> None:
    result, output_dir = _batch_result(tmp_path)

    first = write_batch_artifacts(
        result,
        output_dir,
    )
    first_manifest = first.manifest_path.read_bytes()
    first_summary = first.summary_path.read_bytes()
    first_audit = first.dataset_audit_path.read_bytes()

    with pytest.raises(
        ManifestWriteError,
        match="already exist",
    ):
        write_batch_artifacts(
            result,
            output_dir,
        )

    assert first.manifest_path.read_bytes() == first_manifest
    assert first.summary_path.read_bytes() == first_summary
    assert first.dataset_audit_path.read_bytes() == first_audit


def test_serialization_failure_leaves_no_partial_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, output_dir = _batch_result(tmp_path)

    def fail_record_serialization(
        record: object,
    ) -> str:
        del record
        raise ManifestSerializationError("simulated serialization failure")

    monkeypatch.setattr(
        "smart_beauty_resize.provenance.writer.record_to_json_line",
        fail_record_serialization,
    )

    with pytest.raises(ManifestSerializationError):
        write_batch_artifacts(
            result,
            output_dir,
        )

    runs_root = output_dir / RUNS_DIRECTORY_NAME

    assert not (runs_root / result.summary.run_id).exists()

    assert list(runs_root.glob(f".{result.summary.run_id}.*")) == []


def test_dataset_audit_failure_leaves_no_partial_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, output_dir = _batch_result(tmp_path)

    def fail_dataset_audit(result_value: object) -> object:
        del result_value
        raise ProvenanceError("simulated dataset audit failure")

    monkeypatch.setattr(
        "smart_beauty_resize.audit.build_dataset_audit_summary",
        fail_dataset_audit,
    )

    with pytest.raises(ProvenanceError, match="simulated dataset audit failure"):
        write_batch_artifacts(result, output_dir)

    runs_root = output_dir / RUNS_DIRECTORY_NAME

    assert not (runs_root / result.summary.run_id).exists()
    assert list(runs_root.glob(f".{result.summary.run_id}.*")) == []


def test_writer_rejects_unsafe_run_id(
    tmp_path: Path,
) -> None:
    result, output_dir = _batch_result(tmp_path)

    unsafe_summary = replace(
        result.summary,
        run_id="../escape",
    )
    unsafe_result = BatchExecutionResult(
        records=result.records,
        summary=unsafe_summary,
    )

    with pytest.raises(
        ManifestWriteError,
        match="run_id",
    ):
        write_batch_artifacts(
            unsafe_result,
            output_dir,
        )


def test_writer_rejects_file_output_root(
    tmp_path: Path,
) -> None:
    result, _ = _batch_result(tmp_path)

    output_file = tmp_path / "not-a-directory"
    output_file.write_text(
        "content",
        encoding="utf-8",
    )

    with pytest.raises(
        ManifestWriteError,
        match="not a directory",
    ):
        write_batch_artifacts(
            result,
            output_file,
        )


def test_manifest_writer_rejects_invalid_inputs(
    tmp_path: Path,
) -> None:
    result, output_dir = _batch_result(tmp_path)

    with pytest.raises(ManifestWriteError):
        write_batch_artifacts(
            object(),  # type: ignore[arg-type]
            output_dir,
        )

    with pytest.raises(ManifestWriteError):
        write_batch_artifacts(
            result,
            "output",  # type: ignore[arg-type]
        )


def test_successful_write_leaves_no_staging_or_lock_files(
    tmp_path: Path,
) -> None:
    result, output_dir = _batch_result(tmp_path)

    write_batch_artifacts(
        result,
        output_dir,
    )

    runs_root = output_dir / RUNS_DIRECTORY_NAME

    leftovers = [path.name for path in runs_root.iterdir() if path.name.startswith(".")]

    assert leftovers == []


def test_manifest_write_error_inherits_package_base() -> None:
    assert issubclass(
        ManifestWriteError,
        SmartBeautyResizeError,
    )
