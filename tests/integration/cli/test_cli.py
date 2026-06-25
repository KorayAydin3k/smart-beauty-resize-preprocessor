from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from smart_beauty_resize.cli import app
from smart_beauty_resize.provenance import RUNS_DIRECTORY_NAME

runner = CliRunner()

_ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(value: str) -> str:
    return _ANSI_ESCAPE_PATTERN.sub("", value)


def _write_image(
    path: Path,
    *,
    value: int = 100,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.full(
        (10, 20, 3),
        value,
        dtype=np.uint8,
    )

    Image.fromarray(image).save(path)


def _batch_arguments(
    input_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "batch",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--target-width",
        "32",
        "--target-height",
        "32",
        "--max-upscale-factor",
        "4.0",
    ]


def _run_directories(output_dir: Path) -> list[Path]:
    runs_root = output_dir / RUNS_DIRECTORY_NAME

    if not runs_root.exists():
        return []

    return sorted(path for path in runs_root.iterdir() if path.is_dir())


def test_cli_help() -> None:
    result = runner.invoke(
        app,
        ["--help"],
        color=False,
        terminal_width=160,
    )

    output = _strip_ansi(result.output)

    assert result.exit_code == 0
    assert "batch" in output.lower()


def test_batch_help() -> None:
    result = runner.invoke(
        app,
        ["batch", "--help"],
        color=False,
        terminal_width=160,
    )

    output = _strip_ansi(result.output)

    assert result.exit_code == 0
    assert "--input-dir" in output
    assert "--output-dir" in output
    assert "--target-width" in output
    assert "--target-height" in output
    assert "--input-policy" in output


def test_successful_batch_command(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_image(
        input_dir / "sample.jpg",
        value=90,
    )

    result = runner.invoke(
        app,
        _batch_arguments(input_dir, output_dir),
    )

    assert result.exit_code == 0, result.output
    assert "Successful" in result.output
    assert "Manifest" in result.output

    assert (output_dir / "sample.png").is_file()

    run_directories = _run_directories(output_dir)

    assert len(run_directories) == 1
    assert (run_directories[0] / "manifest.jsonl").is_file()
    assert (run_directories[0] / "run_summary.json").is_file()

    summary = json.loads((run_directories[0] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["total_discovered"] == 1
    assert summary["successful"] == 1
    assert summary["failed"] == 0
    assert summary["skipped"] == 0
    assert summary["input_policy"] == "audit_only"


def test_corrupt_image_exits_two_and_writes_manifest(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    _write_image(
        input_dir / "valid.png",
        value=120,
    )
    (input_dir / "corrupt.jpg").write_bytes(b"not-an-image")

    result = runner.invoke(
        app,
        _batch_arguments(input_dir, output_dir),
    )

    assert result.exit_code == 2

    run_directories = _run_directories(output_dir)
    assert len(run_directories) == 1

    manifest_lines = (
        run_directories[0].joinpath("manifest.jsonl").read_text(encoding="utf-8").splitlines()
    )

    records = [json.loads(line) for line in manifest_lines]

    assert len(records) == 2
    assert sorted(record["status"] for record in records) == [
        "failed",
        "success",
    ]

    failed_record = next(record for record in records if record["status"] == "failed")

    assert failed_record["error_type"] == "ImageDecodeError"

    summary = json.loads((run_directories[0] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["successful"] == 1
    assert summary["failed"] == 1
    assert summary["skipped"] == 0


def test_second_run_skips_existing_output(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_image(
        input_dir / "sample.png",
        value=80,
    )

    arguments = _batch_arguments(
        input_dir,
        output_dir,
    )

    first = runner.invoke(app, arguments)
    second = runner.invoke(app, arguments)

    assert first.exit_code == 0
    assert second.exit_code == 0

    run_directories = _run_directories(output_dir)

    assert len(run_directories) == 2

    summaries = [
        json.loads((run_directory / "run_summary.json").read_text(encoding="utf-8"))
        for run_directory in run_directories
    ]

    assert sorted(summary["successful"] for summary in summaries) == [0, 1]

    assert sorted(summary["skipped"] for summary in summaries) == [0, 1]


def test_overwrite_reprocesses_existing_output(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source = input_dir / "sample.png"

    _write_image(source, value=40)

    arguments = _batch_arguments(
        input_dir,
        output_dir,
    )

    first = runner.invoke(app, arguments)
    assert first.exit_code == 0

    _write_image(source, value=200)

    overwrite_arguments = [
        *arguments,
        "--overwrite",
    ]

    second = runner.invoke(
        app,
        overwrite_arguments,
    )

    assert second.exit_code == 0

    output_image = np.asarray(Image.open(output_dir / "sample.png").convert("RGB"))

    assert int(output_image[16, 16, 0]) == 200


def test_invalid_resize_configuration_exits_one(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    arguments = _batch_arguments(
        input_dir,
        output_dir,
    )

    width_index = arguments.index("--target-width") + 1
    arguments[width_index] = "0"

    result = runner.invoke(
        app,
        arguments,
    )

    assert result.exit_code == 1
    assert _run_directories(output_dir) == []


def test_missing_input_directory_exits_one(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "missing"
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        _batch_arguments(input_dir, output_dir),
    )

    assert result.exit_code == 1
    assert _run_directories(output_dir) == []


def test_fail_fast_exits_one_without_manifest(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    (input_dir / "corrupt.jpg").write_bytes(b"not-an-image")

    arguments = [
        *_batch_arguments(input_dir, output_dir),
        "--fail-fast",
    ]

    result = runner.invoke(
        app,
        arguments,
    )

    assert result.exit_code == 1
    assert _run_directories(output_dir) == []


def test_verbose_prints_success_record(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_image(
        input_dir / "nested" / "sample.jpg",
        value=150,
    )

    arguments = [
        *_batch_arguments(input_dir, output_dir),
        "--verbose",
    ]

    result = runner.invoke(
        app,
        arguments,
    )

    assert result.exit_code == 0
    assert "SUCCESS" in result.output
    assert "nested/sample.jpg" in result.output


def test_flat_output_generates_hashed_filename(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_image(
        input_dir / "person" / "sample.jpg",
        value=110,
    )

    arguments = [
        *_batch_arguments(input_dir, output_dir),
        "--flat-output",
    ]

    result = runner.invoke(
        app,
        arguments,
    )

    assert result.exit_code == 0

    output_images = [path for path in output_dir.glob("*.png") if path.is_file()]

    assert len(output_images) == 1
    assert len(output_images[0].stem) == 64


def test_empty_input_writes_empty_artifacts(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    result = runner.invoke(
        app,
        _batch_arguments(input_dir, output_dir),
    )

    assert result.exit_code == 0

    run_directories = _run_directories(output_dir)

    assert len(run_directories) == 1
    assert (run_directories[0] / "manifest.jsonl").read_bytes() == b""

    summary = json.loads((run_directories[0] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["total_discovered"] == 0
    assert summary["successful"] == 0


def _write_profile(
    path: Path,
    *,
    target_width: int = 36,
    target_height: int = 28,
) -> None:
    path.write_text(
        "\n".join(
            [
                'schema_version: "1.0"',
                'profile_id: "smart-beauty-test"',
                'profile_version: "1.0.0"',
                'model_family: "test"',
                "resize:",
                f"  target_width: {target_width}",
                f"  target_height: {target_height}",
                "  allow_upscale: true",
                "  max_upscale_factor: 4.0",
                "  padding_value: [127, 127, 127]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_batch_help_includes_profile_option() -> None:
    result = runner.invoke(
        app,
        ["batch", "--help"],
        color=False,
        terminal_width=160,
    )

    output = _strip_ansi(result.output)

    assert result.exit_code == 0
    assert "--profile" in output


def test_profile_driven_batch_command(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    profile_path = tmp_path / "profile.yaml"

    _write_image(input_dir / "sample.png", value=75)
    _write_profile(profile_path, target_width=36, target_height=28)

    result = runner.invoke(
        app,
        [
            "batch",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--profile",
            str(profile_path),
        ],
    )

    assert result.exit_code == 0, result.output

    output_image = np.asarray(Image.open(output_dir / "sample.png").convert("RGB"))
    assert output_image.shape == (28, 36, 3)


def test_profile_cannot_be_combined_with_manual_resize_options(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    profile_path = tmp_path / "profile.yaml"
    input_dir.mkdir()
    _write_profile(profile_path)

    result = runner.invoke(
        app,
        [
            "batch",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--profile",
            str(profile_path),
            "--target-width",
            "32",
        ],
    )

    assert result.exit_code == 1
    assert "cannot be combined" in _strip_ansi(result.output)
    assert _run_directories(output_dir) == []


def test_profile_cannot_be_combined_with_manual_boolean_option(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    profile_path = tmp_path / "profile.yaml"
    input_dir.mkdir()
    _write_profile(profile_path)

    result = runner.invoke(
        app,
        [
            "batch",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--profile",
            str(profile_path),
            "--no-allow-upscale",
        ],
    )

    assert result.exit_code == 1
    assert "cannot be combined" in _strip_ansi(result.output)
    assert _run_directories(output_dir) == []


def test_missing_profile_file_exits_one(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "batch",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--profile",
            str(tmp_path / "missing.yaml"),
        ],
    )

    assert result.exit_code == 1
    assert "ProfileConfigurationError" in _strip_ansi(result.output)
    assert _run_directories(output_dir) == []


def test_invalid_profile_exits_one(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    profile_path = tmp_path / "invalid.yaml"
    input_dir.mkdir()
    profile_path.write_text("not: [valid", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "batch",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--profile",
            str(profile_path),
        ],
    )

    assert result.exit_code == 1
    assert "ProfileConfigurationError" in _strip_ansi(result.output)
    assert _run_directories(output_dir) == []


def test_manual_mode_requires_both_target_dimensions(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "batch",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--target-width",
            "32",
        ],
    )

    assert result.exit_code == 1
    assert "--target-width and --target-height are required" in _strip_ansi(result.output)
    assert _run_directories(output_dir) == []


def test_strict_rgb8_cli_rejects_rgba_and_writes_policy_manifest(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    rgba = np.zeros((10, 20, 4), dtype=np.uint8)
    rgba[:, :, :3] = 120
    rgba[:, :, 3] = 128
    Image.fromarray(rgba, mode="RGBA").save(input_dir / "rgba.png")

    result = runner.invoke(
        app,
        [
            *_batch_arguments(input_dir, output_dir),
            "--input-policy",
            "strict_rgb8",
        ],
    )

    assert result.exit_code == 2, result.output
    run_directories = _run_directories(output_dir)
    assert len(run_directories) == 1

    records = [
        json.loads(line)
        for line in (run_directories[0] / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["error_type"] == "InputPolicyViolationError"
    assert records[0]["input_policy"] == "strict_rgb8"
    assert records[0]["decode_metadata"]["source_mode"] == "RGBA"
    assert not (output_dir / "rgba.png").exists()

    summary = json.loads(
        (run_directories[0] / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["input_policy"] == "strict_rgb8"
    assert summary["failed"] == 1


def test_invalid_input_policy_is_rejected_by_cli(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    result = runner.invoke(
        app,
        [
            *_batch_arguments(input_dir, output_dir),
            "--input-policy",
            "unknown",
        ],
    )

    assert result.exit_code == 2
    assert _run_directories(output_dir) == []


def test_profile_mode_accepts_explicit_input_policy(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    profile_path = tmp_path / "profile.yaml"

    _write_image(input_dir / "sample.png", value=75)
    _write_profile(profile_path)

    result = runner.invoke(
        app,
        [
            "batch",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--profile",
            str(profile_path),
            "--input-policy",
            "strict_rgb8",
        ],
    )

    assert result.exit_code == 0, result.output
    run_directories = _run_directories(output_dir)
    summary = json.loads(
        (run_directories[0] / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["input_policy"] == "strict_rgb8"
