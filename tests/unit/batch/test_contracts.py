from __future__ import annotations

from pathlib import Path

import pytest

from smart_beauty_resize import InputPolicy
from smart_beauty_resize.batch import (
    BatchConfig,
    ProcessingStatus,
)
from smart_beauty_resize.contracts import (
    BatchConfigurationError,
    ManifestSerializationError,
    ProvenanceError,
    ResizeConfig,
    SmartBeautyResizeError,
)


def _resize_config() -> ResizeConfig:
    return ResizeConfig(
        target_width=512,
        target_height=512,
    )


def test_processing_status_values() -> None:
    assert ProcessingStatus.SUCCESS.value == "success"
    assert ProcessingStatus.FAILED.value == "failed"
    assert ProcessingStatus.SKIPPED.value == "skipped"

    assert str(ProcessingStatus.SUCCESS) == "success"


def test_valid_batch_config_normalizes_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = BatchConfig(
        input_dir=Path("raw"),
        output_dir=Path("processed"),
        resize_config=_resize_config(),
        output_format=" PNG ",
    )

    assert config.input_dir == tmp_path / "raw"
    assert config.output_dir == tmp_path / "processed"
    assert config.output_format == "png"
    assert config.overwrite is False
    assert config.fail_fast is False
    assert config.preserve_directory_structure is True
    assert config.input_policy is InputPolicy.AUDIT_ONLY



def test_batch_config_accepts_explicit_input_policy(
    tmp_path: Path,
) -> None:
    config = BatchConfig(
        input_dir=tmp_path / "raw",
        output_dir=tmp_path / "processed",
        resize_config=_resize_config(),
        input_policy=InputPolicy.STRICT_RGB8,
    )

    assert config.input_policy is InputPolicy.STRICT_RGB8


def test_batch_config_rejects_string_input_policy(
    tmp_path: Path,
) -> None:
    with pytest.raises(BatchConfigurationError, match="input_policy"):
        BatchConfig(
            input_dir=tmp_path / "raw",
            output_dir=tmp_path / "processed",
            resize_config=_resize_config(),
            input_policy="strict_rgb8",  # type: ignore[arg-type]
        )

def test_batch_config_does_not_require_existing_directories(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "not-created-input"
    output_dir = tmp_path / "not-created-output"

    config = BatchConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        resize_config=_resize_config(),
    )

    assert not config.input_dir.exists()
    assert not config.output_dir.exists()


def test_batch_config_rejects_identical_paths(
    tmp_path: Path,
) -> None:
    same_dir = tmp_path / "images"

    with pytest.raises(
        BatchConfigurationError,
        match="must be different",
    ):
        BatchConfig(
            input_dir=same_dir,
            output_dir=same_dir,
            resize_config=_resize_config(),
        )


def test_batch_config_rejects_output_inside_input(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "raw"
    output_dir = input_dir / "processed"

    with pytest.raises(
        BatchConfigurationError,
        match="must not be located inside",
    ):
        BatchConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            resize_config=_resize_config(),
        )


def test_batch_config_accepts_sibling_output(
    tmp_path: Path,
) -> None:
    config = BatchConfig(
        input_dir=tmp_path / "raw",
        output_dir=tmp_path / "processed",
        resize_config=_resize_config(),
    )

    assert config.input_dir.parent == config.output_dir.parent


@pytest.mark.parametrize(
    "output_format",
    ["jpg", "jpeg", "webp", "", "tiff"],
)
def test_batch_config_rejects_unsupported_output_format(
    tmp_path: Path,
    output_format: str,
) -> None:
    with pytest.raises(BatchConfigurationError):
        BatchConfig(
            input_dir=tmp_path / "raw",
            output_dir=tmp_path / "processed",
            resize_config=_resize_config(),
            output_format=output_format,
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("overwrite", 1),
        ("overwrite", "false"),
        ("fail_fast", 0),
        ("fail_fast", None),
        ("preserve_directory_structure", 1),
        ("preserve_directory_structure", "true"),
    ],
)
def test_batch_config_rejects_non_boolean_flags(
    tmp_path: Path,
    field_name: str,
    field_value: object,
) -> None:
    kwargs: dict[str, object] = {
        "input_dir": tmp_path / "raw",
        "output_dir": tmp_path / "processed",
        "resize_config": _resize_config(),
        field_name: field_value,
    }

    with pytest.raises(
        BatchConfigurationError,
        match=f"{field_name} must be a boolean",
    ):
        BatchConfig(**kwargs)  # type: ignore[arg-type]


def test_batch_config_rejects_non_path_input() -> None:
    with pytest.raises(
        BatchConfigurationError,
        match="input_dir",
    ):
        BatchConfig(
            input_dir="raw",  # type: ignore[arg-type]
            output_dir=Path("processed"),
            resize_config=_resize_config(),
        )


def test_batch_config_rejects_non_path_output() -> None:
    with pytest.raises(
        BatchConfigurationError,
        match="output_dir",
    ):
        BatchConfig(
            input_dir=Path("raw"),
            output_dir="processed",  # type: ignore[arg-type]
            resize_config=_resize_config(),
        )


def test_batch_config_rejects_invalid_resize_config(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        BatchConfigurationError,
        match="resize_config",
    ):
        BatchConfig(
            input_dir=tmp_path / "raw",
            output_dir=tmp_path / "processed",
            resize_config=object(),  # type: ignore[arg-type]
        )


def test_phase_two_exceptions_inherit_from_package_base() -> None:
    assert issubclass(
        BatchConfigurationError,
        SmartBeautyResizeError,
    )
    assert issubclass(
        ProvenanceError,
        SmartBeautyResizeError,
    )
    assert issubclass(
        ManifestSerializationError,
        SmartBeautyResizeError,
    )
