from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from smart_beauty_resize import decode_image
from smart_beauty_resize.contracts import (
    InvalidImageError,
    OutputExistsError,
    OutputWriteError,
    SmartBeautyResizeError,
)
from smart_beauty_resize.provenance import sha256_file
from smart_beauty_resize.writing import write_png_atomic


def _sample_image(value: int = 90) -> np.ndarray:
    image = np.full(
        (6, 8, 3),
        value,
        dtype=np.uint8,
    )

    image[:, :, 0] = value
    image[:, :, 1] = min(value + 10, 255)
    image[:, :, 2] = min(value + 20, 255)

    return image


def test_write_png_atomic_writes_exact_rgb_pixels(
    tmp_path: Path,
) -> None:
    image = _sample_image()

    output = write_png_atomic(
        image,
        tmp_path / "output",
        Path("sample.png"),
    )

    decoded = decode_image(output)

    assert output.is_absolute()
    assert output.is_file()
    assert np.array_equal(decoded, image)


def test_write_png_atomic_creates_nested_directories(
    tmp_path: Path,
) -> None:
    output = write_png_atomic(
        _sample_image(),
        tmp_path / "output",
        Path("person/session/sample.png"),
    )

    assert output == (tmp_path / "output" / "person" / "session" / "sample.png")
    assert output.is_file()


def test_write_png_atomic_does_not_modify_input(
    tmp_path: Path,
) -> None:
    image = _sample_image()
    before = image.copy()

    write_png_atomic(
        image,
        tmp_path / "output",
        Path("sample.png"),
    )

    assert np.array_equal(image, before)


def test_write_png_atomic_accepts_non_contiguous_input(
    tmp_path: Path,
) -> None:
    source = _sample_image()
    non_contiguous = source[:, ::-1, :]

    assert not non_contiguous.flags.c_contiguous

    output = write_png_atomic(
        non_contiguous,
        tmp_path / "output",
        Path("reversed.png"),
    )

    assert np.array_equal(
        decode_image(output),
        non_contiguous,
    )


def test_existing_output_is_preserved_without_overwrite(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    relative_path = Path("sample.png")

    first = write_png_atomic(
        _sample_image(40),
        output_root,
        relative_path,
    )
    original_bytes = first.read_bytes()

    with pytest.raises(
        OutputExistsError,
        match="already exists",
    ):
        write_png_atomic(
            _sample_image(200),
            output_root,
            relative_path,
            overwrite=False,
        )

    assert first.read_bytes() == original_bytes
    assert np.array_equal(
        decode_image(first),
        _sample_image(40),
    )


def test_overwrite_atomically_replaces_existing_output(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    relative_path = Path("sample.png")

    output = write_png_atomic(
        _sample_image(40),
        output_root,
        relative_path,
    )

    replaced = write_png_atomic(
        _sample_image(200),
        output_root,
        relative_path,
        overwrite=True,
    )

    assert replaced == output
    assert np.array_equal(
        decode_image(replaced),
        _sample_image(200),
    )


def test_equal_images_produce_equal_png_bytes(
    tmp_path: Path,
) -> None:
    image = _sample_image()

    first = write_png_atomic(
        image,
        tmp_path / "output",
        Path("first.png"),
    )
    second = write_png_atomic(
        image,
        tmp_path / "output",
        Path("second.png"),
    )

    assert first.read_bytes() == second.read_bytes()
    assert sha256_file(first) == sha256_file(second)


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("/absolute/sample.png"),
        Path("../outside.png"),
        Path("nested/../../outside.png"),
        Path("."),
        Path("sample.jpg"),
        Path("sample.PNG"),
    ],
)
def test_writer_rejects_unsafe_output_paths(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    with pytest.raises(OutputWriteError):
        write_png_atomic(
            _sample_image(),
            tmp_path / "output",
            relative_path,
        )


@pytest.mark.parametrize(
    "compression_level",
    [
        -1,
        10,
        True,
        1.5,
        "6",
    ],
)
def test_writer_rejects_invalid_compression_levels(
    tmp_path: Path,
    compression_level: object,
) -> None:
    with pytest.raises(
        OutputWriteError,
        match="compression_level",
    ):
        write_png_atomic(
            _sample_image(),
            tmp_path / "output",
            Path("sample.png"),
            compression_level=compression_level,  # type: ignore[arg-type]
        )


def test_writer_rejects_invalid_overwrite_value(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        OutputWriteError,
        match="overwrite",
    ):
        write_png_atomic(
            _sample_image(),
            tmp_path / "output",
            Path("sample.png"),
            overwrite=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.uint8),
    ],
)
def test_writer_rejects_invalid_images(
    tmp_path: Path,
    image: np.ndarray,
) -> None:
    with pytest.raises(InvalidImageError):
        write_png_atomic(
            image,  # type: ignore[arg-type]
            tmp_path / "output",
            Path("sample.png"),
        )


def test_writer_rejects_file_as_output_root(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "not-a-directory"
    output_root.write_text("content", encoding="utf-8")

    with pytest.raises(
        OutputWriteError,
        match="not a directory",
    ):
        write_png_atomic(
            _sample_image(),
            output_root,
            Path("sample.png"),
        )


def test_writer_cleans_temporary_file_after_encoding_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "output"

    def fail_save(
        self: Image.Image,
        *args: object,
        **kwargs: object,
    ) -> None:
        del self, args, kwargs
        raise OSError("simulated encoder failure")

    monkeypatch.setattr(
        Image.Image,
        "save",
        fail_save,
    )

    with pytest.raises(
        OutputWriteError,
        match="encode temporary PNG",
    ):
        write_png_atomic(
            _sample_image(),
            output_root,
            Path("sample.png"),
        )

    assert not (output_root / "sample.png").exists()
    assert list(output_root.glob(".*.tmp")) == []


def test_writer_rejects_symlink_target(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()

    real_file = tmp_path / "real.png"
    real_file.write_bytes(b"existing")

    symlink = output_root / "sample.png"

    try:
        symlink.symlink_to(real_file)
    except OSError:
        pytest.skip("File symlinks are unavailable")

    with pytest.raises(
        OutputWriteError,
        match="symbolic link",
    ):
        write_png_atomic(
            _sample_image(),
            output_root,
            Path("sample.png"),
            overwrite=True,
        )


def test_writer_rejects_symlink_parent_component(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    external = tmp_path / "external"
    output_root.mkdir()
    external.mkdir()

    linked_parent = output_root / "linked"

    try:
        linked_parent.symlink_to(
            external,
            target_is_directory=True,
        )
    except OSError:
        pytest.skip("Directory symlinks are unavailable")

    with pytest.raises(
        OutputWriteError,
        match="symbolic link",
    ):
        write_png_atomic(
            _sample_image(),
            output_root,
            Path("linked/sample.png"),
        )


def test_writer_exceptions_inherit_package_base() -> None:
    assert issubclass(
        OutputWriteError,
        SmartBeautyResizeError,
    )
    assert issubclass(
        OutputExistsError,
        OutputWriteError,
    )
