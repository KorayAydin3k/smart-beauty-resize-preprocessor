from __future__ import annotations

from pathlib import Path

import pytest

from smart_beauty_resize.batch import (
    SUPPORTED_IMAGE_EXTENSIONS,
    build_output_relative_path,
    discover_images,
    is_supported_image_path,
)
from smart_beauty_resize.contracts import (
    DiscoveryError,
    SmartBeautyResizeError,
)


def test_supported_extensions_are_normalized() -> None:
    assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS
    assert ".jpeg" in SUPPORTED_IMAGE_EXTENSIONS
    assert ".png" in SUPPORTED_IMAGE_EXTENSIONS
    assert all(
        extension.startswith(".") and extension == extension.lower()
        for extension in SUPPORTED_IMAGE_EXTENSIONS
    )


@pytest.mark.parametrize(
    "filename",
    [
        "sample.jpg",
        "sample.JPEG",
        "sample.PNG",
        "sample.webp",
        "sample.TIFF",
    ],
)
def test_supported_image_path_is_case_insensitive(
    filename: str,
) -> None:
    assert is_supported_image_path(Path(filename))


@pytest.mark.parametrize(
    "filename",
    [
        "sample.txt",
        "sample.json",
        "sample.csv",
        "sample",
        "sample.jpg.tmp",
    ],
)
def test_unsupported_files_are_rejected(filename: str) -> None:
    assert not is_supported_image_path(Path(filename))


def test_discovery_is_recursive_and_deterministic(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)

    (input_dir / "z.PNG").write_bytes(b"z")
    (input_dir / "A.jpg").write_bytes(b"a")
    (nested / "b.jpeg").write_bytes(b"b")
    (nested / "ignore.txt").write_text(
        "ignore",
        encoding="utf-8",
    )

    first = discover_images(input_dir)
    second = discover_images(input_dir)

    assert first == second
    assert [item.relative_path.as_posix() for item in first] == [
        "A.jpg",
        "nested/b.jpeg",
        "z.PNG",
    ]

    assert all(item.source_path.is_absolute() for item in first)


def test_discovery_returns_empty_tuple_for_empty_directory(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()

    assert discover_images(input_dir) == ()


def test_discovery_ignores_file_symlinks(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    original = input_dir / "original.jpg"
    original.write_bytes(b"image")

    symlink = input_dir / "linked.jpg"

    try:
        symlink.symlink_to(original)
    except OSError:
        pytest.skip("File symlinks are unavailable")

    discovered = discover_images(input_dir)

    assert [item.relative_path.as_posix() for item in discovered] == ["original.jpg"]


def test_discovery_ignores_directory_symlinks(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    external = tmp_path / "external"
    input_dir.mkdir()
    external.mkdir()

    (external / "outside.jpg").write_bytes(b"outside")

    linked_directory = input_dir / "linked"

    try:
        linked_directory.symlink_to(
            external,
            target_is_directory=True,
        )
    except OSError:
        pytest.skip("Directory symlinks are unavailable")

    assert discover_images(input_dir) == ()


def test_discovery_rejects_missing_input(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        DiscoveryError,
        match="does not exist",
    ):
        discover_images(tmp_path / "missing")


def test_discovery_rejects_file_as_input(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "image.jpg"
    file_path.write_bytes(b"image")

    with pytest.raises(
        DiscoveryError,
        match="not a directory",
    ):
        discover_images(file_path)


def test_discovery_rejects_non_path_input() -> None:
    with pytest.raises(DiscoveryError):
        discover_images("input")  # type: ignore[arg-type]


def test_output_path_preserves_directory_structure() -> None:
    output = build_output_relative_path(
        Path("person/session/image.JPEG"),
        preserve_directory_structure=True,
    )

    assert output == Path("person/session/image.png")


def test_flat_output_path_is_deterministic_and_collision_resistant() -> None:
    first = build_output_relative_path(
        Path("person_a/image.jpg"),
        preserve_directory_structure=False,
    )
    repeated = build_output_relative_path(
        Path("person_a/image.jpg"),
        preserve_directory_structure=False,
    )
    second = build_output_relative_path(
        Path("person_b/image.jpg"),
        preserve_directory_structure=False,
    )

    assert first == repeated
    assert first != second
    assert first.parent == Path(".")
    assert first.suffix == ".png"
    assert len(first.stem) == 64


@pytest.mark.parametrize(
    "unsafe_path",
    [
        Path("/absolute/image.jpg"),
        Path("../outside.jpg"),
        Path("nested/../../outside.jpg"),
        Path("."),
    ],
)
def test_output_path_rejects_unsafe_relative_paths(
    unsafe_path: Path,
) -> None:
    with pytest.raises(DiscoveryError):
        build_output_relative_path(
            unsafe_path,
            preserve_directory_structure=True,
        )


def test_output_path_rejects_invalid_options() -> None:
    with pytest.raises(DiscoveryError):
        build_output_relative_path(
            Path("image.jpg"),
            preserve_directory_structure=1,  # type: ignore[arg-type]
        )

    with pytest.raises(DiscoveryError):
        build_output_relative_path(
            Path("image.jpg"),
            preserve_directory_structure=True,
            output_format="jpg",
        )


def test_discovery_error_inherits_package_base() -> None:
    assert issubclass(
        DiscoveryError,
        SmartBeautyResizeError,
    )
