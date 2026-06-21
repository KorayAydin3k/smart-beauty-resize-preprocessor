from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from smart_beauty_resize.contracts import (
    InvalidImageError,
    OutputExistsError,
    OutputWriteError,
)

DEFAULT_PNG_COMPRESSION_LEVEL = 6


def _absolute_without_symlink_resolution(path: Path) -> Path:
    """Return a normalized absolute path without resolving symbolic links."""
    return Path(os.path.abspath(os.fspath(path)))


def _validate_output_relative_path(value: object) -> Path:
    if not isinstance(value, Path):
        raise OutputWriteError("relative_path must be a pathlib.Path instance")

    if value.is_absolute():
        raise OutputWriteError("relative_path must be relative")

    if value == Path(".") or not value.parts:
        raise OutputWriteError("relative_path must identify an output file")

    if ".." in value.parts:
        raise OutputWriteError("relative_path must not contain parent-directory traversal")

    if any("\\" in part for part in value.parts):
        raise OutputWriteError("relative_path must use platform-safe path components")

    if value.suffix != ".png":
        raise OutputWriteError("relative_path must use the lowercase '.png' extension")

    return value


def _validate_image(image: object) -> NDArray[np.uint8]:
    if not isinstance(image, np.ndarray):
        raise InvalidImageError("image must be a NumPy array")

    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError("image must have shape (height, width, 3)")

    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise InvalidImageError("image dimensions must be positive")

    if image.dtype != np.dtype(np.uint8):
        raise InvalidImageError("image must have dtype uint8")

    return cast(
        NDArray[np.uint8],
        np.ascontiguousarray(image),
    )


def _validate_compression_level(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 9:
        raise OutputWriteError("compression_level must be an integer in the range [0, 9]")

    return value


def _prepare_output_root(output_root: object) -> Path:
    if not isinstance(output_root, Path):
        raise OutputWriteError("output_root must be a pathlib.Path instance")

    normalized_root = _absolute_without_symlink_resolution(output_root)

    if normalized_root.is_symlink():
        raise OutputWriteError("output_root must not be a symbolic link")

    if normalized_root.exists() and not normalized_root.is_dir():
        raise OutputWriteError("output_root exists but is not a directory")

    try:
        normalized_root.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise OutputWriteError(f"unable to create output directory: {normalized_root}") from exc

    if normalized_root.is_symlink():
        raise OutputWriteError("output_root must not be a symbolic link")

    if not normalized_root.is_dir():
        raise OutputWriteError("output_root is not a directory")

    return normalized_root


def _reject_symlink_components(
    output_root: Path,
    target_parent: Path,
) -> None:
    try:
        relative_parent = target_parent.relative_to(output_root)
    except ValueError as exc:
        raise OutputWriteError("target path escapes output_root") from exc

    current = output_root

    if current.is_symlink():
        raise OutputWriteError("output_root must not be a symbolic link")

    for component in relative_parent.parts:
        current = current / component

        if current.is_symlink():
            raise OutputWriteError(f"output path contains a symbolic link: {current}")


def _write_temporary_png(
    image: NDArray[np.uint8],
    target_parent: Path,
    target_name: str,
    compression_level: int,
) -> Path:
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=target_parent,
            prefix=f".{target_name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            pil_image = Image.fromarray(image)

            if pil_image.mode != "RGB":
                raise OutputWriteError("validated RGB array produced a non-RGB Pillow image")

            pil_image.save(
                temporary_file,
                format="PNG",
                optimize=False,
                compress_level=compression_level,
            )

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        if temporary_path.stat().st_size <= 0:
            raise OutputWriteError("temporary PNG output is empty")

        return temporary_path

    except OutputWriteError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

        raise OutputWriteError("unable to encode temporary PNG output") from exc


def _publish_temporary_file(
    temporary_path: Path,
    target_path: Path,
    *,
    overwrite: bool,
) -> None:
    try:
        if overwrite:
            os.replace(
                temporary_path,
                target_path,
            )
            return

        try:
            os.link(
                temporary_path,
                target_path,
            )
        except FileExistsError as exc:
            raise OutputExistsError(f"output already exists: {target_path}") from exc

    except OutputExistsError:
        raise

    except OSError as exc:
        raise OutputWriteError(f"unable to publish output atomically: {target_path}") from exc


def write_png_atomic(
    image: NDArray[np.uint8],
    output_root: Path,
    relative_path: Path,
    *,
    overwrite: bool = False,
    compression_level: int = DEFAULT_PNG_COMPRESSION_LEVEL,
) -> Path:
    """Write an RGB uint8 image as a safely published PNG.

    The PNG is fully encoded and flushed into a temporary file in the target
    directory before publication.

    With ``overwrite=False``, an atomic hard-link operation prevents an
    existing output from being replaced. With ``overwrite=True``,
    ``os.replace`` atomically replaces the existing output.

    The returned path is absolute.
    """
    if type(overwrite) is not bool:
        raise OutputWriteError("overwrite must be a boolean")

    validated_image = _validate_image(image)
    validated_relative_path = _validate_output_relative_path(relative_path)
    validated_compression_level = _validate_compression_level(compression_level)
    normalized_root = _prepare_output_root(output_root)

    target_path = normalized_root / validated_relative_path
    target_parent = target_path.parent

    try:
        target_parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise OutputWriteError(
            f"unable to create output parent directory: {target_parent}"
        ) from exc

    _reject_symlink_components(
        normalized_root,
        target_parent,
    )

    if target_path.is_symlink():
        raise OutputWriteError(f"output target must not be a symbolic link: {target_path}")

    temporary_path: Path | None = None

    try:
        temporary_path = _write_temporary_png(
            validated_image,
            target_parent,
            target_path.name,
            validated_compression_level,
        )

        _publish_temporary_file(
            temporary_path,
            target_path,
            overwrite=overwrite,
        )

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    if target_path.is_symlink():
        raise OutputWriteError("published output unexpectedly became a symbolic link")

    if not target_path.is_file():
        raise OutputWriteError("published output file does not exist")

    if target_path.stat().st_size <= 0:
        raise OutputWriteError("published output file is empty")

    return target_path
