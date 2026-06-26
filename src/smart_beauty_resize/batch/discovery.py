from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from smart_beauty_resize.contracts import (
    DiscoveryError,
    OutputPathCollisionError,
)

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {
        ".bmp",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)


def _absolute_without_symlink_resolution(path: Path) -> Path:
    """Return a normalized absolute path without resolving symlinks."""
    return Path(os.path.abspath(os.fspath(path)))


def _validate_relative_file_path(
    name: str,
    value: object,
) -> Path:
    if not isinstance(value, Path):
        raise DiscoveryError(f"{name} must be a pathlib.Path instance")

    if value.is_absolute():
        raise DiscoveryError(f"{name} must be relative")

    if value == Path(".") or not value.parts:
        raise DiscoveryError(f"{name} must identify a file")

    if ".." in value.parts:
        raise DiscoveryError(f"{name} must not contain parent-directory traversal")

    return value


@dataclass(frozen=True, slots=True)
class DiscoveredImage:
    """One deterministically discovered source image."""

    source_path: Path
    relative_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            raise DiscoveryError("source_path must be a pathlib.Path instance")

        if not self.source_path.is_absolute():
            raise DiscoveryError("source_path must be absolute")

        relative_path = _validate_relative_file_path(
            "relative_path",
            self.relative_path,
        )

        object.__setattr__(self, "relative_path", relative_path)


def is_supported_image_path(path: Path) -> bool:
    """Return whether a path has a supported image extension."""
    if not isinstance(path, Path):
        raise DiscoveryError("path must be a pathlib.Path instance")

    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def discover_images(input_dir: Path) -> tuple[DiscoveredImage, ...]:
    """Recursively discover supported non-symlink image files.

    Results are sorted deterministically by POSIX relative path.
    File and directory symlinks are deliberately ignored.
    """
    if not isinstance(input_dir, Path):
        raise DiscoveryError("input_dir must be a pathlib.Path instance")

    normalized_root = _absolute_without_symlink_resolution(input_dir)

    if not normalized_root.exists():
        raise DiscoveryError(f"input directory does not exist: {normalized_root}")

    if not normalized_root.is_dir():
        raise DiscoveryError(f"input path is not a directory: {normalized_root}")

    discovered: list[DiscoveredImage] = []

    try:
        for current_root, directory_names, file_names in os.walk(
            normalized_root,
            topdown=True,
            followlinks=False,
        ):
            current_directory = Path(current_root)

            directory_names[:] = sorted(
                (name for name in directory_names if not (current_directory / name).is_symlink()),
                key=lambda value: (value.casefold(), value),
            )

            for file_name in sorted(
                file_names,
                key=lambda value: (value.casefold(), value),
            ):
                source_path = current_directory / file_name

                if source_path.is_symlink():
                    continue

                if not source_path.is_file():
                    continue

                if not is_supported_image_path(source_path):
                    continue

                relative_path = source_path.relative_to(normalized_root)

                discovered.append(
                    DiscoveredImage(
                        source_path=source_path,
                        relative_path=relative_path,
                    )
                )
    except OSError as exc:
        raise DiscoveryError(f"unable to discover images under: {normalized_root}") from exc

    discovered.sort(
        key=lambda item: (
            item.relative_path.as_posix().casefold(),
            item.relative_path.as_posix(),
        )
    )

    return tuple(discovered)


def build_output_relative_path(
    source_relative_path: Path,
    *,
    preserve_directory_structure: bool,
    output_format: str = "png",
) -> Path:
    """Build a deterministic and safe output-relative path.

    When directory structure is not preserved, the complete source-relative
    path is hashed to prevent collisions between equal source filenames.
    """
    source_relative_path = _validate_relative_file_path(
        "source_relative_path",
        source_relative_path,
    )

    if type(preserve_directory_structure) is not bool:
        raise DiscoveryError("preserve_directory_structure must be a boolean")

    if type(output_format) is not str:
        raise DiscoveryError("output_format must be a string")

    normalized_format = output_format.strip().lower()

    if normalized_format != "png":
        raise DiscoveryError("output_format currently supports only 'png'")

    if preserve_directory_structure:
        return source_relative_path.with_suffix(".png")

    digest = hashlib.sha256(source_relative_path.as_posix().encode("utf-8")).hexdigest()

    return Path(f"{digest}.png")


def validate_unique_output_paths(
    discovered_images: tuple[DiscoveredImage, ...],
    *,
    preserve_directory_structure: bool,
    output_format: str = "png",
) -> None:
    """Reject source sets that would write more than one image to one output path.

    Output paths are compared case-insensitively so the same input set behaves
    consistently on case-sensitive and case-insensitive filesystems. The check
    is deterministic and must run before source hashing, decoding, or writing.
    """
    if not isinstance(discovered_images, tuple):
        raise DiscoveryError("discovered_images must be a tuple")

    outputs_by_key: dict[str, list[tuple[Path, Path]]] = {}

    for discovered in discovered_images:
        if not isinstance(discovered, DiscoveredImage):
            raise DiscoveryError("discovered_images entries must be DiscoveredImage instances")

        output_relative_path = build_output_relative_path(
            discovered.relative_path,
            preserve_directory_structure=preserve_directory_structure,
            output_format=output_format,
        )
        key = output_relative_path.as_posix().casefold()
        outputs_by_key.setdefault(key, []).append((output_relative_path, discovered.relative_path))

    collisions = [entries for entries in outputs_by_key.values() if len(entries) > 1]

    if not collisions:
        return

    collisions.sort(
        key=lambda entries: (
            min(path.as_posix() for path, _ in entries).casefold(),
            min(path.as_posix() for path, _ in entries),
        )
    )

    detail_lines: list[str] = []
    for entries in collisions:
        output_relative_path = min(
            (path for path, _ in entries),
            key=lambda path: (path.as_posix().casefold(), path.as_posix()),
        )
        source_paths = sorted(
            (source for _, source in entries),
            key=lambda path: (path.as_posix().casefold(), path.as_posix()),
        )
        sources = ", ".join(path.as_posix() for path in source_paths)
        detail_lines.append(f"- {output_relative_path.as_posix()} <- [{sources}]")

    details = "\n".join(detail_lines)
    raise OutputPathCollisionError(
        "multiple source images resolve to the same output path:\n" + details
    )
