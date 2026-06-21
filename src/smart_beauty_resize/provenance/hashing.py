from __future__ import annotations

import hashlib
import json
from pathlib import Path

from smart_beauty_resize.contracts import (
    ProvenanceError,
    ResizeConfig,
)

DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


def _validate_sha256_hex(digest: str) -> str:
    """Validate and return a lowercase SHA-256 hexadecimal digest."""
    if len(digest) != 64:
        raise ProvenanceError("SHA-256 digest must contain exactly 64 characters")

    if digest != digest.lower():
        raise ProvenanceError("SHA-256 digest must use lowercase hexadecimal characters")

    if any(character not in "0123456789abcdef" for character in digest):
        raise ProvenanceError("SHA-256 digest contains non-hexadecimal characters")

    return digest


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of an immutable byte sequence."""
    if type(data) is not bytes:
        raise ProvenanceError("data must be a bytes instance")

    digest = hashlib.sha256(data).hexdigest()
    return _validate_sha256_hex(digest)


def sha256_file(
    path: Path,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Stream a file and return its lowercase SHA-256 digest.

    The complete file is never loaded into memory. File-system failures are
    wrapped in ``ProvenanceError`` while preserving the original exception.
    """
    if not isinstance(path, Path):
        raise ProvenanceError("path must be a pathlib.Path instance")

    if type(chunk_size) is not int or chunk_size <= 0:
        raise ProvenanceError("chunk_size must be a positive integer")

    digest = hashlib.sha256()

    try:
        with path.open("rb") as file_handle:
            while True:
                chunk = file_handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ProvenanceError(f"unable to hash file: {path}") from exc

    return _validate_sha256_hex(digest.hexdigest())


def _canonical_resize_config_bytes(config: ResizeConfig) -> bytes:
    """Serialize ResizeConfig into deterministic canonical JSON bytes."""
    if not isinstance(config, ResizeConfig):
        raise ProvenanceError("config must be a ResizeConfig instance")

    payload: dict[str, object] = {
        "allow_upscale": config.allow_upscale,
        "max_upscale_factor": float(config.max_upscale_factor),
        "padding_value": list(config.padding_value),
        "target_height": config.target_height,
        "target_width": config.target_width,
    }

    try:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProvenanceError("unable to serialize resize configuration deterministically") from exc

    return serialized.encode("utf-8")


def sha256_resize_config(config: ResizeConfig) -> str:
    """Return a deterministic SHA-256 digest for a ResizeConfig."""
    return sha256_bytes(_canonical_resize_config_bytes(config))
