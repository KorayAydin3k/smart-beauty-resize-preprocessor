from __future__ import annotations

import re
from pathlib import Path

import pytest

from smart_beauty_resize import ResizeConfig
from smart_beauty_resize.contracts import ProvenanceError
from smart_beauty_resize.provenance import (
    sha256_bytes,
    sha256_file,
    sha256_resize_config,
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def test_sha256_bytes_known_value() -> None:
    digest = sha256_bytes(b"abc")

    assert digest == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert SHA256_PATTERN.fullmatch(digest)


def test_sha256_bytes_empty_payload() -> None:
    digest = sha256_bytes(b"")

    assert digest == ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


def test_sha256_bytes_rejects_non_bytes() -> None:
    with pytest.raises(ProvenanceError, match="bytes"):
        sha256_bytes(bytearray(b"abc"))  # type: ignore[arg-type]


def test_sha256_file_matches_bytes_digest(tmp_path: Path) -> None:
    payload = b"smart-beauty-resize\n" * 1000
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(payload)

    assert sha256_file(file_path) == sha256_bytes(payload)


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 64, 1024])
def test_sha256_file_is_independent_of_chunk_size(
    tmp_path: Path,
    chunk_size: int,
) -> None:
    payload = bytes(range(256)) * 20
    file_path = tmp_path / "chunked.bin"
    file_path.write_bytes(payload)

    assert sha256_file(file_path, chunk_size=chunk_size) == sha256_bytes(payload)


@pytest.mark.parametrize(
    "chunk_size",
    [0, -1, True, 1.5, "1024"],
)
def test_sha256_file_rejects_invalid_chunk_size(
    tmp_path: Path,
    chunk_size: object,
) -> None:
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"content")

    with pytest.raises(
        ProvenanceError,
        match="positive integer",
    ):
        sha256_file(
            file_path,
            chunk_size=chunk_size,  # type: ignore[arg-type]
        )


def test_sha256_file_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.bin"

    with pytest.raises(
        ProvenanceError,
        match="unable to hash file",
    ):
        sha256_file(missing)


def test_sha256_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(
        ProvenanceError,
        match="unable to hash file",
    ):
        sha256_file(tmp_path)


def test_sha256_file_rejects_non_path() -> None:
    with pytest.raises(
        ProvenanceError,
        match="pathlib.Path",
    ):
        sha256_file("sample.bin")  # type: ignore[arg-type]


def test_resize_config_hash_is_deterministic() -> None:
    first = ResizeConfig(
        target_width=512,
        target_height=512,
        allow_upscale=True,
        max_upscale_factor=1.5,
        padding_value=(127, 127, 127),
    )
    second = ResizeConfig(
        target_width=512,
        target_height=512,
        allow_upscale=True,
        max_upscale_factor=1.5,
        padding_value=(127, 127, 127),
    )

    first_digest = sha256_resize_config(first)
    second_digest = sha256_resize_config(second)

    assert first_digest == second_digest
    assert SHA256_PATTERN.fullmatch(first_digest)


def test_resize_config_field_changes_change_hash() -> None:
    configs = [
        ResizeConfig(target_width=512, target_height=512),
        ResizeConfig(target_width=640, target_height=512),
        ResizeConfig(target_width=512, target_height=640),
        ResizeConfig(
            target_width=512,
            target_height=512,
            allow_upscale=False,
        ),
        ResizeConfig(
            target_width=512,
            target_height=512,
            max_upscale_factor=2.0,
        ),
        ResizeConfig(
            target_width=512,
            target_height=512,
            padding_value=(0, 0, 0),
        ),
    ]

    hashes = {sha256_resize_config(config) for config in configs}

    assert len(hashes) == len(configs)


def test_resize_config_hash_rejects_invalid_object() -> None:
    with pytest.raises(
        ProvenanceError,
        match="ResizeConfig",
    ):
        sha256_resize_config(object())  # type: ignore[arg-type]
