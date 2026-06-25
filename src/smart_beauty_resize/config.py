from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

from smart_beauty_resize.contracts import (
    ProfileConfigurationError,
    ResizeConfig,
    SmartBeautyResizeError,
)
from smart_beauty_resize.io.contracts import InputPolicy

CURRENT_PROFILE_SCHEMA_VERSION: Final = "1.1"
SUPPORTED_PROFILE_SCHEMA_VERSIONS: Final = frozenset({"1.0", "1.1"})
_PROFILE_ID_PATTERN: Final = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_SEMANTIC_VERSION_PATTERN: Final = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)
_BASE_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "profile_id",
        "profile_version",
        "model_family",
        "resize",
    }
)
_PROFILE_FIELDS_BY_SCHEMA: Final = {
    "1.0": _BASE_TOP_LEVEL_FIELDS,
    "1.1": _BASE_TOP_LEVEL_FIELDS | {"input_policy"},
}
_RESIZE_FIELDS: Final = frozenset(
    {
        "target_width",
        "target_height",
        "allow_upscale",
        "max_upscale_factor",
        "padding_value",
    }
)


@dataclass(frozen=True, slots=True)
class PreprocessingProfile:
    """Validated, versioned preprocessing contract.

    Schema ``1.0`` profiles contain resize settings only and resolve to the
    historical ``audit_only`` input policy. Schema ``1.1`` profiles make the
    source-image acceptance policy explicit.
    """

    schema_version: str
    profile_id: str
    profile_version: str
    model_family: str
    resize_config: ResizeConfig
    input_policy: InputPolicy = InputPolicy.AUDIT_ONLY

    def __post_init__(self) -> None:
        _validate_supported_schema_version(self.schema_version)
        _validate_identifier("profile_id", self.profile_id)
        _validate_semantic_version(self.profile_version)
        _validate_identifier("model_family", self.model_family)

        if not isinstance(self.resize_config, ResizeConfig):
            raise ProfileConfigurationError("resize_config must be a ResizeConfig instance")
        if not isinstance(self.input_policy, InputPolicy):
            raise ProfileConfigurationError("input_policy must be an InputPolicy instance")


def _validate_mapping(
    name: str,
    value: object,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ProfileConfigurationError(f"{name} must be a mapping")

    if any(type(key) is not str for key in value):
        raise ProfileConfigurationError(f"{name} keys must be strings")

    return value


def _validate_field_set(
    name: str,
    mapping: dict[str, object],
    expected_fields: frozenset[str],
) -> None:
    provided_fields = frozenset(mapping)
    missing = sorted(expected_fields - provided_fields)
    unknown = sorted(provided_fields - expected_fields)

    if missing:
        raise ProfileConfigurationError(f"{name} is missing required fields: {', '.join(missing)}")

    if unknown:
        raise ProfileConfigurationError(f"{name} contains unknown fields: {', '.join(unknown)}")


def _validate_supported_schema_version(value: object) -> str:
    if type(value) is not str:
        raise ProfileConfigurationError("schema_version must be a string")

    if value not in SUPPORTED_PROFILE_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_PROFILE_SCHEMA_VERSIONS))
        raise ProfileConfigurationError(
            f"schema_version must be one of the supported versions: {supported}"
        )

    return value


def _validate_identifier(
    name: str,
    value: object,
) -> str:
    if type(value) is not str or not value:
        raise ProfileConfigurationError(f"{name} must be a non-empty string")

    if value != value.strip():
        raise ProfileConfigurationError(f"{name} must not contain surrounding whitespace")

    if _PROFILE_ID_PATTERN.fullmatch(value) is None:
        raise ProfileConfigurationError(
            f"{name} must use lowercase letters, digits, '.', '_' or '-' separators"
        )

    return value


def _validate_semantic_version(value: object) -> str:
    if type(value) is not str or _SEMANTIC_VERSION_PATTERN.fullmatch(value) is None:
        raise ProfileConfigurationError(
            "profile_version must use MAJOR.MINOR.PATCH numeric semantic versioning"
        )

    return value


def _parse_padding_value(value: object) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ProfileConfigurationError("resize.padding_value must be a list of three integers")

    if any(type(channel) is not int for channel in value):
        raise ProfileConfigurationError("resize.padding_value entries must be integers")

    return (value[0], value[1], value[2])


def _parse_input_policy(value: object) -> InputPolicy:
    if type(value) is not str:
        raise ProfileConfigurationError("input_policy must be a string")

    try:
        return InputPolicy(value)
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in InputPolicy)
        raise ProfileConfigurationError(f"input_policy must be one of: {allowed}") from exc


def profile_from_mapping(payload: object) -> PreprocessingProfile:
    """Build a strict preprocessing profile from an in-memory mapping."""
    root = _validate_mapping("profile", payload)

    if "schema_version" not in root:
        raise ProfileConfigurationError("profile is missing required fields: schema_version")

    schema_version = _validate_supported_schema_version(root["schema_version"])
    _validate_field_set("profile", root, _PROFILE_FIELDS_BY_SCHEMA[schema_version])

    profile_id = _validate_identifier("profile_id", root["profile_id"])
    profile_version = _validate_semantic_version(root["profile_version"])
    model_family = _validate_identifier("model_family", root["model_family"])

    resize_payload = _validate_mapping("resize", root["resize"])
    _validate_field_set("resize", resize_payload, _RESIZE_FIELDS)

    try:
        resize_config = ResizeConfig(
            target_width=resize_payload["target_width"],  # type: ignore[arg-type]
            target_height=resize_payload["target_height"],  # type: ignore[arg-type]
            allow_upscale=resize_payload["allow_upscale"],  # type: ignore[arg-type]
            max_upscale_factor=resize_payload["max_upscale_factor"],  # type: ignore[arg-type]
            padding_value=_parse_padding_value(resize_payload["padding_value"]),
        )
    except SmartBeautyResizeError as exc:
        raise ProfileConfigurationError(f"invalid resize configuration: {exc}") from exc

    input_policy = (
        InputPolicy.AUDIT_ONLY
        if schema_version == "1.0"
        else _parse_input_policy(root["input_policy"])
    )

    return PreprocessingProfile(
        schema_version=schema_version,
        profile_id=profile_id,
        profile_version=profile_version,
        model_family=model_family,
        resize_config=resize_config,
        input_policy=input_policy,
    )


def load_preprocessing_profile(path: Path) -> PreprocessingProfile:
    """Load and validate a UTF-8 YAML preprocessing profile from disk."""
    if not isinstance(path, Path):
        raise ProfileConfigurationError("path must be a pathlib.Path instance")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProfileConfigurationError(f"unable to read preprocessing profile: {path}") from exc
    except UnicodeError as exc:
        raise ProfileConfigurationError(
            f"preprocessing profile must be valid UTF-8: {path}"
        ) from exc

    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ProfileConfigurationError(f"invalid YAML preprocessing profile: {path}") from exc

    return profile_from_mapping(payload)
