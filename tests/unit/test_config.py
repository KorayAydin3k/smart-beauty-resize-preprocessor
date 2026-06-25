from __future__ import annotations

from pathlib import Path

import pytest

from smart_beauty_resize.config import (
    PreprocessingProfile,
    load_preprocessing_profile,
    profile_from_mapping,
)
from smart_beauty_resize.contracts import ProfileConfigurationError, ResizeConfig
from smart_beauty_resize.io.contracts import InputPolicy


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": "1.1",
        "profile_id": "smart-beauty-acne",
        "profile_version": "1.1.0",
        "model_family": "acne",
        "input_policy": "strict_rgb8",
        "resize": {
            "target_width": 512,
            "target_height": 512,
            "allow_upscale": True,
            "max_upscale_factor": 1.5,
            "padding_value": [127, 127, 127],
        },
    }


def _legacy_payload() -> dict[str, object]:
    payload = _valid_payload()
    payload["schema_version"] = "1.0"
    payload["profile_version"] = "1.0.0"
    del payload["input_policy"]
    return payload


def test_profile_from_mapping_builds_current_preprocessing_contract() -> None:
    profile = profile_from_mapping(_valid_payload())

    assert profile == PreprocessingProfile(
        schema_version="1.1",
        profile_id="smart-beauty-acne",
        profile_version="1.1.0",
        model_family="acne",
        resize_config=ResizeConfig(
            target_width=512,
            target_height=512,
            allow_upscale=True,
            max_upscale_factor=1.5,
            padding_value=(127, 127, 127),
        ),
        input_policy=InputPolicy.STRICT_RGB8,
    )


def test_legacy_schema_defaults_to_audit_only() -> None:
    profile = profile_from_mapping(_legacy_payload())

    assert profile.schema_version == "1.0"
    assert profile.input_policy is InputPolicy.AUDIT_ONLY


def test_load_preprocessing_profile_reads_utf8_yaml(tmp_path: Path) -> None:
    profile_path = tmp_path / "acne.yaml"
    profile_path.write_text(
        """schema_version: \"1.1\"
profile_id: \"smart-beauty-acne\"
profile_version: \"1.1.0\"
model_family: \"acne\"
input_policy: \"audit_only\"
resize:
  target_width: 256
  target_height: 320
  allow_upscale: false
  max_upscale_factor: 1.0
  padding_value: [10, 20, 30]
""",
        encoding="utf-8",
    )

    profile = load_preprocessing_profile(profile_path)

    assert profile.resize_config.target_width == 256
    assert profile.resize_config.target_height == 320
    assert profile.resize_config.allow_upscale is False
    assert profile.resize_config.padding_value == (10, 20, 30)
    assert profile.input_policy is InputPolicy.AUDIT_ONLY


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "2.0"),
        ("profile_id", "Smart Beauty Acne"),
        ("profile_version", "1.0"),
        ("model_family", "Acne Model"),
    ],
)
def test_invalid_profile_identity_fields_are_rejected(
    field: str,
    value: object,
) -> None:
    payload = _valid_payload()
    payload[field] = value

    with pytest.raises(ProfileConfigurationError):
        profile_from_mapping(payload)


@pytest.mark.parametrize("value", ["unknown", "STRICT_RGB8", 1, None])
def test_invalid_input_policy_is_rejected(value: object) -> None:
    payload = _valid_payload()
    payload["input_policy"] = value

    with pytest.raises(ProfileConfigurationError, match="input_policy"):
        profile_from_mapping(payload)


def test_current_schema_requires_input_policy() -> None:
    payload = _valid_payload()
    del payload["input_policy"]

    with pytest.raises(ProfileConfigurationError, match="missing required fields: input_policy"):
        profile_from_mapping(payload)


def test_legacy_schema_rejects_future_input_policy_field() -> None:
    payload = _legacy_payload()
    payload["input_policy"] = "audit_only"

    with pytest.raises(ProfileConfigurationError, match="unknown fields: input_policy"):
        profile_from_mapping(payload)


def test_unknown_top_level_field_is_rejected() -> None:
    payload = _valid_payload()
    payload["future_option"] = True

    with pytest.raises(ProfileConfigurationError, match="unknown fields"):
        profile_from_mapping(payload)


def test_missing_top_level_field_is_rejected() -> None:
    payload = _valid_payload()
    del payload["model_family"]

    with pytest.raises(ProfileConfigurationError, match="missing required fields"):
        profile_from_mapping(payload)


def test_missing_schema_version_is_rejected_before_schema_dispatch() -> None:
    payload = _valid_payload()
    del payload["schema_version"]

    with pytest.raises(ProfileConfigurationError, match="schema_version"):
        profile_from_mapping(payload)


def test_unknown_resize_field_is_rejected() -> None:
    payload = _valid_payload()
    resize_payload = payload["resize"]
    assert isinstance(resize_payload, dict)
    resize_payload["interpolation"] = "lanczos"

    with pytest.raises(ProfileConfigurationError, match="unknown fields"):
        profile_from_mapping(payload)


def test_invalid_resize_value_preserves_profile_error_boundary() -> None:
    payload = _valid_payload()
    resize_payload = payload["resize"]
    assert isinstance(resize_payload, dict)
    resize_payload["target_width"] = 0

    with pytest.raises(ProfileConfigurationError, match="invalid resize configuration"):
        profile_from_mapping(payload)


def test_padding_value_must_be_yaml_list() -> None:
    payload = _valid_payload()
    resize_payload = payload["resize"]
    assert isinstance(resize_payload, dict)
    resize_payload["padding_value"] = (127, 127, 127)

    with pytest.raises(ProfileConfigurationError, match="list of three integers"):
        profile_from_mapping(payload)


def test_empty_yaml_document_is_rejected(tmp_path: Path) -> None:
    profile_path = tmp_path / "empty.yaml"
    profile_path.write_text("", encoding="utf-8")

    with pytest.raises(ProfileConfigurationError, match="profile must be a mapping"):
        load_preprocessing_profile(profile_path)


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    profile_path = tmp_path / "invalid.yaml"
    profile_path.write_text("profile: [unterminated", encoding="utf-8")

    with pytest.raises(ProfileConfigurationError, match="invalid YAML"):
        load_preprocessing_profile(profile_path)


def test_missing_profile_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProfileConfigurationError, match="unable to read"):
        load_preprocessing_profile(tmp_path / "missing.yaml")
