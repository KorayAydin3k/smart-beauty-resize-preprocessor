from __future__ import annotations

from smart_beauty_resize.contracts import (
    ImageDecodeError,
    InputPolicyViolationError,
)
from smart_beauty_resize.io.contracts import ImageDecodeMetadata, InputPolicy


def enforce_input_policy(
    metadata: ImageDecodeMetadata,
    policy: InputPolicy,
) -> None:
    """Validate decoded source metadata against one explicit input policy.

    ``audit_only`` preserves the historical behavior and accepts every source
    that the decoder can canonicalize. ``strict_rgb8`` requires an unambiguous
    three-channel RGB source with eight-bit samples and no alpha/transparency.
    """
    if not isinstance(metadata, ImageDecodeMetadata):
        raise ImageDecodeError("metadata must be an ImageDecodeMetadata instance")
    if not isinstance(policy, InputPolicy):
        raise ImageDecodeError("policy must be an InputPolicy instance")

    if policy is InputPolicy.AUDIT_ONLY:
        return

    violations: list[str] = []

    if metadata.source_mode != "RGB":
        violations.append(f"source_mode={metadata.source_mode} (expected RGB)")
    if metadata.source_bit_depth != 8:
        violations.append(
            f"source_bit_depth={metadata.source_bit_depth} (expected 8)"
        )
    if metadata.source_channel_count != 3:
        violations.append(
            f"source_channel_count={metadata.source_channel_count} (expected 3)"
        )
    if metadata.alpha_present:
        violations.append("alpha_present=true (expected false)")

    if violations:
        details = "; ".join(violations)
        raise InputPolicyViolationError(
            f"strict_rgb8 rejected source image: {details}"
        )
