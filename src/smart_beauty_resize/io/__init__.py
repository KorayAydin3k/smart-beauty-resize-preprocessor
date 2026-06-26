"""Image decoding contracts and public helpers."""

from smart_beauty_resize.io.contracts import (
    DecodedImage,
    ImageDecodeMetadata,
    InputPolicy,
    SourceImageLimits,
)
from smart_beauty_resize.io.decoder import decode_image, decode_image_with_metadata
from smart_beauty_resize.io.limits import enforce_source_image_limits
from smart_beauty_resize.io.policy import enforce_input_policy

__all__ = [
    "DecodedImage",
    "ImageDecodeMetadata",
    "InputPolicy",
    "SourceImageLimits",
    "decode_image",
    "decode_image_with_metadata",
    "enforce_input_policy",
    "enforce_source_image_limits",
]
