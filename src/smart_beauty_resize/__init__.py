"""Public package interface for smart-beauty-resize-preprocessor."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from smart_beauty_resize.contracts import (
    ExcessiveUpscaleError,
    InvalidImageDimensionsError,
    ResizeConfig,
    ResizeConfigurationError,
    ResizePlan,
)
from smart_beauty_resize.geometry import (
    apply_matrix_to_point,
    calculate_letterbox_plan,
    round_half_up_positive,
)

__all__ = [
    "__version__",
    "ResizeConfig",
    "ResizePlan",
    "ResizeConfigurationError",
    "InvalidImageDimensionsError",
    "ExcessiveUpscaleError",
    "round_half_up_positive",
    "calculate_letterbox_plan",
    "apply_matrix_to_point",
]

try:
    __version__ = version("smart-beauty-resize-preprocessor")
except PackageNotFoundError:
    __version__ = "0+unknown"
