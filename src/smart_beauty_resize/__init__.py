"""Public package interface for smart-beauty-resize-preprocessor."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("smart-beauty-resize-preprocessor")
except PackageNotFoundError:
    __version__ = "0.1.0"
