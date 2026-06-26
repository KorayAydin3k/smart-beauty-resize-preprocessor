# Changelog

All notable changes to this project are documented in this file.

The project follows semantic versioning. Schema versions for profiles, manifests,
and audit artifacts evolve independently and remain explicitly recorded in each
artifact.

## [1.0.0] - 2026-06-26

### Added

- Deterministic aspect-ratio-preserving letterbox resize with explicit geometry plans.
- OpenCV resize backend using area interpolation for downscaling and cubic interpolation
  for upscaling.
- Configurable upscale limits and deterministic RGB padding.
- EXIF-aware Pillow decoding into contiguous RGB `uint8` arrays.
- Non-breaking decode metadata API covering source format, mode, dimensions, bit depth,
  channel count, alpha, ICC, EXIF orientation, and conversion signals.
- Versioned preprocessing profiles with strict validation and compatibility for schemas
  `1.0`, `1.1`, and `1.2`.
- Input acceptance policies: `audit_only` and `strict_rgb8`.
- Pre-decode source width, height, and pixel-count safety limits.
- Deterministic recursive batch discovery with symlink exclusion.
- Output collision preflight before hashing, decoding, resizing, or writing.
- Atomic PNG output writing with overwrite and skip behavior.
- SHA-256 provenance for source files, output files, and resize configuration.
- JSONL image manifest, run summary, and dataset audit report.
- Deterministic flat-output naming based on source-relative-path SHA-256.
- CLI support for manual configuration and versioned profiles.
- Structured package-specific errors and documented exit codes.
- Golden, regression, failure-injection, stress, reproducibility, robustness, integration,
  property, and performance tests.
- Cross-platform reproducibility checks for Linux, macOS, and Windows.
- Stage-level and end-to-end performance measurement harnesses without hard
  hardware-dependent thresholds.

### Stable contracts

- Package version: `1.0.0`.
- Current preprocessing profile schema: `1.2`.
- Current manifest and run-summary schema: `1.3`.
- Current dataset-audit schema: `1.1`.
- Python support: `>=3.11`.
- Output image contract: RGB PNG with deterministic letterbox geometry.
- `config_sha256` remains intentionally scoped to resize configuration only.

### Handoff

- Added a production integration guide.
- Added a release and private acceptance-test checklist.
- Added contributor rules protecting determinism, schema compatibility, provenance,
  and user-image privacy.
