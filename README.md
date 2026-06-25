# smart-beauty-resize-preprocessor

<!-- BATCH-CLI-DOCS-START -->

## Batch preprocessing CLI

Install the project:

    uv sync --all-groups

Display help:

    uv run smart-beauty-resize --help
    uv run smart-beauty-resize batch --help
    uv run python -m smart_beauty_resize --help

Basic usage with manual resize options:

    uv run smart-beauty-resize batch \
      --input-dir data/raw \
      --output-dir data/resized \
      --target-width 512 \
      --target-height 512

Versioned profile usage:

    uv run smart-beauty-resize batch \
      --input-dir data/raw \
      --output-dir data/resized \
      --profile configs/default.yaml

A profile and manual resize options are mutually exclusive. Batch controls such
as `--input-policy`, `--overwrite`, `--fail-fast`, `--flat-output`, and `--verbose`
remain available in either mode.

The batch pipeline:

- discovers supported images recursively;
- processes files in deterministic order;
- applies EXIF orientation;
- preserves aspect ratio;
- uses letterbox padding without stretching or cropping;
- writes RGB PNG outputs;
- records SHA-256 provenance;
- writes `manifest.jsonl` and `run_summary.json`;
- records source decode metadata in each manifest item when decoding succeeds.

### Decode audit API

The original `decode_image(...)` API still returns the same contiguous RGB
`uint8` array. Callers that need source-format audit information can use the
non-breaking detailed API:

    from smart_beauty_resize import decode_image_with_metadata

    decoded = decode_image_with_metadata("data/raw/example.png")
    image = decoded.image
    metadata = decoded.metadata

The metadata reports source format, mode, geometry, sample bit depth, alpha and
ICC presence, EXIF orientation handling, and RGB/bit-depth conversion signals.
Batch-generated manifest records persist the same metadata under a nested
`decode_metadata` object. Records that fail before decoding store `null`.

### Input acceptance policies

The default policy preserves the historical conversion behavior:

    --input-policy audit_only

For model pipelines that require native three-channel RGB sources with known
8-bit samples and no alpha/transparency, use:

    --input-policy strict_rgb8

Policy violations become per-image failed records unless `--fail-fast` is used.
The selected policy is stored in each manifest item and in `run_summary.json`.
This phase keeps the profile schema at `1.0`; policy selection remains an
explicit batch control and is not included in the resize-only `config_sha256`.

### Exit codes

- `0`: completed without failed images;
- `1`: fatal configuration or processing error;
- `2`: completed, but one or more images failed.

### Development checks

    uv run ruff check .
    uv run mypy src
    uv run pytest
    uv build

<!-- BATCH-CLI-DOCS-END -->
