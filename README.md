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

A profile and manual preprocessing options are mutually exclusive. The profile
owns both resize settings and the source input policy. Runtime controls such as
`--overwrite`, `--fail-fast`, `--flat-output`, and `--verbose` remain available
in either mode.

The batch pipeline:

- discovers supported images recursively;
- processes files in deterministic order;
- applies EXIF orientation;
- preserves aspect ratio;
- uses letterbox padding without stretching or cropping;
- writes RGB PNG outputs;
- records SHA-256 provenance;
- writes `manifest.jsonl`, `run_summary.json`, and `dataset_audit.json`;
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

In manual mode, the default policy preserves the historical conversion behavior:

    --input-policy audit_only

For model pipelines that require native three-channel RGB sources with known
8-bit samples and no alpha/transparency, use:

    --input-policy strict_rgb8

Policy violations become per-image failed records unless `--fail-fast` is used.
The selected policy is stored in each manifest item and in `run_summary.json`.

Profile schema `1.1` stores the policy explicitly:

    schema_version: "1.1"
    profile_id: "smart-beauty-default"
    profile_version: "1.1.0"
    model_family: "shared"
    input_policy: "audit_only"
    resize:
      target_width: 512
      target_height: 512
      allow_upscale: true
      max_upscale_factor: 1.5
      padding_value: [127, 127, 127]

Legacy schema `1.0` profiles remain supported and resolve to `audit_only`.
`--input-policy` is a manual-mode option and cannot override a profile. The
resize-only `config_sha256` semantics remain unchanged.

### Dataset audit summary

Every completed run now writes `dataset_audit.json` beside the manifest and run
summary. The independent audit schema starts at `1.0` and links back to the run
through `run_id`, `config_sha256`, and `input_policy`.

The audit aggregates only observed manifest data. It includes:

- processing status and error-type counts;
- decode-metadata coverage;
- source format, mode, bit-depth, and channel-count distributions;
- alpha, ICC, EXIF, RGB-conversion, and bit-depth-conversion counts;
- source width, height, and pixel-count statistics using deterministic nearest-rank
  `p50`, `p95`, and `p99` values.

Records that fail before decoding remain visible through
`records_without_decode_metadata` and the error distribution. No source pixels,
resize geometry, interpolation, manifest schema, or `config_sha256` semantics are
changed by this report.

Programmatic aggregation is also available:

    from smart_beauty_resize.audit import build_dataset_audit_summary

    audit = build_dataset_audit_summary(batch_result)

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
