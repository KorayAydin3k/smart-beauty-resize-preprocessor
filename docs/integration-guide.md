# Smart Beauty Resize Preprocessor 1.0 Integration Guide

## Purpose

The package converts heterogeneous source images into deterministic, model-ready RGB PNG
files while preserving aspect ratio and recording enough provenance to audit every output.
It is intended to run before acne, texture, pores, firmness, and other Smart Beauty vision
pipelines.

The preprocessor does not perform AI super-resolution, image beautification, semantic
retouching, or model inference. It must not invent skin detail.

## Supported runtime

- Python `>=3.11`
- Package version `1.0.0`
- Supported inputs by extension: BMP, JPEG, PNG, TIFF, and WebP
- Output format: RGB PNG

## Recommended CLI integration

Use the versioned default profile for production-equivalent processing:

```bash
uv run smart-beauty-resize batch \
  --input-dir data/raw \
  --output-dir data/resized \
  --profile configs/default.yaml
```

Runtime controls remain available with a profile:

```bash
uv run smart-beauty-resize batch \
  --input-dir data/raw \
  --output-dir data/resized \
  --profile configs/default.yaml \
  --fail-fast
```

Use `--overwrite` only when replacing existing outputs is intentional. It does not bypass
input-to-input output collisions. Use `--flat-output` when directory preservation is not
required and deterministic hash-based output names are preferred.

## Python batch integration

```python
from pathlib import Path

from smart_beauty_resize import load_preprocessing_profile
from smart_beauty_resize.batch import BatchConfig, process_batch

profile = load_preprocessing_profile(Path("configs/default.yaml"))

result = process_batch(
    BatchConfig(
        input_dir=Path("data/raw"),
        output_dir=Path("data/resized"),
        resize_config=profile.resize_config,
        input_policy=profile.input_policy,
        source_limits=profile.source_limits,
        fail_fast=True,
    )
)

if result.summary.failed:
    raise RuntimeError("Preprocessing completed with failed images")
```

The caller owns run-level orchestration, storage lifecycle, retry policy, observability, and
downstream model invocation.

## Direct image API

```python
from pathlib import Path

from smart_beauty_resize import (
    ResizeConfig,
    decode_image_with_metadata,
    resize_sample,
)

source = decode_image_with_metadata(Path("data/raw/example.jpg"))
result = resize_sample(
    source.image,
    ResizeConfig(target_width=512, target_height=512),
)

model_input = result.image
geometry = result.plan
source_metadata = source.metadata
```

Use the detailed decode API when audit metadata is required. The legacy `decode_image(...)`
API continues to return only the RGB array.

## Output and provenance contract

Each successful batch run writes processed PNG files plus a run directory under
`<output-dir>/_runs/<run-id>/` containing:

- `manifest.jsonl`: one structured record per discovered source image;
- `run_summary.json`: run-level status counts and configuration identity;
- `dataset_audit.json`: aggregate source-format, conversion, geometry, and error statistics.

Current schemas:

- manifest and run summary: `1.3`;
- dataset audit: `1.1`;
- preprocessing profile: `1.2`.

Downstream systems should:

1. Reject unknown future schema versions until compatibility is reviewed.
2. Require the expected profile identifier and profile version.
3. Store `config_sha256`, source SHA-256, output SHA-256, and run ID with model results.
4. Consume only records with `status == "success"`.
5. Use padding fields and original/resized dimensions when mapping coordinates between
   source images and model inputs.
6. Preserve the manifest and audit files for reproducibility and incident analysis.

## Error and exit-code contract

CLI exit codes:

- `0`: all discovered images completed without failure;
- `1`: fatal configuration, discovery, collision, or fail-fast error;
- `2`: the run completed, but at least one image failed.

Important structured errors include:

- `ProfileConfigurationError` for invalid profiles;
- `SourceImageLimitError` for oversized sources rejected before full decode;
- `InputPolicyViolationError` for policy-incompatible sources;
- `OutputPathCollisionError` for multiple sources resolving to one output path;
- `ImageDecodeError` for corrupt or unsupported image payloads.

Do not interpret a skipped output as a newly validated result. When overwrite is disabled,
an existing output can cause a skipped record.

## Determinism and deployment

For the same source bytes, resize configuration, supported dependency environment, and
output mode, output pixels and provenance hashes are expected to remain stable. Deployments
should use the committed lockfile and avoid unreviewed dependency upgrades.

Build artifacts:

```bash
uv sync --all-groups --locked
uv build
```

Install the generated wheel in the target service environment and verify:

```bash
python -c "import smart_beauty_resize; print(smart_beauty_resize.__version__)"
smart-beauty-resize --help
```

## Privacy and data handling

Real Smart Beauty acceptance images must remain in approved private storage. Do not commit
faces, manifests containing personal paths, generated outputs, or benchmark reports. Clean
local acceptance directories after release validation according to the team's data-retention
policy.
