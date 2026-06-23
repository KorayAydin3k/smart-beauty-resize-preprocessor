mkdir -p docs

cat > docs/phase-3-validation-report.md <<'MD'
# Phase 3 — Validation and Production Hardening Report

## 1. Objective

Phase 3 validates the deterministic Smart Beauty resize preprocessor beyond
functional correctness.

The validation scope includes:

- exact-output golden regression tests;
- repeated-run determinism;
- resize-kernel performance measurement;
- complete batch-pipeline performance measurement;
- controlled failure injection;
- large-batch stress testing;
- source-file immutability;
- temporary-file cleanup;
- file-descriptor leak detection;
- cross-version and cross-platform reproducibility.

## 2. Validation environment

Local measurements were collected in the following environment:

- Python: 3.11.15
- Operating system: Linux 6.8 Azure x86_64
- NumPy: 2.4.6
- OpenCV: 4.13.0
- Pillow: 12.2.0

Performance values are environment-specific and are not used as strict
continuous-integration thresholds.

Correctness, hashes, geometry, statuses, and deterministic output are treated
as acceptance requirements.

## 3. Golden regression validation

Six committed golden cases cover:

- landscape upscaling;
- portrait upscaling;
- landscape downscaling;
- odd-dimension rounding;
- identity processing;
- disabled upscaling.

Each case validates:

- exact output pixels;
- deterministic PNG encoding within a single runtime;
- source and expected SHA-256 hashes;
- resized dimensions;
- padding geometry;
- interpolation selection;
- output shape and dtype;
- source-array immutability.

Fixture regeneration was also verified to be idempotent.

Cross-platform acceptance is based on decoded RGB pixel equality. Compressed
PNG container bytes may differ between operating systems because the underlying
compression implementation can vary, even when the decoded image is identical.
Byte-level determinism is still required across repeated writes within the same
runtime.

## 4. Resize-kernel benchmark

The resize-kernel benchmark excludes file discovery, decoding, encoding,
filesystem writes, manifest serialization, and file hashing.

| Case | Run 1 median | Run 2 median | Median drift |
|---|---:|---:|---:|
| Small upscale | 2.227 ms | 2.103 ms | 5.56% |
| Medium downscale | 9.685 ms | 9.844 ms | 1.64% |
| Large downscale | 67.976 ms | 63.814 ms | 6.12% |
| Identity | 1.610 ms | 1.574 ms | 2.20% |

Across repeated runs:

- input hashes matched;
- output pixel hashes matched;
- resize geometry matched;
- interpolation selection matched;
- source images remained unchanged.

Result: **PASS**

## 5. End-to-end batch benchmark

The end-to-end benchmark includes:

- deterministic discovery;
- image decoding;
- EXIF-aware normalization;
- resizing and padding;
- PNG encoding;
- atomic filesystem publication;
- SHA-256 hashing;
- manifest generation;
- run-summary generation.

| Case | Images | Run 1 median | Run 2 median | Drift | Throughput range |
|---|---:|---:|---:|---:|---:|
| Small batch | 32 | 1097.007 ms | 1171.087 ms | 6.75% | 27.33–29.17 images/s |
| Medium batch | 12 | 756.894 ms | 792.767 ms | 4.74% | 15.14–15.85 images/s |
| Large batch | 3 | 1186.820 ms | 1245.093 ms | 4.91% | 2.41–2.53 images/s |

Across repeated runs:

- source aggregate hashes matched;
- output aggregate hashes matched;
- configuration hashes matched.

Result: **PASS**

## 6. Failure-injection validation

Controlled failures verify that:

- a corrupt image does not block valid images;
- expected per-image failures are recorded;
- fail-fast mode propagates the original package error;
- unexpected programming errors are not swallowed;
- atomic-writer failures do not publish incomplete files;
- temporary files are removed after writer failure;
- source bytes remain unchanged.

Result: **PASS**

## 7. Large-batch and resource validation

Stress coverage verifies:

- deterministic processing of a 96-image nested dataset;
- complete output generation;
- deterministic record ordering;
- source-file immutability;
- correct second-run skip behavior;
- continuation after one selective output-write failure;
- repeated overwrite runs without file-descriptor growth.

Result: **PASS**

## 8. Reproducibility matrix

The GitHub Actions reproducibility workflow validates:

- Ubuntu with Python 3.11;
- Ubuntu with Python 3.12;
- macOS with Python 3.11;
- Windows with Python 3.11.

The workflow runs:

- golden regression tests;
- exact-pixel reproducibility tests;
- repeated batch-output reproducibility tests.

Final acceptance requires every matrix job to pass.

## 9. Quality gates

The following project-wide checks are required:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
uv build
git diff --check