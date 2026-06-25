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
as `--overwrite`, `--fail-fast`, `--flat-output`, and `--verbose` remain available
in either mode.

The batch pipeline:

- discovers supported images recursively;
- processes files in deterministic order;
- applies EXIF orientation;
- preserves aspect ratio;
- uses letterbox padding without stretching or cropping;
- writes RGB PNG outputs;
- records SHA-256 provenance;
- writes `manifest.jsonl` and `run_summary.json`.

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
