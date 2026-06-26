# Release Checklist

Use this checklist for the `1.0.0` release and adapt it for later semantic versions.

## 1. Repository state

- [ ] The release branch starts from the latest `main`.
- [ ] The working tree is clean.
- [ ] `pyproject.toml`, `uv.lock`, and `CHANGELOG.md` contain the same release version.
- [ ] Profile and artifact schema versions are documented and intentional.
- [ ] No patch files, private images, generated outputs, benchmark reports, `.venv`, or
      `dist/` files are staged.

## 2. Full validation

```bash
uv sync --all-groups --locked
uv run ruff check .
uv run mypy src
uv run pytest
uv build
git diff --check
```

- [ ] Lint passes.
- [ ] Type checking passes.
- [ ] The full test suite passes.
- [ ] Wheel and source distribution build successfully.
- [ ] The built package reports the expected version.

```bash
uv run python -c "import smart_beauty_resize; print(smart_beauty_resize.__version__)"
```

## 3. Private Smart Beauty acceptance run

Use an approved, representative, private dataset containing the formats, orientations,
resolutions, and edge cases expected in production. Do not copy it into the repository.

```bash
rm -rf data/release-acceptance-output

uv run smart-beauty-resize batch \
  --input-dir "$SMART_BEAUTY_ACCEPTANCE_INPUT" \
  --output-dir data/release-acceptance-output \
  --profile configs/default.yaml \
  --fail-fast
```

- [ ] Exit code is `0`.
- [ ] No unexpected source-policy or decode failures occur.
- [ ] No output collision occurs.
- [ ] All outputs are RGB PNG files with the expected target dimensions.
- [ ] Letterboxing preserves aspect ratio and does not crop facial regions.
- [ ] EXIF-rotated phone images are upright.
- [ ] Hair, clothing, and background content are not altered beyond resize and padding.
- [ ] `manifest.jsonl`, `run_summary.json`, and `dataset_audit.json` are present.
- [ ] Audit distributions and conversion counts are plausible for the dataset.
- [ ] Representative outputs are visually reviewed by the product/ML owner.

## 4. Determinism check

Run the same private dataset into a second empty output directory with the same profile.
Compare successful output hashes by source-relative path.

```bash
rm -rf data/release-acceptance-output-2

uv run smart-beauty-resize batch \
  --input-dir "$SMART_BEAUTY_ACCEPTANCE_INPUT" \
  --output-dir data/release-acceptance-output-2 \
  --profile configs/default.yaml \
  --fail-fast

uv run python - <<'PY'
import json
from pathlib import Path


def latest_manifest(root: Path) -> Path:
    run_dirs = sorted((root / "_runs").iterdir())
    if not run_dirs:
        raise RuntimeError(f"No run directory found under {root}")
    return run_dirs[-1] / "manifest.jsonl"


def output_hashes(root: Path) -> dict[str, str]:
    records = [
        json.loads(line)
        for line in latest_manifest(root).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        record["source_relative_path"]: record["output_sha256"]
        for record in records
        if record["status"] == "success"
    }

first = output_hashes(Path("data/release-acceptance-output"))
second = output_hashes(Path("data/release-acceptance-output-2"))

if first != second:
    raise SystemExit("Determinism check failed: output hash mappings differ")

print(f"Determinism check passed for {len(first)} successful images")
PY
```

- [ ] Output hash mappings are identical.
- [ ] `config_sha256` is identical across both runs.

## 5. Performance sanity check

```bash
uv run python -m benchmarks.benchmark_pipeline_stages --quick
```

- [ ] No extreme regression is visible relative to the same runner class.
- [ ] Any material performance change is explained in the release notes.
- [ ] No hardware-specific measurement is presented as a universal SLA.

## 6. Pull request and release

- [ ] The final pull request documents validation and private acceptance results without
      exposing private data.
- [ ] Required CI checks are green.
- [ ] The pull request is merged into `main`.
- [ ] Local `main` is updated with `git pull --ff-only origin main`.
- [ ] An annotated release tag is created from the merge commit.

```bash
git tag -a v1.0.0 -m "Smart Beauty Resize Preprocessor 1.0.0"
git push origin v1.0.0
```

- [ ] A GitHub Release is created from `v1.0.0` using the changelog as release notes.
- [ ] Built artifacts are attached only when the team requires binary handoff.
- [ ] PyPI or another public registry is not used without explicit authorization.

## 7. Post-release

- [ ] The consuming service installs the tagged wheel or pinned commit.
- [ ] A production-equivalent smoke test passes in the consuming environment.
- [ ] Profile ID, profile version, schema versions, and hashes are visible in observability.
- [ ] Private local acceptance outputs are removed according to data-retention policy.
