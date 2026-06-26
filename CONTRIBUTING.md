# Contributing

This repository is a deterministic preprocessing component used upstream of Smart Beauty
computer-vision models. Changes must preserve reproducibility, auditability, and user trust.

## Development setup

Create a branch from the latest `main` and install the locked environment:

```bash
git switch main
git pull --ff-only origin main
git switch -c <branch-name>
uv sync --all-groups --locked
```

## Required validation

Run the complete validation set before opening a pull request:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
uv build
git diff --check
```

Performance changes must also run the relevant benchmark harness. Benchmark results are
hardware-specific and must not be converted into hard CI thresholds without an approved,
stable runner and an explicit product SLA.

## Non-negotiable invariants

- Do not stretch or crop source images unless a future versioned contract explicitly says so.
- Preserve deterministic geometry, interpolation selection, output pixels, discovery order,
  and serialization behavior for unchanged configurations.
- Apply EXIF orientation before geometry-dependent processing.
- Enforce source limits before full decode or NumPy allocation.
- Run output-collision preflight before source hashing, decoding, resizing, or writing.
- Keep output writes atomic.
- Never silently change profile, manifest, run-summary, audit, or hash semantics.
- Keep profile/manual configuration precedence explicit and tested.
- Unexpected programming errors must propagate rather than being converted into misleading
  image-level failures.

## Schema and compatibility changes

Any externally visible contract change requires all of the following:

1. A schema or semantic-version decision documented in the pull request.
2. Backward-compatibility behavior or a clearly documented breaking change.
3. Unit and integration tests for old and new behavior.
4. README, integration guide, and changelog updates.
5. Confirmation that existing deterministic outputs remain unchanged when the contract is
   not intended to alter pixels.

## Test data and privacy

- Never commit real users, faces, internal datasets, credentials, or customer metadata.
- Use synthetic images or explicitly approved non-sensitive fixtures.
- Keep local acceptance datasets, generated outputs, benchmark reports, and `dist/` artifacts
  out of Git.
- Do not include private filesystem paths or personal data in committed manifests or docs.

## Pull request expectations

A pull request should explain:

- the problem and intended product impact;
- the exact contract being added or changed;
- what remains unchanged;
- validation commands and results;
- manual smoke tests for safety-critical behavior;
- schema, compatibility, privacy, and performance implications.
