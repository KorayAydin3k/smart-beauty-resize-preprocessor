# Performance measurement

## Objective

Performance work follows a measure-first rule. The benchmark suite records
hardware-specific latency and throughput without introducing hard CI thresholds
or changing production behavior.

The project has three complementary benchmark layers:

1. `benchmark_resize` isolates the deterministic resize and padding kernel.
2. `benchmark_pipeline_stages` measures production-equivalent hashing, decode,
   policy, resize, PNG write, and output-hash stages.
3. `benchmark_batch` measures the complete batch flow, including discovery,
   collision preflight, output publication, and audit-artifact persistence.

Dataset generation and cleanup are outside measured sections. Every benchmark
also checks output determinism and source immutability.

## Stable environment

For comparable local runs, pin the package lock and reduce numerical-library
thread variability:

```bash
uv sync --all-groups --locked
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Quick smoke measurement

```bash
uv run python -m benchmarks.benchmark_resize --quick
uv run python -m benchmarks.benchmark_pipeline_stages --quick
uv run python -m benchmarks.benchmark_batch --quick
```

Generated JSON reports are written under `benchmarks/results/`, which is ignored
by Git. Each report records the runtime environment, latency distributions,
throughput, and deterministic hashes.

## Full measurement

```bash
uv run python -m benchmarks.benchmark_resize
uv run python -m benchmarks.benchmark_pipeline_stages
uv run python -m benchmarks.benchmark_batch
```

Run each full suite at least twice on an otherwise idle machine. Compare median
and p95 values only within the same hardware and software environment.

## Stage interpretation

The stage profiler reports:

- `discovery_and_preflight`
- `output_path_planning`
- `source_hash`
- `decode_and_normalize`
- `input_policy`
- `resize_and_pad`
- `png_encode_and_write`
- `output_hash`

`median_share_percent` identifies where measured time is concentrated.
`unattributed_overhead_ms` represents loop and timer overhead between the
instrumented sections. It should be observed, not optimized blindly.

An optimization should be proposed only when the same stage remains dominant
across repeated runs and relevant source resolutions. Any optimization must keep
pixel outputs, geometry, hashes, source immutability, and error behavior
unchanged.
