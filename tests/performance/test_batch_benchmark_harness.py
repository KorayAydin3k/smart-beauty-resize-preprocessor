from benchmarks.benchmark_batch import (
    CASES,
    run_case,
)


def test_batch_benchmark_returns_deterministic_metrics() -> None:
    result = run_case(
        CASES[0],
        image_count=3,
        warmup_iterations=0,
        measured_iterations=2,
    )

    assert result["name"] == "small_batch"
    assert result["image_count"] == 3
    assert result["measured_iterations"] == 2

    latency = result["latency_ms"]

    assert latency["minimum"] > 0
    assert latency["maximum"] >= latency["minimum"]
    assert latency["median"] >= latency["minimum"]
    assert latency["p95"] >= latency["minimum"]

    assert result["throughput_images_per_second"] > 0
    assert result["throughput_source_megapixels_per_second"] > 0

    assert len(result["source_aggregate_sha256"]) == 64
    assert len(result["output_aggregate_sha256"]) == 64
    assert len(result["config_sha256"]) == 64
