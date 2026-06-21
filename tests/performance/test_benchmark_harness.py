from benchmarks.benchmark_resize import (
    CASES,
    run_case,
)


def test_benchmark_case_returns_deterministic_metrics() -> None:
    result = run_case(
        CASES[0],
        warmup_iterations=1,
        measured_iterations=2,
    )

    assert result["name"] == "small_upscale"
    assert result["measured_iterations"] == 2

    latency = result["latency_ms"]

    assert latency["minimum"] >= 0
    assert latency["maximum"] >= latency["minimum"]
    assert latency["p95"] >= latency["minimum"]
    assert result["throughput_images_per_second"] > 0

    assert len(result["input_sha256"]) == 64
    assert len(result["output_pixel_sha256"]) == 64

    assert result["plan"]["resized_width"] > 0
    assert result["plan"]["resized_height"] > 0
