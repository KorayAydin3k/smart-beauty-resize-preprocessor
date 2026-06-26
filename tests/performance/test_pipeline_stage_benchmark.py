from benchmarks.benchmark_pipeline_stages import CASES, STAGE_NAMES, run_case


def test_pipeline_stage_benchmark_returns_structured_measurements() -> None:
    result = run_case(
        CASES[0],
        image_count=2,
        warmup_iterations=0,
        measured_iterations=2,
    )

    assert result["name"] == "small_pipeline"
    assert result["image_count"] == 2
    assert result["measured_iterations"] == 2
    assert result["wall_clock_ms"]["median"] > 0
    assert result["throughput_images_per_second"] > 0
    assert result["throughput_source_megapixels_per_second"] > 0

    assert tuple(result["stages"]) == STAGE_NAMES
    assert all(result["stages"][stage]["median"] >= 0 for stage in STAGE_NAMES)
    assert all(
        result["stages"][stage]["median_share_percent"] >= 0 for stage in STAGE_NAMES
    )
    assert 99.0 <= sum(
        result["stages"][stage]["median_share_percent"] for stage in STAGE_NAMES
    ) <= 101.0

    assert result["bottleneck_stage"]["name"] in STAGE_NAMES
    assert result["bottleneck_stage"]["median_ms"] >= 0
    assert result["unattributed_overhead_ms"]["minimum"] >= 0
    assert len(result["source_aggregate_sha256"]) == 64
    assert len(result["output_aggregate_sha256"]) == 64
    assert len(result["config_sha256"]) == 64
