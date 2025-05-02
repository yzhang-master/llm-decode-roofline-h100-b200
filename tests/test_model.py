import pytest

from decode_roofline.model import (
    GPU_SPECS,
    DecodeConfig,
    GPUSpec,
    estimate_decode_traffic,
    roofline,
)


def small_config() -> DecodeConfig:
    return DecodeConfig(
        parameter_count=1_000_000,
        batch_size=2,
        context_length=128,
        num_layers=2,
        num_attention_heads=8,
        num_kv_heads=2,
        head_dim=64,
    )


def test_traffic_formula_counts_weights_kv_reads_writes_and_flops():
    traffic = estimate_decode_traffic(small_config())
    assert traffic.weight_bytes == 2_000_000
    assert traffic.kv_read_bytes == 262_144
    assert traffic.kv_write_bytes == 2_048
    assert traffic.total_bytes == 2_264_192
    assert traffic.linear_flops == 4_000_000
    assert traffic.attention_flops == 1_048_576
    assert traffic.total_flops == 5_048_576
    assert traffic.arithmetic_intensity == pytest.approx(5_048_576 / 2_264_192)


def test_roofline_uses_slower_of_memory_and_compute():
    config = small_config()
    estimate = roofline(config, GPU_SPECS["h100"])
    traffic = estimate_decode_traffic(config)
    assert estimate.bound == "memory"
    assert estimate.memory_time_ms == pytest.approx(traffic.total_bytes / 3.35e12 * 1e3)
    assert estimate.compute_time_ms == pytest.approx(traffic.total_flops / 989.5e12 * 1e3)
    assert estimate.lower_bound_step_ms == estimate.memory_time_ms
    assert estimate.upper_bound_tokens_per_second == pytest.approx(
        config.batch_size / estimate.memory_time_ms * 1e3
    )


def test_compute_bound_shape_uses_compute_limit():
    config = DecodeConfig(
        parameter_count=1,
        batch_size=1024,
        context_length=1,
        num_layers=1,
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=1,
        bytes_per_weight=1,
        bytes_per_kv_element=1,
    )
    assert roofline(config, GPUSpec("tiny", 1000.0, 0.001)).bound == "compute"


def test_validation_rejects_invalid_gqa_and_sizes():
    with pytest.raises(ValueError, match="divisible"):
        estimate_decode_traffic(DecodeConfig(num_attention_heads=7, num_kv_heads=2))
    with pytest.raises(ValueError, match="batch_size"):
        estimate_decode_traffic(DecodeConfig(batch_size=0))
