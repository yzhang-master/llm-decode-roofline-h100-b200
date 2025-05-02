from decode_roofline.analysis import analyze_measurement, recommendations
from decode_roofline.model import GPU_SPECS, DecodeConfig, GPUSpec
from decode_roofline.parser import NcuMeasurement


def measurement(duration_ms: float, read: int, write: int = 0) -> NcuMeasurement:
    return NcuMeasurement(duration_ms, read, write, 1, "test.csv", "ncu")


def test_measured_analysis_uses_observed_bytes_and_duration():
    config = DecodeConfig(
        parameter_count=1000,
        batch_size=2,
        context_length=16,
        num_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=8,
    )
    spec = GPUSpec("test", 100.0, 100.0)
    observed = measurement(duration_ms=0.001, read=10_000)
    result = analyze_measurement(config, spec, observed)
    assert result.achieved_hbm_gbps == 10.0
    assert result.hbm_utilization == 0.1
    assert result.measured_tokens_per_second == 2_000_000.0
    assert result.measured_arithmetic_intensity > 0
    assert result.excess_dram_bytes > 0


def test_recommendations_change_with_measured_bandwidth():
    config = DecodeConfig(context_length=128, bytes_per_kv_element=2)
    spec = GPUSpec("test", 100.0, 100.0)
    low = measurement(duration_ms=1.0, read=1_000_000)
    high = measurement(duration_ms=0.01, read=1_000_000)
    assert "below 80%" in recommendations(config, spec, low)[0]
    assert "near the nominal HBM roofline" in recommendations(config, spec, high)[0]


def test_long_context_recommendation_is_explicitly_accuracy_gated():
    config = DecodeConfig(context_length=8192)
    observed = measurement(duration_ms=1.0, read=1)
    tuning = recommendations(config, GPU_SPECS["h100"], observed)
    assert any("compare accuracy" in item for item in tuning)
