from dataclasses import asdict, dataclass

from .model import DecodeConfig, GPUSpec, estimate_decode_traffic, roofline
from .parser import NcuMeasurement


@dataclass(frozen=True)
class MeasuredAnalysis:
    achieved_hbm_gbps: float
    hbm_utilization: float
    measured_arithmetic_intensity: float
    measured_tokens_per_second: float
    measured_to_roofline_latency_ratio: float
    excess_dram_bytes: int
    recommendations: tuple[str, ...]


def recommendations(
    config: DecodeConfig,
    spec: GPUSpec,
    measurement: NcuMeasurement,
) -> tuple[str, ...]:
    traffic = estimate_decode_traffic(config)
    estimate = roofline(config, spec)
    utilization = measurement.achieved_hbm_gbps / spec.hbm_gbps
    excess = measurement.dram_bytes - traffic.total_bytes
    result = []
    if estimate.bound == "compute":
        result.append(
            "The analytical decode shape is compute-bound; profile Tensor Core utilization "
            "before spending effort on HBM tuning."
        )
    elif utilization >= 0.8:
        result.append(
            "Measured DRAM traffic is near the nominal HBM roofline; reduce weight/KV bytes "
            "with quantization or cache reuse instead of relying on launch tuning."
        )
    else:
        result.append(
            "Measured bandwidth is below 80% of nominal HBM; inspect coalescing, occupancy, "
            "short kernels and exposed launch gaps before changing the arithmetic."
        )
    if excess > 0.2 * traffic.total_bytes:
        result.append(
            "Measured DRAM bytes exceed the weight+KV model by more than 20%; locate "
            "activation spills, workspace traffic or unintended KV-cache rereads."
        )
    if config.bytes_per_kv_element > 1 and config.context_length >= 4096:
        result.append(
            "Long-context KV traffic is material; evaluate FP8 KV and compare accuracy "
            "before adopting it."
        )
    if not result:
        result.append("No tuning signal crossed the configured thresholds.")
    return tuple(result)


def analyze_measurement(
    config: DecodeConfig, spec: GPUSpec, measurement: NcuMeasurement
) -> MeasuredAnalysis:
    traffic = estimate_decode_traffic(config)
    estimate = roofline(config, spec)
    utilization = measurement.achieved_hbm_gbps / spec.hbm_gbps
    return MeasuredAnalysis(
        achieved_hbm_gbps=measurement.achieved_hbm_gbps,
        hbm_utilization=utilization,
        measured_arithmetic_intensity=traffic.total_flops / measurement.dram_bytes,
        measured_tokens_per_second=config.batch_size / measurement.duration_ms * 1e3,
        measured_to_roofline_latency_ratio=measurement.duration_ms
        / estimate.lower_bound_step_ms,
        excess_dram_bytes=measurement.dram_bytes - traffic.total_bytes,
        recommendations=recommendations(config, spec, measurement),
    )


def analysis_dict(
    config: DecodeConfig, spec: GPUSpec, measurement: NcuMeasurement
) -> dict[str, object]:
    traffic = estimate_decode_traffic(config)
    estimate = roofline(config, spec)
    measured = analyze_measurement(config, spec, measurement)
    return {
        "gpu": asdict(spec),
        "config": asdict(config),
        "model": {
            **asdict(traffic),
            "total_bytes": traffic.total_bytes,
            "total_flops": traffic.total_flops,
            **asdict(estimate),
        },
        "measurement": {
            **asdict(measurement),
            "dram_bytes": measurement.dram_bytes,
        },
        "analysis": asdict(measured),
        "nominal_peaks_are_not_measurements": True,
    }
