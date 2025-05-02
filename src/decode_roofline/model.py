from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPUSpec:
    name: str
    hbm_gbps: float
    bf16_tflops: float


GPU_SPECS = {
    "h100": GPUSpec("H100 SXM", 3350.0, 989.5),
    "b200": GPUSpec("B200 SXM", 8000.0, 2250.0),
}


@dataclass(frozen=True)
class DecodeConfig:
    parameter_count: int = 8_030_000_000
    batch_size: int = 1
    context_length: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    bytes_per_weight: int = 2
    bytes_per_kv_element: int = 2

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError("attention heads must be divisible by KV heads")


@dataclass(frozen=True)
class DecodeTraffic:
    weight_bytes: int
    kv_read_bytes: int
    kv_write_bytes: int
    linear_flops: int
    attention_flops: int

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.kv_read_bytes + self.kv_write_bytes

    @property
    def total_flops(self) -> int:
        return self.linear_flops + self.attention_flops

    @property
    def arithmetic_intensity(self) -> float:
        return self.total_flops / self.total_bytes


def estimate_decode_traffic(config: DecodeConfig) -> DecodeTraffic:
    config.validate()
    weight_bytes = config.parameter_count * config.bytes_per_weight
    kv_elements_per_token = (
        2 * config.num_layers * config.num_kv_heads * config.head_dim
    )
    kv_read_bytes = (
        config.batch_size
        * config.context_length
        * kv_elements_per_token
        * config.bytes_per_kv_element
    )
    kv_write_bytes = config.batch_size * kv_elements_per_token * config.bytes_per_kv_element
    linear_flops = 2 * config.batch_size * config.parameter_count
    attention_flops = (
        4
        * config.batch_size
        * config.context_length
        * config.num_layers
        * config.num_attention_heads
        * config.head_dim
    )
    return DecodeTraffic(
        weight_bytes=weight_bytes,
        kv_read_bytes=kv_read_bytes,
        kv_write_bytes=kv_write_bytes,
        linear_flops=linear_flops,
        attention_flops=attention_flops,
    )


@dataclass(frozen=True)
class RooflineEstimate:
    arithmetic_intensity_flops_per_byte: float
    memory_time_ms: float
    compute_time_ms: float
    bound: str
    lower_bound_step_ms: float
    upper_bound_tokens_per_second: float
    attainable_tflops: float


def roofline(config: DecodeConfig, spec: GPUSpec) -> RooflineEstimate:
    if spec.hbm_gbps <= 0 or spec.bf16_tflops <= 0:
        raise ValueError("GPU peak rates must be positive")
    traffic = estimate_decode_traffic(config)
    memory_time_ms = traffic.total_bytes / (spec.hbm_gbps * 1e9) * 1e3
    compute_time_ms = traffic.total_flops / (spec.bf16_tflops * 1e12) * 1e3
    step_ms = max(memory_time_ms, compute_time_ms)
    return RooflineEstimate(
        arithmetic_intensity_flops_per_byte=traffic.arithmetic_intensity,
        memory_time_ms=memory_time_ms,
        compute_time_ms=compute_time_ms,
        bound="memory" if memory_time_ms >= compute_time_ms else "compute",
        lower_bound_step_ms=step_ms,
        upper_bound_tokens_per_second=config.batch_size / step_ms * 1e3,
        attainable_tflops=traffic.total_flops / step_ms / 1e9,
    )
