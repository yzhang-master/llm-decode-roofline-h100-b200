# llm-decode-roofline-h100-b200

> **Era:** 2025+ · Staff AI Engineer focus area (measurement-driven LLM decode optimization). Technique reimplemented in 2026 with Nsight Compute and a dependency-free Python analyzer.

An arithmetic-intensity and HBM roofline model for autoregressive decode, a parser for Nsight Compute raw CSV and binary `.ncu-rep` reports, and tuning recommendations driven by measured duration and DRAM traffic.

## Problem

Low-batch decode is usually limited by reading weights and the KV cache, but “memory-bound” alone does not identify what to change. A useful analysis must distinguish insufficient achieved bandwidth from excess traffic, and must compare measured bytes with the bytes implied by the model configuration.

## Method

For one decode step producing `B` tokens, the analytical traffic is:

```
weight bytes   = P * weight_element_bytes
KV read bytes  = 2 * B * T * L * Hkv * D * kv_element_bytes
KV write bytes = 2 * B * L * Hkv * D * kv_element_bytes
linear FLOPs   = 2 * B * P
attention FLOPs = 4 * B * T * L * Hq * D
AI             = total FLOPs / total bytes
step roofline  = max(bytes / peak_HBM, FLOPs / peak_BF16)
```

The K/V factor 2 appears in both reads and writes; attention FLOPs cover score and value matmuls. The default shape is Llama-3-8B-like GQA: 8.03B parameters, 32 layers, 32 query heads, 8 KV heads and head dimension 128.

Nominal presets are H100 SXM (3350 GB/s, 989.5 dense BF16 TFLOPS) and B200 SXM (8000 GB/s, 2250 dense BF16 TFLOPS). These are hardware specification inputs, **not local measurements**. All can be replaced through `GPUSpec` in Python.

The Nsight parser sums unique kernel instances of:

- `gpu__time_duration.sum`, with ns/us/ms/s unit conversion;
- `dram__bytes_read.sum` and `dram__bytes_write.sum`, with binary KiB/MiB/GiB conversion.

It computes achieved HBM bandwidth, utilization, measured-byte arithmetic intensity, tokens/s and measured/roofline latency ratio. `.ncu-rep` files are imported by invoking `ncu --import <report> --csv --page raw`; no report is parsed by guessing binary bytes.

Recommendations respond to measured input: below 80% of nominal bandwidth triggers coalescing/occupancy/launch-gap investigation; at least 80% shifts priority to reducing bytes; measured traffic more than 20% above the model triggers an activation/workspace/KV-reread audit. Long-context FP8 KV advice explicitly requires an accuracy comparison.

## Results

Analytical defaults, batch 1, context 4096, BF16 weights and KV:

| Quantity | H100 SXM | B200 SXM |
|---|---:|---:|
| modeled weight + KV traffic | 16,597,001,984 bytes | 16,597,001,984 bytes |
| modeled FLOPs | 18,207,483,648 | 18,207,483,648 |
| arithmetic intensity | 1.0970344925 FLOP/byte | 1.0970344925 FLOP/byte |
| memory-limited lower-bound step | 4.9543289504 ms | 2.0746252480 ms |
| upper-bound tokens/s | 201.8436825657 | 482.0147643359 |

These are deterministic model outputs, not observed GPU latencies. **No H100 or B200 profiling has been run locally; achieved bandwidth and measured tokens/s are pending.**

`fixtures/synthetic_h100.csv` is hand-written synthetic parser input. It contains no real H100 data, and output preserves `origin: "synthetic"`. Its numbers must never be copied into a GPU results table.

## Reproduce

Run tests, the parser smoke and analytical model:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest -q
python -m decode_roofline.cli fixtures/synthetic_h100.csv --gpu h100
python -m decode_roofline.cli --model-only --gpu b200
```

For a real capture, wrap exactly one model decode step in an NVTX range named `decode_step` in the target application. With PyTorch, for example, use `torch.cuda.nvtx.range("decode_step")` around that step after warmup. Then run one of these exact commands:

```bash
# H100
ncu --target-processes all --nvtx --nvtx-include 'decode_step/' \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
  -o results/decode_h100 python your_decode_workload.py

# B200
ncu --target-processes all --nvtx --nvtx-include 'decode_step/' \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
  -o results/decode_b200 python your_decode_workload.py
```

Equivalent wrappers are `scripts/ncu_h100.sh` and `scripts/ncu_b200.sh`. Analyze either binary report:

```bash
python -m decode_roofline.cli results/decode_h100.ncu-rep --gpu h100 \
  --parameters 8030000000 --batch-size 1 --context-length 4096 \
  --layers 32 --attention-heads 32 --kv-heads 8 --head-dim 128 \
  --weight-bytes 2 --kv-bytes 2
```

Add `--output results/analysis.json` to preserve the analysis. Raw CSV exported with the same three metrics is accepted directly.

## Limitations

- The model assumes one ideal weight read per step and covers weights plus KV, not activations, workspaces, allocator traffic or host overhead. The excess-byte diagnostic makes some of those omissions visible.
- FLOPs are analytical; the profiler input supplies duration and DRAM bytes, not a measured Tensor Core FLOP count.
- Summed profiled-kernel durations exclude unprofiled kernels and gaps between kernels. Use a timeline tool for end-to-end latency and synchronize before/after the NVTX step in the workload.
- Nsight Compute replay and cache control can change cache behavior. Keep settings fixed when comparing tuning variants.
- Roofline peaks are nominal dense BF16/HBM rates and vary by board/form factor. Recommendations are triage heuristics, not proof that a proposed change improves accuracy or speed.
- No real GPU report is included; only the explicitly labeled synthetic parser fixture is present.

## References

- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* 2009. DOI:10.1145/1498765.1498785.
- NVIDIA. *Nsight Compute CLI.* https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- NVIDIA. *H100 Tensor Core GPU Architecture.* https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
- NVIDIA. *Blackwell Architecture Technical Brief.* https://resources.nvidia.com/en-us-blackwell-architecture

## License

MIT
