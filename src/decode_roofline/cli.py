import argparse
import json
from dataclasses import replace
from pathlib import Path

from .analysis import analysis_dict
from .model import GPU_SPECS, DecodeConfig, estimate_decode_traffic, roofline
from .parser import parse_ncu_report


def _config(args: argparse.Namespace) -> DecodeConfig:
    config = DecodeConfig()
    return replace(
        config,
        parameter_count=args.parameters,
        batch_size=args.batch_size,
        context_length=args.context_length,
        num_layers=args.layers,
        num_attention_heads=args.attention_heads,
        num_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        bytes_per_weight=args.weight_bytes,
        bytes_per_kv_element=args.kv_bytes,
    )


def main() -> None:
    defaults = DecodeConfig()
    parser = argparse.ArgumentParser(
        description="Decode roofline model and Nsight Compute CSV/.ncu-rep analyzer"
    )
    parser.add_argument("report", nargs="?", type=Path, help="raw CSV or .ncu-rep report")
    parser.add_argument("--gpu", choices=sorted(GPU_SPECS), default="h100")
    parser.add_argument("--parameters", type=int, default=defaults.parameter_count)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--context-length", type=int, default=defaults.context_length)
    parser.add_argument("--layers", type=int, default=defaults.num_layers)
    parser.add_argument("--attention-heads", type=int, default=defaults.num_attention_heads)
    parser.add_argument("--kv-heads", type=int, default=defaults.num_kv_heads)
    parser.add_argument("--head-dim", type=int, default=defaults.head_dim)
    parser.add_argument("--weight-bytes", type=int, default=defaults.bytes_per_weight)
    parser.add_argument("--kv-bytes", type=int, default=defaults.bytes_per_kv_element)
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.model_only and args.report:
        parser.error("--model-only does not take a report")
    if not args.model_only and not args.report:
        parser.error("provide an Nsight Compute report, or use --model-only")

    config = _config(args)
    spec = GPU_SPECS[args.gpu]
    if args.model_only:
        traffic = estimate_decode_traffic(config)
        result: dict[str, object] = {
            "gpu": spec.__dict__,
            "config": config.__dict__,
            "model": {
                **traffic.__dict__,
                "total_bytes": traffic.total_bytes,
                "total_flops": traffic.total_flops,
                **roofline(config, spec).__dict__,
            },
            "nominal_peaks_are_not_measurements": True,
        }
    else:
        measurement = parse_ncu_report(args.report)
        result = analysis_dict(config, spec, measurement)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
