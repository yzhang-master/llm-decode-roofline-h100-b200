#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command running an NVTX range named decode_step>" >&2
  exit 2
fi
mkdir -p "$ROOT/results"
ncu --target-processes all \
  --nvtx --nvtx-include 'decode_step/' \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
  -o "$ROOT/results/decode_h100" "$@"
echo "analyze with: python -m decode_roofline.cli $ROOT/results/decode_h100.ncu-rep --gpu h100"
