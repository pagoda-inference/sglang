#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

export NCCL_SOCKET_IFNAME=YW-bond4
export GLOO_SOCKET_IFNAME=YW-bond4

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 16 \
  --ep-size 16 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 192.168.4.3:20001 \
  --trust-remote-code \
  --mem-fraction-static 0.80 \
  --disaggregation-mode prefill \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --enable-dsa-cache-layer-split \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 20002 \
  --disaggregation-ib-device mlx5_bond_0 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --disable-cuda-graph \
  --watchdog-timeout 1800 \
  --enable-layerwise-nvtx-marker
