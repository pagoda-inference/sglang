#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 16 \
  --ep-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 192.168.4.7:21001 \
  --trust-remote-code \
  --mem-fraction-static 0.80 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 21002 \
  --disaggregation-ib-device mlx5_bond_0 \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --disable-cuda-graph \
  --watchdog-timeout 1800
