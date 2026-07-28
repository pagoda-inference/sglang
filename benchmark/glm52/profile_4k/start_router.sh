#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

exec python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --mini-lb \
  --prefill http://192.168.4.3:30000 20002 \
  --decode http://192.168.4.7:30000 \
  --host 0.0.0.0 \
  --port 31000
