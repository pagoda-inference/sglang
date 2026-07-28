#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

OUT_DIR=${OUT_DIR:-/tmp/glm52-profile-4k-b1}
mkdir -p "$OUT_DIR"

python3 -m sglang.benchmark.serving \
  --backend sglang-oai \
  --host 127.0.0.1 \
  --port 31000 \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group 1 \
  --gsp-system-prompt-len 0 \
  --gsp-question-len 4096 \
  --gsp-output-len 1 \
  --request-rate inf \
  --max-concurrency 1 \
  --num-prompts 1 \
  --warmup-requests 1 \
  --seed 123 \
  --disable-tqdm \
  --profile \
  --pd-separated \
  --profile-prefill-url http://127.0.0.1:30000 \
  --profile-activities CPU GPU \
  --profile-by-stage \
  --profile-stages prefill \
  --profile-output-dir "$OUT_DIR/traces" \
  --profile-prefix glm52-4k-b1-prefill \
  --output-file "$OUT_DIR/result.jsonl" \
  2>&1 | tee "$OUT_DIR/bench.log"
