#!/usr/bin/env bash

set -u

export PYTHONPATH="${PYTHONPATH:-/tmp/sglang/python}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-31000}"
MODEL="${MODEL:-/models/public/modellist/452/V1}"
OUT_DIR="${OUT_DIR:-/tmp/bench-pd-noflush}"
SEED="${SEED:-123}"
ONLY_SCENARIOS="${ONLY_SCENARIOS:-}"

mkdir -p "$OUT_DIR"

scenarios=(
  "4k_0pct 0 4096"
  "16k_90pct 14745 1639"
  "50k_90pct 46080 5120"
)

for scenario in "${scenarios[@]}"; do
  read -r name prefix_len question_len <<< "$scenario"
  if [[ -n "$ONLY_SCENARIOS" && " $ONLY_SCENARIOS " != *" $name "* ]]; then
    continue
  fi
  input_len=$((prefix_len + question_len))
  if (( input_len == 0 )); then
    input_len=4096
    question_len=4096
  fi

  for concurrency in 1 2 4 8 16; do
    num_prompts=$((concurrency * 3))
    result="$OUT_DIR/${name}-c${concurrency}.jsonl"
    log="$OUT_DIR/${name}-c${concurrency}.log"
    rm -f "$result"

    echo "[$(date '+%F %T')] scenario=$name input=$input_len prefix=$prefix_len concurrency=$concurrency prompts=$num_prompts"
    python3 -m sglang.benchmark.serving \
      --backend sglang-oai \
      --host "$HOST" \
      --port "$PORT" \
      --model "$MODEL" \
      --tokenizer "$MODEL" \
      --dataset-name generated-shared-prefix \
      --gsp-num-groups 1 \
      --gsp-prompts-per-group "$num_prompts" \
      --gsp-system-prompt-len "$prefix_len" \
      --gsp-question-len "$question_len" \
      --gsp-output-len 1 \
      --request-rate inf \
      --max-concurrency "$concurrency" \
      --num-prompts "$num_prompts" \
      --warmup-requests 0 \
      --seed "$SEED" \
      --disable-tqdm \
      --output-file "$result" \
      2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    echo "[$(date '+%F %T')] scenario=$name concurrency=$concurrency rc=$rc"
  done
done

python3 - "$OUT_DIR" <<'PY'
import csv
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
columns = [
    "prefix_cache", "prefix_len", "input", "output", "request_rate",
    "num_prompts", "max_concurrency", "mean_concurrency",
    "Peak_concurrent_requests", "duration_s", "rps",
    "generate_throughput_tok_s", "total_throughput_tok_s",
    "Peak_output_token_throughput", "mean_ttft_ms", "p95_ttft_ms",
    "p99_ttft_ms", "mean_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p95_itl_ms", "p99_itl_ms",
]
scenarios = {
    "4k_0pct": (0, 0, 4096),
    "16k_90pct": (90, 14745, 16384),
    "50k_90pct": (90, 46080, 51200),
}
mapping = {
    "mean_concurrency": "concurrency",
    "Peak_concurrent_requests": "max_concurrent_requests",
    "duration_s": "duration",
    "rps": "request_throughput",
    "generate_throughput_tok_s": "output_throughput",
    "total_throughput_tok_s": "total_throughput",
    "Peak_output_token_throughput": "max_output_tokens_per_s",
}

rows = []
for name, (prefix_cache, prefix_len, input_len) in scenarios.items():
    for concurrency in (1, 2, 4, 8, 16):
        path = out_dir / f"{name}-c{concurrency}.jsonl"
        if not path.exists():
            continue
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if not lines:
            continue
        result = json.loads(lines[-1])
        row = {
            "prefix_cache": prefix_cache,
            "prefix_len": prefix_len,
            "input": input_len,
            "output": 1,
            "request_rate": result.get("request_rate", "inf"),
            "num_prompts": concurrency * 3,
            "max_concurrency": result.get("max_concurrency", concurrency),
        }
        for column in columns:
            if column in row:
                continue
            row[column] = result.get(mapping.get(column, column), "")
        rows.append(row)

with (out_dir / "all.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
PY

cat "$OUT_DIR/all.csv"
