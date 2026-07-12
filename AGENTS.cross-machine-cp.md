# Cross-machine CP16 implementation guide

This file is the task guide for branch `cross-machine-cp16`. It describes how to add experimental cross-machine context parallelism support for DSA models such as GLM5/GLM5.2, starting with `cp_size=16`. Follow this guide before making code changes for this effort.

## Goal

Enable opt-in cross-machine prefill context parallelism for DSA models, with an initial target of:

- NVIDIA Hopper/H800 CUDA systems.
- `tp_size=16`.
- `dp_size=1`.
- `attn_cp_size=16`.
- `pp_size=1`.
- DSA prefill CP enabled.

Initial support is allowed to ignore strict precision parity. The first milestone is to run without startup guards or obvious runtime failures.

## Non-goals for the first milestone

- Do not claim production correctness.
- Do not remove the default single-machine protection for normal users.
- Do not attempt broad support for arbitrary `tp_size`, `dp_size`, `pp_size`, disaggregation, speculative decoding, or mixed hardware.
- Do not tune accuracy unless runtime failures are resolved first.

## Current code facts

- GLM DSA models use the DeepSeek DSA implementation path. `GlmMoeDsaForCausalLM` inherits from `DeepseekV2ForCausalLM`.
- `GlmMoeDsaForCausalLM` is handled by the DSA branch in `python/sglang/srt/server_args.py`.
- The current cross-machine blocker is the `tp_size <= 8` guard in:
  - `python/sglang/srt/server_args.py`
  - `python/sglang/srt/arg_groups/deepseek_v4_hook.py`
- `zigzag` is the new name for legacy `in-seq-split`.
- `interleave` is the new name for legacy `round-robin-split`.
- DSA default legacy mode is `round-robin-split`, which maps to `interleave`.
- For DSA CP:
  - `interleave` requires `dp_size == 1`.
  - `zigzag` forces `moe_a2a_backend=deepep`, `ep_size=tp_size`, `moe_dense_tp_size=1`, and effectively requires batch size 1.
- With `tp_size=16`, `dp_size=1`, and DSA CP, `attn_cp_size` becomes 16. The attention CP group spans all TP ranks.

## Implementation stages

### Stage 1: Add opt-in guard

Add an explicit experimental switch, preferably in the existing env system:

```text
SGLANG_ALLOW_CROSS_MACHINE_CP=1
```

Use it to replace the hard `tp_size <= 8` assertions. Default behavior must remain unchanged: if `tp_size > 8` and the env var is not set, startup must fail with a clear message.

When the env var is set, allow startup and log a warning that cross-machine CP is experimental and precision is not guaranteed.

#### Stage 1 tests

- Unit/config test: DSA CP with `tp_size=16` fails when the env var is absent.
- Unit/config test: DSA CP with `tp_size=16` passes the cross-machine guard when the env var is set.
- Unit/config test: DSA CP with `tp_size=8` behaves as before.
- Search test: no remaining `tp_size <= 8` DSA CP guard blocks the new env path.

### Stage 2: Validate strategy defaults and CLI aliases

Confirm that new and legacy flags map consistently:

- `--cp-strategy interleave` maps to `dsa_prefill_cp_mode=round-robin-split`.
- `--cp-strategy zigzag` maps to `dsa_prefill_cp_mode=in-seq-split`.
- Legacy `--dsa-prefill-cp-mode round-robin-split` still maps to `interleave`.
- Legacy `--dsa-prefill-cp-mode in-seq-split` still maps to `zigzag`.

For the first cross-machine smoke path, prefer `interleave` with `dp_size=1` unless a model-specific launch recipe requires `zigzag`.

#### Stage 2 tests

- Argument-normalization test for `--cp-strategy interleave`.
- Argument-normalization test for `--cp-strategy zigzag`.
- Argument-normalization test for legacy DSA mode aliases.
- Negative test: `interleave` with `dp_size > 1` still fails.

### Stage 3: Add runtime observability

Add startup logs for the experimental path. Include:

- `tp_size`.
- `dp_size`.
- `pp_size`.
- `attn_cp_size`.
- `attn_tp_size`.
- `cp_strategy`.
- `dsa_prefill_cp_mode`.
- `moe_a2a_backend`.
- `ep_size`.
- `moe_dp_size`.
- `kv_cache_dtype`.

Keep logs concise and avoid per-request noise unless debugging a runtime failure.

#### Stage 3 tests

- Unit/config test or captured-log test that the experimental warning is emitted when `SGLANG_ALLOW_CROSS_MACHINE_CP=1` and `tp_size > 8`.
- Unit/config test that no experimental warning appears for normal single-machine CP.

### Stage 4: Single-node simulation smoke

Before multi-node H800 runs, validate that `tp_size > 8` no longer fails due to argument guards in a lightweight test or mocked config path. This is not a real distributed runtime test.

#### Stage 4 tests

- Lightweight Python test constructing/parsing server args for DSA CP with `tp_size=16`, `dp_size=1`, `cp_strategy=interleave`, env var enabled.
- Confirm computed values:
  - `enable_dsa_prefill_context_parallel=True`.
  - `attn_cp_size=16`.
  - `tp_size=16`.
  - `dp_size=1`.
  - `dsa_prefill_cp_mode=round-robin-split`.

### Stage 5: Multi-node runtime smoke

Run on 2 x 8 H800 nodes with contiguous rank mapping:

```text
node0: ranks 0-7
node1: ranks 8-15
```

Start with:

- `tp_size=16`.
- `dp_size=1`.
- `pp_size=1`.
- `cp_strategy=interleave`.
- batch size 1.
- short output length.
- no speculative decoding.
- no complex prefix-cache hit scenario.

#### Stage 5 tests

- Launch server successfully.
- Send a 4K-token prompt and generate at least 16 tokens.
- Send an 8K-token prompt and generate at least 16 tokens.
- Send a 32K-token prompt if memory allows.
- Repeat the same short request three times to check for hangs, empty output, and obvious runtime instability.

Record failures by category:

- Distributed initialization.
- NCCL collective hang/failure.
- DeepEP/MoE A2A failure.
- CP all-gather/reorder failure.
- KV cache write failure.
- DSA indexer/top-k failure.
- OOM.

#### Lightweight model strategy

Do not use an unrelated small dense model as the main smoke test for this task. A
normal small model can validate generic distributed launch behavior, but it does
not exercise the GLM/DeepSeek DSA CP code path.

For fast iteration, use the real GLM DSA model config with dummy weights:

```bash
env SGLANG_ALLOW_CROSS_MACHINE_CP=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 \
python -m sglang.launch_server \
  --model-path /ssd/GLM-5.2-FP8 --load-format dummy \
  --host 0.0.0.0 --port 30000 \
  --tp-size 16 --nnodes 2 --node-rank ${NODE_RANK} \
  --dist-init-addr 192.168.4.6:20000 \
  --enable-prefill-cp --cp-strategy interleave \
  --max-running-requests 1 --disable-cuda-graph
```

Start node rank 1 first, then node rank 0. This keeps startup and memory cost low
while still exercising GLM DSA argument handling, CP group construction,
distributed initialization, and a basic prefill/generate request.

The first successful smoke used `/ssd/GLM-5.2-FP8` with `--load-format dummy` on:

- node rank 0: `root@192.168.4.6`, container `lzt_sglang514`.
- node rank 1: `root@192.168.4.14`, container `lzt_sglang514`.

Observed result:

- `tp_size=16`, `attn_cp_size=16`, `cp_strategy=interleave`.
- `CustomAllreduce` disabled automatically because the process group spans nodes.
- `/model_info` succeeded.
- A one-token `/generate` request succeeded and CP prefill batches appeared on all ranks.

### Stage 6: Minimal correctness smoke

This stage does not require strict precision parity, but it should catch obviously broken outputs.

#### Stage 6 tests

- Compare generated text from `tp_size=8` single-node CP and `tp_size=16` cross-machine CP for a few simple prompts.
- Confirm output is non-empty, not repeated garbage, and follows the instruction.
- Capture logits/top-k diffs only if practical; do not block the first milestone on exact equality.

## Recommended first implementation patch

Keep the first patch small:

1. Add `SGLANG_ALLOW_CROSS_MACHINE_CP`.
2. Replace the two `tp_size <= 8` assertions with env-gated errors.
3. Add warning logs for experimental cross-machine CP.
4. Add config/argument tests for guarded and allowed paths.

## Commands to prefer

- Use `rg` for searches.
- Use focused `pytest` targets for new tests.
- Do not run full distributed multi-node tests unless explicitly requested or available.

## Acceptance criteria for the first milestone

- Default behavior is unchanged.
- `tp_size=16` DSA CP can pass argument validation when `SGLANG_ALLOW_CROSS_MACHINE_CP=1`.
- The experimental path is clearly logged.
- Unit tests cover guard behavior and strategy mapping.
- A manual H800 2-node smoke checklist exists in this file and is followed before claiming runtime support.
