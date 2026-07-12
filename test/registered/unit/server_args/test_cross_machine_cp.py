from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.srt.server_args import (
    CROSS_MACHINE_CP_ERROR,
    ServerArgs,
    validate_cross_machine_context_parallel,
)


def test_cross_machine_cp_guard_blocks_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_ALLOW_CROSS_MACHINE_CP", raising=False)

    with pytest.raises(ValueError, match="SGLANG_ALLOW_CROSS_MACHINE_CP=1"):
        validate_cross_machine_context_parallel(
            16,
            context="unit-test",
        )


def test_cross_machine_cp_guard_allows_single_machine_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_ALLOW_CROSS_MACHINE_CP", raising=False)

    validate_cross_machine_context_parallel(
        8,
        context="unit-test",
    )


def test_cross_machine_cp_guard_allows_when_env_enabled(monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_ALLOW_CROSS_MACHINE_CP", "1")

    with caplog.at_level("WARNING", logger="sglang.srt.server_args"):
        validate_cross_machine_context_parallel(
            16,
            context="unit-test",
        )

    assert any(
        "Experimental cross-machine context parallelism is enabled" in message
        for message in caplog.messages
    )


def _make_server_args_for_dsa_cp(*, tp_size: int, cp_strategy: str) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = "fake-glm-dsa"
    server_args.tp_size = tp_size
    server_args.dp_size = 1
    server_args.pp_size = 1
    server_args.attn_cp_size = 1
    server_args.attention_backend = "dsa"
    server_args.enable_prefill_cp = True
    server_args.cp_strategy = cp_strategy
    server_args.dsa_prefill_cp_mode = "round-robin-split"
    server_args.prefill_cp_mode = "in-seq-split"
    server_args.moe_dp_size = 1
    server_args.moe_dense_tp_size = 1
    server_args.ep_size = 1
    server_args.kv_cache_dtype = "auto"
    server_args.cuda_graph_config = default_cuda_graph_config()
    return server_args


def _install_fake_glm_dsa_config(monkeypatch, server_args: ServerArgs) -> None:
    fake_hf_config = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk_freq=1,
        index_topk_pattern=None,
    )
    fake_model_config = SimpleNamespace(hf_config=fake_hf_config)

    monkeypatch.setattr(server_args, "get_model_config", lambda: fake_model_config)
    monkeypatch.setattr(
        "sglang.srt.configs.model_config.is_deepseek_dsa",
        lambda _: True,
    )
    monkeypatch.setattr("sglang.srt.server_args.is_npu", lambda: False)
    monkeypatch.setattr("sglang.srt.server_args.is_xpu", lambda: False)


def test_glm_dsa_cross_machine_cp_blocks_without_env(monkeypatch):
    server_args = _make_server_args_for_dsa_cp(
        tp_size=16,
        cp_strategy="interleave",
    )
    _install_fake_glm_dsa_config(monkeypatch, server_args)
    server_args._handle_legacy_cp_arguments()

    monkeypatch.delenv("SGLANG_ALLOW_CROSS_MACHINE_CP", raising=False)
    monkeypatch.setenv("SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD", "2048")

    with pytest.raises(ValueError, match="SGLANG_ALLOW_CROSS_MACHINE_CP=1"):
        server_args._handle_model_specific_adjustments()

    assert CROSS_MACHINE_CP_ERROR.startswith(
        "Context parallel only supports single machine"
    )


def test_glm_dsa_cross_machine_cp_allows_with_env(monkeypatch, caplog):
    server_args = _make_server_args_for_dsa_cp(
        tp_size=16,
        cp_strategy="interleave",
    )
    _install_fake_glm_dsa_config(monkeypatch, server_args)
    server_args._handle_legacy_cp_arguments()

    monkeypatch.setenv("SGLANG_ALLOW_CROSS_MACHINE_CP", "1")
    monkeypatch.setenv("SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD", "2048")

    with caplog.at_level("WARNING", logger="sglang.srt.server_args"):
        server_args._handle_model_specific_adjustments()

    assert server_args.attn_cp_size == 16
    assert server_args.dsa_prefill_cp_mode == "round-robin-split"
    assert server_args.enable_dsa_prefill_context_parallel is True
    assert any(
        "Experimental cross-machine context parallelism is enabled" in message
        for message in caplog.messages
    )


def test_cp_strategy_interleave_maps_to_round_robin_for_dsa():
    server_args = _make_server_args_for_dsa_cp(
        tp_size=16,
        cp_strategy="interleave",
    )

    server_args._handle_legacy_cp_arguments()

    assert server_args.enable_dsa_prefill_context_parallel is True
    assert server_args.enable_prefill_context_parallel is False
    assert server_args.dsa_prefill_cp_mode == "round-robin-split"
    assert server_args.prefill_cp_mode == "round-robin-split"


def test_cp_strategy_zigzag_maps_to_in_seq_for_dsa():
    server_args = _make_server_args_for_dsa_cp(
        tp_size=16,
        cp_strategy="zigzag",
    )

    server_args._handle_legacy_cp_arguments()

    assert server_args.enable_dsa_prefill_context_parallel is True
    assert server_args.enable_prefill_context_parallel is False
    assert server_args.dsa_prefill_cp_mode == "in-seq-split"
    assert server_args.prefill_cp_mode == "in-seq-split"


def test_interleave_dsa_cp_still_rejects_dp_attention(monkeypatch):
    server_args = _make_server_args_for_dsa_cp(
        tp_size=8,
        cp_strategy="interleave",
    )
    server_args.dp_size = 2
    _install_fake_glm_dsa_config(monkeypatch, server_args)
    server_args._handle_legacy_cp_arguments()

    monkeypatch.setenv("SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD", "2048")

    with pytest.raises(AssertionError, match="interleave DSA CP"):
        server_args._handle_model_specific_adjustments()
