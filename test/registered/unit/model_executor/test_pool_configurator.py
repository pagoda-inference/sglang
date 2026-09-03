"""Unit tests for pool_configurator.py -- CPU only, no GPU required.

Tests the end-to-end computation: available_bytes -> MemoryPoolConfig,
verifying tokens are correct, constraints are respected, and memory
invariants hold (tokens * per_token_cost <= available_bytes).
"""

import contextlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@contextlib.contextmanager
def mock_cpu_env(kv_size=2, tp_size=1, swa_eviction_interval=4):
    """Mock GPU-dependent functions for CPU-only testing.

    swa_eviction_interval pins SGLANG_SWA_EVICTION_INTERVAL (decode batches between
    SWA evictions) to a small value so the chunk-cap formula stays hand-computable;
    only SWAChunkCapPoolConfigurator reads it.
    """
    from sglang.srt.environ import envs

    def element_size(dtype):
        if dtype is torch.uint8:
            return 1
        return kv_size

    with (
        patch("torch._utils._element_size", side_effect=element_size),
        get_parallel().override(attn_tp_size=tp_size),
        envs.SGLANG_SWA_EVICTION_INTERVAL.override(swa_eviction_interval),
    ):
        yield


def _make_model_runner(
    *,
    num_kv_heads=4,
    head_dim=64,
    v_head_dim=64,
    num_layers=32,
    use_mla_backend=False,
    is_hybrid_swa=False,
    full_attention_layer_ids=None,
    swa_attention_layer_ids=None,
    swa_num_kv_heads=None,
    swa_head_dim=None,
    swa_v_head_dim=None,
    swa_full_tokens_ratio=0.5,
    page_size=1,
    mambaish_config=None,
    disable_radix_cache=False,
    chunked_prefill_size=None,
    disable_overlap_schedule=False,
    sliding_window_size=None,
    speculative_num_draft_tokens=None,
    max_speculative_num_draft_tokens=None,
    speculative_algorithm=None,
    speculative_num_steps=None,
    speculative_eagle_topk=None,
    disaggregation_mode="null",
    max_running_requests=None,
    disaggregation_decode_extra_slots=0,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    index_head_dim=128,
    hf_config=None,
    enable_hisparse=False,
    hisparse_config=None,
    kv_cache_dtype="fake_bf16",
):
    """Create a mock ModelRunner with the fields configurators need."""
    mr = MagicMock()

    mr.use_mla_backend = use_mla_backend
    mr.is_draft_worker = False
    mr.num_effective_layers = num_layers
    mr.start_layer = 0
    mr.end_layer = num_layers
    mr.dp_size = 1
    mr.page_size = page_size
    mr.mambaish_config = mambaish_config
    mr.is_hybrid_swa = is_hybrid_swa
    mr.sliding_window_size = sliding_window_size

    mc = SimpleNamespace()
    mc.head_dim = head_dim
    mc.v_head_dim = v_head_dim
    mc.kv_lora_rank = kv_lora_rank
    mc.qk_rope_head_dim = qk_rope_head_dim
    mc.index_head_dim = index_head_dim
    mc.is_hybrid_swa = is_hybrid_swa
    mc.full_attention_layer_ids = (
        full_attention_layer_ids
        if full_attention_layer_ids is not None
        else list(range(num_layers))
    )
    mc.swa_attention_layer_ids = (
        swa_attention_layer_ids if swa_attention_layer_ids is not None else []
    )
    mc.swa_head_dim = swa_head_dim or head_dim
    mc.swa_v_head_dim = swa_v_head_dim or v_head_dim
    mc.get_num_kv_heads = lambda tp_size: num_kv_heads
    mc.get_swa_num_kv_heads = lambda tp_size: swa_num_kv_heads or num_kv_heads
    mc.hf_config = hf_config or SimpleNamespace(architectures=["LlamaForCausalLM"])
    if not hasattr(mc.hf_config, "get_text_config"):
        mc.hf_config.get_text_config = lambda: mc.hf_config
    mc.linear_attn_registry_result = None
    mr.model_config = mc

    mr.kv_cache_dtype = kv_cache_dtype

    sa = SimpleNamespace()
    sa.swa_full_tokens_ratio = swa_full_tokens_ratio
    sa.page_size = page_size
    sa.disable_radix_cache = disable_radix_cache
    sa.chunked_prefill_size = chunked_prefill_size
    sa.disable_overlap_schedule = disable_overlap_schedule
    sa.speculative_num_draft_tokens = speculative_num_draft_tokens
    sa.max_speculative_num_draft_tokens = (
        max_speculative_num_draft_tokens or speculative_num_draft_tokens
    )
    sa.speculative_algorithm = speculative_algorithm
    sa.speculative_num_steps = speculative_num_steps
    sa.speculative_eagle_topk = speculative_eagle_topk
    sa.disaggregation_mode = disaggregation_mode
    sa.max_running_requests = max_running_requests
    sa.disaggregation_decode_extra_slots = disaggregation_decode_extra_slots
    sa.enable_dsa_cache_layer_split = False
    sa.enable_hisparse = enable_hisparse
    sa.hisparse_config = hisparse_config
    sa.dsa_prefill_backend = "flashmla_sparse"
    sa.dsa_decode_backend = "flashmla_sparse"
    mr.server_args = sa

    spec = MagicMock()
    spec.is_eagle.return_value = False
    spec.is_standalone.return_value = False
    spec.is_dflash.return_value = False
    spec.is_dflash_family.return_value = False
    spec.is_none.return_value = True
    mr.spec_algorithm = spec

    mr.layer_info = SimpleNamespace(
        start_layer=0, end_layer=num_layers, num_effective_layers=num_layers
    )
    mr.ps = ParallelState.trivial()
    mr.pp_group = SimpleNamespace(rank_in_group=0)
    mr.spec_aux_config = SimpleNamespace(
        eagle_draft_num_layers=None, dflash_draft_num_layers=None
    )

    return mr


KV_SIZE = 2  # bf16


def _full_per_token(mr):
    mc = mr.model_config
    return mc.get_num_kv_heads(1) * (mc.head_dim + mc.v_head_dim) * KV_SIZE


def _swa_per_token(mr):
    mc = mr.model_config
    return mc.get_swa_num_kv_heads(1) * (mc.swa_head_dim + mc.swa_v_head_dim) * KV_SIZE


def _actual_memory_used(mr, config):
    """Compute actual memory consumed by the pool sizes in config."""
    mc = mr.model_config

    if mr.use_mla_backend:
        mla_per_token = (mc.kv_lora_rank + mc.qk_rope_head_dim) * KV_SIZE
        indexer_per_token = 0
        if getattr(mc.hf_config, "index_topk", None) is not None:
            index_head_dim = getattr(mc.hf_config, "index_head_dim", mc.index_head_dim)
            indexer_per_token = index_head_dim + index_head_dim // 128 * 4
        return (
            config.max_total_num_tokens
            * (mla_per_token + indexer_per_token)
            * mr.num_effective_layers
        )

    full_pt = _full_per_token(mr)
    swa_pt = _swa_per_token(mr)
    nf = len(mc.full_attention_layer_ids)
    ns = len(mc.swa_attention_layer_ids)

    if mr.is_hybrid_swa:
        full = config.full_max_total_num_tokens or 0
        swa = config.swa_max_total_num_tokens or 0
        return full * full_pt * nf + swa * swa_pt * ns
    else:
        return config.max_total_num_tokens * full_pt * (nf + ns)


class TestDefaultConfigurator(unittest.TestCase):
    """Default (MHA): available_bytes -> tokens, memory invariant holds."""

    def _run(self, available_bytes, page_size=1, **kwargs):
        mr = _make_model_runner(page_size=page_size, **kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, cfg, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_page_alignment(self):
        available = 10_000_000
        _, _, config = self._run(available, page_size=128)
        self.assertEqual(config.max_total_num_tokens % 128, 0)

    def test_constraint_respected(self):
        """calculate_pool_sizes_from_max_tokens respects the limit."""
        mr, cfg, config = self._run(10_000_000)
        with mock_cpu_env():
            constrained = cfg.calculate_pool_sizes_from_max_tokens(100, page_size=1)
        self.assertEqual(constrained.max_total_num_tokens, 100)

    def test_constraint_page_aligned(self):
        mr, cfg, _ = self._run(10_000_000, page_size=128)
        with mock_cpu_env():
            constrained = cfg.calculate_pool_sizes_from_max_tokens(1000, page_size=128)
        self.assertEqual(constrained.max_total_num_tokens, 896)  # 1000 // 128 * 128

    def test_no_swa_fields(self):
        _, _, config = self._run(10_000_000)
        self.assertIsNone(config.full_max_total_num_tokens)
        self.assertIsNone(config.swa_max_total_num_tokens)


class TestMLAConfigurator(unittest.TestCase):
    """MLA: available_bytes -> tokens, memory invariant holds."""

    NUM_LAYERS = 32
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64

    def _run(self, available_bytes, page_size=1):
        mr = _make_model_runner(
            num_layers=self.NUM_LAYERS,
            use_mla_backend=True,
            page_size=page_size,
            kv_lora_rank=self.KV_LORA_RANK,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM,
            hf_config=SimpleNamespace(architectures=["DeepseekV2ForCausalLM"]),
            kv_cache_dtype=torch.bfloat16,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def _main_kv_cell_size(self):
        return (self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM) * self.NUM_LAYERS * KV_SIZE

    def test_memory_utilization(self):
        available = 10_000_000
        mr, _, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_page_alignment(self):
        available = 10_000_000
        _, _, config = self._run(available, page_size=128)
        self.assertEqual(config.max_total_num_tokens % 128, 0)

    def test_without_dsa_index_memory(self):
        available = 10_000_000
        _, _, config = self._run(available)
        self.assertEqual(
            config.max_total_num_tokens, available // self._main_kv_cell_size()
        )


class TestDSAModelConfigurator(unittest.TestCase):
    """GLM-5 DSA/NSA MLA memory sizing invariants."""

    PAGE_SIZE = 64
    NUM_LAYERS = 78
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    INDEX_HEAD_DIM = 128
    AVAILABLE_GPU_GB = (32, 64, 96)
    DEVICE_BUFFER_SIZES = (4096, 6144, 8192)
    BATCH_SIZES = (32, 64, 96)
    HOST_TO_DEVICE_RATIOS = (1, 2, 5, 8)

    @staticmethod
    def _make_dsa_hf_config(index_head_dim=INDEX_HEAD_DIM):
        return SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_head_dim=index_head_dim,
        )

    def _make_dsa_runner(
        self,
        *,
        enable_hisparse=False,
        max_running_requests=None,
        device_buffer_size=None,
        host_to_device_ratio=None,
        attn_dp_size=1,
        kv_cache_dtype=torch.bfloat16,
    ):
        hisparse_config = None
        if enable_hisparse:
            hisparse_config = json.dumps(
                {
                    "top_k": 2048,
                    "device_buffer_size": device_buffer_size,
                    "host_to_device_ratio": host_to_device_ratio,
                }
            )

        mr = _make_model_runner(
            num_layers=self.NUM_LAYERS,
            use_mla_backend=True,
            page_size=self.PAGE_SIZE,
            kv_lora_rank=self.KV_LORA_RANK,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM,
            index_head_dim=self.INDEX_HEAD_DIM,
            hf_config=self._make_dsa_hf_config(),
            enable_hisparse=enable_hisparse,
            max_running_requests=max_running_requests,
            hisparse_config=hisparse_config,
            kv_cache_dtype=kv_cache_dtype,
            disaggregation_mode="decode" if enable_hisparse else "null",
        )
        mr.ps = SimpleNamespace(attn_dp_size=attn_dp_size)
        return mr

    def _calculate_config(self, mr, available_bytes):
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, mr.server_args.page_size)
        return cfg, config

    def _main_kv_cell_size(self):
        return (self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM) * self.NUM_LAYERS * KV_SIZE

    def _index_k_cell_size(self):
        return (self.INDEX_HEAD_DIM + self.INDEX_HEAD_DIM // 128 * 4) * self.NUM_LAYERS

    @staticmethod
    def _align_up(value, page_size):
        return ((value + page_size - 1) // page_size) * page_size

    def _expected_hisparse_pool_size(self, buffer_size, batch_size):
        return self._align_up(buffer_size * batch_size, self.PAGE_SIZE)

    def _actual_hisparse_gpu_memory(self, config):
        return (
            config.hisparse_device_num_tokens * self._main_kv_cell_size()
            + config.max_total_num_tokens * self._index_k_cell_size()
        )

    def _actual_hisparse_cpu_memory(self, config, host_to_device_ratio):
        return config.max_total_num_tokens * self._main_kv_cell_size()

    def test_non_hisparse_full_gpu_pool_fits_available_memory(self):
        mr = self._make_dsa_runner()
        for available_gpu_gb in self.AVAILABLE_GPU_GB:
            available_gpu = available_gpu_gb * 1024**3
            with self.subTest(available_gpu_gb=available_gpu_gb):
                _, config = self._calculate_config(mr, available_gpu)
                self.assertLessEqual(_actual_memory_used(mr, config), available_gpu)
                self.assertEqual(config.max_total_num_tokens % self.PAGE_SIZE, 0)

    def test_fp8_main_kv_size_uses_actual_mla_layout(self):
        mr = self._make_dsa_runner(kv_cache_dtype=torch.float8_e4m3fn)
        with mock_cpu_env(kv_size=1):
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            configurator = create_memory_pool_configurator(mr)

        expected_mla_dim = 512 + 512 // 128 * 4 + 64 * 2
        self.assertEqual(
            configurator._main_kv_size,
            expected_mla_dim * self.NUM_LAYERS,
        )

    def test_hisparse_gpu_and_cpu_pool_fit_memory_budget(self):
        for available_gpu_gb in self.AVAILABLE_GPU_GB:
            available_gpu = available_gpu_gb * 1024**3
            for buffer_size in self.DEVICE_BUFFER_SIZES:
                for batch_size in self.BATCH_SIZES:
                    hot_tokens = self._expected_hisparse_pool_size(
                        buffer_size, batch_size
                    )
                    if hot_tokens * self._main_kv_cell_size() > available_gpu:
                        continue
                    for host_to_device_ratio in self.HOST_TO_DEVICE_RATIOS:
                        available_cpu = available_gpu * host_to_device_ratio
                        minimal_config = SimpleNamespace(
                            max_total_num_tokens=hot_tokens * host_to_device_ratio,
                            hisparse_device_num_tokens=hot_tokens,
                        )
                        if (
                            self._actual_hisparse_gpu_memory(
                                minimal_config,
                            )
                            > available_gpu
                            or self._actual_hisparse_cpu_memory(
                                minimal_config, host_to_device_ratio
                            )
                            > available_cpu
                        ):
                            continue

                        with self.subTest(
                            available_gpu_gb=available_gpu_gb,
                            buffer_size=buffer_size,
                            batch_size=batch_size,
                            host_to_device_ratio=host_to_device_ratio,
                        ):
                            mr = self._make_dsa_runner(
                                enable_hisparse=True,
                                max_running_requests=batch_size,
                                device_buffer_size=buffer_size,
                                host_to_device_ratio=host_to_device_ratio,
                            )
                            _, config = self._calculate_config(mr, available_gpu)
                            self.assertEqual(
                                config.hisparse_device_num_tokens, hot_tokens
                            )
                            self.assertGreaterEqual(
                                config.max_total_num_tokens,
                                hot_tokens * host_to_device_ratio,
                            )
                            self.assertLessEqual(
                                self._actual_hisparse_gpu_memory(
                                    config,
                                ),
                                available_gpu,
                            )
                            self.assertLessEqual(
                                self._actual_hisparse_cpu_memory(
                                    config, host_to_device_ratio
                                ),
                                available_cpu,
                            )
                            self.assertEqual(
                                config.max_total_num_tokens % self.PAGE_SIZE, 0
                            )

    def test_host_ratio_increases_logical_capacity_until_gpu_limited(self):
        available_gpu = 32 * 1024**3
        capacities = []
        for host_to_device_ratio in self.HOST_TO_DEVICE_RATIOS:
            mr = self._make_dsa_runner(
                enable_hisparse=True,
                max_running_requests=32,
                device_buffer_size=4096,
                host_to_device_ratio=host_to_device_ratio,
            )
            _, config = self._calculate_config(mr, available_gpu)
            capacities.append(config.max_total_num_tokens)

        self.assertLess(capacities[0], capacities[1])
        self.assertLess(capacities[1], capacities[2])
        self.assertLessEqual(capacities[2], capacities[3])

    def test_hisparse_reserves_eagle_draft_kv_on_gpu(self):
        available_gpu = 32 * 1024**3
        base_mr = self._make_dsa_runner(
            enable_hisparse=True,
            max_running_requests=32,
            device_buffer_size=4096,
            host_to_device_ratio=2,
        )
        mtp_mr = self._make_dsa_runner(
            enable_hisparse=True,
            max_running_requests=32,
            device_buffer_size=4096,
            host_to_device_ratio=2,
        )
        mtp_mr.spec_algorithm.is_eagle.return_value = True
        mtp_mr.spec_aux_config.eagle_draft_num_layers = 1

        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            base_configurator = create_memory_pool_configurator(base_mr)
            mtp_configurator = create_memory_pool_configurator(mtp_mr)
            base_config = base_configurator.calculate_pool_sizes(
                available_gpu, self.PAGE_SIZE
            )
            mtp_config = mtp_configurator.calculate_pool_sizes(
                available_gpu, self.PAGE_SIZE
            )

        draft_bytes_per_token = (
            self._main_kv_cell_size() + self._index_k_cell_size()
        ) // self.NUM_LAYERS
        self.assertEqual(mtp_configurator._draft_kv_size, draft_bytes_per_token)
        self.assertLess(
            mtp_config.max_total_num_tokens, base_config.max_total_num_tokens
        )
        actual_gpu_bytes = (
            mtp_config.hisparse_device_num_tokens
            * self._main_kv_cell_size()
            + mtp_config.max_total_num_tokens
            * (self._index_k_cell_size() + draft_bytes_per_token)
        )
        self.assertLessEqual(actual_gpu_bytes, available_gpu)

    def test_larger_device_buffer_reduces_logical_capacity(self):
        available_gpu = 32 * 1024**3
        capacities = []
        for device_buffer_size in (4096, 8192):
            mr = self._make_dsa_runner(
                enable_hisparse=True,
                max_running_requests=32,
                device_buffer_size=device_buffer_size,
                host_to_device_ratio=8,
            )
            _, config = self._calculate_config(mr, available_gpu)
            capacities.append(config.max_total_num_tokens)

        self.assertGreater(capacities[0], capacities[1])

    def test_dp32_uses_one_hot_buffer_per_worker(self):
        available_gpu = 32 * 1024**3
        capacities = []
        for host_to_device_ratio in (1, 2, 5):
            mr = self._make_dsa_runner(
                enable_hisparse=True,
                max_running_requests=32,
                device_buffer_size=2048,
                host_to_device_ratio=host_to_device_ratio,
                attn_dp_size=32,
            )
            _, config = self._calculate_config(mr, available_gpu)
            self.assertEqual(config.hisparse_device_num_tokens, 2048)
            capacities.append(config.max_total_num_tokens)

        self.assertLess(capacities[0], capacities[1])
        self.assertLess(capacities[1], capacities[2])


class TestHybridSWAConfigurator(unittest.TestCase):
    """Hybrid SWA: full/swa split, ratio, memory invariant."""

    def _make_swa_runner(self, full_layers=16, swa_layers=16, ratio=0.5, page_size=1):
        return _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(full_layers)),
            swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
            swa_num_kv_heads=4,
            page_size=page_size,
            swa_full_tokens_ratio=ratio,
        )

    def _run(self, available_bytes, **kwargs):
        mr = self._make_swa_runner(**kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, mr.server_args.page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_ratio_respected(self):
        """swa_tokens ~= full_tokens * ratio (within page alignment)"""
        available = 10_000_000
        for ratio in [0.25, 0.5, 0.75, 1.0]:
            mr, _, config = self._run(available, ratio=ratio, page_size=1)
            full = config.full_max_total_num_tokens
            swa = config.swa_max_total_num_tokens
            self.assertEqual(swa, int(full * ratio), f"ratio={ratio}")

    def test_ratio_with_page_alignment(self):
        """With page alignment, swa_tokens = align(full_tokens * ratio)"""
        available = 10_000_000
        mr, _, config = self._run(available, ratio=0.5, page_size=128)
        full = config.full_max_total_num_tokens
        swa = config.swa_max_total_num_tokens
        self.assertEqual(full % 128, 0)
        self.assertEqual(swa % 128, 0)
        self.assertEqual(swa, (int(full * 0.5) // 128) * 128)

    def test_max_total_equals_full(self):
        """For hybrid, max_total_num_tokens = full_max_total_num_tokens"""
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.full_max_total_num_tokens)

    def test_constraint_respected(self):
        """full_tokens = constrained value after re-run"""
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(200, page_size=1)
        self.assertEqual(config.full_max_total_num_tokens, 200)
        self.assertEqual(config.swa_max_total_num_tokens, 100)

    def test_constraint_memory_within_budget(self):
        """After constraint, memory <= original budget (but less than profiled due to constraint)."""
        available = 10_000_000
        mr, cfg, original = self._run(available, page_size=1)
        user_limit = original.full_max_total_num_tokens // 2
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(
                user_limit, mr.server_args.page_size
            )
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        # constrained should use roughly half the memory
        original_used = _actual_memory_used(mr, original)
        self.assertAlmostEqual(used / original_used, 0.5, delta=0.01)

    def test_different_layer_counts(self):
        """Asymmetric full/swa layer counts"""
        available = 10_000_000
        mr, _, config = self._run(available, full_layers=24, swa_layers=8, ratio=0.5)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertEqual(
            config.swa_max_total_num_tokens,
            int(config.full_max_total_num_tokens * 0.5),
        )

    def test_chunk_cache_cap_accounts_for_spec_topk_page_rounding(self):
        available = 1_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=4,
            sliding_window_size=8,
            page_size=4,
            max_running_requests=2,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=5,
            disable_overlap_schedule=True,  # spec-v1: no double allocation
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=4)

        # spec-v1 (overlap off): decode_alloc = max(ceil_align(3+4,4)*2,
        # ceil_align(5,4)) = 16. trailing = 8 + 20 + page(4) = 32; per req =
        # 32 + 16 = 48. Global prefill = 1*chunk(4) + page(4) = 8.
        # cap = 48 * 2 + 8 = 104 -> ceil_align(104, 4) = 104.
        self.assertEqual(config.swa_max_total_num_tokens, 104)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_doubles_decode_alloc_for_spec_v2_overlap(self):
        # Overlap on -> spec-v2: decode_alloc = 2 * get_alloc_len_per_decode =
        # 2 * max(steps*topk, max_draft) = 2 * max(6, 5) = 12 (page=1, since the
        # v2 allocator does not support page>1 & topk>1). trailing = 8 + 20 +
        # page(1) = 29; per req = 29 + 12 = 41. Global prefill =
        # 2*chunk(4) + page(1) = 9; cap = 41 * 2 + 9 = 91.
        available = 1_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=4,
            sliding_window_size=8,
            page_size=1,
            max_running_requests=2,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=5,
            disable_overlap_schedule=False,  # spec-v2: 2 * get_alloc_len_per_decode
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        self.assertEqual(config.swa_max_total_num_tokens, 91)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_drops_prefill_for_disagg_decode(self):
        available = 1_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=1000,
            sliding_window_size=4,
            page_size=1,
            max_running_requests=10,
            disaggregation_mode="decode",
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        # disagg decode drops the prefill term: per req = 4 + 1 + 4 + 1 = 10 (as above).
        self.assertEqual(config.swa_max_total_num_tokens, 100)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_prefill_holds_window_plus_chunk(self):
        # Non-decode (prefill) engine: each request keeps its decode footprint, while
        # in-flight chunked-prefill tokens are a global batch budget -- two chunks
        # under overlap.
        available = 1_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=16,
            sliding_window_size=8,
            page_size=4,
            max_running_requests=2,
            disaggregation_mode="prefill",
            disable_overlap_schedule=False,  # overlap -> 2 chunks in flight
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=4)

        # per req = trailing(window(8) + eviction(4) + page(4)) + decode_alloc(4)
        # = 20. Global prefill = 2*chunk(16) + page(4) = 36.
        # cap = 20 * max_running_requests(2) + 36 = 76.
        self.assertEqual(config.swa_max_total_num_tokens, 76)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_disagg_decode_pre_alloc(self):
        # decode adds disaggregation_decode_extra_slots in-transfer slots to the
        # request count (num_reserved_decode_tokens is a full-pool concern, not SWA).
        available = 2_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=1000,
            sliding_window_size=4,
            page_size=1,
            max_running_requests=10,
            disaggregation_mode="decode",
            disaggregation_decode_extra_slots=2,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        # active per req = 4 + 1 + 4 + 1 = 10 for the 10 running requests; the 2
        # in-transfer extra slots hold only window + page = 4 + 1 = 5 each.
        # cap = 10 * 10 + 5 * 2 = 110.
        self.assertEqual(config.swa_max_total_num_tokens, 110)
        self.assertLessEqual(_actual_memory_used(mr, config), available)


class TestAllSWAConfigurator(unittest.TestCase):
    """All-SWA (full_layers=0): special case."""

    def _run(self, available_bytes, ratio=0.5, page_size=1, **kwargs):
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[],
            swa_attention_layer_ids=list(range(32)),
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=ratio,
            page_size=page_size,
            **kwargs,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def test_full_max_is_zero(self):
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.full_max_total_num_tokens, 0)

    def test_max_total_equals_swa(self):
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.swa_max_total_num_tokens)

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        swa_pt = _swa_per_token(mr)
        ns = len(mr.model_config.swa_attention_layer_ids)
        used = config.swa_max_total_num_tokens * swa_pt * ns
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_constraint_respected(self):
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(500, page_size=1)
        self.assertEqual(config.max_total_num_tokens, 500)
        self.assertEqual(config.swa_max_total_num_tokens, 500)


class TestEagleConfigurator(unittest.TestCase):
    """EAGLE: draft KV cache must be accounted for so total allocation fits in budget."""

    def test_eagle_does_not_exceed_budget(self):
        """Total memory (target + draft KV cache) must not exceed available."""
        available = 10_000_000
        num_layers = 32
        eagle_draft_num_layers = 4

        mr = _make_model_runner(num_layers=num_layers)
        mr.spec_algorithm.is_eagle.return_value = True
        mr.spec_algorithm.is_standalone.return_value = False
        mr.spec_algorithm.is_none.return_value = False
        mr.spec_aux_config.eagle_draft_num_layers = eagle_draft_num_layers

        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, 1)

        full_pt = _full_per_token(mr)
        total_layers = num_layers + eagle_draft_num_layers
        used = config.max_total_num_tokens * full_pt * total_layers
        self.assertLessEqual(used, available)


class TestFactory(unittest.TestCase):
    def test_default_for_non_swa(self):
        mr = _make_model_runner(is_hybrid_swa=False)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                DefaultPoolConfigurator,
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
        self.assertIsInstance(cfg, DefaultPoolConfigurator)

    def test_swa_for_hybrid(self):
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(16)),
            swa_attention_layer_ids=list(range(16, 32)),
            swa_num_kv_heads=4,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                HybridSWAPoolConfigurator,
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
        self.assertIsInstance(cfg, HybridSWAPoolConfigurator)

    def test_chunk_cap_configurator_selection(self):
        # SWAChunkCapPoolConfigurator is selected only when max_running_requests is set.
        def _cfg(max_running_requests):
            mr = _make_model_runner(
                is_hybrid_swa=True,
                full_attention_layer_ids=[0],
                swa_attention_layer_ids=[1],
                swa_num_kv_heads=4,
                disable_radix_cache=True,
                chunked_prefill_size=4,
                sliding_window_size=8,
                max_running_requests=max_running_requests,
            )
            with mock_cpu_env():
                from sglang.srt.model_executor.pool_configurator import (
                    create_memory_pool_configurator,
                )

                return create_memory_pool_configurator(mr)

        from sglang.srt.model_executor.pool_configurator import (
            SWAChunkCapPoolConfigurator,
        )

        self.assertIsInstance(_cfg(2), SWAChunkCapPoolConfigurator)
        self.assertNotIsInstance(_cfg(None), SWAChunkCapPoolConfigurator)


if __name__ == "__main__":
    unittest.main()
