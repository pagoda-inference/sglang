import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiSparsePoolConfigurator(CustomTestCase):
    def _compute_cell_size(
        self,
        kv_cache_dtype: torch.dtype,
        *,
        enable_hisparse: bool,
        host_to_device_ratio: int = 1,
    ) -> int:
        num_layers = 2
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_head_dim=128,
        )
        hf_config.get_text_config = lambda: hf_config

        override = get_context().override_server_args(
            enable_hisparse=enable_hisparse,
            hisparse_config=f'{{"host_to_device_ratio": {host_to_device_ratio}}}',
            enable_hierarchical_cache=False,
            disaggregation_mode="null",
            dsa_prefill_backend="flashmla_sparse",
            dsa_decode_backend="flashmla_sparse",
        )
        server_args = override.install()
        self.addCleanup(override.restore)

        kvc = MagicMock(
            use_mla_backend=True,
            kv_cache_dtype=kv_cache_dtype,
            is_draft_worker=False,
            model_config=SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                hf_config=hf_config,
            ),
            layer_info=SimpleNamespace(start_layer=0, end_layer=num_layers),
            server_args=server_args,
        )

        with get_parallel().override(attn_tp_size=1):
            configurator = object.__new__(DefaultPoolConfigurator)
            return configurator._compute_cell_size(kvc, num_layers=num_layers)

    def test_mla_layout_without_hisparse(self):
        for kv_cache_dtype, expected_cell_size in (
            (torch.bfloat16, 2568),
            (torch.float8_e4m3fn, 1576),
        ):
            with self.subTest(kv_cache_dtype=kv_cache_dtype):
                cell_size = self._compute_cell_size(
                    kv_cache_dtype,
                    enable_hisparse=False,
                )
                self.assertEqual(cell_size, expected_cell_size)

    def test_hisparse_indexer_scales_with_ratio(self):
        for host_to_device_ratio, expected_cell_size in (
            (2, 1840),
            (4, 2368),
        ):
            with self.subTest(host_to_device_ratio=host_to_device_ratio):
                cell_size = self._compute_cell_size(
                    torch.float8_e4m3fn,
                    enable_hisparse=True,
                    host_to_device_ratio=host_to_device_ratio,
                )
                self.assertEqual(cell_size, expected_cell_size)

    def _make_hisparse_sizing_configurator(
        self, *, host_to_device_ratio: int
    ) -> DefaultPoolConfigurator:
        configurator = object.__new__(DefaultPoolConfigurator)
        configurator.use_hisparse_memory_config = True
        configurator._main_kv_size = 1024
        configurator._indexer_kv_base_size = 128
        configurator._hisparse_device_buffer_size = 4096
        configurator._hisparse_host_to_device_ratio = host_to_device_ratio
        configurator._hisparse_max_running_requests = 32
        return configurator

    def test_hisparse_ratio_expands_logical_capacity(self):
        configs = [
            self._make_hisparse_sizing_configurator(
                host_to_device_ratio=host_to_device_ratio
            ).calculate_pool_sizes(32 * (1 << 30), page_size=64)
            for host_to_device_ratio in (2, 4, 8)
        ]

        logical_capacities = [config.max_total_num_tokens for config in configs]
        self.assertLess(logical_capacities[0], logical_capacities[1])
        self.assertLessEqual(logical_capacities[1], logical_capacities[2])
        self.assertEqual(
            [config.hisparse_device_num_tokens for config in configs],
            [4096 * 32] * 3,
        )

    def test_hisparse_constrained_capacity_keeps_device_hot_buffer(self):
        configurator = self._make_hisparse_sizing_configurator(host_to_device_ratio=2)
        config = configurator.calculate_pool_sizes_from_max_tokens(
            1_000_000, page_size=64
        )

        self.assertEqual(config.max_total_num_tokens, 999_936)
        self.assertEqual(config.hisparse_device_num_tokens, 4096 * 32)


if __name__ == "__main__":
    unittest.main()
