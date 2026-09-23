import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler_components import dp_attn
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiSparseMtpCudaGraphVote(unittest.TestCase):
    def test_index_share_requires_a_real_topk_seed(self):
        model_runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(index_share_for_mtp_iteration=True)
            )
        )
        decode_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
        )
        memory = SimpleNamespace(enable_hisparse=True)

        with patch.object(dp_attn, "get_memory", return_value=memory):
            decode_batch.spec_info = SimpleNamespace(
                future_indices=[1],
                future_dsa_topk_indices_available=False,
                dsa_topk_indices=None,
            )
            self.assertFalse(
                dp_attn._can_run_hisparse_mtp_cuda_graph(model_runner, decode_batch)
            )

            decode_batch.spec_info = SimpleNamespace(
                future_indices=[1],
                future_dsa_topk_indices_available=True,
                dsa_topk_indices=None,
            )
            self.assertTrue(
                dp_attn._can_run_hisparse_mtp_cuda_graph(model_runner, decode_batch)
            )

            decode_batch.spec_info = SimpleNamespace(
                future_indices=None,
                future_dsa_topk_indices_available=False,
                dsa_topk_indices=[1],
            )
            self.assertTrue(
                dp_attn._can_run_hisparse_mtp_cuda_graph(model_runner, decode_batch)
            )


class TestHiSparseRealNumReqs(unittest.TestCase):
    def test_uses_batch_size_before_dp_padding(self):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            _original_batch_size=2,
            batch_size=8,
        )
        self.assertEqual(ModelRunner._get_hisparse_real_num_reqs(batch), 2)

    def test_idle_batch_has_no_real_requests(self):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: True),
            _original_batch_size=2,
            batch_size=8,
        )
        self.assertEqual(ModelRunner._get_hisparse_real_num_reqs(batch), 0)

    def test_falls_back_when_batch_is_not_padded(self):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            _original_batch_size=None,
            batch_size=3,
        )
        self.assertEqual(ModelRunner._get_hisparse_real_num_reqs(batch), 3)


if __name__ == "__main__":
    unittest.main()
