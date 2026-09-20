import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite='stage-a-test-cpu')


class TestMooncakeHiSparseTransfer(unittest.TestCase):
    def test_hisparse_transfer_uses_only_target_kv_group(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100, 200, 300],
            kv_item_lens=[40, 80, 120],
            target_kv_data_ptr_count=2,
        )
        captured = {}

        def capture_send(**kwargs):
            captured.update(kwargs)
            return 0

        manager._send_kvcache_generic = capture_send
        prefill_kv_indices = np.array([5], dtype=np.int32)
        dst_kv_indices = np.array([10], dtype=np.int32)

        ret = manager.send_kvcache_hisparse(
            mooncake_session_id='session',
            prefill_kv_indices=prefill_kv_indices,
            dst_kv_ptrs=[1000, 2000],
            dst_kv_indices=dst_kv_indices,
            executor=None,
        )

        self.assertEqual(ret, 0)
        self.assertEqual(captured['src_data_ptrs'], [100, 200])
        self.assertEqual(captured['dst_data_ptrs'], [1000, 2000])
        self.assertEqual(captured['item_lens'], [40, 80])
        np.testing.assert_array_equal(
            captured['prefill_data_indices'], prefill_kv_indices
        )
        np.testing.assert_array_equal(captured['dst_data_indices'], dst_kv_indices)


if __name__ == '__main__':
    unittest.main()
