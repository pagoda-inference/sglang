import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.fake.conn import FakeKVReceiver, FakeKVSender
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.mem_cache.hisparse_spec import build_hisparse_spec_layout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestMooncakeTargetOnlyHiSparseTransfer(unittest.TestCase):
    def test_mtp_layout_allows_repeated_topk_occurrences(self):
        layout = build_hisparse_spec_layout(
            hot_slots=8192,
            tail_capacity_slots=64,
            compress_ratio=1,
            verify_width=4,
            top_k=2048,
        )

        self.assertEqual(layout.occurrences, 8192)
        self.assertEqual(layout.speculative_slots, 4)
        self.assertEqual(layout.scratch_slots, 59)

        with self.assertRaisesRegex(ValueError, "Invalid HiSparse speculative layout"):
            build_hisparse_spec_layout(
                hot_slots=2048,
                tail_capacity_slots=64,
                compress_ratio=1,
                verify_width=4,
                top_k=2048,
            )

    def test_prefill_send_kv_chunk_uses_scheduler_draft_pool(self):
        class Scheduler(SchedulerDisaggregationPrefillMixin):
            pass

        scheduler = Scheduler()
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(page_size=64),
            translate_kv_indices_for_transfer=lambda indices: indices,
        )
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(128, dtype=torch.int64).reshape(1, 128)
        )
        scheduler.draft_token_to_kv_pool = object()
        scheduler.enable_hisparse = False
        scheduler.enable_staging = False
        scheduler.disagg_prefill_pending_chunk_rids = set()
        sends = []
        sender = SimpleNamespace(
            should_send_kv_chunk=lambda page_count, is_last_chunk: True,
            send=lambda *args, **kwargs: sends.append((args, kwargs)),
        )
        req = SimpleNamespace(
            req_pool_idx=0,
            rid="test",
            start_send_idx=0,
            origin_input_ids=[0] * 128,
            extend_range=SimpleNamespace(end=128),
            disagg_kv_sender=sender,
        )

        scheduler.send_kv_chunk(req, last_chunk=False, end_idx=128)

        self.assertEqual(req.start_send_idx, 128)
        self.assertEqual(len(sends), 1)
        np.testing.assert_array_equal(sends[0][1]["draft_kv_indices"], [0, 1])

    def test_fake_transfer_components_accept_draft_indices(self):
        receiver = FakeKVReceiver.__new__(FakeKVReceiver)
        receiver.has_sent_metadata = False
        receiver.send_metadata(
            np.array([1, 2], dtype=np.int32),
            draft_kv_indices=np.array([3, 4], dtype=np.int32),
        )

        sender = FakeKVSender.__new__(FakeKVSender)
        sender.has_sent = False
        sender.send(
            np.array([1, 2], dtype=np.int32),
            draft_kv_indices=np.array([3, 4], dtype=np.int32),
        )

        self.assertTrue(receiver.has_sent_metadata)
        self.assertTrue(sender.has_sent)

    def test_target_and_draft_use_separate_pointer_index_spaces(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100, 101, 200],
            kv_item_lens=[40, 40, 24],
            kv_layer_ids=[],
            target_kv_data_ptr_count=2,
            draft_kv_data_ptr_count=1,
        )
        calls = []

        def capture_send(**kwargs):
            calls.append(kwargs)
            return 0

        def validate_layout(*args, **kwargs):
            return None

        manager._send_kvcache_generic = capture_send
        manager._validate_envelope_kv_layout = validate_layout

        ret = manager.send_kvcache(
            mooncake_session_id="session",
            prefill_kv_indices=np.array([2, 3], dtype=np.int32),
            dst_kv_ptrs=[300, 301, 400],
            dst_kv_indices=np.array([10, 11], dtype=np.int32),
            executor=None,
            dst_draft_kv_indices=np.array([20, 21], dtype=np.int32),
            dst_target_kv_data_ptr_count=2,
            dst_draft_kv_data_ptr_count=1,
            prefill_draft_kv_indices=np.array([4, 5], dtype=np.int32),
        )

        self.assertEqual(ret, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["src_data_ptrs"], [100, 101])
        self.assertEqual(calls[0]["dst_data_ptrs"], [300, 301])
        np.testing.assert_array_equal(calls[0]["prefill_data_indices"], [2, 3])
        np.testing.assert_array_equal(calls[0]["dst_data_indices"], [10, 11])
        self.assertEqual(calls[1]["src_data_ptrs"], [200])
        self.assertEqual(calls[1]["dst_data_ptrs"], [400])
        np.testing.assert_array_equal(calls[1]["prefill_data_indices"], [4, 5])
        np.testing.assert_array_equal(calls[1]["dst_data_indices"], [20, 21])

    def test_hisparse_path_expands_pages_to_token_indices(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100, 200],
            kv_item_lens=[8, 8],
            kv_layer_ids=[],
            page_size=4,
            target_kv_data_ptr_count=1,
            draft_kv_data_ptr_count=1,
        )
        calls = []
        manager._send_kvcache_generic = lambda **kwargs: calls.append(kwargs) or 0

        ret = manager.send_kvcache_hisparse(
            mooncake_session_id="session",
            prefill_kv_indices=np.array([2], dtype=np.int32),
            dst_kv_ptrs=[300, 400],
            dst_kv_indices=np.array([7], dtype=np.int32),
            page_index_slice=slice(0, 1),
            executor=None,
            dst_draft_kv_indices=np.array([40, 41, 42, 43], dtype=np.int32),
            dst_target_kv_data_ptr_count=1,
            dst_draft_kv_data_ptr_count=1,
            prefill_draft_kv_indices=np.array([3], dtype=np.int32),
        )

        self.assertEqual(ret, 0)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0]["prefill_data_indices"], [8, 9, 10, 11])
        np.testing.assert_array_equal(calls[0]["dst_data_indices"], [28, 29, 30, 31])
        np.testing.assert_array_equal(
            calls[1]["prefill_data_indices"], [12, 13, 14, 15]
        )
        np.testing.assert_array_equal(calls[1]["dst_data_indices"], [40, 41, 42, 43])

    def test_legacy_dense_path_keeps_appended_draft_pointers(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100, 101, 200],
            kv_item_lens=[40, 40, 24],
            kv_layer_ids=[],
            target_kv_data_ptr_count=2,
            draft_kv_data_ptr_count=1,
        )
        calls = []

        def capture_send(**kwargs):
            calls.append(kwargs)
            return 0

        manager._send_kvcache_generic = capture_send
        manager._validate_envelope_kv_layout = lambda *args, **kwargs: None

        ret = manager.send_kvcache(
            mooncake_session_id="session",
            prefill_kv_indices=np.array([2, 3], dtype=np.int32),
            dst_kv_ptrs=[300, 301, 400],
            dst_kv_indices=np.array([10, 11], dtype=np.int32),
            executor=None,
        )

        self.assertEqual(ret, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["src_data_ptrs"], [100, 101, 200])
        self.assertEqual(calls[0]["dst_data_ptrs"], [300, 301, 400])

    def test_draft_transfer_rejects_pointer_group_mismatch(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100, 101, 200],
            kv_item_lens=[40, 40, 24],
            kv_layer_ids=[],
            target_kv_data_ptr_count=2,
            draft_kv_data_ptr_count=1,
        )
        manager._send_kvcache_generic = lambda **kwargs: 0
        manager._validate_envelope_kv_layout = lambda *args, **kwargs: None

        with self.assertRaisesRegex(RuntimeError, "target/draft KV pointer count"):
            manager.send_kvcache(
                mooncake_session_id="session",
                prefill_kv_indices=np.array([2, 3], dtype=np.int32),
                dst_kv_ptrs=[300, 301, 400],
                dst_kv_indices=np.array([10, 11], dtype=np.int32),
                executor=None,
                dst_draft_kv_indices=np.array([20, 21], dtype=np.int32),
                dst_draft_kv_data_ptr_count=2,
                dst_target_kv_data_ptr_count=1,
                prefill_draft_kv_indices=np.array([4, 5], dtype=np.int32),
            )

    def test_missing_draft_metadata_rejects_new_registration(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100, 101, 200],
            kv_item_lens=[40, 40, 24],
            kv_layer_ids=[],
            target_kv_data_ptr_count=2,
            draft_kv_data_ptr_count=1,
        )
        calls = []
        manager._send_kvcache_generic = lambda **kwargs: calls.append(kwargs) or 0
        manager._validate_envelope_kv_layout = lambda *args, **kwargs: None

        with self.assertRaisesRegex(RuntimeError, "destination metadata is missing"):
            manager.send_kvcache(
                mooncake_session_id="session",
                prefill_kv_indices=np.array([2, 3], dtype=np.int32),
                dst_kv_ptrs=[300, 301, 400],
                dst_kv_indices=np.array([10, 11], dtype=np.int32),
                executor=None,
                dst_target_kv_data_ptr_count=2,
                dst_draft_kv_data_ptr_count=1,
            )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
