import unittest
from unittest import mock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode


class TestMooncakeEarlySendWait(unittest.TestCase):
    def test_transfer_chunk_carries_wait_event(self):
        manager = object.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {1: KVPoll.Transferring}
        manager.transfer_infos = {1: {"session:1": None}}
        manager.transfer_queues = [FastQueue()]
        wait_event = mock.Mock()

        manager.add_transfer_request(
            1,
            np.array([1, 2, 3], dtype=np.int32),
            slice(None),
            False,
            aux_index=7,
            wait_event=wait_event,
        )

        chunk = manager.transfer_queues[0].get()
        self.assertIs(chunk.wait_event, wait_event)

    def test_worker_synchronizes_wait_event(self):
        wait_event = mock.Mock()
        chunk = TransferKVChunk(
            room=1,
            prefill_kv_indices=np.array([1], dtype=np.int32),
            index_slice=slice(None),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
            wait_event=wait_event,
        )

        MooncakeKVManager._wait_for_producer(chunk)

        wait_event.synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
