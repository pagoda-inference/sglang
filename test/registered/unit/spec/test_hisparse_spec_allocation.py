import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.mem_cache.allocation as allocation
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite='base-a-test-cpu')


class HiSparseSpecAllocationTest(unittest.TestCase):
    def test_hisparse_spec_extension_allocates_logical_slots_only(self):
        logical_locs = torch.arange(64, 72, dtype=torch.int64)
        allocator = SimpleNamespace(
            page_size=8,
            alloc_logical_only=MagicMock(return_value=logical_locs),
        )
        req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=64))
        req_to_token = torch.zeros((1, 128), dtype=torch.int64)

        last_loc = torch.tensor([11], dtype=torch.int64)
        assigned = []

        def assign_req_to_token(*args, **kwargs):
            assigned.append((args, kwargs))
            req_to_token[0, 64:72] = logical_locs

        with (
            patch.object(allocation, 'get_last_loc', return_value=last_loc),
            patch.object(
                allocation,
                'assign_req_to_token_pool_func',
                side_effect=assign_req_to_token,
            ),
        ):
            allocation.alloc_for_spec_decode(
                SimpleNamespace(token_to_kv_pool_allocator=allocator),
                SimpleNamespace(req_to_token=req_to_token),
                reqs=[req],
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                cur_kv_lens=torch.tensor([64], dtype=torch.int32),
                cur_kv_lens_cpu=torch.tensor([64], dtype=torch.int32),
                nxt_kv_lens=torch.tensor([72], dtype=torch.int32),
                nxt_kv_lens_cpu=torch.tensor([72], dtype=torch.int32),
                num_needed_tokens=8,
                batch=SimpleNamespace(device=torch.device('cpu')),
            )

        allocator.alloc_logical_only.assert_called_once_with(
            prefix_lens=torch.tensor([64], dtype=torch.int32),
            prefix_lens_cpu=torch.tensor([64], dtype=torch.int32),
            seq_lens=torch.tensor([72], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([72], dtype=torch.int32),
            last_loc=last_loc,
            extend_num_tokens=8,
        )
        self.assertEqual(req.kv.kv_allocated_len, 72)
        self.assertEqual(len(assigned), 1)
        self.assertTrue(torch.equal(req_to_token[0, 64:72], logical_locs))


if __name__ == '__main__':
    unittest.main()
