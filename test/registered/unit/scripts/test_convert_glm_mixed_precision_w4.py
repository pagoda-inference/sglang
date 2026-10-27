import importlib.util
import json
from pathlib import Path
import tempfile

import torch
from safetensors import torch as safetensors_torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int8_triton_moe import (
    reduce_repeated_group_scale_to_channel,
    unpack_packed_int4_as_int8,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


SCRIPT_PATH = (
    Path(__file__).parents[4]
    / "scripts"
    / "convert_glm_mixed_precision_w4_to_compressed_tensors.py"
)
SPEC = importlib.util.spec_from_file_location("glm_w4_converter", SCRIPT_PATH)
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class TestQLeanExpertRepack(CustomTestCase):
    def test_qlean_storage_interleave_is_removed(self):
        logical_chunk_codes = torch.arange(8, dtype=torch.uint8)
        storage_chunk_codes = logical_chunk_codes[[0, 4, 1, 5, 2, 6, 3, 7]]
        storage_chunks = storage_chunk_codes[:, None].repeat(1, 4).reshape(-1)
        source_pairs = storage_chunks.reshape(-1, 2)
        source_pairs = source_pairs[:, 0] | (source_pairs[:, 1] << 4)
        source = source_pairs[None, :].to(torch.int8)

        packed = CONVERTER.repack_expert_weight(source)
        packed_pairs = packed.view(torch.uint8).reshape(-1)
        codes = torch.stack((packed_pairs & 0xF, packed_pairs >> 4), dim=1).reshape(-1)
        expected = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.uint8)
        expected = expected[:, None].repeat(1, 4).reshape(-1)

        self.assertEqual(packed.shape, (1, 4))
        self.assertTrue(torch.equal(codes, expected))

    def test_qlean_signed_nibble_round_trip(self):
        logical_values = torch.arange(-8, 8, dtype=torch.int8).repeat(2)
        logical_codes = logical_values.to(torch.uint8) + 8
        storage_chunks = logical_codes.reshape(-1, 4)[[0, 4, 1, 5, 2, 6, 3, 7]]
        encoded = (storage_chunks.reshape(-1) ^ 0x8).to(torch.uint8)
        source = (
            encoded[0::2] | (encoded[1::2] << 4)
        ).to(torch.int8)[None]

        packed = CONVERTER.repack_expert_weight(source)
        unpacked = unpack_packed_int4_as_int8(packed)

        self.assertTrue(torch.equal(unpacked.reshape(-1), logical_values))

    def test_repeated_group_scale_reduces_to_channel(self):
        scale = torch.tensor([[[2.0, 2.0, 2.0]]], dtype=torch.float32)
        self.assertTrue(
            torch.equal(
                reduce_repeated_group_scale_to_channel(scale, "test_scale"),
                torch.full((1, 1, 1), 2.0),
            )
        )

        non_uniform = torch.tensor([[[2.0, 3.0]]], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "non-uniform group scales"):
            reduce_repeated_group_scale_to_channel(non_uniform, "test_scale")

    def test_partial_repack_preserves_existing_index_entries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            first_shard = "model-00001-of-00002.safetensors"
            second_shard = "model-00002-of-00002.safetensors"
            safetensors_torch.save_file({"dense": torch.zeros(1)}, str(output / first_shard))
            safetensors_torch.save_file({"expert": torch.zeros(1)}, str(output / second_shard))
            index_path = output / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 0},
                        "weight_map": {"dense": first_shard, "expert": second_shard},
                    }
                )
            )

            CONVERTER.finalize_index(output, {second_shard})
            index = json.loads(index_path.read_text())

            self.assertEqual(
                index["weight_map"],
                {"dense": first_shard, "expert": second_shard},
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
