#!/usr/bin/env python3
"""Convert GLM mixed_precision_w4 checkpoints to compressed-tensors W4A8."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file, save_file


EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
BLOCK_SIZE = 128
FP8_MAX = 448.0
QLEAN_LOGICAL_CHUNK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument(
        "--shards", nargs="*", default=[], help="Convert only matching shard names"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repack_expert_weight(weight: torch.Tensor) -> torch.Tensor:
    """Repack QLean's 32-column expert storage interleave into logical order."""
    if weight.ndim != 2 or weight.shape[-1] % 16 != 0:
        raise ValueError(f"Invalid expert packed weight: {tuple(weight.shape)}")
    pairs = weight.view(torch.uint8)
    pair_shape = pairs.shape
    pairs = (
        pairs.reshape(pair_shape[0], -1, 8, 2)
        .index_select(2, torch.tensor(QLEAN_LOGICAL_CHUNK_ORDER, device=pairs.device))
        .reshape(pair_shape)
    )
    pairs = pairs ^ 0x88
    packed = (
        pairs[:, 0::4].to(torch.int32)
        | (pairs[:, 1::4].to(torch.int32) << 8)
        | (pairs[:, 2::4].to(torch.int32) << 16)
        | (pairs[:, 3::4].to(torch.int32) << 24)
    )
    return packed.contiguous()


def expand_expert_scale(scale: torch.Tensor, input_features: int) -> torch.Tensor:
    if scale.ndim == 1:
        scale = scale.unsqueeze(-1)
    if scale.ndim != 2 or scale.shape[1] != 1:
        raise ValueError(f"Expected channel expert scale: {tuple(scale.shape)}")
    if input_features % BLOCK_SIZE:
        raise ValueError(f"Expert input not divisible by 128: {input_features}")
    return scale.to(torch.bfloat16).repeat(1, input_features // BLOCK_SIZE).contiguous()


def quantize_dense_to_fp8_block(
    weight: torch.Tensor, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or scale.shape != (weight.shape[0], 1):
        raise ValueError(f"Bad dense weight/scale: {tuple(weight.shape)}, {tuple(scale.shape)}")
    dequantized = weight.to(torch.float32) * scale.to(torch.float32)
    padded_rows = (weight.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    padded_cols = (weight.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    padded = torch.zeros(
        (padded_rows, padded_cols), dtype=dequantized.dtype, device=dequantized.device
    )
    padded[: weight.shape[0], : weight.shape[1]].copy_(dequantized)
    blocked = padded.reshape(
        padded_rows // BLOCK_SIZE,
        BLOCK_SIZE,
        padded_cols // BLOCK_SIZE,
        BLOCK_SIZE,
    ).permute(0, 2, 1, 3)
    amax = blocked.abs().amax(dim=(2, 3))
    quant_scale = torch.where(amax > 0, FP8_MAX / amax, torch.ones_like(amax))
    quantized = torch.clamp(
        blocked * quant_scale[:, :, None, None], -FP8_MAX, FP8_MAX
    ).to(torch.float8_e4m3fn)
    quantized = (
        quantized.permute(0, 2, 1, 3)
        .reshape(padded_rows, padded_cols)[: weight.shape[0], : weight.shape[1]]
        .contiguous()
    )
    scale_inv = (amax / FP8_MAX).to(torch.float32).contiguous()
    return quantized, scale_inv


def convert_shard(source_path: Path, output_path: Path) -> None:
    weights = load_file(str(source_path))
    converted: Dict[str, torch.Tensor] = {}

    for name, tensor in weights.items():
        expert_match = EXPERT_WEIGHT_RE.fullmatch(name)
        if expert_match is None:
            if name.endswith(".weight_scale"):
                continue
            if name.endswith(".weight") and tensor.dtype == torch.int8:
                scale_name = name + "_scale"
                if scale_name not in weights:
                    raise KeyError(f"Missing paired scale for {name}")
                converted[name], converted[scale_name] = quantize_dense_to_fp8_block(
                    tensor, weights[scale_name]
                )
            else:
                converted[name] = tensor
            continue

        scale_name = name + "_scale"
        if scale_name not in weights:
            raise KeyError(f"Missing paired scale for {name}")
        input_features = tensor.shape[-1] * 2
        converted[name + "_packed"] = repack_expert_weight(tensor)
        converted[scale_name] = expand_expert_scale(weights[scale_name], input_features)
        converted[name + "_shape"] = torch.tensor(
            [tensor.shape[0], input_features], dtype=torch.int32
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_path), metadata={"format": "pt"})


def tensor_nbytes(metadata: Dict[str, object]) -> int:
    dtype_sizes = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1, "I32": 4}
    dtype = metadata["dtype"]
    if dtype not in dtype_sizes:
        raise ValueError(f"Unsupported output dtype: {dtype}")
    count = 1
    for dimension in metadata["shape"]:
        count *= dimension
    return count * dtype_sizes[dtype]


def finalize_index(output: Path, selected_files: set[str]) -> None:
    from safetensors import safe_open

    index_path = output.joinpath("model.safetensors.index.json")
    weight_map: Dict[str, str] = {}
    if index_path.exists():
        existing_index = json.loads(index_path.read_text())
        weight_map = {
            name: shard
            for name, shard in existing_index.get("weight_map", {}).items()
            if shard not in selected_files
        }
    total_size = 0
    for shard in sorted(selected_files):
        path = output / shard
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for name in f.keys():
                weight_map[name] = shard
                tensor_slice = f.get_slice(name)
                total_size += tensor_nbytes(
                    {"dtype": tensor_slice.get_dtype(), "shape": tensor_slice.get_shape()}
                )

    index_path.write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_config(source: Path, output: Path, reference: Path) -> None:
    source_config = json.loads(source.joinpath("config.json").read_text())
    reference_config = json.loads(reference.read_text())
    quantization_config = reference_config["quantization_config"]
    ignored = set(quantization_config.get("ignore", []))
    source_ignored = set(
        source_config.get("quantization_config", {}).get("ignore", [])
    )
    for ignored_name in source_config.get("quantization_config", {}).get("ignore", []):
        if ignored_name.startswith("re:"):
            ignored.add(ignored_name)
        else:
            # Anchor exact source ignores. Otherwise an entry such as
            # `...mlp.gate` can prefix-match `...mlp.gate_proj`.
            ignored.add("re:^" + re.escape(ignored_name) + "$")
            nextn_prefix = f"model.layers.{source_config['num_hidden_layers']}."
            if ignored_name.startswith(nextn_prefix):
                suffix = ignored_name[len(nextn_prefix) :]
                if (
                    suffix.endswith(".gate_proj")
                    and ignored_name[: -len("gate_proj")] + "up_proj"
                    in source_ignored
                ):
                    suffix = suffix[: -len("gate_proj")] + "gate_up_proj"
                target_prefix = "model"
                if suffix.split(".", 1)[0] not in {
                    "eh_proj",
                    "shared_head",
                    "enorm",
                    "hnorm",
                }:
                    target_prefix = "model.decoder"
                ignored.add(
                    "re:^" + re.escape(f"{target_prefix}.{suffix}") + "$"
                )
            if (
                ignored_name.endswith(".gate_proj")
                and ignored_name[: -len("gate_proj")] + "up_proj" in source_ignored
            ):
                ignored.add(
                    "re:^"
                    + re.escape(ignored_name[: -len("gate_proj")] + "gate_up_proj")
                    + "$"
                )
    quantization_config["ignore"] = sorted(ignored)

    converted_config = dict(source_config)
    converted_config["quantization_config"] = quantization_config
    output.mkdir(parents=True, exist_ok=True)
    output.joinpath("config.json").write_text(
        json.dumps(converted_config, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    args = parse_args()
    source_index = json.loads(
        args.source.joinpath("model.safetensors.index.json").read_text()
    )
    source_shards = sorted(set(source_index["weight_map"].values()))
    if args.shards:
        selected = [
            shard for shard in source_shards if any(token in shard for token in args.shards)
        ]
    else:
        selected = source_shards
    if not selected:
        raise ValueError("No source shards matched --shards")

    args.output.mkdir(parents=True, exist_ok=True)
    write_config(args.source, args.output, args.reference_config)

    skipped = {
        "model.safetensors.index.json",
        "config.json",
        *(path.name for path in args.source.iterdir() if path.suffix == ".safetensors"),
    }
    for path in args.source.iterdir():
        if path.is_file() and path.name not in skipped:
            destination = args.output / path.name
            if not destination.exists():
                shutil.copy2(path, destination)

    for shard in selected:
        output_path = args.output / shard
        if output_path.exists() and not args.overwrite:
            print(f"SKIP {shard}", flush=True)
            continue
        if output_path.exists():
            output_path.unlink()
        print(f"CONVERT {shard}", flush=True)
        convert_shard(args.source / shard, output_path)

    finalize_index(args.output, set(selected))
    print(f"Converted {len(selected)} shard(s) under {args.output}", flush=True)


if __name__ == "__main__":
    main()
