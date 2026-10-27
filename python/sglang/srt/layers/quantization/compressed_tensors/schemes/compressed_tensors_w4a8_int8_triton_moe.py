from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationArgs

    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

__all__ = ["CompressedTensorsW4A8Int8TritonMoE"]

logger = logging.getLogger(__name__)


def unpack_packed_int4_as_int8(weight: torch.Tensor) -> torch.Tensor:
    packed_shape = weight.shape
    packed_pairs = (
        weight.reshape(-1).view(torch.uint8).reshape(*packed_shape, 4)
    )
    output_shape = (*packed_pairs.shape[:-1], packed_pairs.shape[-1] * 2)
    unpacked = torch.empty(output_shape, dtype=torch.int8, device=weight.device)
    unpacked[..., 0::2] = (packed_pairs & 0xF).to(torch.int8) - 8
    unpacked[..., 1::2] = (packed_pairs >> 4).to(torch.int8) - 8
    return unpacked.reshape(*unpacked.shape[:-2], -1).contiguous()


def reduce_repeated_group_scale_to_channel(
    scale: torch.Tensor, name: str
) -> torch.Tensor:
    if scale.ndim != 3:
        raise ValueError(
            f"Invalid packed MoE scale shape for {name}: {tuple(scale.shape)}"
        )

    if (
        scale.shape[-1] > 1
        and not torch.equal(scale, scale[:, :, :1].expand_as(scale))
    ):
        raise ValueError(
            f"{name} has non-uniform group scales; the exact W4A8-INT8 "
            "Triton fallback only supports per-output-channel scales."
        )
    return (
        scale[:, :, :1]
        .float()
        .contiguous()
    )


class CompressedTensorsW4A8Int8TritonMoE(CompressedTensorsMoEScheme):
    """Exact CUDA fallback that unpacks INT4 to INT8 after loading.

    This path intentionally trades expert-weight memory for implementation
    simplicity. It preserves native W4A8-INT8 numerics while reusing the
    existing dynamic per-token INT8 Triton fused-MoE kernel.
    """

    def __init__(
        self,
        quant_config: CompressedTensorsConfig,
        weight_quant: QuantizationArgs,
    ):
        self.quant_config = quant_config
        self.num_bits = weight_quant.num_bits
        self.packed_factor = 32 // self.num_bits
        self.strategy = weight_quant.strategy
        self.group_size = weight_quant.group_size
        self.is_k_full = True

        if self.num_bits != 4 or not weight_quant.symmetric or weight_quant.dynamic:
            raise ValueError(
                "The W4A8-INT8 Triton fallback requires static symmetric 4-bit weights."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def _create_weight(
        self,
        layer: torch.nn.Module,
        name: str,
        packed_input_size: int,
        output_size: int,
        extra_weight_attrs: dict,
    ) -> None:
        weight = torch.nn.Parameter(
            torch.empty(
                layer.num_experts,
                output_size,
                packed_input_size // self.packed_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter(name, weight)
        set_weight_attrs(weight, extra_weight_attrs)

        weight_shape = torch.nn.Parameter(
            torch.empty(layer.num_experts, 2, dtype=torch.int32), requires_grad=False
        )
        layer.register_parameter(f"{name.removesuffix('_packed')}_shape", weight_shape)
        set_weight_attrs(weight_shape, extra_weight_attrs)

    def _create_scale(
        self,
        layer: torch.nn.Module,
        name: str,
        input_size: int,
        output_size: int,
        extra_weight_attrs: dict,
    ) -> None:
        num_groups = 1 if self.strategy == "channel" else input_size // self.group_size
        scale = torch.nn.Parameter(
            torch.ones(
                layer.num_experts,
                output_size,
                num_groups,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter(name, scale)
        set_weight_attrs(scale, extra_weight_attrs)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        layer.num_experts = num_experts
        extra_weight_attrs.update({"quant_method": self.strategy})

        self._create_weight(
            layer,
            "w13_weight_packed",
            hidden_size,
            2 * intermediate_size_per_partition,
            extra_weight_attrs,
        )
        self._create_weight(
            layer,
            "w2_weight_packed",
            intermediate_size_per_partition,
            hidden_size,
            extra_weight_attrs,
        )
        self._create_scale(
            layer,
            "w13_weight_scale",
            hidden_size,
            2 * intermediate_size_per_partition,
            extra_weight_attrs,
        )
        self._create_scale(
            layer,
            "w2_weight_scale",
            intermediate_size_per_partition,
            hidden_size,
            extra_weight_attrs,
        )

        logger.info_once(
            "W4A8-INT8 fallback weights: num_experts=%s hidden_size=%s "
            "intermediate_size=%s strategy=%s group_size=%s "
            "w13_packed=%s w13_scale=%s w2_packed=%s w2_scale=%s",
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            self.strategy,
            self.group_size,
            tuple(layer.w13_weight_packed.shape),
            tuple(layer.w13_weight_scale.shape),
            tuple(layer.w2_weight_packed.shape),
            tuple(layer.w2_weight_scale.shape),
        )

        layer.w13_input_scale = None
        layer.w2_input_scale = None
        layer.is_w4a8_int8_unpack_converted = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "is_w4a8_int8_unpack_converted", False):
            return

        layer.w13_weight = torch.nn.Parameter(
            unpack_packed_int4_as_int8(layer.w13_weight_packed.data),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            unpack_packed_int4_as_int8(layer.w2_weight_packed.data),
            requires_grad=False,
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            reduce_repeated_group_scale_to_channel(
                layer.w13_weight_scale.data, "w13_weight_scale"
            ),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            reduce_repeated_group_scale_to_channel(
                layer.w2_weight_scale.data, "w2_weight_scale"
            ),
            requires_grad=False,
        )

        layer.w13_weight_packed = torch.nn.Parameter(
            torch.empty(0, dtype=torch.int32, device=layer.w13_weight.device),
            requires_grad=False,
        )
        layer.w2_weight_packed = torch.nn.Parameter(
            torch.empty(0, dtype=torch.int32, device=layer.w13_weight.device),
            requires_grad=False,
        )
        layer.is_w4a8_int8_unpack_converted = True
        logger.info_once(
            "Unpacked compressed-tensors W4A8-INT8 routed experts to INT8 for Triton"
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def get_triton_quant_info(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )

    def apply_weights(
        self, layer: torch.nn.Module, dispatch_output: StandardDispatchOutput
    ) -> CombineInput:
        if self.moe_runner_config.activation != "silu":
            raise ValueError("W4A8-INT8 Triton fallback only supports SiLU activation.")
        return self.runner.run(dispatch_output, self.get_triton_quant_info(layer))
