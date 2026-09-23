from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_GATHER_BLOCK_SIZE = 64
_LINEAR_TRANSFER = "linear"
_DSV4_PAGED_TRANSFER = "dsv4_paged"


class HiSparseSpecState(NamedTuple):
    """Persistent cache state and reusable miss workspace for speculative swap.

    ``cache_index`` stores the two int64 hash banks as
    ``[num_requests, 2, hash_size]``. ``cache_policy`` uses a control-plane row
    for the packed CLOCK states followed by one reference-epoch row per
    request: ``[1 + num_requests, hot_buffer_size]``.

    ``scratch_locs`` and ``scratch_state`` hold reusable miss locations,
    counters, and metadata shared by all layers.
    """

    cache_index: torch.Tensor
    cache_policy: torch.Tensor
    scratch_locs: torch.Tensor
    scratch_state: torch.Tensor


def _spec_transfer_policy(
    transfer_kind: str, item_size_bytes: int | None = None
) -> str:
    if transfer_kind == _LINEAR_TRANSFER:
        if item_size_bytes is None:
            raise ValueError("Linear HiSparse KV transfer requires item_size_bytes.")
        return f"LinearKVTransferPolicy<{item_size_bytes}>"
    if transfer_kind == _DSV4_PAGED_TRANSFER:
        return "Dsv4PagedKVTransferPolicy"
    raise ValueError(f"Unsupported HiSparse KV transfer kind: {transfer_kind!r}.")


@cache_once
def _jit_spec_module(
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    num_steps: int,
    record_miss_plan: bool,
    transfer_policy: str,
) -> Module:
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        num_steps,
        record_miss_plan,
        transfer_policy,
        is_arch_support_pdl(),
    )
    return load_jit(
        "hisparse_spec",
        *template_args,
        cuda_files=["kvcacheio/hisparse_spec.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer_spec",
                f"load_cache_to_device_buffer_spec<{template_args}>",
            )
        ],
    )


@cache_once
def _jit_spec_state_module(hot_buffer_size: int) -> Module:
    template_args = make_cpp_args(hot_buffer_size)
    return load_jit(
        "hisparse_spec_state",
        hot_buffer_size,
        cuda_files=["kvcacheio/hisparse_spec.cuh"],
        cuda_wrappers=[
            (
                "initialize_hisparse_spec_state",
                f"initialize_hisparse_spec_state<{template_args}>",
            )
        ],
    )


@cache_once
def _jit_hisparse_spec_metadata_module(
    verify_width: int,
    compress_ratio: int,
    hot_buffer_size: int,
    speculative_slots: int,
) -> Module:
    template_args = make_cpp_args(
        verify_width,
        compress_ratio,
        hot_buffer_size,
        speculative_slots,
    )
    return load_jit(
        "hisparse_spec_metadata",
        verify_width,
        compress_ratio,
        hot_buffer_size,
        speculative_slots,
        cuda_files=["kvcacheio/hisparse_spec.cuh"],
        cuda_wrappers=[
            (
                "prepare_hisparse_spec_verify",
                f"prepare_hisparse_spec_verify<{template_args}>",
            ),
            (
                "complete_hisparse_spec_finalize",
                f"complete_hisparse_spec_finalize<{template_args}>",
            ),
        ],
    )


@cache_once
def _jit_hisparse_spec_transfer_module(
    compress_ratio: int,
    hot_buffer_size: int,
    speculative_slots: int,
    transfer_policy: str,
) -> Module:
    template_args = make_cpp_args(
        compress_ratio,
        hot_buffer_size,
        speculative_slots,
        transfer_policy,
    )
    return load_jit(
        "hisparse_spec_transfer",
        compress_ratio,
        hot_buffer_size,
        speculative_slots,
        transfer_policy,
        cuda_files=["kvcacheio/hisparse_spec.cuh"],
        cuda_wrappers=[
            (
                "transfer_hisparse_spec_finalize",
                f"transfer_hisparse_spec_finalize<{template_args}>",
            )
        ],
    )


def load_cache_to_device_buffer_spec_mla(
    *,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    state: HiSparseSpecState,
    num_real_reqs: torch.Tensor,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    transfer_kind: str = _LINEAR_TRANSFER,
    item_size_bytes: int | None = None,
) -> None:
    """Resolve all speculative steps and swap unique misses in one launch pair.

    Optional miss-plan outputs use the same protocol as the single-step HiSparse
    kernel, so shared-index layers can replay only the Host-to-GPU copies with
    ``copy_cache_planned_mla``.
    """
    _, num_steps, num_top_k = top_k_tokens.shape
    if num_steps <= 0 or num_top_k <= 0 or num_steps * num_top_k > 8192:
        raise ValueError(
            "HiSparse speculative swap requires positive steps/top-k and "
            f"steps * top-k <= 8192, got {num_steps} * {num_top_k}."
        )
    hot_buffer_size = state.cache_policy.size(1)
    active_tail_slots = device_buffer_tokens.size(1) - hot_buffer_size
    record_miss_plan = miss_src is not None
    if record_miss_plan:
        if miss_dst is None or miss_count is None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together."
            )
        if miss_src.dtype != torch.int64 or miss_dst.dtype != torch.int32:
            raise ValueError("miss_src must be int64 and miss_dst must be int32.")
        if miss_count.dtype != torch.int32:
            raise ValueError("miss_count must be int32.")
        plan_capacity = num_steps * num_top_k
        batch_size = top_k_tokens.size(0)
        if (
            miss_src.ndim != 2
            or miss_dst.ndim != 2
            or miss_src.size(0) < batch_size
            or miss_dst.size(0) < batch_size
            or miss_src.size(1) < plan_capacity
            or miss_dst.size(1) < plan_capacity
        ):
            raise ValueError(
                "speculative miss_src/miss_dst must have shape "
                f"[batch, >= steps * top_k] (capacity {plan_capacity})."
            )
        if miss_count.ndim != 1 or miss_count.numel() < batch_size:
            raise ValueError("speculative miss_count must have shape [batch].")
        if miss_src.stride(0) != miss_dst.stride(0):
            raise ValueError("miss_src/miss_dst row strides must match.")
    else:
        if miss_dst is not None or miss_count is not None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together."
            )
        empty = torch.empty(0)
        miss_src = miss_dst = miss_count = empty

    if transfer_kind == _LINEAR_TRANSFER and item_size_bytes is None:
        item_size_bytes = host_cache.stride(0) * host_cache.element_size()

    module = _jit_spec_module(
        _GATHER_BLOCK_SIZE,
        num_top_k,
        hot_buffer_size,
        num_steps,
        record_miss_plan,
        _spec_transfer_policy(transfer_kind, item_size_bytes),
    )

    module.load_cache_to_device_buffer_spec(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        state.cache_index,
        state.cache_policy,
        state.scratch_locs,
        state.scratch_state,
        num_real_reqs,
        active_tail_slots,
        miss_src,
        miss_dst,
        miss_count,
    )


def initialize_hisparse_spec_state(
    *,
    device_buffer_tokens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    state: HiSparseSpecState,
) -> None:
    hot_buffer_size = state.cache_policy.size(1)
    module = _jit_spec_state_module(hot_buffer_size)
    module.initialize_hisparse_spec_state(
        device_buffer_tokens,
        req_pool_indices,
        state.cache_index,
        state.cache_policy,
        state.scratch_state,
    )


def prepare_hisparse_spec_verify(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    hisparse_out_locs: torch.Tensor,
    req_to_device_buffer: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    num_real_reqs: torch.Tensor,
    logical_to_device_mapping: torch.Tensor,
    verify_width: int,
    compress_ratio: int,
    hot_buffer_size: int,
    speculative_slots: int,
) -> None:
    module = _jit_hisparse_spec_metadata_module(
        verify_width,
        compress_ratio,
        hot_buffer_size,
        speculative_slots,
    )
    module.prepare_hisparse_spec_verify(
        req_pool_indices,
        prefix_lens,
        hisparse_out_locs,
        req_to_device_buffer,
        device_buffer_tokens,
        num_real_reqs,
        logical_to_device_mapping,
    )


def transfer_hisparse_spec_finalize(
    *,
    device_ptrs: torch.Tensor,
    host_ptrs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    req_to_host_pool: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    compress_ratio: int,
    hot_buffer_size: int,
    speculative_slots: int,
    transfer_kind: str,
    item_size_bytes: int | None = None,
) -> None:
    module = _jit_hisparse_spec_transfer_module(
        compress_ratio,
        hot_buffer_size,
        speculative_slots,
        _spec_transfer_policy(transfer_kind, item_size_bytes),
    )
    module.transfer_hisparse_spec_finalize(
        device_ptrs,
        host_ptrs,
        req_pool_indices,
        prefix_lens,
        commit_lens,
        req_to_host_pool,
        device_buffer_locs,
    )


def complete_hisparse_spec_finalize(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    logical_to_device_mapping: torch.Tensor,
    verify_width: int,
    compress_ratio: int,
    hot_buffer_size: int,
    speculative_slots: int,
) -> None:
    module = _jit_hisparse_spec_metadata_module(
        verify_width,
        compress_ratio,
        hot_buffer_size,
        speculative_slots,
    )
    module.complete_hisparse_spec_finalize(
        req_pool_indices,
        prefix_lens,
        commit_lens,
        req_to_token,
        device_buffer_tokens,
        logical_to_device_mapping,
    )
