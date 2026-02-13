# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Helix parallelism operations for attention.

Helix uses All-to-All communication instead of AllGather+ReduceScatter
for context parallel attention, which can reduce communication overhead
for long-context scenarios.

Reference: https://arxiv.org/abs/2507.07120
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

# ============================================================================
# Packed A2A Buffer Cache
# ============================================================================
# Helix uses a single packed A2A call per layer, fusing output and LSE into
# one tensor. Send/recv buffers are cached by shape to avoid per-call
# allocation. In vLLM continuous batching, B (decode token count) varies
# per step, but the scheduler stabilizes B at a few values during steady
# state. The cache typically holds 2-3 entries. Capped at _A2A_CACHE_LIMIT
# entries to bound GPU memory; oldest entries evicted on overflow.
_A2A_CACHE_LIMIT = 8
_a2a_buffers: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()


def _lse_weighted_combine(
    outputs: torch.Tensor,
    lses: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference implementation for LSE-weighted combination.

    This is a pure PyTorch implementation for testing purposes.
    For GPU execution, use helix_lse_combine_triton instead.

    Args:
        outputs: Partial attention outputs [N, B, H, D]
                 N = number of KV shards (ranks)
                 B = batch size
                 H = number of heads
                 D = head dimension
        lses: Log-sum-exp values [N, B, H]
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H, D], and optionally global LSE [B, H]
    """
    N, B, H, D = outputs.shape

    # Handle NaN and inf in LSEs
    lses = torch.where(
        torch.isnan(lses) | torch.isinf(lses),
        torch.tensor(float("-inf"), device=lses.device, dtype=lses.dtype),
        lses,
    )

    # Compute max LSE for numerical stability
    lse_max, _ = lses.max(dim=0)  # [B, H]
    lse_max = torch.where(
        lse_max == float("-inf"),
        torch.zeros_like(lse_max),
        lse_max,
    )

    # Compute weights: softmax over the N dimension
    if is_lse_base_on_e:
        weights = torch.exp(lses - lse_max.unsqueeze(0))  # [N, B, H]
    else:
        weights = torch.pow(2.0, lses - lse_max.unsqueeze(0))  # [N, B, H]

    # Handle NaN weights
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)

    # Normalize weights
    weight_sum = weights.sum(dim=0, keepdim=True)  # [1, B, H]
    weights = weights / weight_sum.clamp(min=1e-10)  # [N, B, H]

    # Weighted combination: sum over N dimension
    # outputs: [N, B, H, D], weights: [N, B, H] -> need to expand weights
    result = (outputs * weights.unsqueeze(-1)).sum(dim=0)  # [B, H, D]

    if return_lse:
        # Compute global LSE: logsumexp over N dimension
        if is_lse_base_on_e:
            global_lse = torch.log(weight_sum.squeeze(0)) + lse_max  # [B, H]
        else:
            global_lse = torch.log2(weight_sum.squeeze(0)) + lse_max  # [B, H]
        return result, global_lse

    return result


@triton.jit
def _helix_lse_combine_kernel(
    # Input pointers
    recv_output_ptr,
    recv_lse_ptr,
    # Output pointers
    out_ptr,
    out_lse_ptr,
    # Strides for recv_output [N, B, H_local, D]
    ro_stride_N,
    ro_stride_B,
    ro_stride_H,
    ro_stride_D,
    # Strides for recv_lse [N, B, H_local]
    rl_stride_N,
    rl_stride_B,
    rl_stride_H,
    # Strides for output [B, H_local, D]
    o_stride_B,
    o_stride_H,
    o_stride_D,
    # Constants
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """
    Triton kernel for Helix LSE-weighted combination.

    After All-to-All, each rank has:
    - recv_output [N, B, H_local, D]: partial outputs from all KV shards
    - recv_lse [N, B, H_local]: partial LSEs from all KV shards

    This kernel computes the weighted combination locally (no communication).

    Grid: (B, H_local)
    Each program handles one (batch, head) and processes all D elements.
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)

    # Base offset for this (batch, head)
    base_lse_offset = batch_idx * rl_stride_B + head_idx * rl_stride_H
    base_out_offset = batch_idx * ro_stride_B + head_idx * ro_stride_H

    # Step 1: Load all LSEs and compute weights
    # We need to load LSEs one by one and compute global LSE
    # First pass: find max LSE
    lse_max = -float("inf")
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        # Handle NaN and inf
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    # Second pass: compute sum of exp(lse - max)
    lse_sum = 0.0
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    # Compute global LSE (Triton kernel - keep if/else for clarity)
    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    # Step 2: Weighted combination across D dimension
    d_offsets = tl.arange(0, HEAD_DIM)

    # Initialize accumulator
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Third pass: weighted sum
    for n in tl.static_range(N):
        # Compute weight for this shard
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            weight = tl.exp(lse_val - global_lse)
        else:
            weight = tl.exp2(lse_val - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)

        # Load output for this shard and accumulate
        out_offsets = n * ro_stride_N + base_out_offset + d_offsets * ro_stride_D
        out_vals = tl.load(recv_output_ptr + out_offsets)
        acc += out_vals.to(tl.float32) * weight

    # Store result
    final_offsets = (
        batch_idx * o_stride_B + head_idx * o_stride_H + d_offsets * o_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)

    # Optional: store global LSE
    if RETURN_LSE:
        tl.store(out_lse_ptr + base_lse_offset, global_lse)


def helix_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated LSE-weighted combination for Helix.

    Args:
        recv_output: [N, B, H_local, D] - partial outputs from all KV shards
        recv_lse: [N, B, H_local] - partial LSEs from all KV shards
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H_local, D]
        If return_lse=True, also returns global_lse [B, H_local]
    """
    N, B, H_local, D = recv_output.shape

    # Allocate output tensors
    out = torch.empty(
        (B, H_local, D), device=recv_output.device, dtype=recv_output.dtype
    )

    if return_lse:
        out_lse = torch.empty(
            (B, H_local), device=recv_lse.device, dtype=recv_lse.dtype
        )
    else:
        # Dummy tensor (not used, but kernel expects it)
        out_lse = torch.empty(1, device=recv_lse.device, dtype=recv_lse.dtype)

    # Get strides
    ro_stride_N, ro_stride_B, ro_stride_H, ro_stride_D = recv_output.stride()
    rl_stride_N, rl_stride_B, rl_stride_H = recv_lse.stride()
    o_stride_B, o_stride_H, o_stride_D = out.stride()

    # Launch kernel (grid must be 3-tuple)
    grid = (B, H_local, 1)

    _helix_lse_combine_kernel[grid](
        recv_output,
        recv_lse,
        out,
        out_lse,
        ro_stride_N,
        ro_stride_B,
        ro_stride_H,
        ro_stride_D,
        rl_stride_N,
        rl_stride_B,
        rl_stride_H,
        o_stride_B,
        o_stride_H,
        o_stride_D,
        N=N,
        HEAD_DIM=D,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
    )

    if return_lse:
        return out, out_lse
    return out


def helix_alltoall_lse_reduce(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    kvp_group: GroupCoordinator,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Helix-style attention output combination using packed All-to-All.

    Uses a single packed A2A call per layer, fusing output and LSE into one
    tensor to minimize NCCL call count. This reduces NCCL calls from 3
    (standard DCP: AG+AG+RS) to 1 per layer, providing significant latency
    savings on multi-node deployments where per-call NCCL overhead is high.

    Packed A2A communication:
        Output [B,H,D] and LSE [B,H] (fp32) are packed into a single tensor
        [N,B,H/N,D+K] where K extra elements carry LSE via bit-exact dtype
        reinterpretation (K=2 for bf16/fp16, K=1 for fp32). This preserves
        full fp32 LSE precision with only +0.2% data volume overhead (D=512).

    Send/recv buffers are cached by shape to avoid per-call allocation.
    Cache is bounded to 8 entries; oldest evicted on overflow.

    Tensor flow:
        Input:  local_output [B, H, D] - all heads, local KV shard
        Pack:   [N, B, H/N, D+K] - output + LSE packed into one tensor
        A2A:    Single all_to_all_single call
        Unpack: recv_output [N, B, H/N, D], recv_lse [N, B, H/N]
        Combine: output [B, H/N, D] - LSE-weighted sum (Triton kernel)

    Args:
        local_output: Local attention output [B, H, D] where:
                      B = num_tokens, H = gathered_heads, D = kv_lora_rank
                      Each rank has output for the SAME tokens but computed
                      with DIFFERENT KV cache shards.
        local_lse: Local log-sum-exp values [B, H]
        kvp_group: GroupCoordinator for KV parallel communication
        return_lse: If True, also return the local portion of global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined attention output [B, H/N, D] (scattered along head dimension)
        If return_lse=True, also returns local_lse [B, H/N]
    """
    world_size = kvp_group.world_size

    if world_size == 1:
        if return_lse:
            return local_output, local_lse
        return local_output

    # Ensure inputs are contiguous for reshape operations
    local_output = local_output.contiguous()
    local_lse = local_lse.contiguous()

    B, H, D = local_output.shape
    H_per_rank = H // world_size

    # Compute packing dimensions.
    # fp32 LSE (4 bytes) is stored as K output-dtype elements via byte
    # reinterpretation: K=2 for bf16/fp16 (2 bytes each), K=1 for fp32.
    # This preserves bit-exact LSE precision (no value casting).
    out_elem_size = local_output.element_size()
    lse_extra_elems = 4 // out_elem_size
    packed_D = D + lse_extra_elems

    # Get or allocate cached send/recv buffers.
    cache_key = (
        B,
        H_per_rank,
        packed_D,
        world_size,
        local_output.dtype,
        str(local_output.device),
    )
    if cache_key in _a2a_buffers:
        # Move to end (most recently used) for LRU eviction
        _a2a_buffers.move_to_end(cache_key)
        send_packed, recv_packed = _a2a_buffers[cache_key]
    else:
        packed_shape = (world_size, B, H_per_rank, packed_D)
        send_packed = torch.empty(
            packed_shape, dtype=local_output.dtype, device=local_output.device
        )
        recv_packed = torch.empty(
            packed_shape, dtype=local_output.dtype, device=local_output.device
        )
        _a2a_buffers[cache_key] = (send_packed, recv_packed)
        logger.debug(
            "Helix A2A: allocated buffers for B=%d, packed_D=%d (cache entries: %d)",
            B,
            packed_D,
            len(_a2a_buffers),
        )
        # Evict oldest entries if cache exceeds limit
        while len(_a2a_buffers) > _A2A_CACHE_LIMIT:
            evicted_key, _ = _a2a_buffers.popitem(last=False)
            logger.debug("Helix A2A: evicted buffer cache entry B=%d", evicted_key[0])

    # Pack output: [B, H, D] -> [B, N, H/N, D] -> [N, B, H/N, D]
    # .copy_() from non-contiguous permuted view reuses the send buffer.
    send_packed[:, :, :, :D].copy_(
        local_output.view(B, world_size, H_per_rank, D).permute(1, 0, 2, 3)
    )

    # Pack LSE with bit-exact preservation via dtype reinterpretation.
    # fp32 bytes are reinterpreted as output-dtype elements (no value cast).
    # After A2A, view(fp32) reconstructs exact original values.
    lse_permuted = (
        local_lse.view(B, world_size, H_per_rank).permute(1, 0, 2).contiguous()
    )  # [N, B, H/N] fp32

    if out_elem_size == 4:
        send_packed[:, :, :, D].copy_(lse_permuted)
    else:
        lse_reinterp = lse_permuted.view(local_output.dtype)
        lse_reinterp = lse_reinterp.view(world_size, B, H_per_rank, lse_extra_elems)
        send_packed[:, :, :, D:packed_D].copy_(lse_reinterp)

    # Single A2A call (fused output + LSE)
    dist.all_to_all_single(
        recv_packed.view(-1), send_packed.view(-1), group=kvp_group.device_group
    )

    # Unpack output: [N, B, H/N, D]
    recv_output = recv_packed[:, :, :, :D].contiguous()

    # Unpack LSE with reverse reinterpretation: [N, B, H/N]
    if out_elem_size == 4:
        recv_lse = recv_packed[:, :, :, D].contiguous()
    else:
        recv_lse_raw = recv_packed[:, :, :, D:packed_D].contiguous()
        recv_lse = recv_lse_raw.view(world_size, B, H_per_rank * lse_extra_elems).view(
            torch.float32
        )

    # LSE-weighted combination via Triton kernel (local, no communication)
    return helix_lse_combine_triton(
        recv_output,
        recv_lse,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )
