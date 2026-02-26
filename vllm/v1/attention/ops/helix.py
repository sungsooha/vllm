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
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

# ============================================================================
# Static Buffers for Packed A2A
# ============================================================================
# Helix packs output + LSE into a single tensor for one NCCL A2A call per
# layer (instead of 2). Buffers are allocated once for the largest batch size
# seen and reused across all layers within a step (layers execute sequentially).
# CUDA graph warmup runs with padded batch sizes, so max B is seen early.
_helix_a2a_buffers: dict[str, torch.Tensor] | None = None
_helix_a2a_max_B: int = 0


def _ensure_a2a_buffers(
    N: int,
    B: int,
    H_per_rank: int,
    D: int,
    out_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate or grow static A2A buffers to fit batch size B."""
    global _helix_a2a_buffers, _helix_a2a_max_B

    out_elem_size = torch.tensor([], dtype=out_dtype).element_size()
    lse_extra = 4 // out_elem_size  # K=2 for bf16/fp16, K=1 for fp32
    packed_D = D + lse_extra

    if _helix_a2a_buffers is not None and _helix_a2a_max_B >= B:
        return _helix_a2a_buffers

    _helix_a2a_max_B = B
    _helix_a2a_buffers = {
        "send_packed": torch.empty(
            (N, B, H_per_rank, packed_D), dtype=out_dtype, device=device
        ),
        "recv_packed": torch.empty(
            (N, B, H_per_rank, packed_D), dtype=out_dtype, device=device
        ),
        "recv_lse": torch.empty((N, B, H_per_rank), dtype=torch.float32, device=device),
        "out": torch.empty((B, H_per_rank, D), dtype=out_dtype, device=device),
        "out_lse": torch.empty((B, H_per_rank), dtype=torch.float32, device=device),
    }
    logger.debug(
        "Helix A2A: allocated static buffers for B_max=%d, packed_D=%d",
        B,
        packed_D,
    )
    return _helix_a2a_buffers


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
    out: torch.Tensor | None = None,
    out_lse: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated LSE-weighted combination for Helix.

    Args:
        recv_output: [N, B, H_local, D] - partial outputs from all KV shards
        recv_lse: [N, B, H_local] - partial LSEs from all KV shards
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2
        out: Optional pre-allocated output buffer [B, H_local, D]
        out_lse: Optional pre-allocated LSE output buffer [B, H_local]

    Returns:
        Combined output [B, H_local, D]
        If return_lse=True, also returns global_lse [B, H_local]
    """
    N, B, H_local, D = recv_output.shape

    # Use provided buffers or allocate new ones
    if out is None:
        out = torch.empty(
            (B, H_local, D), device=recv_output.device, dtype=recv_output.dtype
        )
    if return_lse and out_lse is None:
        out_lse = torch.empty(
            (B, H_local), device=recv_lse.device, dtype=recv_lse.dtype
        )
    elif not return_lse and out_lse is None:
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
    tensor to minimize NCCL kernel count. Static buffers are allocated once
    for the largest batch size seen and shared across all layers in a step.

    Packed A2A communication:
        Output [B,H,D] and LSE [B,H] (fp32) are packed into a single tensor
        [N,B,H/N,D+K] where K extra elements carry LSE via bit-exact dtype
        reinterpretation (K=2 for bf16/fp16, K=1 for fp32). This preserves
        full fp32 LSE precision with only +0.2% data volume overhead (D=128).

    Tensor flow:
        Input:  local_output [B, H, D] - all heads, local KV shard
        Pack:   [N, B, H/N, D+K] - output + LSE packed into one tensor
        A2A:    Single all_to_all_single call
        Unpack: recv_output [N, B, H/N, D] (view), recv_lse [N, B, H/N]
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
    out_elem_size = local_output.element_size()
    lse_extra = 4 // out_elem_size  # K=2 for bf16/fp16, K=1 for fp32
    packed_D = D + lse_extra

    # Get static buffers (allocated once, reused across layers and steps)
    bufs = _ensure_a2a_buffers(
        world_size, B, H_per_rank, D, local_output.dtype, local_output.device
    )

    # Use full-sized buffers for A2A to guarantee contiguity.
    # When B < B_max (warmup only), extra rows carry garbage but are never read.
    # During CUDA graph execution B == B_max, so no bandwidth is wasted.
    send_packed = bufs["send_packed"]  # [N, B_max, H_per_rank, packed_D]
    recv_packed = bufs["recv_packed"]  # [N, B_max, H_per_rank, packed_D]

    # Step 1: Pack output into send buffer (first B rows)
    # [B, H, D] -> view [B, N, H/N, D] -> permute [N, B, H/N, D] -> copy
    send_packed[:, :B, :, :D].copy_(
        local_output.view(B, world_size, H_per_rank, D).permute(1, 0, 2, 3)
    )

    # Pack LSE via bit-exact dtype reinterpretation (preserves fp32 precision)
    lse_permuted = (
        local_lse.view(B, world_size, H_per_rank).permute(1, 0, 2).contiguous()
    )  # [N, B, H/N] fp32
    if out_elem_size == 4:
        # fp32 output: LSE is same dtype, just copy
        send_packed[:, :B, :, D].copy_(lse_permuted)
    else:
        # bf16/fp16 output: reinterpret fp32 bytes as K output-dtype elements
        lse_reinterp = lse_permuted.view(local_output.dtype).view(
            world_size, B, H_per_rank, lse_extra
        )
        send_packed[:, :B, :, D:packed_D].copy_(lse_reinterp)

    # Step 2: Single packed A2A call (replaces 2 separate calls)
    # Full-sized buffers are always contiguous, so reshape(-1) is a view.
    dist.all_to_all_single(
        recv_packed.reshape(-1),
        send_packed.reshape(-1),
        group=kvp_group.device_group,
    )

    # Step 3: Unpack received data (first B rows only)
    # Output: view of packed buffer (no .contiguous() — Triton reads via strides)
    # The view has stride ro_stride_H = packed_D (not D), which the kernel
    # handles correctly via d_offsets * ro_stride_D where ro_stride_D = 1.
    recv_output_view = recv_packed[:, :B, :, :D]

    # LSE: unpack into static fp32 buffer (Triton kernel needs float32 pointer)
    recv_lse = bufs["recv_lse"][:, :B, :]
    if out_elem_size == 4:
        recv_lse.copy_(recv_packed[:, :B, :, D])
    else:
        # Reverse reinterpretation: K bf16 elements -> fp32
        recv_lse_raw = recv_packed[:, :B, :, D:packed_D].contiguous()
        recv_lse.copy_(
            recv_lse_raw.view(world_size, B, H_per_rank * lse_extra).view(torch.float32)
        )

    # Step 4: LSE-weighted combination via Triton kernel (local, no communication)
    # Use static output buffers to avoid per-call allocation.
    out = bufs["out"][:B, :, :]
    out_lse = bufs["out_lse"][:B, :] if return_lse else None

    return helix_lse_combine_triton(
        recv_output_view,
        recv_lse,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
        out=out,
        out_lse=out_lse,
    )
