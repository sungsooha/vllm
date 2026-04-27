# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DCP All-to-All communication backend for attention.

Provides All-to-All (A2A) communication as an alternative to
AllGather + ReduceScatter (AG+RS) for Decode Context Parallel (DCP).
Instead of gathering the full Q tensor and scattering partial outputs,
A2A exchanges partial attention outputs and their LSE values across
ranks, then combines them with exact LSE-weighted reduction.

This reduces the number of NCCL calls per attention layer from 3
(AG for Q, AG for K metadata, RS for output) to 2 (A2A for output,
A2A for LSE), lowering per-step communication overhead for long-context
decode where NCCL latency is a significant fraction of step time.

Usage:
    vllm serve model --tp 16 --dcp 16 --dcp-comm-backend a2a

Reference: https://arxiv.org/abs/2507.07120
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm import envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.attention.ops.common import CPTritonContext

logger = init_logger(__name__)


@dataclass
class _DCPA2AProfileStats:
    calls: int = 0
    skipped_warmup: int = 0
    bytes_output: int = 0
    bytes_lse: int = 0
    totals_ms: dict[str, float] = field(default_factory=dict)

    def add(
        self,
        timings_ms: dict[str, float],
        bytes_output: int,
        bytes_lse: int,
    ) -> None:
        self.calls += 1
        self.bytes_output += bytes_output
        self.bytes_lse += bytes_lse
        for key, value in timings_ms.items():
            self.totals_ms[key] = self.totals_ms.get(key, 0.0) + value

    def avg(self, key: str) -> float:
        if self.calls == 0:
            return 0.0
        return self.totals_ms.get(key, 0.0) / self.calls


_DCP_A2A_PROFILE_STATS: dict[
    tuple[int, int, int, int, bool, bool], _DCPA2AProfileStats
] = {}


def _dcp_a2a_profile_enabled() -> bool:
    return bool(envs.VLLM_DCP_A2A_PROFILE)


def _dcp_a2a_should_log_rank(cp_group: GroupCoordinator) -> bool:
    if envs.VLLM_DCP_A2A_PROFILE_ALL_RANKS:
        return True
    return getattr(cp_group, "rank_in_group", 0) == 0


def _dcp_a2a_mark(sync: bool) -> float:
    if sync:
        torch.accelerator.synchronize()
    return time.perf_counter()


def _dcp_a2a_profile_record(
    cp_group: GroupCoordinator,
    key: tuple[int, int, int, int, bool, bool],
    timings_ms: dict[str, float],
    bytes_output: int,
    bytes_lse: int,
) -> None:
    stats = _DCP_A2A_PROFILE_STATS.setdefault(key, _DCPA2AProfileStats())
    warmup = max(envs.VLLM_DCP_A2A_PROFILE_WARMUP, 0)
    if stats.skipped_warmup < warmup:
        stats.skipped_warmup += 1
        return

    stats.add(timings_ms, bytes_output, bytes_lse)
    interval = max(envs.VLLM_DCP_A2A_PROFILE_INTERVAL, 1)
    if stats.calls % interval != 0 or not _dcp_a2a_should_log_rank(cp_group):
        return

    world_size, B, H, D, return_lse, is_lse_base_on_e = key
    rank = getattr(cp_group, "rank_in_group", -1)
    total_mb = (stats.bytes_output + stats.bytes_lse) / (1024 * 1024)
    logger.info(
        "DCP_A2A_PROFILE rank=%s world=%d shape=B%d_H%d_D%d "
        "calls=%d warmup=%d sync=%s return_lse=%s lse_base_e=%s "
        "avg_ms(total=%.3f,copy=%.3f,pack=%.3f,a2a_enqueue=%.3f,"
        "wait_output=%.3f,wait_lse=%.3f,combine=%.3f) "
        "avg_comm_mb=%.3f",
        rank,
        world_size,
        B,
        H,
        D,
        stats.calls,
        stats.skipped_warmup,
        envs.VLLM_DCP_A2A_PROFILE_SYNC,
        return_lse,
        is_lse_base_on_e,
        stats.avg("total"),
        stats.avg("copy"),
        stats.avg("pack"),
        stats.avg("a2a_enqueue"),
        stats.avg("wait_output"),
        stats.avg("wait_lse"),
        stats.avg("combine"),
        total_mb / stats.calls,
    )


def _lse_weighted_combine(
    outputs: torch.Tensor,
    lses: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference implementation for LSE-weighted combination.

    This is a pure PyTorch implementation used for testing and validation.
    For GPU execution, use dcp_lse_combine_triton instead.

    Args:
        outputs: Partial attention outputs [N, B, H, D]
                 N = number of KV shards (ranks)
                 B = batch size (num_tokens)
                 H = number of heads per rank
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
    result = (outputs * weights.unsqueeze(-1)).sum(dim=0)  # [B, H, D]

    if return_lse:
        if is_lse_base_on_e:
            global_lse = torch.log(weight_sum.squeeze(0)) + lse_max  # [B, H]
        else:
            global_lse = torch.log2(weight_sum.squeeze(0)) + lse_max  # [B, H]
        return result, global_lse

    return result


@triton.jit
def _dcp_lse_combine_kernel(
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
    Triton kernel for LSE-weighted combination of partial attention outputs.

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

    # First pass: find max LSE for numerical stability
    lse_max = -float("inf")
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
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

    # Compute global LSE
    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    # Third pass: weighted combination across D dimension
    d_offsets = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for n in tl.static_range(N):
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

        out_offsets = n * ro_stride_N + base_out_offset + d_offsets * ro_stride_D
        out_vals = tl.load(recv_output_ptr + out_offsets)
        acc += out_vals.to(tl.float32) * weight

    # Store result
    final_offsets = (
        batch_idx * o_stride_B + head_idx * o_stride_H + d_offsets * o_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)

    if RETURN_LSE:
        tl.store(out_lse_ptr + base_lse_offset, global_lse)


def dcp_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated LSE-weighted combination for DCP A2A.

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

    out = torch.empty(
        (B, H_local, D), device=recv_output.device, dtype=recv_output.dtype
    )

    if return_lse:
        out_lse = torch.empty(
            (B, H_local), device=recv_lse.device, dtype=recv_lse.dtype
        )
    else:
        out_lse = torch.empty(1, device=recv_lse.device, dtype=recv_lse.dtype)

    ro_stride_N, ro_stride_B, ro_stride_H, ro_stride_D = recv_output.stride()
    rl_stride_N, rl_stride_B, rl_stride_H = recv_lse.stride()
    o_stride_B, o_stride_H, o_stride_D = out.stride()

    grid = (B, H_local, 1)

    _dcp_lse_combine_kernel[grid](
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


def _dcp_a2a_lse_pack_dim(output_dtype: torch.dtype) -> int:
    """Number of output-dtype slots needed to bit-pack one fp32 LSE value."""
    bits = torch.finfo(output_dtype).bits
    if bits > 32 or 32 % bits != 0:
        raise ValueError(f"Cannot pack fp32 LSE into output dtype {output_dtype}.")
    return 32 // bits


@triton.jit
def _dcp_a2a_pack_send_kernel(
    # Input: output [B, H, D] and LSE [B, H]
    out_ptr,
    lse_ptr,
    # Output: packed send buffer [N, B, H_per_rank, D + LSE_PACK_DIM]
    send_ptr,
    # Strides for output [B, H, D]
    out_stride_B,
    out_stride_H,
    out_stride_D,
    # Strides for LSE [B, H]
    lse_stride_B,
    lse_stride_H,
    # Strides for send buffer [N, B, H_per_rank, D + LSE_PACK_DIM]
    send_stride_N,
    send_stride_B,
    send_stride_H,
    send_stride_D,
    # Constants
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H_PER_RANK: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    """
    Fused pack for DCP A2A.

    Converts output [B, H, D] and fp32 LSE [B, H] into a single buffer
    [N, B, H/N, D + LSE_PACK_DIM]. For bf16/fp16 output, fp32 LSE is stored
    as two raw 16-bit halves in the output dtype lanes.
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    for rank_idx in tl.static_range(N):
        src_head_idx = rank_idx * H_PER_RANK + local_head_idx
        send_base = (
            rank_idx * send_stride_N
            + batch_idx * send_stride_B
            + local_head_idx * send_stride_H
        )

        out_offsets = (
            batch_idx * out_stride_B
            + src_head_idx * out_stride_H
            + d_offsets * out_stride_D
        )
        out_vals = tl.load(out_ptr + out_offsets)
        tl.store(send_ptr + send_base + d_offsets * send_stride_D, out_vals)

        lse_offset = batch_idx * lse_stride_B + src_head_idx * lse_stride_H
        lse_val = tl.load(lse_ptr + lse_offset)

        if LSE_PACK_DIM == 1:
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lse_val.to(send_ptr.dtype.element_ty),
            )
        else:
            lse_bits = lse_val.to(tl.uint32, bitcast=True)
            lo = (lse_bits & 0xFFFF).to(tl.uint16)
            hi = ((lse_bits >> 16) & 0xFFFF).to(tl.uint16)
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lo.to(send_ptr.dtype.element_ty, bitcast=True),
            )
            tl.store(
                send_ptr + send_base + (HEAD_DIM + 1) * send_stride_D,
                hi.to(send_ptr.dtype.element_ty, bitcast=True),
            )


@triton.jit
def _dcp_a2a_fused_unpack_combine_kernel(
    # Input: packed recv buffer [N, B, H_per_rank, D + LSE_PACK_DIM]
    recv_ptr,
    # Output: combined output [B, H_per_rank, D] and optional global LSE
    out_ptr,
    out_lse_ptr,
    # Strides for recv buffer [N, B, H_per_rank, D + LSE_PACK_DIM]
    recv_stride_N,
    recv_stride_B,
    recv_stride_H,
    recv_stride_D,
    # Strides for output [B, H_per_rank, D]
    out_stride_B,
    out_stride_H,
    out_stride_D,
    # Strides for output LSE [B, H_per_rank]
    out_lse_stride_B,
    out_lse_stride_H,
    # Constants
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    """
    Fused unpack plus LSE-weighted combine for packed DCP A2A.

    Reads output lanes and bit-packed fp32 LSE directly from the packed receive
    buffer. This preserves the global LSE return contract required by DSV4's
    attention sink.
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_max = -float("inf")
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)

        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    lse_sum = 0.0
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)

        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)

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

        out_offsets = recv_base + d_offsets * recv_stride_D
        out_vals = tl.load(recv_ptr + out_offsets).to(tl.float32)
        acc += out_vals * weight

    final_offsets = (
        batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)

    if RETURN_LSE:
        out_lse_offset = batch_idx * out_lse_stride_B + head_idx * out_lse_stride_H
        tl.store(out_lse_ptr + out_lse_offset, global_lse)


def _dcp_a2a_pack_send_triton(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    send_combined: torch.Tensor,
    world_size: int,
    H_per_rank: int,
    D: int,
    lse_pack_dim: int,
) -> None:
    B = cp_attn_out.shape[0]
    grid = (B, H_per_rank, 1)
    _dcp_a2a_pack_send_kernel[grid](
        cp_attn_out,
        cp_attn_lse,
        send_combined,
        cp_attn_out.stride(0),
        cp_attn_out.stride(1),
        cp_attn_out.stride(2),
        cp_attn_lse.stride(0),
        cp_attn_lse.stride(1),
        send_combined.stride(0),
        send_combined.stride(1),
        send_combined.stride(2),
        send_combined.stride(3),
        N=world_size,
        HEAD_DIM=D,
        H_PER_RANK=H_per_rank,
        LSE_PACK_DIM=lse_pack_dim,
    )


def _dcp_a2a_fused_unpack_combine_triton(
    recv_combined: torch.Tensor,
    head_dim: int,
    lse_pack_dim: int,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    N, B, H_per_rank, _ = recv_combined.shape
    out = torch.empty(
        (B, H_per_rank, head_dim),
        device=recv_combined.device,
        dtype=recv_combined.dtype,
    )
    if return_lse:
        out_lse = torch.empty(
            (B, H_per_rank), device=recv_combined.device, dtype=torch.float32
        )
    else:
        out_lse = torch.empty((1, 1), device=recv_combined.device, dtype=torch.float32)

    grid = (B, H_per_rank, 1)
    _dcp_a2a_fused_unpack_combine_kernel[grid](
        recv_combined,
        out,
        out_lse,
        recv_combined.stride(0),
        recv_combined.stride(1),
        recv_combined.stride(2),
        recv_combined.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out_lse.stride(0),
        out_lse.stride(1),
        N=N,
        HEAD_DIM=head_dim,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
        LSE_PACK_DIM=lse_pack_dim,
    )

    if return_lse:
        return out, out_lse
    return out


def dcp_a2a_lse_reduce_packed(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Experimental packed DCP A2A path.

    Packs attention output and fp32 LSE into one output-dtype buffer, exchanges it
    with one all_to_all_single, then fuses LSE unpack and global LSE combine.
    """
    del ctx
    world_size = cp_group.world_size
    if world_size == 1:
        if return_lse:
            return cp_attn_out, cp_attn_lse
        return cp_attn_out

    B, H, D = cp_attn_out.shape
    if H % world_size != 0:
        raise ValueError(f"H={H} must be divisible by DCP world size {world_size}.")
    H_per_rank = H // world_size
    lse_pack_dim = _dcp_a2a_lse_pack_dim(cp_attn_out.dtype)

    profile = _dcp_a2a_profile_enabled()
    profile_sync = profile and envs.VLLM_DCP_A2A_PROFILE_SYNC

    timings_ms: dict[str, float] = {"copy": 0.0, "wait_lse": 0.0}
    t_total = _dcp_a2a_mark(profile_sync) if profile else 0.0

    t0 = _dcp_a2a_mark(profile_sync) if profile else 0.0
    send_combined = torch.empty(
        (world_size, B, H_per_rank, D + lse_pack_dim),
        device=cp_attn_out.device,
        dtype=cp_attn_out.dtype,
    )
    recv_combined = torch.empty_like(send_combined)
    _dcp_a2a_pack_send_triton(
        cp_attn_out,
        cp_attn_lse,
        send_combined,
        world_size,
        H_per_rank,
        D,
        lse_pack_dim,
    )
    if profile:
        t1 = _dcp_a2a_mark(profile_sync)
        timings_ms["pack"] = (t1 - t0) * 1000.0

    t0 = time.perf_counter() if profile else 0.0
    work = dist.all_to_all_single(
        recv_combined.view(-1),
        send_combined.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    if profile:
        t1 = time.perf_counter()
        timings_ms["a2a_enqueue"] = (t1 - t0) * 1000.0

    t0 = time.perf_counter() if profile else 0.0
    work.wait()
    if profile:
        t1 = _dcp_a2a_mark(profile_sync)
        timings_ms["wait_output"] = (t1 - t0) * 1000.0

    t0 = _dcp_a2a_mark(profile_sync) if profile else 0.0
    result = _dcp_a2a_fused_unpack_combine_triton(
        recv_combined,
        D,
        lse_pack_dim,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    if profile:
        t1 = _dcp_a2a_mark(profile_sync)
        timings_ms["combine"] = (t1 - t0) * 1000.0
        timings_ms["total"] = (t1 - t_total) * 1000.0
        profile_key = (world_size, B, H, D, return_lse, is_lse_base_on_e)
        _dcp_a2a_profile_record(
            cp_group,
            profile_key,
            timings_ms,
            bytes_output=send_combined.numel() * send_combined.element_size(),
            bytes_lse=0,
        )

    return result


def dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Combine partial attention outputs across DCP ranks using All-to-All.

    Each rank holds attention output for all heads but only a local shard
    of the KV cache. This function:
    1. Exchanges partial outputs across ranks via All-to-All
    2. Exchanges LSE values via All-to-All
    3. Combines them with exact LSE-weighted reduction (Triton kernel)

    Tensor flow:
        Input:  cp_attn_out [B, H, D] - all heads, local KV shard
        Reshape: [N, B, H/N, D] - split heads across ranks
        A2A:    Two all_to_all_single calls (output and LSE)
        Combine: recv [N, B, H/N, D] + lse [N, B, H/N] -> [B, H/N, D]

    Args:
        cp_attn_out: [B, H, D] where B=num_tokens, H=total_heads, D=head_dim
        cp_attn_lse: [B, H] log-sum-exp values (fp32)
        cp_group: GroupCoordinator for DCP communication
        ctx: CPTritonContext (unused, for signature compatibility)
        return_lse: If True, also return the combined global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H/N, D] (head-scattered)
        If return_lse=True, also returns global_lse [B, H/N]
    """
    world_size = cp_group.world_size

    if world_size == 1:
        if return_lse:
            return cp_attn_out, cp_attn_lse
        return cp_attn_out

    # Large prefill chunks can make A2A temporary buffers several GiB each:
    # [world_size, tokens, heads_per_rank, head_dim]. Bound the transient
    # allocation while keeping the low-token decode path on the fast path.
    max_tokens = envs.VLLM_DCP_A2A_MAX_TOKENS
    if max_tokens > 0 and cp_attn_out.shape[0] > max_tokens:
        B, H, D = cp_attn_out.shape
        if H % world_size != 0:
            raise ValueError(f"H={H} must be divisible by DCP world size {world_size}.")
        H_per_rank = H // world_size
        logger.info_once(
            "Chunking DCP A2A LSE reduce for large token batch: "
            "tokens=%d, max_tokens=%d, world_size=%d, heads=%d, head_dim=%d",
            B,
            max_tokens,
            world_size,
            H,
            D,
        )
        out = torch.empty(
            (B, H_per_rank, D), device=cp_attn_out.device, dtype=cp_attn_out.dtype
        )
        out_lse = (
            torch.empty((B, H_per_rank), device=cp_attn_out.device, dtype=torch.float32)
            if return_lse
            else None
        )
        for start in range(0, B, max_tokens):
            end = min(start + max_tokens, B)
            chunk_result = dcp_a2a_lse_reduce(
                cp_attn_out[start:end],
                cp_attn_lse[start:end],
                cp_group,
                ctx=ctx,
                return_lse=return_lse,
                is_lse_base_on_e=is_lse_base_on_e,
            )
            if return_lse:
                assert isinstance(chunk_result, tuple)
                chunk_out, chunk_lse = chunk_result
                out[start:end].copy_(chunk_out)
                assert out_lse is not None
                out_lse[start:end].copy_(chunk_lse)
            else:
                assert isinstance(chunk_result, torch.Tensor)
                out[start:end].copy_(chunk_result)
        if return_lse:
            assert out_lse is not None
            return out, out_lse
        return out

    if envs.VLLM_DCP_A2A_PACKED:
        return dcp_a2a_lse_reduce_packed(
            cp_attn_out,
            cp_attn_lse,
            cp_group,
            ctx=ctx,
            return_lse=return_lse,
            is_lse_base_on_e=is_lse_base_on_e,
        )

    profile = _dcp_a2a_profile_enabled()
    profile_sync = profile and envs.VLLM_DCP_A2A_PROFILE_SYNC

    timings_ms: dict[str, float] = {}
    t_total = _dcp_a2a_mark(profile_sync) if profile else 0.0

    t0 = _dcp_a2a_mark(profile_sync) if profile else 0.0
    local_output = cp_attn_out.contiguous()
    local_lse = cp_attn_lse.contiguous()
    if profile:
        t1 = _dcp_a2a_mark(profile_sync)
        timings_ms["copy"] = (t1 - t0) * 1000.0

    B, H, D = local_output.shape
    H_per_rank = H // world_size

    t0 = _dcp_a2a_mark(profile_sync) if profile else 0.0
    # Reshape for All-to-All: [B, H, D] -> [N, B, H/N, D]
    # Split heads into N chunks, each destined for a different rank
    send_output = (
        local_output.view(B, world_size, H_per_rank, D).permute(1, 0, 2, 3).contiguous()
    )
    recv_output = torch.empty_like(send_output)

    # Same for LSE: [B, H] -> [N, B, H/N]
    send_lse = local_lse.view(B, world_size, H_per_rank).permute(1, 0, 2).contiguous()
    recv_lse = torch.empty_like(send_lse)
    if profile:
        t1 = _dcp_a2a_mark(profile_sync)
        timings_ms["pack"] = (t1 - t0) * 1000.0

    # All-to-All for partial attention outputs and LSE values (async overlap)
    t0 = time.perf_counter() if profile else 0.0
    work_output = dist.all_to_all_single(
        recv_output.view(-1),
        send_output.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    work_lse = dist.all_to_all_single(
        recv_lse.view(-1),
        send_lse.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    if profile:
        t1 = time.perf_counter()
        timings_ms["a2a_enqueue"] = (t1 - t0) * 1000.0

    t0 = time.perf_counter() if profile else 0.0
    work_output.wait()
    if profile:
        t1 = _dcp_a2a_mark(profile_sync)
        timings_ms["wait_output"] = (t1 - t0) * 1000.0

    t0 = time.perf_counter() if profile else 0.0
    work_lse.wait()
    if profile:
        t1 = _dcp_a2a_mark(profile_sync)
        timings_ms["wait_lse"] = (t1 - t0) * 1000.0

    # LSE-weighted combination via Triton kernel (local, no communication)
    t0 = _dcp_a2a_mark(profile_sync) if profile else 0.0
    result = dcp_lse_combine_triton(
        recv_output,
        recv_lse,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    if profile:
        t1 = _dcp_a2a_mark(profile_sync)
        timings_ms["combine"] = (t1 - t0) * 1000.0
        timings_ms["total"] = (t1 - t_total) * 1000.0
        bytes_output = send_output.numel() * send_output.element_size()
        bytes_lse = send_lse.numel() * send_lse.element_size()
        profile_key = (world_size, B, H, D, return_lse, is_lse_base_on_e)
        _dcp_a2a_profile_record(
            cp_group,
            profile_key,
            timings_ms,
            bytes_output,
            bytes_lse,
        )
    return result
