# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.parallel_state import GroupCoordinator


def sharded_rms_norm(
    x_local: torch.Tensor,
    gamma_local: torch.Tensor,
    eps: float,
    group: GroupCoordinator,
) -> torch.Tensor:
    """Apply RMSNorm to one KV-latent shard.

    ``x_local`` holds the local slice of a full latent vector along its last
    dimension. The normalization denominator is computed over the full latent
    vector by summing local squared norms and all-reducing those per-token
    scalars over the latent-parallel group. ``gamma_local`` is sharded the same
    way as ``x_local``.
    """
    if x_local.shape[-1] != gamma_local.numel():
        raise ValueError(
            "gamma_local must match the local latent dimension, got "
            f"x_local.shape[-1]={x_local.shape[-1]} and "
            f"gamma_local.numel()={gamma_local.numel()}."
        )

    orig_dtype = x_local.dtype
    x_float = x_local.to(torch.float32)
    local_sum_sq = (x_float * x_float).sum(dim=-1, keepdim=True)
    global_sum_sq = group.all_reduce(local_sum_sq)

    full_latent_dim = gamma_local.numel() * group.world_size
    inv_rms = torch.rsqrt(global_sum_sq / full_latent_dim + eps)

    output = (x_float * inv_rms).to(orig_dtype)
    return output * gamma_local.to(dtype=orig_dtype)


def all_gather_latent_cache(
    cache_local: torch.Tensor,
    group: GroupCoordinator,
) -> torch.Tensor:
    """All-gather a dense latent-sharded cache tensor along its last dimension."""
    return group.all_gather(cache_local, dim=-1)
