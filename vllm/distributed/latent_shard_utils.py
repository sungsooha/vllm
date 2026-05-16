# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.parallel_state import GroupCoordinator

_DSV4_NOPE_BLOCKS = 7
_DSV4_NOPE_BLOCK_BYTES = 64
_DSV4_ROPE_BYTES = 128
_DSV4_SCALE_BYTES = 8
_DSV4_TOKEN_DATA_BYTES = _DSV4_NOPE_BLOCKS * _DSV4_NOPE_BLOCK_BYTES + _DSV4_ROPE_BYTES
_DSV4_PAYLOAD_BYTES = _DSV4_TOKEN_DATA_BYTES + _DSV4_SCALE_BYTES


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


def _dsv4_nope_block_partition(
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    """Return the half-open NoPE quant-block range owned by one latent rank."""
    if world_size < 1 or world_size > _DSV4_NOPE_BLOCKS:
        raise ValueError(
            "DSv4 NoPE block sharding requires 1 <= world_size <= "
            f"{_DSV4_NOPE_BLOCKS}, got {world_size}."
        )
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid latent rank {rank} for world_size={world_size}.")

    blocks_per_rank, remainder = divmod(_DSV4_NOPE_BLOCKS, world_size)
    start = rank * blocks_per_rank + min(rank, remainder)
    count = blocks_per_rank + (1 if rank < remainder else 0)
    return start, start + count


def gather_top_k_dsv4_payload(
    cache_local_nope_blocks: torch.Tensor,
    cache_local_scales: torch.Tensor,
    cache_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    group: GroupCoordinator,
) -> torch.Tensor:
    """Gather DSv4 selected-token latent shards into compact 584B payloads.

    DSv4's ``fp8_ds_mla`` payload is logically:

    * 448 NoPE FP8 bytes = seven 64-byte quant blocks;
    * 128 RoPE BF16 bytes, kept replicated for the Phase B v1 prototype;
    * eight scale bytes = seven real UE8M0 scale bytes plus one pad byte.

    Each latent rank owns a whole-number range of NoPE quant blocks and their
    matching scale bytes. This helper gathers the selected local blocks across
    the latent-parallel group and returns compact per-token payloads with shape
    ``topk_indices.shape + (584,)``. The returned payload is convenient for
    tests and temporary selected-token assembly. FlashMLA's paged cache stores
    scale bytes in a page-tail region, so integration code must still page-pack
    this compact representation before calling the real kernel.
    """
    if cache_local_nope_blocks.dtype is not torch.uint8:
        raise ValueError("cache_local_nope_blocks must be torch.uint8.")
    if cache_local_scales.dtype is not torch.uint8:
        raise ValueError("cache_local_scales must be torch.uint8.")
    if cache_rope.dtype is not torch.uint8:
        raise ValueError("cache_rope must be torch.uint8.")
    if cache_local_nope_blocks.dim() != 3:
        raise ValueError(
            "cache_local_nope_blocks must have shape [num_tokens, local_blocks, 64]."
        )
    if cache_local_scales.dim() != 2:
        raise ValueError(
            "cache_local_scales must have shape [num_tokens, local_blocks]."
        )
    if cache_rope.dim() != 2:
        raise ValueError("cache_rope must have shape [num_tokens, 128].")
    if cache_local_nope_blocks.shape[0] != cache_local_scales.shape[0]:
        raise ValueError("NoPE block cache and scale cache token counts differ.")
    if cache_local_nope_blocks.shape[0] != cache_rope.shape[0]:
        raise ValueError("NoPE block cache and RoPE cache token counts differ.")
    if cache_local_nope_blocks.shape[-1] != _DSV4_NOPE_BLOCK_BYTES:
        raise ValueError(
            f"The last NoPE block dimension must be {_DSV4_NOPE_BLOCK_BYTES} bytes."
        )
    if cache_rope.shape[-1] != _DSV4_ROPE_BYTES:
        raise ValueError(f"RoPE cache must have {_DSV4_ROPE_BYTES} bytes/token.")

    world_size = group.world_size
    rank = group.rank_in_group
    start_block, end_block = _dsv4_nope_block_partition(rank, world_size)
    local_blocks = end_block - start_block
    if cache_local_nope_blocks.shape[-2] != local_blocks:
        raise ValueError(
            "cache_local_nope_blocks local block count does not match the "
            "rank's DSv4 NoPE block partition: expected "
            f"{local_blocks}, got {cache_local_nope_blocks.shape[-2]}."
        )
    if cache_local_scales.shape[-1] != local_blocks:
        raise ValueError(
            "cache_local_scales local block count does not match the rank's "
            f"partition: expected {local_blocks}, got "
            f"{cache_local_scales.shape[-1]}."
        )

    topk_indices = topk_indices.to(
        device=cache_local_nope_blocks.device,
        dtype=torch.long,
    )
    selected_nope = cache_local_nope_blocks[topk_indices]
    selected_scales = cache_local_scales[topk_indices]
    selected_rope = cache_rope[topk_indices]

    max_blocks_per_rank = (_DSV4_NOPE_BLOCKS + world_size - 1) // world_size
    if local_blocks < max_blocks_per_rank:
        padded_shape = (
            *selected_nope.shape[:-2],
            max_blocks_per_rank,
            _DSV4_NOPE_BLOCK_BYTES,
        )
        padded_nope = selected_nope.new_zeros(padded_shape)
        padded_nope[..., :local_blocks, :] = selected_nope

        padded_scale_shape = (*selected_scales.shape[:-1], max_blocks_per_rank)
        padded_scales = selected_scales.new_zeros(padded_scale_shape)
        padded_scales[..., :local_blocks] = selected_scales
    else:
        padded_nope = selected_nope
        padded_scales = selected_scales

    gathered_nope = group.all_gather(padded_nope, dim=-2)
    gathered_scales = group.all_gather(padded_scales, dim=-1)

    nope_pieces = []
    scale_pieces = []
    for source_rank in range(world_size):
        source_start, source_end = _dsv4_nope_block_partition(source_rank, world_size)
        source_blocks = source_end - source_start
        gather_start = source_rank * max_blocks_per_rank
        gather_end = gather_start + source_blocks
        nope_pieces.append(gathered_nope[..., gather_start:gather_end, :])
        scale_pieces.append(gathered_scales[..., gather_start:gather_end])

    full_nope = torch.cat(nope_pieces, dim=-2)
    full_scales = torch.cat(scale_pieces, dim=-1)
    if full_nope.shape[-2] != _DSV4_NOPE_BLOCKS:
        raise AssertionError("Internal DSv4 NoPE gather produced wrong block count.")
    if full_scales.shape[-1] != _DSV4_NOPE_BLOCKS:
        raise AssertionError("Internal DSv4 scale gather produced wrong block count.")

    output_shape = (*topk_indices.shape, _DSV4_PAYLOAD_BYTES)
    payload = torch.empty(output_shape, dtype=torch.uint8, device=cache_rope.device)
    payload[..., : _DSV4_NOPE_BLOCKS * _DSV4_NOPE_BLOCK_BYTES] = full_nope.reshape(
        *topk_indices.shape, _DSV4_NOPE_BLOCKS * _DSV4_NOPE_BLOCK_BYTES
    )
    payload[
        ..., _DSV4_NOPE_BLOCKS * _DSV4_NOPE_BLOCK_BYTES : _DSV4_TOKEN_DATA_BYTES
    ] = selected_rope
    payload[
        ..., _DSV4_TOKEN_DATA_BYTES : _DSV4_TOKEN_DATA_BYTES + _DSV4_NOPE_BLOCKS
    ] = full_scales
    payload[..., _DSV4_TOKEN_DATA_BYTES + _DSV4_NOPE_BLOCKS] = 0
    return payload
