# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.distributed.latent_shard_utils import (
    all_gather_latent_cache,
    dsv4_paged_cache_to_latent_components,
    gather_top_k_dsv4_payload,
    gather_top_k_dsv4_payload_from_paged_cache,
    pack_dsv4_payload_to_paged_cache,
    sharded_rms_norm,
)


class _FakeLatentGroup:
    """Single-process stand-in for the latent-parallel group.

    The real distributed path calls all-reduce on per-token local squared sums
    and all-gather on local latent shards. These tests precompute the expected
    collective results and return them through the same helper APIs.
    """

    def __init__(
        self,
        *,
        world_size: int,
        rank_in_group: int = 0,
        all_reduce_result: torch.Tensor | None = None,
        all_gather_result: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> None:
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self._all_reduce_result = all_reduce_result
        if isinstance(all_gather_result, list):
            self._all_gather_results = list(all_gather_result)
        elif all_gather_result is None:
            self._all_gather_results = []
        else:
            self._all_gather_results = [all_gather_result]

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self._all_reduce_result is not None
        assert tensor.shape == self._all_reduce_result.shape
        return self._all_reduce_result.clone()

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert self._all_gather_results
        result = self._all_gather_results.pop(0)
        assert tensor.shape[:dim] == result.shape[:dim]
        return result.clone()


def _rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def _mock_mla_backend(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
) -> torch.Tensor:
    """Small deterministic backend used to verify backend-visible tensors.

    Step 5 should restore the full latent vector before the backend sees it.
    This mock consumes q, normalized latent KV, and RoPE KV so the test fails
    if any of the three tensors differs from the unsharded path.
    """

    q_term = q.float().sum(dim=-1, keepdim=True)
    kv_term = kv_c_normed.float().sum(dim=-1, keepdim=True)
    rope_term = k_pe.float().sum(dim=-1, keepdim=True)
    return q_term + kv_term + rope_term


def test_sharded_rms_norm_matches_full_rms_norm() -> None:
    torch.manual_seed(0)
    tokens = 7
    kv_lora_rank = 12
    latent_parallel_size = 2
    eps = 1e-6

    x = torch.randn(tokens, kv_lora_rank, dtype=torch.float32) * 0.25
    gamma = torch.randn(kv_lora_rank, dtype=torch.float32) * 0.1 + 1.0
    ref = _rms_norm_ref(x, gamma, eps)

    x_shards = x.chunk(latent_parallel_size, dim=-1)
    gamma_shards = gamma.chunk(latent_parallel_size, dim=-1)
    global_sum_sq = sum(
        shard.float().pow(2).sum(dim=-1, keepdim=True) for shard in x_shards
    )

    local_outputs = [
        sharded_rms_norm(
            shard,
            gamma_shard,
            eps,
            _FakeLatentGroup(
                world_size=latent_parallel_size,
                all_reduce_result=global_sum_sq,
            ),
        )
        for shard, gamma_shard in zip(x_shards, gamma_shards)
    ]
    gathered = torch.cat(local_outputs, dim=-1)

    torch.testing.assert_close(gathered, ref, atol=1e-6, rtol=1e-6)


def test_mla_latent_shard_step5_matches_unsharded_projection_path() -> None:
    torch.manual_seed(1)
    tokens = 5
    hidden_size = 16
    kv_lora_rank = 12
    qk_rope_head_dim = 4
    q_output_size = 20
    latent_parallel_size = 2
    eps = 1e-6

    hidden_states = torch.randn(tokens, hidden_size, dtype=torch.float32) * 0.2
    kv_a_proj_with_mqa_weight = (
        torch.randn(kv_lora_rank + qk_rope_head_dim, hidden_size) * 0.1
    )
    q_proj_weight = torch.randn(q_output_size, hidden_size) * 0.1
    gamma = torch.randn(kv_lora_rank, dtype=torch.float32) * 0.1 + 1.0

    # Baseline path: one merged ReplicatedLinear for latent KV and RoPE KV.
    kv_lora = hidden_states @ kv_a_proj_with_mqa_weight.t()
    kv_c, k_pe = kv_lora.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    q = hidden_states @ q_proj_weight.t()
    kv_c_normed = _rms_norm_ref(kv_c, gamma, eps)
    ref_backend_out = _mock_mla_backend(q, kv_c_normed, k_pe)

    # Step 4 path: split the checkpoint weight into latent and RoPE projections.
    kv_a_proj_latent_weight = kv_a_proj_with_mqa_weight[:kv_lora_rank, :]
    kv_a_proj_rope_weight = kv_a_proj_with_mqa_weight[kv_lora_rank:, :]
    kv_c_projected = hidden_states @ kv_a_proj_latent_weight.t()
    k_pe_projected = hidden_states @ kv_a_proj_rope_weight.t()

    torch.testing.assert_close(kv_c_projected, kv_c, atol=0, rtol=0)
    torch.testing.assert_close(k_pe_projected, k_pe, atol=0, rtol=0)

    # Step 4 sharded RMSNorm: each rank owns one latent slice.
    kv_c_shards = kv_c_projected.chunk(latent_parallel_size, dim=-1)
    gamma_shards = gamma.chunk(latent_parallel_size, dim=-1)
    global_sum_sq = sum(
        shard.float().pow(2).sum(dim=-1, keepdim=True) for shard in kv_c_shards
    )
    local_normed_shards = [
        sharded_rms_norm(
            shard,
            gamma_shard,
            eps,
            _FakeLatentGroup(
                world_size=latent_parallel_size,
                all_reduce_result=global_sum_sq,
            ),
        )
        for shard, gamma_shard in zip(kv_c_shards, gamma_shards)
    ]

    # Step 5 path: AllGather restores full latent KV before the backend call.
    expected_gather = torch.cat(local_normed_shards, dim=-1)
    kv_c_normed_gathered = all_gather_latent_cache(
        local_normed_shards[0],
        _FakeLatentGroup(
            world_size=latent_parallel_size,
            all_gather_result=expected_gather,
        ),
    )
    sharded_backend_out = _mock_mla_backend(q, kv_c_normed_gathered, k_pe_projected)

    torch.testing.assert_close(kv_c_normed_gathered, kv_c_normed, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        sharded_backend_out,
        ref_backend_out,
        atol=1e-6,
        rtol=1e-6,
    )


def test_gather_top_k_dsv4_payload_reconstructs_full_payload() -> None:
    num_tokens = 8
    batch = 2
    topk = 3
    world_size = 2

    full_nope = (
        torch.arange(num_tokens * 7 * 64, dtype=torch.int64).reshape(num_tokens, 7, 64)
        % 251
    ).to(torch.uint8)
    full_scales = (
        torch.arange(num_tokens * 7, dtype=torch.int64).reshape(num_tokens, 7) + 17
    ).to(torch.uint8)
    rope = (
        torch.arange(num_tokens * 128, dtype=torch.int64).reshape(num_tokens, 128) + 31
    ).to(torch.uint8)
    topk_indices = torch.tensor([[7, 0, 3], [2, 5, 1]], dtype=torch.long)

    rank0_nope = full_nope[:, :4, :]
    rank0_scales = full_scales[:, :4]
    rank1_nope = full_nope[:, 4:, :]
    rank1_scales = full_scales[:, 4:]

    selected_rank0_nope = rank0_nope[topk_indices]
    selected_rank0_scales = rank0_scales[topk_indices]
    selected_rank1_nope = rank1_nope[topk_indices]
    selected_rank1_scales = rank1_scales[topk_indices]

    padded_rank1_nope = torch.zeros(
        *selected_rank1_nope.shape[:-2], 4, 64, dtype=torch.uint8
    )
    padded_rank1_nope[..., :3, :] = selected_rank1_nope
    gathered_nope = torch.cat([selected_rank0_nope, padded_rank1_nope], dim=-2)

    padded_rank1_scales = torch.zeros(
        *selected_rank1_scales.shape[:-1], 4, dtype=torch.uint8
    )
    padded_rank1_scales[..., :3] = selected_rank1_scales
    gathered_scales = torch.cat([selected_rank0_scales, padded_rank1_scales], dim=-1)

    payload = gather_top_k_dsv4_payload(
        rank0_nope,
        rank0_scales,
        rope,
        topk_indices,
        _FakeLatentGroup(
            world_size=world_size,
            rank_in_group=0,
            all_gather_result=[gathered_nope, gathered_scales],
        ),
    )

    expected = torch.empty(batch, topk, 584, dtype=torch.uint8)
    expected[..., :448] = full_nope[topk_indices].reshape(batch, topk, 448)
    expected[..., 448:576] = rope[topk_indices]
    expected[..., 576:583] = full_scales[topk_indices]
    expected[..., 583] = 0

    torch.testing.assert_close(payload, expected, atol=0, rtol=0)


def test_dsv4_payload_pack_roundtrips_page_tail_scale_layout() -> None:
    batch = 2
    topk = 5
    block_size = 4
    compact = (
        torch.arange(batch * topk * 584, dtype=torch.int64).reshape(batch, topk, 584)
        % 251
    ).to(torch.uint8)

    paged_cache, dense_indices = pack_dsv4_payload_to_paged_cache(
        compact,
        block_size,
    )
    assert paged_cache.shape == (3, block_size, 584)
    torch.testing.assert_close(
        dense_indices,
        torch.arange(batch * topk, dtype=torch.int32).reshape(batch, topk),
    )

    token_data, scales, rope = dsv4_paged_cache_to_latent_components(
        paged_cache,
        _FakeLatentGroup(world_size=1, rank_in_group=0),
    )
    topk_indices = dense_indices.to(torch.long)
    gathered = gather_top_k_dsv4_payload(
        token_data,
        scales,
        rope,
        topk_indices,
        _FakeLatentGroup(
            world_size=1,
            rank_in_group=0,
            all_gather_result=[
                token_data[topk_indices],
                scales[topk_indices],
            ],
        ),
    )
    torch.testing.assert_close(gathered, compact, atol=0, rtol=0)


def test_gather_top_k_dsv4_payload_from_paged_cache_handles_uneven_shards() -> None:
    num_tokens = 9
    block_size = 4
    world_size = 2
    topk_indices = torch.tensor([[8, 3, -1], [2, 6, 0]], dtype=torch.long)

    compact = (
        torch.arange(num_tokens * 584, dtype=torch.int64).reshape(num_tokens, 584) % 251
    ).to(torch.uint8)
    paged_cache, _ = pack_dsv4_payload_to_paged_cache(compact, block_size)

    rank0_nope, rank0_scales, _ = dsv4_paged_cache_to_latent_components(
        paged_cache,
        _FakeLatentGroup(world_size=world_size, rank_in_group=0),
    )
    rank1_nope, rank1_scales, _ = dsv4_paged_cache_to_latent_components(
        paged_cache,
        _FakeLatentGroup(world_size=world_size, rank_in_group=1),
    )

    safe_indices = topk_indices.clamp_min(0)
    selected_rank0_nope = rank0_nope[safe_indices]
    selected_rank0_scales = rank0_scales[safe_indices]
    selected_rank1_nope = rank1_nope[safe_indices]
    selected_rank1_scales = rank1_scales[safe_indices]

    padded_rank1_nope = torch.zeros(
        *selected_rank1_nope.shape[:-2], 4, 64, dtype=torch.uint8
    )
    padded_rank1_nope[..., :3, :] = selected_rank1_nope
    gathered_nope = torch.cat([selected_rank0_nope, padded_rank1_nope], dim=-2)

    padded_rank1_scales = torch.zeros(
        *selected_rank1_scales.shape[:-1], 4, dtype=torch.uint8
    )
    padded_rank1_scales[..., :3] = selected_rank1_scales
    gathered_scales = torch.cat([selected_rank0_scales, padded_rank1_scales], dim=-1)

    payload = gather_top_k_dsv4_payload_from_paged_cache(
        paged_cache,
        topk_indices,
        _FakeLatentGroup(
            world_size=world_size,
            rank_in_group=0,
            all_gather_result=[gathered_nope, gathered_scales],
        ),
    )

    expected = compact[safe_indices]
    torch.testing.assert_close(payload, expected, atol=0, rtol=0)
