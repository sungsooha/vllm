# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.distributed.latent_shard_utils import (
    all_gather_latent_cache,
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
        all_reduce_result: torch.Tensor | None = None,
        all_gather_result: torch.Tensor | None = None,
    ) -> None:
        self.world_size = world_size
        self._all_reduce_result = all_reduce_result
        self._all_gather_result = all_gather_result

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self._all_reduce_result is not None
        assert tensor.shape == self._all_reduce_result.shape
        return self._all_reduce_result.clone()

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1
        assert self._all_gather_result is not None
        assert tensor.shape[:-1] == self._all_gather_result.shape[:-1]
        return self._all_gather_result.clone()


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
