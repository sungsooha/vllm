# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import CacheConfig
from vllm.distributed.latent_shard_utils import (
    all_gather_latent_cache,
    sharded_rms_norm,
)
from vllm.distributed.parallel_state import (
    get_latent_parallel_group,
    get_latent_parallel_rank,
    get_latent_parallel_world_size,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None
    kv_a_proj_latent: torch.nn.Module | None = None
    kv_a_proj_rope: torch.nn.Module | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.kv_a_proj_latent = mla_modules.kv_a_proj_latent
        self.kv_a_proj_rope = mla_modules.kv_a_proj_rope
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.kv_latent_parallel_group = get_latent_parallel_group()
        self.kv_latent_parallel_size = get_latent_parallel_world_size()
        self.kv_latent_parallel_rank = get_latent_parallel_rank()
        if self.kv_lora_rank % self.kv_latent_parallel_size != 0:
            raise ValueError(
                "kv_lora_rank must be divisible by kv_latent_parallel_size, got "
                f"kv_lora_rank={self.kv_lora_rank} and "
                f"kv_latent_parallel_size={self.kv_latent_parallel_size}."
            )
        self.kv_lora_rank_per_partition = (
            self.kv_lora_rank // self.kv_latent_parallel_size
        )

        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )

        self.prefix = prefix

    def _apply_kv_a_layernorm(self, kv_c: torch.Tensor) -> torch.Tensor:
        if self.kv_latent_parallel_size == 1:
            return self.kv_a_layernorm(kv_c)

        start = self.kv_latent_parallel_rank * self.kv_lora_rank_per_partition
        kv_c_local = kv_c.narrow(-1, start, self.kv_lora_rank_per_partition)
        gamma_local = self.kv_a_layernorm.weight.narrow(
            0, start, self.kv_lora_rank_per_partition
        )
        return sharded_rms_norm(
            kv_c_local,
            gamma_local,
            self.kv_a_layernorm.variance_epsilon,
            self.kv_latent_parallel_group,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None
        kv_c = None
        k_pe = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            if self.kv_latent_parallel_size > 1:
                assert self.kv_a_proj_latent is not None, (
                    "kv_a_proj_latent is required when kv_latent_parallel_size > 1"
                )
                assert self.kv_a_proj_rope is not None, (
                    "kv_a_proj_rope is required when kv_latent_parallel_size > 1"
                )
                kv_c = self.kv_a_proj_latent(hidden_states)[0]
                k_pe = self.kv_a_proj_rope(hidden_states)[0]
            else:
                kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        if kv_lora is not None:
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
        assert kv_c is not None
        assert k_pe is not None
        kv_c_normed = self._apply_kv_a_layernorm(kv_c)
        if self.kv_latent_parallel_size > 1:
            # Phase A P1: restore the full latent vector before the backend
            # updates cache or plans attention. Step 6 moves this gather to
            # the read side after the cache stores only local latent shards.
            kv_c_normed = all_gather_latent_cache(
                kv_c_normed, self.kv_latent_parallel_group
            )

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse:
            _topk_indices = self.indexer(
                hidden_states, q_c, positions, self.indexer_rope_emb
            )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )

        return self.o_proj(attn_out)[0]
