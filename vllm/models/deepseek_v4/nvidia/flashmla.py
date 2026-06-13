# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import (
    DeepseekV4Attention,
    _apply_attn_sink_with_lse,
    _get_dcp_padded_head_counts,
)
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_cp_local_seq_lens
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import (
    dcp_a2a_lse_reduce,
    dcp_a2a_packed_workspace_specs,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV4FlashMLAAttention(DeepseekV4Attention):
    """FlashMLA sparse MLA attention layer for DeepSeek V4 (CUDA)."""

    backend_cls = DeepseekV4FlashMLABackend

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()
        try:
            dcp_world_size = get_dcp_group().world_size
            dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            dcp_world_size = (
                self.vllm_config.parallel_config.decode_context_parallel_size
            )
            dcp_rank = 0
        self.dcp_world_size = dcp_world_size
        self.dcp_rank = dcp_rank
        try:
            pcp_world_size = get_pcp_group().world_size
            pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            pcp_world_size = 1
            pcp_rank = 0
        self.total_cp_world_size = pcp_world_size * dcp_world_size
        self.total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
        self.cp_kv_cache_interleave_size = (
            self.vllm_config.parallel_config.cp_kv_cache_interleave_size
        )
        self.dcp_combine = (
            dcp_a2a_lse_reduce
            if self.vllm_config.parallel_config.dcp_comm_backend == "a2a"
            else cp_lse_ag_out_rs
        )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # FP8 decode kernel only supports h_q = 64 or 128.
        if num_heads > 128:
            raise ValueError(
                f"DeepseekV4 FlashMLA does not support {num_heads} heads "
                "(FP8 decode kernel requires h_q in {64, 128})."
            )
        return 64 if num_heads <= 64 else 128

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )
        if q_dcp_replicated is not None:
            _, padded_global_heads = _get_dcp_padded_head_counts(
                self.n_local_heads,
                self.dcp_world_size,
            )
            assert q_dcp_replicated.shape == (
                q.shape[0],
                padded_global_heads,
                self.head_dim,
            ), (
                "replicated DCP Q shape "
                f"{tuple(q_dcp_replicated.shape)} does not match expected "
                f"{(q.shape[0], padded_global_heads, self.head_dim)}"
            )
            assert q_dcp_replicated.dtype == q.dtype, (
                "replicated DCP Q dtype "
                f"{q_dcp_replicated.dtype} must match local Q dtype {q.dtype}"
            )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            workspaces: list[tuple[tuple[int, ...], torch.dtype]] = [
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            ]
            if self.dcp_world_size > 1:
                _, padded_global_heads = _get_dcp_padded_head_counts(
                    self.n_local_heads,
                    self.dcp_world_size,
                )
                workspaces.append(
                    (
                        (
                            self.max_num_batched_tokens,
                            padded_global_heads,
                            q.shape[-1],
                        ),
                        q.dtype,
                    )
                )
                workspaces.extend(
                    dcp_a2a_packed_workspace_specs(
                        num_tokens=self.max_num_batched_tokens,
                        padded_global_heads=padded_global_heads,
                        head_dim=q.shape[-1],
                        dcp_world_size=self.dcp_world_size,
                        output_dtype=q.dtype,
                    )
                )
            current_workspace_manager().get_simultaneous(*workspaces)
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
                q_dcp_replicated=(
                    q_dcp_replicated[num_decode_tokens:]
                    if q_dcp_replicated is not None
                    else None
                ),
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
                q_dcp_replicated=(
                    q_dcp_replicated[:num_decode_tokens]
                    if q_dcp_replicated is not None
                    else None
                ),
            )

    def _prepare_dcp_query(
        self,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        padded_local_heads, _ = _get_dcp_padded_head_counts(
            self.n_local_heads,
            self.dcp_world_size,
        )
        q_local = q[:, : self.n_local_heads, :]
        if padded_local_heads > self.n_local_heads:
            q_local = F.pad(
                q_local,
                (0, 0, 0, padded_local_heads - self.n_local_heads),
            )
        q_across_dcp = get_dcp_group().all_gather(q_local.contiguous(), dim=1)
        return q_across_dcp, padded_local_heads

    @staticmethod
    def _normalize_flashmla_output(
        out: torch.Tensor,
        num_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        if out.ndim == 4 and out.shape[1] == 1:
            out = out.squeeze(1)
        elif out.ndim == 4 and out.shape[0] == 1:
            out = out.squeeze(0)
        assert out.shape == (
            num_tokens,
            num_heads,
            out.shape[-1],
        ), f"unexpected FlashMLA output shape {tuple(out.shape)}"
        return out

    @staticmethod
    def _normalize_flashmla_lse(
        lse: torch.Tensor,
        num_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        if lse.ndim == 3:
            if lse.shape == (num_tokens, num_heads, 1):
                lse = lse.squeeze(-1)
            elif lse.shape == (num_tokens, 1, num_heads):
                lse = lse.squeeze(1)
            elif lse.shape == (1, num_tokens, num_heads):
                lse = lse.squeeze(0)
            elif lse.shape == (num_heads, num_tokens, 1):
                lse = lse.squeeze(-1).transpose(0, 1)
        elif lse.ndim == 2 and lse.shape == (num_heads, num_tokens):
            lse = lse.transpose(0, 1)
        assert lse.shape == (
            num_tokens,
            num_heads,
        ), f"unexpected FlashMLA LSE shape {tuple(lse.shape)}"
        return lse.contiguous()

    def _dcp_lse_combine(
        self,
        partial_out: torch.Tensor,
        partial_lse: torch.Tensor,
        output: torch.Tensor,
        padded_local_heads: int,
        send_buffer: torch.Tensor | None = None,
        recv_buffer: torch.Tensor | None = None,
    ) -> None:
        if self.dcp_combine is dcp_a2a_lse_reduce:
            combined_out, combined_lse = self.dcp_combine(
                partial_out,
                partial_lse,
                get_dcp_group(),
                return_lse=True,
                send_buffer=send_buffer,
                recv_buffer=recv_buffer,
            )
        else:
            combined_out, combined_lse = self.dcp_combine(
                partial_out,
                partial_lse,
                get_dcp_group(),
                return_lse=True,
            )
        assert combined_out.shape == (
            output.shape[0],
            padded_local_heads,
            self.head_dim,
        ), f"unexpected DCP-combined output shape {tuple(combined_out.shape)}"
        assert combined_lse.shape == (
            output.shape[0],
            padded_local_heads,
        ), f"unexpected DCP-combined LSE shape {tuple(combined_lse.shape)}"
        combined_out = _apply_attn_sink_with_lse(
            combined_out,
            combined_lse,
            self.attn_sink[:padded_local_heads],
        )
        output[:, :padded_local_heads, :].copy_(combined_out)

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # We treat queries in the same seq as different queries and later we
        # only attend by generated indices. Under DCP, FlashMLA computes over
        # all DCP-replicated Q heads and the LSE combine scatters local heads.
        if self.dcp_world_size > 1:
            padded_local_heads, _ = _get_dcp_padded_head_counts(
                self.n_local_heads,
                self.dcp_world_size,
            )
            if q_dcp_replicated is not None:
                q_flash = q_dcp_replicated
            else:
                q_flash, padded_local_heads = self._prepare_dcp_query(q)
        else:
            q_flash = q
            padded_local_heads = self.padded_heads
        q_flash = q_flash.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        flash_output = output
        a2a_send_buffer = None
        a2a_recv_buffer = None
        if self.dcp_world_size > 1:
            decode_specs: list[tuple[tuple[int, ...], torch.dtype]] = [
                ((num_decode_tokens, q_flash.shape[2], self.head_dim), q.dtype),
            ]
            decode_specs.extend(
                dcp_a2a_packed_workspace_specs(
                    num_tokens=num_decode_tokens,
                    padded_global_heads=q_flash.shape[2],
                    head_dim=self.head_dim,
                    dcp_world_size=self.dcp_world_size,
                    output_dtype=q.dtype,
                )
            )
            decode_workspaces = current_workspace_manager().get_simultaneous(
                *decode_specs
            )
            flash_output = decode_workspaces[0]
            if len(decode_workspaces) >= 3:
                a2a_send_buffer = decode_workspaces[1]
                a2a_recv_buffer = decode_workspaces[2]

        out, lse = flash_mla_with_kvcache(
            q=q_flash,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=None if self.dcp_world_size > 1 else self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=flash_output.unsqueeze(1),
        )
        if self.dcp_world_size > 1:
            partial_out = self._normalize_flashmla_output(
                out,
                num_decode_tokens,
                q_flash.shape[2],
            )
            partial_lse = self._normalize_flashmla_lse(
                lse,
                num_decode_tokens,
                q_flash.shape[2],
            )
            self._dcp_lse_combine(
                partial_out,
                partial_lse,
                output,
                padded_local_heads,
                send_buffer=a2a_send_buffer,
                recv_buffer=a2a_recv_buffer,
            )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        swa_seq_lens = swa_metadata.prefill_swa_seq_lens
        swa_gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert swa_seq_lens is not None
        assert swa_gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            # Compressed region must fit the full compressed pool (seq_len //
            # compress_ratio), not just top_k. top_k bounds how many indices
            # the indexer selects, not the pool size it indexes into.
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        chunk_size_const = self.PREFILL_CHUNK_SIZE
        num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const

        workspace_manager = current_workspace_manager()
        dcp_padded_global_heads = 0
        max_chunk_tokens = 0
        if self.dcp_world_size > 1:
            _, dcp_padded_global_heads = _get_dcp_padded_head_counts(
                self.n_local_heads,
                self.dcp_world_size,
            )
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size_const
                chunk_end = min(chunk_start + chunk_size_const, num_prefills)
                query_start = (
                    query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
                ).item()
                query_end = (
                    query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
                ).item()
                max_chunk_tokens = max(max_chunk_tokens, query_end - query_start)

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            chunk_size = chunk_end - chunk_start
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            ).item()
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            ).item()

            prefill_workspaces: list[tuple[tuple[int, ...], torch.dtype]] = [
                ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
            ]
            a2a_send_buffer: torch.Tensor | None = None
            a2a_recv_buffer: torch.Tensor | None = None
            if self.dcp_world_size > 1:
                prefill_workspaces.append(
                    (
                        (
                            max_chunk_tokens,
                            dcp_padded_global_heads,
                            self.head_dim,
                        ),
                        q.dtype,
                    )
                )
                prefill_workspaces.extend(
                    dcp_a2a_packed_workspace_specs(
                        num_tokens=max_chunk_tokens,
                        padded_global_heads=dcp_padded_global_heads,
                        head_dim=self.head_dim,
                        dcp_world_size=self.dcp_world_size,
                        output_dtype=q.dtype,
                    )
                )
            workspaces = workspace_manager.get_simultaneous(*prefill_workspaces)
            kv = workspaces[0]
            if self.dcp_world_size > 1:
                flash_output_workspace = workspaces[1]
                if len(workspaces) >= 4:
                    a2a_send_buffer = workspaces[2]
                    a2a_recv_buffer = workspaces[3]
            else:
                flash_output_workspace = None

            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                compressed_seq_lens = get_cp_local_seq_lens(
                    seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    self.total_cp_world_size,
                    self.total_cp_rank,
                    self.cp_kv_cache_interleave_size,
                )
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=compressed_seq_lens,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=swa_seq_lens[chunk_start:chunk_end],
                gather_lens=swa_gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                swa_gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
                total_cp_world_size=self.total_cp_world_size,
                total_cp_rank=self.total_cp_rank,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            )

            q_chunk = q[query_start:query_end]
            if self.dcp_world_size > 1:
                assert flash_output_workspace is not None
                padded_local_heads, _ = _get_dcp_padded_head_counts(
                    self.n_local_heads,
                    self.dcp_world_size,
                )
                if q_dcp_replicated is not None:
                    q_chunk = q_dcp_replicated[query_start:query_end]
                else:
                    q_chunk, padded_local_heads = self._prepare_dcp_query(q_chunk)
                flash_output = flash_output_workspace[: q_chunk.shape[0]]
            else:
                padded_local_heads = self.padded_heads
                flash_output = output[query_start:query_end]

            output_chunk, lse, _ = flash_mla_sparse_fwd(
                q=q_chunk,
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=None if self.dcp_world_size > 1 else self.attn_sink,
                topk_length=combined_lens,
                out=flash_output,
            )
            if self.dcp_world_size > 1:
                partial_out = self._normalize_flashmla_output(
                    output_chunk,
                    q_chunk.shape[0],
                    q_chunk.shape[1],
                )
                partial_lse = self._normalize_flashmla_lse(
                    lse,
                    q_chunk.shape[0],
                    q_chunk.shape[1],
                )
                self._dcp_lse_combine(
                    partial_out,
                    partial_lse,
                    output[query_start:query_end],
                    padded_local_heads,
                    send_buffer=a2a_send_buffer,
                    recv_buffer=a2a_recv_buffer,
                )
