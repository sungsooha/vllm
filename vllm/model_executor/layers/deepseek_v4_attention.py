# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm import envs
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.ops.deepseek_v4_ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    fused_indexer_q_rope_quant,
    fused_inv_rope_fp8_quant,
    fused_q_kv_rmsnorm,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.deepseek_compressor import DeepseekCompressor
from vllm.model_executor.layers.deepseek_v4_debug import (
    dsv4_debug_dump,
    dsv4_debug_layer_idx,
    dsv4_debug_should_dump,
)
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import (
    QuantFP8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.compressor_utils import get_cp_local_seq_lens
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

# Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
# workspace allocated at _forward_prefill (and the matching profile-time
# reservation in attention_impl's dummy-run branch).
PREFILL_CHUNK_SIZE = 4


@dataclass
class _DSV4DCPProfileStats:
    calls: int = 0
    skipped_warmup: int = 0
    totals_ms: dict[str, float] = field(default_factory=dict)

    def add(self, timings_ms: dict[str, float]) -> None:
        self.calls += 1
        for key, value in timings_ms.items():
            self.totals_ms[key] = self.totals_ms.get(key, 0.0) + value

    def avg(self, key: str) -> float:
        if self.calls == 0:
            return 0.0
        return self.totals_ms.get(key, 0.0) / self.calls


_DSV4_DCP_LAYER_PROFILE_STATS: dict[
    tuple[int, str, int, int, int, int], _DSV4DCPProfileStats
] = {}


def _dsv4_dcp_layer_profile_enabled() -> bool:
    return bool(envs.VLLM_DSV4_DCP_LAYER_PROFILE)


def _dsv4_dcp_layer_profile_should_log_rank(dcp_rank: int) -> bool:
    if envs.VLLM_DSV4_DCP_LAYER_PROFILE_ALL_RANKS:
        return True
    return dcp_rank == 0


def _dsv4_dcp_layer_profile_mark(sync: bool) -> float:
    if sync:
        torch.accelerator.synchronize()
    return time.perf_counter()


def _dsv4_dcp_layer_profile_record(
    *,
    dcp_rank: int,
    layer_idx: int,
    mode: str,
    compress_ratio: int,
    B: int,
    H: int,
    D: int,
    timings_ms: dict[str, float],
) -> None:
    key = (layer_idx, mode, compress_ratio, B, H, D)
    stats = _DSV4_DCP_LAYER_PROFILE_STATS.setdefault(key, _DSV4DCPProfileStats())
    warmup = max(envs.VLLM_DSV4_DCP_LAYER_PROFILE_WARMUP, 0)
    if stats.skipped_warmup < warmup:
        stats.skipped_warmup += 1
        return

    stats.add(timings_ms)
    interval = max(envs.VLLM_DSV4_DCP_LAYER_PROFILE_INTERVAL, 1)
    if stats.calls % interval != 0 or not _dsv4_dcp_layer_profile_should_log_rank(
        dcp_rank
    ):
        return

    logger.info(
        "DSV4_DCP_LAYER_PROFILE rank=%d layer=%d mode=%s "
        "compress_ratio=%d shape=B%d_H%d_D%d calls=%d warmup=%d sync=%s "
        "avg_ms(q_gather=%.3f,flash=%.3f,normalize=%.3f,"
        "a2a_reduce=%.3f,sink_scale=%.3f,output_copy=%.3f,"
        "sink_copy=%.3f,total=%.3f)",
        dcp_rank,
        layer_idx,
        mode,
        compress_ratio,
        B,
        H,
        D,
        stats.calls,
        stats.skipped_warmup,
        envs.VLLM_DSV4_DCP_LAYER_PROFILE_SYNC,
        stats.avg("q_gather"),
        stats.avg("flash"),
        stats.avg("normalize"),
        stats.avg("a2a_reduce"),
        stats.avg("sink_scale"),
        stats.avg("output_copy"),
        stats.avg("sink_scale") + stats.avg("output_copy"),
        stats.avg("total"),
    )


def _get_dcp_padded_head_counts(
    local_heads: int,
    dcp_world_size: int,
) -> tuple[int, int]:
    global_heads = local_heads * dcp_world_size
    for padded_global_heads in DeepseekV4MLAAttention.SUPPORTED_HEAD_COUNTS:
        if (
            global_heads <= padded_global_heads
            and padded_global_heads % dcp_world_size == 0
        ):
            padded_local_heads = padded_global_heads // dcp_world_size
            if local_heads <= padded_local_heads:
                return padded_local_heads, padded_global_heads
    raise ValueError(
        "DeepseekV4 DCP requires gathered attention heads to fit a FlashMLA "
        f"supported head count {DeepseekV4MLAAttention.SUPPORTED_HEAD_COUNTS}, "
        f"got local_heads={local_heads}, dcp_world_size={dcp_world_size}."
    )


def _apply_attn_sink_with_lse(
    output: torch.Tensor,
    global_lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    sink = attn_sink[: global_lse.shape[1]].to(global_lse.dtype)
    sink_scale = torch.sigmoid(global_lse - sink.unsqueeze(0))
    return output * sink_scale.unsqueeze(-1).to(output.dtype)


def _dsv4_debug_sample_rows(num_rows: int) -> list[int]:
    if num_rows <= 0:
        return []
    max_rows = int(os.environ.get("VLLM_DSV4_DEBUG_TOKEN_ROWS", "16"))
    rows = list(range(min(max_rows, num_rows)))
    last_idx = num_rows - 1
    if last_idx not in rows:
        rows.append(last_idx)
    return rows


def _dsv4_debug_prefill_kv_probe(
    *,
    layer_idx: int | None,
    kv: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    extra: dict[str, object],
) -> None:
    stage = "attn.prefill.kv_probe"
    if not dsv4_debug_should_dump(stage, layer_idx):
        return

    rows = _dsv4_debug_sample_rows(combined_indices.shape[0])
    if not rows:
        return

    device = combined_indices.device
    row_tensor = torch.tensor(rows, dtype=torch.int32, device=device)
    row_lens = combined_lens[row_tensor.long()].to(torch.int64)
    num_cols = int(os.environ.get("VLLM_DSV4_DEBUG_KV_PROBE_COLS", "16"))
    dim = min(kv.shape[-1], int(os.environ.get("VLLM_DSV4_DEBUG_KV_PROBE_DIMS", "32")))
    col_offsets = torch.arange(num_cols, dtype=torch.int64, device=device)

    prefix_cols = col_offsets.unsqueeze(0).expand(len(rows), -1)
    prefix_valid = prefix_cols < row_lens.unsqueeze(1)
    prefix_indices = combined_indices[row_tensor.long()].gather(
        1, prefix_cols.clamp(max=combined_indices.shape[1] - 1).to(torch.long)
    )
    prefix_indices = torch.where(
        prefix_valid, prefix_indices, torch.full_like(prefix_indices, -1)
    )

    suffix_start = torch.clamp(row_lens - num_cols, min=0)
    suffix_cols = suffix_start.unsqueeze(1) + col_offsets.unsqueeze(0)
    suffix_valid = suffix_cols < row_lens.unsqueeze(1)
    suffix_indices = combined_indices[row_tensor.long()].gather(
        1, suffix_cols.clamp(max=combined_indices.shape[1] - 1).to(torch.long)
    )
    suffix_indices = torch.where(
        suffix_valid, suffix_indices, torch.full_like(suffix_indices, -1)
    )

    kv_flat = kv.reshape(-1, kv.shape[-1])

    def gather_prefix(indices: torch.Tensor) -> torch.Tensor:
        safe_indices = indices.clamp(min=0, max=kv_flat.shape[0] - 1).to(torch.long)
        gathered = kv_flat.index_select(0, safe_indices.reshape(-1))
        gathered = gathered.reshape(*indices.shape, kv.shape[-1])[..., :dim]
        valid = (indices >= 0).unsqueeze(-1)
        return torch.where(valid, gathered, torch.zeros_like(gathered)).contiguous()

    dsv4_debug_dump(
        stage,
        layer_idx=layer_idx,
        tensors={
            "probe_rows": row_tensor,
            "probe_lens": row_lens.to(torch.int32),
            "prefix_indices": prefix_indices,
            "prefix_kv_prefix": gather_prefix(prefix_indices),
            "suffix_indices": suffix_indices,
            "suffix_kv_prefix": gather_prefix(suffix_indices),
        },
        extra={
            **extra,
            "probe_cols": num_cols,
            "probe_dims": dim,
            "kv_shape": list(kv.shape),
            "combined_indices_shape": list(combined_indices.shape),
        },
    )


@dataclass
class DeepseekV4MLAModules:
    """Modules used in DeepseekV4 MLA."""

    vllm_config: VllmConfig
    fused_wqa_wkv: torch.nn.Module
    q_norm: torch.nn.Module
    wq_b: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    rotary_emb: torch.nn.Module
    indexer: torch.nn.Module | None
    indexer_rotary_emb: torch.nn.Module
    topk_indices_buffer: torch.Tensor | None
    aux_stream: torch.cuda.Stream | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("deepseek_v4_multi_head_latent_attention")
class DeepseekV4MultiHeadLatentAttentionWrapper(PluggableLayer):
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
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        o_lora_rank: int | None,
        mla_modules: DeepseekV4MLAModules,
        window_size: int,
        compress_ratio: int | None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

        # FlashMLA sparse kernel only supports 64 or 128 heads; pad up to the
        # next supported size. Must match DeepseekV4MLAAttention.padded_heads.
        if num_heads <= 64:
            self.padded_heads = 64
        elif num_heads <= 128:
            self.padded_heads = 128
        else:
            raise ValueError(
                f"DeepseekV4 attention does not support {num_heads} heads "
                "(must be <= 128)."
            )

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.window_size = window_size
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        self.prefix = prefix

        # Extract config from vllm_config
        config = mla_modules.vllm_config.model_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()

        # DeepseekV4-specific attributes (num_heads is already TP-adjusted)
        self.eps = config.rms_norm_eps
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - self.rope_head_dim
        self.n_local_groups = config.o_groups // tp_size
        self.o_lora_rank = config.o_lora_rank

        # Store projection modules
        self.fused_wqa_wkv = mla_modules.fused_wqa_wkv
        self.q_norm = mla_modules.q_norm
        self.wq_b = mla_modules.wq_b

        self.kv_norm = mla_modules.kv_norm
        self.wo_a = mla_modules.wo_a

        self._wo_a_act_quant = QuantFP8(
            static=False,
            group_shape=GroupShape(1, 128),
            use_ue8m0=True,
        )
        # Bypass packed-for-deepgemm path — we need FP32 scales (not packed
        # INT32) so fp8_einsum can handle layout transform internally.
        self._wo_a_act_quant.use_deep_gemm_supported = False
        self.wo_b = mla_modules.wo_b

        # Pick fp8_einsum recipe based on GPU arch:
        # SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128
        # SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1
        from vllm.platforms import current_platform

        cap = current_platform.get_device_capability()
        self._einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
        self._tma_aligned_scales = cap.major >= 10

        self.rotary_emb = mla_modules.rotary_emb
        self.indexer_rotary_emb = mla_modules.indexer_rotary_emb
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.indexer = mla_modules.indexer

        # Per-head RMS normalization for Q (no learnable weights)
        self.q_head_norm = RMSNorm(head_dim, eps=self.eps, has_weight=False)

        # TODO(yifan): currently hardcoded for FP8 sparse, make it more generic
        head_bytes = (
            self.nope_head_dim  # 448 fp8 NoPE
            + self.rope_head_dim * 2  # 64 bf16 RoPE
            + self.nope_head_dim // 64  # 7B scale factors
            + 1  # 1B pad
        )

        self.aux_stream = mla_modules.aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=torch.uint8,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        self.mla_attn = DeepseekV4MLAAttention(
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            compress_ratio=self.compress_ratio,
            window_size=self.window_size,
            head_bytes=head_bytes,
            swa_cache_layer=self.swa_cache_layer,
            attn_sink=mla_modules.attn_sink,  # already padded with -inf
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            indexer=self.indexer,
            topk_indices_buffer=self.topk_indices_buffer,
        )
        # Register this layer in the compilation config's static forward context
        # This allows the custom op to retrieve the layer during execution
        compilation_config = mla_modules.vllm_config.compilation_config
        # HACK
        self.layer_name = prefix + ".deepseek_v4_multi_head_latent_attention"
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

        # Create the compressor for layers with compress_ratio > 1; after
        # creating the DeepseekV4MLAAttention layer to get its cache.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=mla_modules.vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.mla_attn.prefix,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)

        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Attention (inside custom op for torch.compile boundary)
        torch.ops.vllm.deepseek_v4_attention(
            hidden_states,
            qr,
            kv,
            positions,
            o_padded,
            self.layer_name,
        )
        o = o_padded[:, : self.n_local_heads, :]
        dsv4_debug_dump(
            "attn.wrapper.o_local",
            layer_idx=dsv4_debug_layer_idx(self.prefix),
            tensors={
                "o": o,
                "o_padded": o_padded,
            },
            extra={
                "prefix": self.prefix,
                "n_local_heads": self.n_local_heads,
                "padded_heads": self.padded_heads,
            },
        )

        # O projection: inverse RoPE + FP8 quant + einsum + wo_b
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            tma_aligned_scales=self._tma_aligned_scales,
        )

        wo_a_fp8 = self.wo_a.weight
        wo_a_scale = self.wo_a.weight_scale_inv

        z = torch.empty(
            (num_tokens, self.n_local_groups, self.o_lora_rank),
            device=o.device,
            dtype=torch.bfloat16,
        )
        torch.ops.vllm.deepseek_v4_fp8_einsum(
            o_fp8,
            o_scale,
            wo_a_fp8,
            wo_a_scale,
            z,
            "bhr,hdr->bhd",
            list(self._einsum_recipe),
        )

        return self.wo_b(z.flatten(1))

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)

        # Overlap kv_insert with whichever of indexer/compressor is present.
        # Indexer implies compressor; when both exist, compressor rides on the
        # aux stream alongside kv_insert so the heavy indexer owns default.
        if self.indexer is not None:
            # Local ref so the closure keeps a non-None type for mypy.
            assert self.compressor is not None
            compressor = self.compressor

            def kv_insert_and_compress() -> None:
                self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                compressor(hidden_states, positions, self.rotary_emb)

            maybe_execute_in_parallel(
                lambda: self.indexer(
                    hidden_states, qr, positions, self.indexer_rotary_emb
                ),
                kv_insert_and_compress,
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        elif self.compressor is not None:
            # Compressor on default, kv_insert on aux.
            maybe_execute_in_parallel(
                lambda: self.compressor(hidden_states, positions, self.rotary_emb),
                lambda: self._fused_qnorm_rope_kv_insert(
                    q, kv, positions, attn_metadata
                ),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        else:
            # SWA-only layer: no compressor, no overlap.
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        # Handle dummy run (no metadata).
        if not isinstance(attn_metadata, dict):
            # Reserve _forward_prefill's bf16-gather workspace; the dummy
            # run returns before mla_attn runs, so without this the shared
            # workspace locks below the real prefill size.
            sub = self.mla_attn
            swa_only = sub.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (sub.max_model_len + sub.compress_ratio - 1) // sub.compress_ratio
            )
            M = N + sub.window_size + sub.max_num_batched_tokens
            prefill_workspaces = [
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16)
            ]
            if sub.dcp_world_size > 1:
                _, padded_global_heads = _get_dcp_padded_head_counts(
                    sub.num_heads,
                    sub.dcp_world_size,
                )
                prefill_workspaces.append(
                    (
                        (
                            sub.max_num_batched_tokens,
                            padded_global_heads,
                            q.shape[-1],
                        ),
                        q.dtype,
                    )
                )
            current_workspace_manager().get_simultaneous(*prefill_workspaces)
            out.zero_()
            return

        # Pad q to FlashMLA-required head count (64 or 128)
        if self.n_local_heads < self.padded_heads:
            pad_size = self.padded_heads - self.n_local_heads
            q = F.pad(q, (0, 0, 0, pad_size), value=0.0)

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.mla_attn(q, kv, positions, output=out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
    ) -> None:
        if not isinstance(attn_metadata, dict):
            return

        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)

        # Horizontally fused:
        #   Q side:  q_head_norm (per-head RMSNorm, no weight) + GPT-J RoPE
        #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert
        # kv is unchanged; mla_attn reads kv solely via swa_kv_cache.
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )


def deepseek_v4_attention(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.attention_impl(hidden_states, qr, kv, positions, out)


def deepseek_v4_attention_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_attention",
    op_func=deepseek_v4_attention,
    mutates_args=["out"],
    fake_impl=deepseek_v4_attention_fake,
)


def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))


def deepseek_v4_fp8_einsum_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_fp8_einsum",
    op_func=deepseek_v4_fp8_einsum,
    mutates_args=["out"],
    fake_impl=deepseek_v4_fp8_einsum_fake,
)


class DeepseekV4MLAAttention(nn.Module, AttentionLayerBase):
    # FlashMLA FP8 sparse only supports 64 or 128 heads
    SUPPORTED_HEAD_COUNTS = (64, 128)

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        compress_ratio: int,
        window_size: int,
        head_bytes: int,
        swa_cache_layer: DeepseekV4SWACache,
        attn_sink: torch.Tensor,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        # Sparse MLA Args
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream: torch.cuda.Stream | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.head_dim = head_dim
        self.scale = scale
        self.window_size = window_size
        self.head_bytes = head_bytes
        self.compress_ratio = compress_ratio
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.nope_head_dim = qk_nope_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.indexer = indexer
        self.topk_indices_buffer = topk_indices_buffer

        self.prefix = prefix  # Alias for compatibility with compressor
        self.debug_layer_idx = dsv4_debug_layer_idx(prefix)

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        # Determine padded head count for FlashMLA
        if num_heads not in self.SUPPORTED_HEAD_COUNTS:
            if num_heads < 64:
                self.padded_heads = 64
            elif num_heads < 128:
                self.padded_heads = 128
            else:
                raise ValueError(
                    f"DeepseekV4MLAAttention does not support {num_heads} heads. "
                    f"Supported: <= 128 (will be padded to 64 or 128)"
                )
        else:
            self.padded_heads = num_heads

        # Store attention sink
        assert attn_sink is not None
        self.attn_sink: torch.Tensor = attn_sink
        # Store SWA cache
        assert swa_cache_layer is not None
        self.swa_cache_layer: DeepseekV4SWACache = swa_cache_layer

        # Get vllm config for cache setup
        vllm_config = get_current_vllm_config()
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        try:
            dcp_world_size = get_dcp_group().world_size
            dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            dcp_world_size = 1
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
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )
        self.dcp_combine = (
            dcp_a2a_lse_reduce
            if vllm_config.parallel_config.dcp_comm_backend == "a2a"
            else cp_lse_ag_out_rs
        )
        # DeepseekV4 only supports fp8 kv-cache format for now
        kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "fp8"

        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 only supports fp8 kv-cache format for now, "
            f"got {kv_cache_dtype}"
        )
        assert issubclass(self.get_attn_backend(), FlashMLASparseBackend), (
            "Only FlashMLA Sparse Attention backend is supported for DeepseekV4 for now"
        )
        # FlashMLA Sparse Attention fp8 backend uses "fp8_ds_mla" kv-cache format
        # Automatically convert fp8 kv-cache format to "fp8_ds_mla"
        if (
            issubclass(self.get_attn_backend(), FlashMLASparseBackend)
            and kv_cache_dtype.startswith("fp8")
            and kv_cache_dtype != "fp8_ds_mla"
        ):
            assert cache_config is not None
            cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once(
                "Using DeepSeek's fp8_ds_mla KV cache format. To use standard "
                "fp8 kv-cache format, please set `--attention-backend "
                "FLASHINFER_MLA_SPARSE`"
            )

        self.kv_cache_dtype = kv_cache_dtype

        # Register with compilation context for metadata lookup
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self

        self.kv_cache = torch.tensor([])

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4FlashMLASparseBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
            model_version="deepseek_v4",
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        flashmla_metadata = attn_metadata.get(self.prefix)
        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
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
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _prepare_dcp_query(
        self,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        padded_local_heads, _ = _get_dcp_padded_head_counts(
            self.num_heads,
            self.dcp_world_size,
        )
        q_local = q[:, : self.num_heads, :]
        if padded_local_heads > self.num_heads:
            q_local = F.pad(q_local, (0, 0, 0, padded_local_heads - self.num_heads))
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
        profile_timings_ms: dict[str, float] | None = None,
    ) -> None:
        profile = profile_timings_ms is not None
        profile_sync = profile and envs.VLLM_DSV4_DCP_LAYER_PROFILE_SYNC
        t0 = _dsv4_dcp_layer_profile_mark(profile_sync) if profile else 0.0
        combined_out, combined_lse = self.dcp_combine(
            partial_out,
            partial_lse,
            get_dcp_group(),
            return_lse=True,
        )
        if profile_timings_ms is not None:
            t1 = _dsv4_dcp_layer_profile_mark(profile_sync)
            profile_timings_ms["a2a_reduce"] = (t1 - t0) * 1000.0
        assert combined_out.shape == (
            output.shape[0],
            padded_local_heads,
            self.head_dim,
        ), f"unexpected DCP-combined output shape {tuple(combined_out.shape)}"
        assert combined_lse.shape == (
            output.shape[0],
            padded_local_heads,
        ), f"unexpected DCP-combined LSE shape {tuple(combined_lse.shape)}"
        # FlashMLA sink scaling does not affect returned LSE, so under DCP it
        # must be applied once after the global LSE is known.
        combined_out_pre_sink = combined_out
        attn_sink = self.attn_sink[:padded_local_heads]
        t0 = _dsv4_dcp_layer_profile_mark(profile_sync) if profile else 0.0
        combined_out = _apply_attn_sink_with_lse(
            combined_out_pre_sink,
            combined_lse,
            attn_sink,
        )
        if profile_timings_ms is not None:
            t1 = _dsv4_dcp_layer_profile_mark(profile_sync)
            profile_timings_ms["sink_scale"] = (t1 - t0) * 1000.0
        dsv4_debug_dump(
            "attn.dcp_lse_combine",
            layer_idx=self.debug_layer_idx,
            tensors={
                "partial_out": partial_out,
                "partial_lse": partial_lse,
                "combined_out_pre_sink": combined_out_pre_sink,
                "combined_out": combined_out,
                "combined_lse": combined_lse,
                "attn_sink": attn_sink,
            },
            extra={
                "prefix": self.prefix,
                "compress_ratio": self.compress_ratio,
                "dcp_world_size": self.dcp_world_size,
                "dcp_rank": self.dcp_rank,
                "padded_local_heads": padded_local_heads,
                "head_chunk_start": self.dcp_rank * padded_local_heads,
                "head_chunk_end": (self.dcp_rank + 1) * padded_local_heads,
            },
        )
        t0 = _dsv4_dcp_layer_profile_mark(profile_sync) if profile else 0.0
        output[:, :padded_local_heads, :].copy_(combined_out)
        if profile_timings_ms is not None:
            t1 = _dsv4_dcp_layer_profile_mark(profile_sync)
            profile_timings_ms["output_copy"] = (t1 - t0) * 1000.0

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        profile = self.dcp_world_size > 1 and _dsv4_dcp_layer_profile_enabled()
        profile_sync = profile and envs.VLLM_DSV4_DCP_LAYER_PROFILE_SYNC
        profile_timings_ms: dict[str, float] = {}
        t_total = _dsv4_dcp_layer_profile_mark(profile_sync) if profile else 0.0

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
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
        # only attend by generated indices. Under DCP, gather actual local
        # query heads across the DCP group before FlashMLA; the cross-rank LSE
        # combine scatters the result back to this rank's local heads.
        if self.dcp_world_size > 1:
            t0 = _dsv4_dcp_layer_profile_mark(profile_sync) if profile else 0.0
            q_flash, padded_local_heads = self._prepare_dcp_query(q)
            if profile:
                t1 = _dsv4_dcp_layer_profile_mark(profile_sync)
                profile_timings_ms["q_gather"] = (t1 - t0) * 1000.0
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
        if self.dcp_world_size > 1:
            (flash_output,) = current_workspace_manager().get_simultaneous(
                ((num_decode_tokens, q_flash.shape[2], self.head_dim), q.dtype),
            )

        dsv4_debug_dump(
            "attn.decode.before_flash",
            layer_idx=self.debug_layer_idx,
            tensors={
                "q_flash": q_flash,
                "swa_indices": swa_indices,
                "swa_lens": swa_lens,
                "topk_indices": topk_indices,
                "topk_lens": topk_lens,
            },
            extra={
                "prefix": self.prefix,
                "compress_ratio": self.compress_ratio,
                "swa_only": swa_only,
                "num_decode_tokens": num_decode_tokens,
                "dcp_world_size": self.dcp_world_size,
                "dcp_rank": self.dcp_rank,
                "padded_local_heads": padded_local_heads,
            },
        )

        t0 = _dsv4_dcp_layer_profile_mark(profile_sync) if profile else 0.0
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
        if profile:
            t1 = _dsv4_dcp_layer_profile_mark(profile_sync)
            profile_timings_ms["flash"] = (t1 - t0) * 1000.0
        dsv4_debug_dump(
            "attn.decode.after_flash",
            layer_idx=self.debug_layer_idx,
            tensors={
                "out": out,
                "lse": lse,
                "output": output,
            },
            extra={
                "prefix": self.prefix,
                "compress_ratio": self.compress_ratio,
                "swa_only": swa_only,
                "dcp_world_size": self.dcp_world_size,
                "dcp_rank": self.dcp_rank,
            },
        )
        if self.dcp_world_size > 1:
            t0 = _dsv4_dcp_layer_profile_mark(profile_sync) if profile else 0.0
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
            if profile:
                t1 = _dsv4_dcp_layer_profile_mark(profile_sync)
                profile_timings_ms["normalize"] = (t1 - t0) * 1000.0
            self._dcp_lse_combine(
                partial_out,
                partial_lse,
                output,
                padded_local_heads,
                profile_timings_ms if profile else None,
            )
            if profile:
                t1 = _dsv4_dcp_layer_profile_mark(profile_sync)
                profile_timings_ms["total"] = (t1 - t_total) * 1000.0
                _dsv4_dcp_layer_profile_record(
                    dcp_rank=self.dcp_rank,
                    layer_idx=self.debug_layer_idx,
                    mode="decode",
                    compress_ratio=self.compress_ratio,
                    B=num_decode_tokens,
                    H=q_flash.shape[2],
                    D=self.head_dim,
                    timings_ms=profile_timings_ms,
                )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        profile = self.dcp_world_size > 1 and _dsv4_dcp_layer_profile_enabled()
        profile_sync = profile and envs.VLLM_DSV4_DCP_LAYER_PROFILE_SYNC

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
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE

        workspace_manager = current_workspace_manager()
        dcp_padded_global_heads = 0
        max_chunk_tokens = 0
        if self.dcp_world_size > 1:
            _, dcp_padded_global_heads = _get_dcp_padded_head_counts(
                self.num_heads,
                self.dcp_world_size,
            )
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
                chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
                query_start = (
                    query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
                ).item()
                query_end = (
                    query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
                ).item()
                max_chunk_tokens = max(max_chunk_tokens, query_end - query_start)

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            ).item()
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            ).item()

            prefill_workspaces = [
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            ]
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
            workspaces = workspace_manager.get_simultaneous(*prefill_workspaces)
            kv = workspaces[0]
            flash_output_workspace = workspaces[1] if self.dcp_world_size > 1 else None

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
            profile_timings_ms: dict[str, float] = {}
            t_total = _dsv4_dcp_layer_profile_mark(profile_sync) if profile else 0.0
            if self.dcp_world_size > 1:
                assert flash_output_workspace is not None
                t0 = _dsv4_dcp_layer_profile_mark(profile_sync) if profile else 0.0
                q_chunk, padded_local_heads = self._prepare_dcp_query(q_chunk)
                if profile:
                    t1 = _dsv4_dcp_layer_profile_mark(profile_sync)
                    profile_timings_ms["q_gather"] = (t1 - t0) * 1000.0
                flash_output = flash_output_workspace[: q_chunk.shape[0]]
            else:
                padded_local_heads = self.padded_heads
                flash_output = output[query_start:query_end]

            dsv4_debug_dump(
                "attn.prefill.before_flash",
                layer_idx=self.debug_layer_idx,
                tensors={
                    "q_chunk": q_chunk,
                    "kv": kv[:chunk_size],
                    "topk_indices": topk_indices[query_start:query_end],
                    "combined_indices": combined_indices,
                    "combined_lens": combined_lens,
                    "seq_lens": seq_lens[chunk_start:chunk_end],
                    "swa_seq_lens": swa_seq_lens[chunk_start:chunk_end],
                    "swa_gather_lens": swa_gather_lens[chunk_start:chunk_end],
                },
                extra={
                    "prefix": self.prefix,
                    "compress_ratio": self.compress_ratio,
                    "swa_only": swa_only,
                    "chunk_idx": chunk_idx,
                    "chunk_size": chunk_size,
                    "query_start": query_start,
                    "query_end": query_end,
                    "M": M,
                    "N": N,
                    "top_k": top_k,
                    "dcp_world_size": self.dcp_world_size,
                    "dcp_rank": self.dcp_rank,
                    "total_cp_world_size": self.total_cp_world_size,
                    "total_cp_rank": self.total_cp_rank,
                    "padded_local_heads": padded_local_heads,
                },
            )
            _dsv4_debug_prefill_kv_probe(
                layer_idx=self.debug_layer_idx,
                kv=kv[:chunk_size],
                combined_indices=combined_indices,
                combined_lens=combined_lens,
                extra={
                    "prefix": self.prefix,
                    "compress_ratio": self.compress_ratio,
                    "swa_only": swa_only,
                    "chunk_idx": chunk_idx,
                    "chunk_size": chunk_size,
                    "query_start": query_start,
                    "query_end": query_end,
                    "M": M,
                    "N": N,
                    "top_k": top_k,
                    "dcp_world_size": self.dcp_world_size,
                    "dcp_rank": self.dcp_rank,
                    "total_cp_world_size": self.total_cp_world_size,
                    "total_cp_rank": self.total_cp_rank,
                    "cp_kv_cache_interleave_size": self.cp_kv_cache_interleave_size,
                    "padded_local_heads": padded_local_heads,
                },
            )

            t0 = _dsv4_dcp_layer_profile_mark(profile_sync) if profile else 0.0
            output_chunk, lse, _ = flash_mla_sparse_fwd(
                q=q_chunk,
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=None if self.dcp_world_size > 1 else self.attn_sink,
                topk_length=combined_lens,
                out=flash_output,
            )
            if profile:
                t1 = _dsv4_dcp_layer_profile_mark(profile_sync)
                profile_timings_ms["flash"] = (t1 - t0) * 1000.0
            dsv4_debug_dump(
                "attn.prefill.after_flash",
                layer_idx=self.debug_layer_idx,
                tensors={
                    "output_chunk": output_chunk,
                    "lse": lse,
                    "output_slice": output[query_start:query_end],
                },
                extra={
                    "prefix": self.prefix,
                    "compress_ratio": self.compress_ratio,
                    "swa_only": swa_only,
                    "chunk_idx": chunk_idx,
                    "query_start": query_start,
                    "query_end": query_end,
                    "dcp_world_size": self.dcp_world_size,
                    "dcp_rank": self.dcp_rank,
                },
            )
            if self.dcp_world_size > 1:
                t0 = _dsv4_dcp_layer_profile_mark(profile_sync) if profile else 0.0
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
                if profile:
                    t1 = _dsv4_dcp_layer_profile_mark(profile_sync)
                    profile_timings_ms["normalize"] = (t1 - t0) * 1000.0
                self._dcp_lse_combine(
                    partial_out,
                    partial_lse,
                    output[query_start:query_end],
                    padded_local_heads,
                    profile_timings_ms if profile else None,
                )
                if profile:
                    t1 = _dsv4_dcp_layer_profile_mark(profile_sync)
                    profile_timings_ms["total"] = (t1 - t_total) * 1000.0
                    _dsv4_dcp_layer_profile_record(
                        dcp_rank=self.dcp_rank,
                        layer_idx=self.debug_layer_idx,
                        mode="prefill",
                        compress_ratio=self.compress_ratio,
                        B=q_chunk.shape[0],
                        H=q_chunk.shape[1],
                        D=self.head_dim,
                        timings_ms=profile_timings_ms,
                    )


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # DeepseekV4 aligns indexer pages to FlashMLA's 576B so they can pack with
            # the indexer's compressor state cache. V3.2 keeps the legacy layout.
            alignment=576,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lighening Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indxer cache use the same layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
        # but only use the first half of the memory.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = DeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        k = self.compressor(hidden_states, positions, rotary_emb)
        weights, _ = self.weights_proj(hidden_states)
        q_quant, weights = fused_indexer_q_rope_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            weights,
            self.softmax_scale,
            self.n_head**-0.5,
            use_fp4=self.use_fp4_kv,
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)
