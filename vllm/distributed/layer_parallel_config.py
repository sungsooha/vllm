# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer parallel configuration resolver.

A typed resolver that returns a per-layer ``LayerParallelConfig`` for shared
layer code (``QKVParallelLinear``, attention modules, KV cache spec builders,
DCP metadata builders, …) to query at construction time.

Mirrors the shape of ``quant_config.get_quant_method(layer, prefix)`` —
already an accepted upstream pattern (see
``vllm/model_executor/layers/quantization/base_config.py``).

v1 scope: lower the global ``tensor_parallel_size_attention`` (TPA) CLI flag
into per-attention-layer config. Non-attention layers and the no-TPA case
return a default ``LayerParallelConfig`` (all fields ``None``), meaning
"fall back to the global TP world".

Public API:
    ``init_layer_parallel_resolver(...)`` — called once from
    ``initialize_model_parallel`` at engine init.
    ``get_layer_parallel_config(layer, prefix) -> LayerParallelConfig`` — the
    resolver function callable by any shared layer code. **Module-level so
    that v2.1 KV-cache/attention-metadata consumers can reuse the same
    public API without churn.**
    ``clear_layer_parallel_resolver()`` — testing only.

v2+ additions (additive, no rename of public API):
    - ``LayerParallelConfig.dcp_size`` / ``kvp_size`` (per-layer DCP/KVP)
    - ``LayerParallelConfig.q_projection`` (Q-rep policy)
    - ``LayerParallelConfig.ep_size`` (per-layer EP)
    - Optional ``descriptor=None`` keyword on the resolver function
"""

from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class LayerParallelConfig:
    """Resolved per-layer parallel configuration.

    Every field defaults to ``None``, meaning "use the global TP world
    behavior at this layer".

    v1 fields:
        tp_size: Effective TP world size this layer should shard with.
        tp_rank: This rank's index within ``tp_size``. Differs from the
            full TP rank when TPA < TP (attention TP groups are smaller).
            QKV weight loading uses this rank, not the full TP rank.
    """

    tp_size: int | None = None
    tp_rank: int | None = None
    # v2+ fields are intentionally NOT declared yet. Adding them later is
    # additive and non-breaking; declaring them now without a consumer is
    # over-engineering for v1.


# --------------------------------------------------------------------- #
# Internal resolver state. Not exported.
# --------------------------------------------------------------------- #
# v1 holds a single set of values lowered from the
# ``tensor_parallel_size_attention`` CLI flag. v2+ may replace this with a
# rule list, glob registry, or descriptor-keyed table — without changing
# the public ``get_layer_parallel_config(layer, prefix)`` API.

# Prefix substring patterns identifying an attention sub-module. These are
# the same as the prototype's heuristic — fragile but adequate for the v1
# whitelist (Llama-3.x and Nemotron-49B-GQA both use ``self_attn``).
# Will be replaced with class-based dispatch in v2.x when needed.
_ATTENTION_PREFIX_PATTERNS: tuple[str, ...] = (
    ".self_attn.",
    ".attention.",
    ".attn.",
)


@dataclass(frozen=True)
class _ResolverState:
    """Private holder for the resolver's configuration."""

    full_tp_size: int
    full_tp_rank: int
    attn_tp_size: int  # == full_tp_size when TPA is unset
    attn_tp_rank: int  # within the attention TP group


_resolver_state: _ResolverState | None = None


def init_layer_parallel_resolver(
    *,
    full_tp_size: int,
    full_tp_rank: int,
    attn_tp_size: int,
    attn_tp_rank: int,
) -> None:
    """Initialize the resolver with the engine's lowered parallel config.

    Called once from ``initialize_model_parallel`` at engine init time after
    the TP and DCP groups are constructed. Subsequent calls reset the
    resolver — useful for tests, harmless in production.

    Args:
        full_tp_size: The full tensor-parallel world size
            (``tensor_parallel_size``).
        full_tp_rank: This worker's rank in the full TP group.
        attn_tp_size: The attention-only TP size (``tpa_size`` if set,
            else ``full_tp_size``). Always satisfies
            ``attn_tp_size <= full_tp_size`` and
            ``full_tp_size % attn_tp_size == 0``.
        attn_tp_rank: This worker's rank within the attention TP group.
            When ``attn_tp_size == full_tp_size`` this equals
            ``full_tp_rank``.
    """
    global _resolver_state
    if full_tp_size % attn_tp_size != 0:
        raise ValueError(
            "full_tp_size must be divisible by attn_tp_size: "
            f"got {full_tp_size=}, {attn_tp_size=}"
        )
    if not 0 <= attn_tp_rank < attn_tp_size:
        raise ValueError(f"attn_tp_rank out of range: {attn_tp_rank=}, {attn_tp_size=}")
    _resolver_state = _ResolverState(
        full_tp_size=full_tp_size,
        full_tp_rank=full_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=attn_tp_rank,
    )
    if attn_tp_size != full_tp_size:
        logger.info(
            "Per-layer parallel resolver initialized: "
            "full_tp=%d, attn_tp=%d (TPA mode active)",
            full_tp_size,
            attn_tp_size,
        )


def clear_layer_parallel_resolver() -> None:
    """Reset the resolver state. Tests only."""
    global _resolver_state
    _resolver_state = None


def get_layer_parallel_config(
    layer: Any,  # noqa: ARG001  # reserved for v2.x class-based dispatch
    prefix: str,
) -> LayerParallelConfig:
    """Resolve the per-layer parallel config for a layer at this prefix.

    Mirrors ``quant_config.get_quant_method(layer, prefix)`` — two args,
    layer object + module path prefix. ``layer`` is currently unused (v1
    relies on prefix-based attention detection); reserved for v2.x where
    class-based dispatch will replace the prefix heuristic.

    Returns:
        ``LayerParallelConfig`` with fields populated when this layer
        differs from the global TP world (currently: attention layers
        under TPA mode). Otherwise all fields are ``None`` and callers
        should fall back to ``get_tensor_model_parallel_world_size()`` /
        ``get_tensor_model_parallel_rank()``.
    """
    if _resolver_state is None:
        # Resolver not initialized — happens in unit tests that don't
        # set up distributed; return defaults so callers fall back.
        return LayerParallelConfig()

    state = _resolver_state
    if state.attn_tp_size == state.full_tp_size:
        # No TPA — every layer uses the global TP world.
        return LayerParallelConfig()

    if not _is_attention_prefix(prefix):
        # Non-attention layer keeps the global TP world.
        return LayerParallelConfig()

    return LayerParallelConfig(
        tp_size=state.attn_tp_size,
        tp_rank=state.attn_tp_rank,
    )


def _is_attention_prefix(prefix: str) -> bool:
    """Return True if the prefix names an attention sub-module.

    v1 heuristic — wraps the prefix with dots for boundary matching and
    checks against ``_ATTENTION_PREFIX_PATTERNS``. v2.x will replace this
    with class-based dispatch (using the ``layer`` argument).
    """
    wrapped = f".{prefix}."
    return any(p in wrapped for p in _ATTENTION_PREFIX_PATTERNS)
