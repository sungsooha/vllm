# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer parallel configuration registry.

Maps module prefix patterns to parallel settings. Layers query the
registry to get their effective TP size, EP, DCP, and CUDA graph mode.

Usage:
    # During model init (automatic via CLI flags):
    register_layer_parallel_config("*.self_attn.*", LayerParallelConfig(tp_size=4))

    # In layer __init__:
    config = get_layer_parallel_config(prefix)
    tp_size = config.tp_size or get_tensor_model_parallel_world_size()
"""

from dataclasses import dataclass
from fnmatch import fnmatch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class LayerParallelConfig:
    """Parallel configuration for a layer or layer type."""

    tp_size: int | None = None  # None = use global TP
    tp_rank: int | None = None  # None = derive from tp_size
    # Future extensions (leave as None for now):
    # ep_enabled: Optional[bool] = None
    # dcp_enabled: Optional[bool] = None
    # cuda_graph_mode: Optional[str] = None


# Global registry: list of (pattern, config) tuples, checked in order
_LAYER_PARALLEL_REGISTRY: list[tuple[str, LayerParallelConfig]] = []


def register_layer_parallel_config(
    pattern: str,
    config: LayerParallelConfig,
) -> None:
    """Register a parallel config for layers matching the pattern.

    Args:
        pattern: Glob-like pattern matched against module prefix.
            Uses fnmatch-style matching (*, ? wildcards).
            Examples: "*.self_attn.*", "*.attention.*", "*.mlp.*"
        config: Parallel configuration for matching layers.
    """
    _LAYER_PARALLEL_REGISTRY.append((pattern, config))
    logger.info(
        "Registered layer parallel config: %s -> tp_size=%s", pattern, config.tp_size
    )


def get_layer_parallel_config(prefix: str) -> LayerParallelConfig:
    """Get the parallel config for a module at the given prefix.

    Checks patterns in registration order, returns first match.
    Returns default config (all None) if no pattern matches.
    """
    # Wrap prefix with dots for boundary matching
    wrapped = f".{prefix}."
    for pattern, config in _LAYER_PARALLEL_REGISTRY:
        if fnmatch(wrapped, f"*{pattern}*"):
            return config
    return LayerParallelConfig()


def clear_layer_parallel_registry() -> None:
    """Clear all registered configs. Used in testing."""
    _LAYER_PARALLEL_REGISTRY.clear()


# --- Convenience: detect attention layers ---
_ATTENTION_PATTERNS = (".self_attn.", ".attention.", ".attn.")


def is_attention_layer(prefix: str) -> bool:
    """Check if a module prefix indicates an attention layer."""
    wrapped = f".{prefix}."
    return any(p in wrapped for p in _ATTENTION_PATTERNS)
