# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated DeepSeek V4 parity debug dumps.

This module is intentionally lightweight and dormant by default. It writes
small JSONL tensor summaries only when VLLM_DSV4_DEBUG_DUMP_DIR is set.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

_DUMP_COUNTS: dict[str, int] = {}


def dsv4_debug_layer_idx(prefix: str) -> int | None:
    parts = prefix.split(".")
    try:
        layer_pos = parts.index("layers")
    except ValueError:
        return None
    if layer_pos + 1 >= len(parts):
        return None
    try:
        return int(parts[layer_pos + 1])
    except ValueError:
        return None


def dsv4_debug_enabled() -> bool:
    return bool(os.environ.get("VLLM_DSV4_DEBUG_DUMP_DIR"))


def dsv4_debug_should_dump(stage: str, layer_idx: int | None = None) -> bool:
    if not dsv4_debug_enabled():
        return False

    stage_filter = os.environ.get("VLLM_DSV4_DEBUG_STAGES", "").strip()
    if stage_filter:
        allowed_stages = {item.strip() for item in stage_filter.split(",") if item}
        if stage not in allowed_stages:
            return False

    layer_filter = os.environ.get("VLLM_DSV4_DEBUG_LAYERS", "").strip()
    if layer_filter and layer_filter.lower() != "all" and layer_idx is not None:
        allowed_layers = {
            int(item.strip())
            for item in layer_filter.split(",")
            if item.strip().lstrip("-").isdigit()
        }
        if layer_idx not in allowed_layers:
            return False

    return True


def _identity() -> dict[str, str | int | None]:
    return {
        "pid": os.getpid(),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "world_size": os.environ.get("WORLD_SIZE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _sample_rows(tensor: torch.Tensor) -> list[int]:
    if tensor.ndim == 0 or tensor.shape[0] == 0:
        return []
    max_rows = int(os.environ.get("VLLM_DSV4_DEBUG_TOKEN_ROWS", "16"))
    rows = list(range(min(max_rows, tensor.shape[0])))
    last_idx = tensor.shape[0] - 1
    if last_idx not in rows:
        rows.append(last_idx)
    return rows


def _tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None

    detached = tensor.detach()
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
    }
    if detached.numel() == 0:
        return summary

    sample_values = int(os.environ.get("VLLM_DSV4_DEBUG_VALUES", "8"))
    flat_sample = detached.reshape(-1)[:sample_values]
    if flat_sample.is_cuda:
        flat_sample = flat_sample.cpu()
    summary["flat_sample"] = flat_sample.tolist()

    full_stats = os.environ.get("VLLM_DSV4_DEBUG_FULL_STATS", "0") == "1"
    cpu = detached.cpu() if detached.is_cuda and full_stats else detached
    if full_stats:
        flat = cpu.reshape(-1)
        try:
            stats = flat.float()
            finite = torch.isfinite(stats)
            summary["finite_count"] = int(finite.sum().item())
            summary["numel"] = int(stats.numel())
            if bool(finite.any().item()):
                finite_stats = stats[finite]
                summary["mean"] = float(finite_stats.mean().item())
                summary["std"] = float(finite_stats.std(unbiased=False).item())
                summary["min"] = float(finite_stats.min().item())
                summary["max"] = float(finite_stats.max().item())
                summary["norm"] = float(torch.linalg.vector_norm(finite_stats).item())
        except RuntimeError as exc:
            summary["stats_error"] = str(exc)

    if detached.ndim >= 1:
        rows = []
        for row_idx in _sample_rows(detached):
            row = detached[row_idx].reshape(-1)
            if row.is_cuda:
                row = row.cpu()
            row_summary: dict[str, Any] = {
                "idx": row_idx,
                "sample": row[:sample_values].tolist(),
            }
            try:
                row_float = row.float()
                row_summary["mean"] = float(row_float.mean().item())
                row_summary["norm"] = float(torch.linalg.vector_norm(row_float).item())
            except RuntimeError as exc:
                row_summary["stats_error"] = str(exc)
            rows.append(row_summary)
        summary["rows"] = rows

    if detached.ndim == 2 and detached.shape[1] > 1024:
        topk = min(
            int(os.environ.get("VLLM_DSV4_DEBUG_TOPK", "10")),
            detached.shape[1],
        )
        top_rows = []
        for row_idx in _sample_rows(detached):
            values, indices = torch.topk(detached[row_idx].float(), k=topk)
            if values.is_cuda:
                values = values.cpu()
                indices = indices.cpu()
            top_rows.append(
                {
                    "idx": row_idx,
                    "top_indices": indices.tolist(),
                    "top_values": values.tolist(),
                }
            )
        summary["topk_rows"] = top_rows

    return summary


def dsv4_debug_dump(
    stage: str,
    *,
    layer_idx: int | None = None,
    tensors: dict[str, torch.Tensor | None] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if not dsv4_debug_should_dump(stage, layer_idx):
        return

    identity = _identity()
    count_key = f"{identity['pid']}:{layer_idx}:{stage}"
    count = _DUMP_COUNTS.get(count_key, 0)
    max_calls = int(os.environ.get("VLLM_DSV4_DEBUG_MAX_CALLS", "4"))
    if count >= max_calls:
        return
    _DUMP_COUNTS[count_key] = count + 1

    dump_dir = Path(os.environ["VLLM_DSV4_DEBUG_DUMP_DIR"])
    dump_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "time": time.time(),
        "stage": stage,
        "layer_idx": layer_idx,
        "call": count,
        "identity": identity,
        "extra": extra or {},
        "tensors": {},
    }
    for name, tensor in (tensors or {}).items():
        record["tensors"][name] = _tensor_summary(tensor)

    rank = identity.get("rank") or "na"
    local_rank = identity.get("local_rank") or "na"
    path = dump_dir / f"pid{identity['pid']}_rank{rank}_local{local_rank}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
