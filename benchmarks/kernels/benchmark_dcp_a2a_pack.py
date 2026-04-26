# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DCP A2A two-collective vs packed one-collective paths.

Usage:
    torchrun --nproc_per_node=4 benchmarks/kernels/benchmark_dcp_a2a_pack.py \
        --batch-sizes 1,8,16,512,1024,4096,8192 --heads 64 --head-dim 512

The benchmark intentionally uses the real distributed A2A functions so it can
catch packing, global-LSE, and collective regressions before server integration.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

# Make sure the default dcp_a2a_lse_reduce call uses the current two-collective
# path even when the caller's shell has the experimental env enabled.
os.environ["VLLM_DCP_A2A_PACKED"] = "0"

from vllm.v1.attention.ops.dcp_alltoall import (  # noqa: E402
    dcp_a2a_lse_reduce,
    dcp_a2a_lse_reduce_packed,
)


@dataclass
class _BenchGroup:
    world_size: int
    rank_in_group: int
    device_group: dist.ProcessGroup


def _parse_batch_sizes(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_dtype(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _init_dist() -> tuple[int, int, int, torch.device]:
    if torch.accelerator.device_count() < 1:
        raise RuntimeError("benchmark_dcp_a2a_pack.py requires CUDA")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.accelerator.set_device_index(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _make_inputs(
    *,
    batch_size: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260426 + rank * 1009 + batch_size)
    out = torch.randn(
        (batch_size, heads, head_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    lse = torch.randn(
        (batch_size, heads),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    # Keep values in a realistic-but-stable range for exp/log comparisons.
    lse = lse * 4.0
    return out, lse


def _time_ms(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    dist.barrier()

    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    dist.barrier()
    return start.elapsed_time(end) / iters


def _max_error(
    ref: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    opt: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[float, float, float, float]:
    if isinstance(ref, tuple):
        ref_out, ref_lse = ref
        opt_out, opt_lse = opt  # type: ignore[misc]
    else:
        ref_out, opt_out = ref, opt  # type: ignore[assignment]
        ref_lse = opt_lse = None

    out_abs = (ref_out.float() - opt_out.float()).abs().max().item()
    out_rel = (
        (
            (ref_out.float() - opt_out.float()).abs()
            / ref_out.float().abs().clamp(min=1e-6)
        )
        .max()
        .item()
    )
    if ref_lse is None or opt_lse is None:
        return out_abs, out_rel, 0.0, 0.0
    lse_abs = (ref_lse.float() - opt_lse.float()).abs().max().item()
    lse_rel = (
        (
            (ref_lse.float() - opt_lse.float()).abs()
            / ref_lse.float().abs().clamp(min=1e-6)
        )
        .max()
        .item()
    )
    return out_abs, out_rel, lse_abs, lse_rel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=[1, 8, 16])
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--return-lse", action="store_true")
    parser.add_argument("--base2-lse", action="store_true")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rank, _local_rank, world_size, device = _init_dist()
    if args.heads % world_size != 0:
        raise ValueError(f"--heads {args.heads} must be divisible by {world_size}")

    dtype = _parse_dtype(args.dtype)
    group = _BenchGroup(world_size, rank, dist.group.WORLD)
    rows: list[dict[str, object]] = []

    for batch_size in args.batch_sizes:
        out, lse = _make_inputs(
            batch_size=batch_size,
            heads=args.heads,
            head_dim=args.head_dim,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        kwargs = {
            "return_lse": args.return_lse,
            "is_lse_base_on_e": not args.base2_lse,
        }
        ref = dcp_a2a_lse_reduce(out, lse, group, **kwargs)
        opt = dcp_a2a_lse_reduce_packed(out, lse, group, **kwargs)
        torch.accelerator.synchronize()
        out_abs, out_rel, lse_abs, lse_rel = _max_error(ref, opt)

        ref_ms = _time_ms(
            lambda out=out, lse=lse, kwargs=kwargs: dcp_a2a_lse_reduce(
                out, lse, group, **kwargs
            ),
            args.warmup,
            args.iters,
        )
        packed_ms = _time_ms(
            lambda out=out, lse=lse, kwargs=kwargs: dcp_a2a_lse_reduce_packed(
                out, lse, group, **kwargs
            ),
            args.warmup,
            args.iters,
        )

        row = {
            "world_size": world_size,
            "batch_size": batch_size,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "return_lse": args.return_lse,
            "lse_base": "base2" if args.base2_lse else "basee",
            "two_collective_ms": ref_ms,
            "packed_ms": packed_ms,
            "speedup": ref_ms / packed_ms if packed_ms > 0 else float("nan"),
            "out_max_abs": out_abs,
            "out_max_rel": out_rel,
            "lse_max_abs": lse_abs,
            "lse_max_rel": lse_rel,
        }
        rows.append(row)

        if rank == 0:
            print(
                "DCP_A2A_PACK_BENCH "
                + " ".join(f"{key}={value}" for key, value in row.items()),
                flush=True,
            )

    if args.csv and rank == 0:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"benchmark_dcp_a2a_pack.py failed: {exc}", file=sys.stderr)
        raise
