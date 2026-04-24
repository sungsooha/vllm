# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def _owner(pos: int, cp_world_size: int, interleave: int) -> int:
    return (pos % (cp_world_size * interleave)) // interleave


def _local_count(
    length: int,
    cp_world_size: int,
    cp_rank: int,
    interleave: int,
) -> int:
    base = length // interleave // cp_world_size * interleave
    remainder = length - base * cp_world_size
    extra = min(max(remainder - cp_rank * interleave, 0), interleave)
    return base + extra


def test_deepseek_v4_swa_dcp_shards_cover_window_once():
    for cp_world_size, interleave in [(2, 1), (2, 2), (4, 1), (4, 8)]:
        for window_size in [1, 7, 64, 257]:
            for query_pos in [0, 1, 15, 63, 64, 255, 1024]:
                start = max(query_pos - window_size + 1, 0)
                expected = list(range(start, query_pos + 1))

                shards: list[list[int]] = []
                for rank in range(cp_world_size):
                    shard = [
                        pos
                        for pos in expected
                        if _owner(pos, cp_world_size, interleave) == rank
                    ]
                    shards.append(shard)

                    local_start = _local_count(start, cp_world_size, rank, interleave)
                    packed_offsets = [
                        _local_count(pos + 1, cp_world_size, rank, interleave)
                        - 1
                        - local_start
                        for pos in shard
                    ]
                    assert packed_offsets == list(range(len(shard)))

                flattened = [pos for shard in shards for pos in shard]
                assert sorted(flattened) == expected
                assert len(flattened) == len(set(flattened))


def test_deepseek_v4_cp_local_count_matches_enumeration():
    for cp_world_size, interleave in [(1, 1), (2, 1), (2, 4), (4, 8)]:
        for length in [0, 1, 2, 7, 64, 255, 1024, 4097]:
            for rank in range(cp_world_size):
                expected = sum(
                    1
                    for pos in range(length)
                    if _owner(pos, cp_world_size, interleave) == rank
                )
                assert _local_count(length, cp_world_size, rank, interleave) == expected
