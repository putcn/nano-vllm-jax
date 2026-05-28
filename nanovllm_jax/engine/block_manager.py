"""Paged block manager (Phase 6).

Manages a pool of fixed-size KV cache blocks across sequences.
Pure Python — no JAX/numpy needed here; block indices are ints.

Key differences vs PyTorch original:
- BlockTable stored as plain list[int]
- No CUDA device handling
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from nanovllm_jax.engine.sequence import Sequence, SequenceStatus


class BlockAllocator:
    """Free-list allocator for a fixed pool of blocks."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free: deque[int] = deque(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("Out of KV cache blocks")
        return self._free.popleft()

    def free(self, block_id: int) -> None:
        self._free.append(block_id)


class BlockManager:
    """Manages paged KV cache block allocation for all sequences.

    Args:
        num_blocks: Total number of paged blocks in the KV cache.
        block_size: Tokens per block.
    """

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._allocator = BlockAllocator(num_blocks)

    # ----------------------------------------------------------------
    # Queries
    # ----------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return self._allocator.num_free

    def can_allocate(self, seq: Sequence) -> bool:
        """Can we allocate all blocks needed for seq's current total_len?"""
        needed = self._blocks_needed(seq.total_len)
        already = len(seq.block_table)
        return (needed - already) <= self._allocator.num_free

    def can_append(self, seq: Sequence) -> bool:
        """Can we append one more token (possibly allocating a new block)?"""
        # A new block is needed when the next token would exceed current capacity
        current_capacity = len(seq.block_table) * self.block_size
        if seq.total_len < current_capacity:
            return True
        return self._allocator.num_free >= 1

    # ----------------------------------------------------------------
    # Allocation
    # ----------------------------------------------------------------

    def allocate(self, seq: Sequence) -> None:
        """Allocate all blocks needed for seq (called at prefill start)."""
        needed = self._blocks_needed(seq.total_len)
        while len(seq.block_table) < needed:
            seq.block_table.append(self._allocator.allocate())

    def append_slot(self, seq: Sequence) -> None:
        """Ensure there is a free slot for the next token; allocate if needed."""
        current_capacity = len(seq.block_table) * self.block_size
        if seq.total_len >= current_capacity:
            seq.block_table.append(self._allocator.allocate())

    def free(self, seq: Sequence) -> None:
        """Return all blocks of seq to the free pool."""
        for block_id in seq.block_table:
            self._allocator.free(block_id)
        seq.block_table.clear()

    # ----------------------------------------------------------------
    # Block table helpers (for building JAX arrays)
    # ----------------------------------------------------------------

    def get_block_table(self, seq: Sequence, max_blocks: Optional[int] = None) -> list[int]:
        """Return block table padded to max_blocks with 0."""
        bt = list(seq.block_table)
        if max_blocks is not None:
            bt = bt[:max_blocks]
            bt += [0] * (max_blocks - len(bt))
        return bt

    def get_position_for_token(self, seq: Sequence, token_idx: int) -> tuple[int, int]:
        """Return (block_id, offset_within_block) for token at position token_idx."""
        block_num = token_idx // self.block_size
        offset = token_idx % self.block_size
        return seq.block_table[block_num], offset

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size
