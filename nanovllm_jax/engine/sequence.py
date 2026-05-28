"""Sequence and SequenceGroup dataclasses (Phase 6).

Direct port of nanovllm/engine/sequence.py with minimal changes:
- No torch dependency
- block_table stored as plain Python list of ints
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from nanovllm_jax.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    SWAPPED = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    # Paged KV cache: list of block indices allocated to this sequence
    block_table: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def last_token_id(self) -> int:
        return self.all_token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    # ----------------------------------------------------------------
    # Mutations
    # ----------------------------------------------------------------

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def check_stop(self) -> None:
        """Mark finished if EOS or max_tokens reached."""
        sp = self.sampling_params
        if self.output_len >= sp.max_tokens:
            self.status = SequenceStatus.FINISHED
            return
        if sp.stop_token_ids and self.last_token_id in sp.stop_token_ids:
            self.status = SequenceStatus.FINISHED


@dataclass
class SequenceGroup:
    """Groups sequences that share the same prompt (beam search / parallel sampling)."""
    group_id: int
    sequences: list[Sequence] = field(default_factory=list)

    def add(self, seq: Sequence) -> None:
        self.sequences.append(seq)

    @property
    def is_finished(self) -> bool:
        return all(s.is_finished for s in self.sequences)

    @property
    def num_seqs(self) -> int:
        return len(self.sequences)
