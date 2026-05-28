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
    block_table: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

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

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def check_stop(self, eos_token_id: int = 0) -> None:
        """Mark finished if max_tokens reached, EOS emitted, or stop_token_id hit.

        Args:
            eos_token_id: Engine-level EOS token (0 = disabled).
        """
        sp = self.sampling_params
        if self.output_len >= sp.max_tokens:
            self.status = SequenceStatus.FINISHED
            return
        last = self.last_token_id
        if eos_token_id and last == eos_token_id and not sp.ignore_eos:
            self.status = SequenceStatus.FINISHED
            return
        if sp.stop_token_ids and last in sp.stop_token_ids:
            self.status = SequenceStatus.FINISHED


@dataclass
class SequenceGroup:
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
