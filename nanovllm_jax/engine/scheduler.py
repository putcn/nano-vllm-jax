"""Continuous-batching scheduler (Phase 6).

Port of nanovllm/engine/scheduler.py.
Pure Python priority-queue logic; no JAX.

Scheduling policy:
  1. Try to fit WAITING sequences into the running batch (FCFS).
  2. If a running sequence cannot append (no free blocks), preempt
     the lowest-priority running sequence (swap-out / abort).
  3. Return a SchedulerOutput describing what to run this step.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from nanovllm_jax.engine.block_manager import BlockManager
from nanovllm_jax.engine.sequence import Sequence, SequenceStatus
from nanovllm_jax.config import EngineConfig


@dataclass
class SchedulerOutput:
    """What the model runner should execute this step."""
    # Sequences doing prefill this step (status just became RUNNING)
    prefill_seqs: list[Sequence] = field(default_factory=list)
    # Sequences doing a single decode step
    decode_seqs: list[Sequence] = field(default_factory=list)
    # Sequences that were preempted (freed, moved back to WAITING)
    preempted_seqs: list[Sequence] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill_seqs and not self.decode_seqs

    @property
    def num_seqs(self) -> int:
        return len(self.prefill_seqs) + len(self.decode_seqs)


class Scheduler:
    """FCFS continuous-batching scheduler.

    Args:
        config:        EngineConfig (max_num_seqs, max_num_batched_tokens, etc.)
        block_manager: Manages paged KV cache block allocation.
    """

    def __init__(self, config: EngineConfig, block_manager: BlockManager) -> None:
        self.config = config
        self.block_manager = block_manager
        self._waiting: list[Sequence] = []   # FCFS queue
        self._running: list[Sequence] = []   # currently scheduled

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def add(self, seq: Sequence) -> None:
        """Enqueue a new sequence."""
        self._waiting.append(seq)

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def has_work(self) -> bool:
        return bool(self._waiting or self._running)

    def step(self) -> SchedulerOutput:
        """Produce one SchedulerOutput for the current step.

        Steps:
          1. Remove finished sequences.
          2. Ensure running sequences can append; preempt if not.
          3. Promote waiting sequences to running (prefill).
        """
        output = SchedulerOutput()

        # 1. Clean up finished sequences
        finished = [s for s in self._running if s.is_finished]
        for seq in finished:
            self.block_manager.free(seq)
            self._running.remove(seq)

        # 2. Ensure decode sequences have a free slot
        #    Preempt from the back (lowest priority = last added)
        for seq in list(reversed(self._running)):
            if not self.block_manager.can_append(seq):
                self.block_manager.free(seq)
                seq.status = SequenceStatus.WAITING
                self._running.remove(seq)
                self._waiting.insert(0, seq)  # re-queue at front
                output.preempted_seqs.append(seq)

        # Reserve slots for all decode sequences
        for seq in self._running:
            self.block_manager.append_slot(seq)

        # 3. Promote waiting -> running (prefill)
        while self._waiting:
            seq = self._waiting[0]
            # Check batch limits
            if len(self._running) >= self.config.max_num_seqs:
                break
            if not self.block_manager.can_allocate(seq):
                break
            self._waiting.pop(0)
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self._running.append(seq)
            output.prefill_seqs.append(seq)

        output.decode_seqs = [
            s for s in self._running if s not in output.prefill_seqs
        ]
        return output

    def finish(self, seq: Sequence) -> None:
        """Mark a sequence finished and free its blocks."""
        seq.status = SequenceStatus.FINISHED
        if seq in self._running:
            self.block_manager.free(seq)
            self._running.remove(seq)
