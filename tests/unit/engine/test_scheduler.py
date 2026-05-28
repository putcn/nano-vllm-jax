"""Unit tests for the continuous-batching Scheduler (Phase 6)."""
import pytest
from nanovllm_jax.engine.block_manager import BlockManager
from nanovllm_jax.engine.scheduler import Scheduler, SchedulerOutput
from nanovllm_jax.engine.sequence import Sequence, SequenceStatus
from nanovllm_jax.sampling_params import SamplingParams
from nanovllm_jax.config import EngineConfig


def make_engine_config(**kwargs):
    defaults = dict(
        model="test",
        max_num_seqs=4,
        max_num_batched_tokens=512,
        max_model_len=128,
        block_size=8,
        num_gpu_blocks=32,
    )
    defaults.update(kwargs)
    return EngineConfig(**defaults)


def make_seq(seq_id, prompt_len=4, max_tokens=8):
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def make_scheduler(max_num_seqs=4, num_blocks=32, block_size=8):
    cfg = make_engine_config(max_num_seqs=max_num_seqs, num_gpu_blocks=num_blocks, block_size=block_size)
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    return Scheduler(cfg, bm), bm


# ---------------------------------------------------------------------------
# Basic scheduling
# ---------------------------------------------------------------------------

def test_empty_step():
    sched, _ = make_scheduler()
    out = sched.step()
    assert out.is_empty


def test_single_prefill():
    sched, _ = make_scheduler()
    seq = make_seq(0, prompt_len=4)
    sched.add(seq)
    out = sched.step()
    assert len(out.prefill_seqs) == 1
    assert out.prefill_seqs[0].seq_id == 0
    assert seq.status == SequenceStatus.RUNNING


def test_prefill_then_decode():
    sched, _ = make_scheduler()
    seq = make_seq(0)
    sched.add(seq)
    # Step 1: prefill
    out1 = sched.step()
    assert len(out1.prefill_seqs) == 1
    assert len(out1.decode_seqs) == 0
    # Step 2: decode (no new seqs added)
    out2 = sched.step()
    assert len(out2.prefill_seqs) == 0
    assert len(out2.decode_seqs) == 1


def test_max_num_seqs_respected():
    sched, _ = make_scheduler(max_num_seqs=2)
    for i in range(4):
        sched.add(make_seq(i))
    out = sched.step()
    assert len(out.prefill_seqs) <= 2
    assert sched.num_waiting >= 2


def test_finished_seq_cleaned_up():
    sched, _ = make_scheduler()
    seq = make_seq(0)
    sched.add(seq)
    sched.step()  # prefill -> RUNNING
    seq.status = SequenceStatus.FINISHED
    sched.step()  # should clean up
    assert sched.num_running == 0


def test_preemption_on_no_free_blocks():
    """When blocks run out, scheduler preempts a running sequence."""
    sched, bm = make_scheduler(num_blocks=4, block_size=4)
    # Fill up blocks with 2 seqs of 8 tokens each (2 blocks each = 4 total)
    s0 = make_seq(0, prompt_len=8)
    s1 = make_seq(1, prompt_len=8)
    sched.add(s0)
    sched.add(s1)
    out = sched.step()  # both prefill, all 4 blocks used
    assert bm.num_free_blocks == 0
    # Simulate decode: s0 needs a new block but none available
    # Scheduler should preempt s1 (last added)
    out2 = sched.step()
    assert len(out2.preempted_seqs) >= 1


def test_has_work():
    sched, _ = make_scheduler()
    assert not sched.has_work
    sched.add(make_seq(0))
    assert sched.has_work
