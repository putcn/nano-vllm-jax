"""Unit tests for Sequence and SequenceGroup (Phase 6)."""
import pytest
from nanovllm_jax.engine.sequence import Sequence, SequenceGroup, SequenceStatus
from nanovllm_jax.sampling_params import SamplingParams


def make_seq(seq_id=0, prompt=None, max_tokens=16):
    if prompt is None:
        prompt = [1, 2, 3]
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def test_prompt_len():
    s = make_seq(prompt=[10, 20, 30])
    assert s.prompt_len == 3


def test_total_len_with_output():
    s = make_seq(prompt=[1, 2])
    s.append_token(99)
    s.append_token(100)
    assert s.total_len == 4
    assert s.output_len == 2


def test_all_token_ids():
    s = make_seq(prompt=[1, 2, 3])
    s.append_token(4)
    assert s.all_token_ids == [1, 2, 3, 4]


def test_last_token_id():
    s = make_seq(prompt=[5, 6])
    s.append_token(7)
    assert s.last_token_id == 7


def test_check_stop_max_tokens():
    s = make_seq(max_tokens=2)
    s.append_token(10)
    s.check_stop()
    assert not s.is_finished
    s.append_token(11)
    s.check_stop()
    assert s.is_finished


def test_check_stop_eos():
    s = Sequence(
        seq_id=0,
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=100, stop_token_ids=[2]),
    )
    s.append_token(2)
    s.check_stop()
    assert s.is_finished


def test_initial_status():
    s = make_seq()
    assert s.status == SequenceStatus.WAITING


def test_sequence_group_is_finished():
    g = SequenceGroup(group_id=0)
    s1 = make_seq(0)
    s2 = make_seq(1)
    g.add(s1)
    g.add(s2)
    assert not g.is_finished
    s1.status = SequenceStatus.FINISHED
    assert not g.is_finished
    s2.status = SequenceStatus.FINISHED
    assert g.is_finished


def test_sequence_group_num_seqs():
    g = SequenceGroup(group_id=1)
    for i in range(3):
        g.add(make_seq(i))
    assert g.num_seqs == 3
