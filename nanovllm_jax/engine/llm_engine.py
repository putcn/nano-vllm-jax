"""LLMEngine: ties Scheduler + BlockManager + ModelRunner into one step loop.

Status: ✅ Done (Phase 7)

Usage::

    from nanovllm_jax.engine.llm_engine import LLMEngine
    from nanovllm_jax.config import EngineConfig
    from nanovllm_jax.sampling_params import SamplingParams

    engine = LLMEngine(EngineConfig(model="...", num_gpu_blocks=512))
    req_id = engine.add_request("Hello world", SamplingParams(max_tokens=64))
    while engine.has_unfinished:
        engine.step()
    output = engine.get_outputs(req_id)
"""
from __future__ import annotations
from typing import Dict, List, Optional

from nanovllm_jax.config import EngineConfig
from nanovllm_jax.sampling_params import SamplingParams
from nanovllm_jax.engine.sequence import Sequence, SequenceStatus
from nanovllm_jax.engine.block_manager import BlockManager
from nanovllm_jax.engine.scheduler import Scheduler, SchedulerOutput
from nanovllm_jax.engine.model_runner import ModelRunner


class RequestOutput:
    def __init__(self, request_id: str, prompt: str, token_ids: List[int], text: str):
        self.request_id = request_id
        self.prompt = prompt
        self.token_ids = token_ids
        self.text = text

    def __repr__(self) -> str:
        return f"RequestOutput(id={self.request_id!r}, tokens={len(self.token_ids)}, text={self.text[:40]!r})"


class LLMEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.block_manager = BlockManager(
            num_blocks=config.num_gpu_blocks,
            block_size=config.block_size,
        )
        self.scheduler = Scheduler(config, self.block_manager)
        self.model_runner = ModelRunner(config)
        self._sequences: Dict[str, Sequence] = {}
        self._next_seq_id: int = 0
        self._outputs: Dict[str, RequestOutput] = {}

    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        *,
        request_id: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
    ) -> str:
        if request_id is None:
            request_id = str(self._next_seq_id)
        if prompt_token_ids is None:
            prompt_token_ids = [min(ord(c), 255) for c in prompt]
        seq = Sequence(
            seq_id=self._next_seq_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        self._next_seq_id += 1
        self._sequences[request_id] = seq
        self.scheduler.add(seq)
        return request_id

    def step(self) -> List[RequestOutput]:
        out: SchedulerOutput = self.scheduler.step()
        if out.is_empty:
            return []

        new_tokens: Dict[int, int] = self.model_runner.run(
            prefill_seqs=out.prefill_seqs,
            decode_seqs=out.decode_seqs,
        )

        eos = self.config.eos_token_id
        finished: List[RequestOutput] = []
        for seq in out.prefill_seqs + out.decode_seqs:
            token_id = new_tokens.get(seq.seq_id)
            if token_id is not None:
                seq.append_token(token_id)
                seq.check_stop(eos_token_id=eos)   # <-- pass EOS

        for req_id, seq in list(self._sequences.items()):
            if seq.is_finished and req_id not in self._outputs:
                ro = RequestOutput(
                    request_id=req_id,
                    prompt="",
                    token_ids=seq.output_token_ids[:],
                    text="".join(chr(min(t, 127)) for t in seq.output_token_ids),
                )
                self._outputs[req_id] = ro
                finished.append(ro)

        return finished

    def get_outputs(self, request_id: str) -> Optional[RequestOutput]:
        return self._outputs.get(request_id)

    @property
    def has_unfinished(self) -> bool:
        return self.scheduler.has_work

    def generate_all(self) -> Dict[str, RequestOutput]:
        while self.has_unfinished:
            self.step()
        return dict(self._outputs)
