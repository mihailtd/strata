"""Tensor-Level Recurrent State Handoff ($S_t$) & Hybrid Dual Protocol.

Enables zero-token, sub-millisecond context handoffs between specialized domain
experts on hybrid recurrent architectures (e.g. Qwen3.5 with GatedDeltaNet SSMs).
Employs the Hybrid Dual Protocol: emits a concise human-auditable summary for UI/logs
while propagating the full 52.5 MB mathematical state tensor ($S_t$) in VRAM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from novel_peft import FoldableExpert, WeightFoldingEngine
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache


def clone_hybrid_cache(cache: DynamicCache) -> DynamicCache:
    """Clones a DynamicCache preserving LinearAttentionLayer and DynamicLayer types."""
    new_cache = DynamicCache()
    setattr(new_cache, "_seen_tokens", getattr(cache, "_seen_tokens", 0))  # noqa: B010
    for layer in cache.layers:
        new_layer = object.__new__(type(layer))
        for k, v in layer.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(new_layer, k, v.clone())
            elif isinstance(v, list) and all(isinstance(x, torch.Tensor) for x in v):
                setattr(new_layer, k, [x.clone() for x in v])
            elif isinstance(v, list):
                setattr(new_layer, k, list(v))
            elif isinstance(v, dict):
                setattr(
                    new_layer, k, {dk: (dv.clone() if isinstance(dv, torch.Tensor) else dv) for dk, dv in v.items()}
                )
            else:
                setattr(new_layer, k, v)
        new_cache.layers.append(new_layer)
    return new_cache


@dataclass
class RecurrentStateSnapshot:
    """Immutable, cloneable snapshot of a hybrid recurrent cache ($S_t$)."""

    cache_instance: DynamicCache
    seq_length: int
    created_at_s: float = field(default_factory=time.perf_counter)
    total_bytes: int = 0

    def __post_init__(self):
        if not self.total_bytes:
            self.total_bytes = self._calculate_bytes()

    def _calculate_bytes(self) -> int:
        total = 0
        for layer in self.cache_instance.layers:
            for val in layer.__dict__.values():
                if isinstance(val, torch.Tensor):
                    total += val.numel() * val.element_size()
                elif isinstance(val, list):
                    for x in val:
                        if isinstance(x, torch.Tensor):
                            total += x.numel() * x.element_size()
                elif isinstance(val, dict):
                    for x in val.values():
                        if isinstance(x, torch.Tensor):
                            total += x.numel() * x.element_size()
        return total

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    def clone(self) -> RecurrentStateSnapshot:
        """Deep-copy all tensor states in VRAM with minimal overhead for parallel agent branching."""
        cloned_cache = clone_hybrid_cache(self.cache_instance)
        return RecurrentStateSnapshot(
            cache_instance=cloned_cache,
            seq_length=self.seq_length,
            created_at_s=time.perf_counter(),
            total_bytes=self.total_bytes,
        )

    def create_cache(self) -> DynamicCache:
        """Construct a fresh DynamicCache instance initialized from this snapshot."""
        return clone_hybrid_cache(self.cache_instance)


def capture_recurrent_state(cache: DynamicCache, seq_length: int = 0) -> RecurrentStateSnapshot:
    """Capture the full recurrent state tensor ($S_t$) from a DynamicCache."""
    cloned = clone_hybrid_cache(cache)
    if seq_length == 0 and len(cache.layers) > 0:
        for layer in cache.layers:
            keys = getattr(layer, "keys", None)
            if isinstance(keys, torch.Tensor):
                seq_length = max(seq_length, keys.shape[-2])
    return RecurrentStateSnapshot(cache_instance=cloned, seq_length=seq_length)


@dataclass
class DualProtocolResult:
    """Output container for a multi-agent execution turn under the Hybrid Dual Protocol."""

    full_output_text: str
    human_summary: str
    state_snapshot: RecurrentStateSnapshot
    generated_tokens: int
    prompt_tokens: int
    prefill_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float
    expert_name: str


class AgentHandoffSession:
    """Coordinates multi-agent execution with in-place adapter folding and tensor state handoff."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        folding_engine: WeightFoldingEngine,
        domain_experts: dict[str, FoldableExpert],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.folding_engine = folding_engine
        self.domain_experts = domain_experts
        self.current_state: RecurrentStateSnapshot | None = None
        self.active_expert_name: str | None = None

    @torch.no_grad()
    def execute_turn(
        self,
        expert_name: str,
        instruction: str,
        state_handoff: RecurrentStateSnapshot | None = None,
        generate_human_summary: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> DualProtocolResult:
        """Execute an agent turn, seamlessly continuing from a tensor state handoff."""
        t_start = time.perf_counter()

        # 1. Activate requested domain expert in-place via WeightFoldingEngine
        if self.active_expert_name != expert_name:
            expert = self.domain_experts.get(expert_name)
            if expert is None:
                raise ValueError(f"Unknown expert '{expert_name}'. Registered: {list(self.domain_experts.keys())}")
            self.folding_engine.activate(expert)
            self.active_expert_name = expert_name

        # 2. Determine state continuity (inherited snapshot vs fresh prompt)
        inherited_state = state_handoff or self.current_state

        if inherited_state is not None:
            # We have an existing mathematical state S_t.
            # Format minimal steering prompt to guide the new expert
            steering_prompt = f"<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            inputs = self.tokenizer(steering_prompt, return_tensors="pt").to(self.model.device)
            prompt_tokens = inputs.input_ids.shape[1]

            # Clone state for isolated execution
            cache = inherited_state.create_cache()

            # Execute fast steering prefill
            t_prefill_0 = time.perf_counter()
            out = self.model(**inputs, past_key_values=cache, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t_prefill_0) * 1000.0
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        else:
            # Fresh initial turn
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            prompt_tokens = inputs.input_ids.shape[1]

            t_prefill_0 = time.perf_counter()
            out = self.model(**inputs, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t_prefill_0) * 1000.0
            cache = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        # 3. Autoregressive token generation
        t_decode_0 = time.perf_counter()
        gen_tokens = [next_token]
        curr_token = next_token

        for _ in range(max_new_tokens):
            if curr_token.item() == self.tokenizer.eos_token_id:
                break
            step_out = self.model(curr_token, past_key_values=cache, use_cache=True)
            if temperature > 0.0:
                probs = torch.softmax(step_out.logits[:, -1, :] / temperature, dim=-1)
                curr_token = torch.multinomial(probs, num_samples=1)
            else:
                curr_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
            gen_tokens.append(curr_token)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t_decode_0) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0

        all_ids = torch.cat(gen_tokens, dim=-1)
        decoded = self.tokenizer.decode(all_ids[0], skip_special_tokens=True)
        full_text = " ".join(decoded) if isinstance(decoded, list) else decoded

        # 4. Hybrid Dual Protocol: Extract concise human audit summary
        human_summary = ""
        if generate_human_summary:
            lines = [line.strip() for line in full_text.splitlines() if line.strip()]
            summary_candidates = [
                line for line in lines if not line.startswith("```") and not line.startswith("#") and len(line) > 10
            ]
            if summary_candidates:
                human_summary = summary_candidates[0]
            else:
                human_summary = f"Generated {len(gen_tokens)} tokens of specialized {expert_name} output."

        # 5. Capture ending recurrent state S_t for next turn / handoff
        new_snapshot = capture_recurrent_state(cache)
        self.current_state = new_snapshot

        return DualProtocolResult(
            full_output_text=full_text,
            human_summary=human_summary,
            state_snapshot=new_snapshot,
            generated_tokens=len(gen_tokens),
            prompt_tokens=prompt_tokens,
            prefill_latency_ms=prefill_ms,
            decode_latency_ms=decode_ms,
            total_latency_ms=total_ms,
            expert_name=expert_name,
        )
