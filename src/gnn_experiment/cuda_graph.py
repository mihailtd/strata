"""Draw-Call Batching & In-Place Weight Folding Synergy: CUDA Graph Capture for Micro-Experts.

Combines In-Place Weight Folding (WeightFoldingEngine) with CUDA/HIP Graph Capture:
1. Weight Folding unwraps model layers completely, removing dynamic PEFT hooks.
2. CUDA Graph captures the unwrapped model forward step once into a static execution graph,
   eliminating per-token CPU kernel launch tax (30+ layers x 10+ C++ launches/token).
3. Expert Swapping (WeightFoldingEngine.activate): In-place weight mutation W_live = W0 + s*(U@V)
   preserves static weight VRAM pointers (data_ptr()), allowing graph.replay() to execute
   new micro-experts instantly WITHOUT graph re-capture penalty.
"""

from __future__ import annotations

import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StaticCache

from gnn_experiment.novel_peft import FoldableExpert, WeightFoldingEngine


class FoldedCudaGraphDecoder:
    """Static CUDA/HIP Graph execution engine for folded micro-expert single-token decode."""

    def __init__(
        self,
        model: PreTrainedModel | torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 32768,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = torch.device(device) if isinstance(device, str) else device

        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_input_ids: torch.Tensor | None = None
        self.static_position_ids: torch.Tensor | None = None
        self.static_cache_position: torch.Tensor | None = None
        self.static_logits: torch.Tensor | None = None
        self.past_key_values: StaticCache | None = None
        self._is_captured = False
        self._capture_count = 0
        self._is_locked = False

    @property
    def capture_count(self) -> int:
        return self._capture_count

    def capture(self, prompt_tokens: torch.Tensor, warmup_steps: int = 3) -> None:
        """Prefill prompt, allocate static buffers, and capture static CUDA Graph once."""
        if self._is_locked:
            raise RuntimeError("Graph re-capture attempted after initialization! Single-capture lock active.")

        if self.device.type != "cuda":
            raise RuntimeError("CUDA Graph capture requires a GPU (cuda/hip) device.")

        self.model.eval()
        prompt_tokens = prompt_tokens.to(self.device)
        cur_pos = prompt_tokens.shape[1]

        # 1. Instantiate StaticCache
        dtype = getattr(self.model, "dtype", torch.bfloat16)
        self.past_key_values = StaticCache(
            config=self.model.config,
            max_batch_size=1,
            max_cache_len=self.max_seq_len,
            device=self.device,
            dtype=dtype,
        )

        # 2. Prefill pass using StaticCache
        with torch.no_grad():
            cache_pos = torch.arange(0, cur_pos, device=self.device, dtype=torch.long)
            outputs = self.model(
                prompt_tokens, past_key_values=self.past_key_values, cache_position=cache_pos, use_cache=True
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        # 3. Allocate static input, position, cache_position, and 2D attention_mask buffers
        self.static_input_ids = next_token.clone().to(self.device)
        self.static_position_ids = torch.tensor([[cur_pos]], dtype=torch.long, device=self.device)
        self.static_cache_position = torch.tensor([cur_pos], dtype=torch.long, device=self.device)
        # FULL-LENGTH and never sliced. Slicing to `cur_pos + 1` bakes a fixed
        # mask width into the graph: replay then decodes every later position
        # against a mask that stops short of it, and the outputs silently
        # diverge (measured: divergence from plain greedy at token 12).
        # Width must equal the StaticCache width so the causal mask the model
        # builds has a shape that does not change between capture and replay;
        # positions beyond the current one are masked by the model's own
        # `kv_arange > cache_position` term, which reads the *static*
        # cache_position tensor and therefore re-evaluates on every replay.
        self.static_attention_mask = torch.ones((1, self.max_seq_len), dtype=torch.long, device=self.device)

        # 4. Warmup passes on dedicated CUDA stream
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(device=self.device))
        with torch.cuda.stream(s):
            for _ in range(warmup_steps):
                with torch.no_grad():
                    out = self.model(
                        self.static_input_ids,
                        attention_mask=self.static_attention_mask,
                        position_ids=self.static_position_ids,
                        cache_position=self.static_cache_position,
                        past_key_values=self.past_key_values,
                        use_cache=True,
                    )
                    _ = out.logits
                    # advance AFTER the forward -- the decode loop must use this
                    # same order or the first replayed token lands one slot late
                    self.static_position_ids += 1
                    self.static_cache_position += 1
                    cur_pos += 1

        torch.cuda.current_stream(device=self.device).wait_stream(s)

        # 5. Record static CUDA Graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s), torch.no_grad():
            out = self.model(
                self.static_input_ids,
                attention_mask=self.static_attention_mask,
                position_ids=self.static_position_ids,
                cache_position=self.static_cache_position,
                past_key_values=self.past_key_values,
                use_cache=True,
            )
            self.static_logits = out.logits

        torch.cuda.current_stream(device=self.device).wait_stream(s)
        self._is_captured = True
        self._capture_count += 1
        self._is_locked = True

    def prefill(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """Prefills prompt KV cache in-place into StaticCache without re-capturing CUDA Graph."""
        prompt_tokens = prompt_tokens.to(self.device)
        cur_pos = prompt_tokens.shape[1]

        # Reset StaticCache tensors for new turn prefill
        if self.past_key_values is not None:
            self.past_key_values.reset()

        with torch.no_grad():
            cache_pos = torch.arange(0, cur_pos, device=self.device, dtype=torch.long)
            outputs = self.model(
                prompt_tokens, past_key_values=self.past_key_values, cache_position=cache_pos, use_cache=True
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        self.static_input_ids.copy_(next_token)
        self.static_position_ids.copy_(torch.tensor([[cur_pos]], dtype=torch.long, device=self.device))
        self.static_cache_position.copy_(torch.tensor([cur_pos], dtype=torch.long, device=self.device))
        return outputs.logits

    def generate_with_graph(
        self,
        prompt_tokens: torch.Tensor,
        engine: WeightFoldingEngine | None = None,
        expert: FoldableExpert | None = None,
        max_new_tokens: int = 64,
    ) -> tuple[list[int], float, float, float]:
        """Executes single-token decode via CUDA Graph replay.

        If expert is provided, swaps expert in-place via WeightFoldingEngine before graph replay.
        Returns (generated_token_ids, decode_time_s, tokens_per_second, swap_ms).
        """
        if not self._is_captured:
            self.capture(prompt_tokens)

        # Swap expert in-place (mutates W_live at static VRAM pointers)
        swap_ms = 0.0
        if engine is not None and expert is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            engine.activate(expert)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            swap_ms = (time.perf_counter() - t0) * 1000.0

        # Prefill prompt into StaticCache
        self.prefill(prompt_tokens)

        generated_tensors = [self.static_input_ids.clone()]

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        start_t = time.perf_counter()

        # Collect stop token IDs for Qwen 3.5 (<|im_end|>, <|endoftext|>, eos_token)
        stop_token_ids = set()
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.add(self.tokenizer.eos_token_id)
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and isinstance(tid, int) and tid > 0:
                stop_token_ids.add(tid)

        # Order matters and must match capture(): REPLAY FIRST, then advance.
        # `prefill` leaves static_input_ids holding the token that belongs at
        # static_cache_position, so incrementing before the first replay wrote
        # that token one slot late and left a zeroed KV entry behind it.
        for _ in range(max_new_tokens - 1):
            self.graph.replay()
            next_token = torch.argmax(self.static_logits[:, -1, :], dim=-1, keepdim=True)
            self.static_input_ids.copy_(next_token)
            self.static_position_ids += 1
            self.static_cache_position += 1
            tok_id = next_token.item()
            if tok_id in stop_token_ids:
                break
            generated_tensors.append(next_token.clone())

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        elapsed_s = time.perf_counter() - start_t

        generated = [t.item() for t in generated_tensors]
        tok_s = (len(generated) - 1) / max(1e-5, elapsed_s)

        return generated, elapsed_s, tok_s, swap_ms

    @torch.no_grad()
    def verify_against_eager(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 32) -> dict:
        """Assert graph replay emits the SAME tokens as plain greedy decode.

        Both paths are greedy over identical weights, so they must agree token
        for token. Any speed number reported without this passing is
        meaningless -- a graph that decodes against a stale mask is faster
        precisely because it is doing the wrong thing (it also misses EOS and
        runs to the token cap, which flatters tok/s by amortising fixed cost).
        """
        ref = self.model.generate(
            prompt_tokens.to(self.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        ref_toks = ref[0][prompt_tokens.shape[1] :].tolist()

        got_toks, _, _, _ = self.generate_with_graph(prompt_tokens, max_new_tokens=max_new_tokens)

        n = min(len(ref_toks), len(got_toks))
        first_div = next((i for i in range(n) if ref_toks[i] != got_toks[i]), None)

        # `generate` returns the terminating stop token; this decoder breaks
        # before appending it. That is a reporting convention, not a numerical
        # difference, so a graph run that is an exact prefix of the reference
        # with only stop tokens left over still counts as a match. Anything
        # else -- any mismatched token, or extra NON-stop tokens -- does not.
        tail = ref_toks[len(got_toks) :]
        tail_is_only_stops = all(t in self._stop_token_ids() for t in tail)
        return {
            "match": first_div is None and len(got_toks) <= len(ref_toks) and tail_is_only_stops,
            "first_divergence": first_div,
            "n_ref": len(ref_toks),
            "n_graph": len(got_toks),
            "unmatched_tail": [self.tokenizer.convert_ids_to_tokens(t) for t in tail],
            "ref_text": self.tokenizer.decode(ref_toks, skip_special_tokens=True),
            "graph_text": self.tokenizer.decode(got_toks, skip_special_tokens=True),
        }

    def _stop_token_ids(self) -> set[int]:
        ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        for tok in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid > 0:
                ids.add(tid)
        return ids

    def generate_tokens_stream(
        self,
        prompt_tokens: torch.Tensor,
        engine: WeightFoldingEngine | None = None,
        expert: FoldableExpert | None = None,
        max_new_tokens: int = 64,
    ):
        """Yields decoded token text pieces token-by-token during CUDA Graph replay."""
        if not self._is_captured:
            self.capture(prompt_tokens)

        if engine is not None and expert is not None:
            engine.activate(expert)

        self.prefill(prompt_tokens)

        stop_token_ids = set()
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.add(self.tokenizer.eos_token_id)
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and isinstance(tid, int) and tid > 0:
                stop_token_ids.add(tid)

        first_tok_id = self.static_input_ids.item()
        if first_tok_id not in stop_token_ids:
            yield self.tokenizer.decode([first_tok_id])

        for _ in range(max_new_tokens - 1):
            self.graph.replay()  # replay first, then advance -- see generate_with_graph
            next_token = torch.argmax(self.static_logits[:, -1, :], dim=-1, keepdim=True)
            self.static_input_ids.copy_(next_token)
            self.static_position_ids += 1
            self.static_cache_position += 1
            tok_id = next_token.item()
            if tok_id in stop_token_ids:
                break
            yield self.tokenizer.decode([tok_id])
