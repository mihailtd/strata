"""Draw-Call Batching: CUDA Graph Capture & Replay for Single-Stream Decode (T-12).

Eliminates per-token CPU kernel launch overhead (30+ transformer layers x 10+ launches/layer)
by recording the entire decode forward step into a static CUDA / HIP execution graph.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class CudaGraphDecoder:
    """Static CUDA/HIP Graph execution wrapper for batch-1 decode."""

    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_input_ids: torch.Tensor | None = None
        self.static_position_ids: torch.Tensor | None = None
        self.static_logits: torch.Tensor | None = None
        self.past_key_values = None

    def capture(self, prompt_tokens: torch.Tensor, warmup_steps: int = 3) -> None:
        """Prefill prompt, then capture the graph for single-token decode steps."""
        self.model.eval()

        if self.device.type != "cuda":
            raise RuntimeError("CUDA Graph capture requires a GPU (cuda/hip) device.")

        # 1. Prefill pass to compute prompt KV cache
        prompt_tokens = prompt_tokens.to(self.device)
        with torch.no_grad():
            outputs = self.model(prompt_tokens, use_cache=True)
            self.past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            cur_pos = prompt_tokens.shape[1]

        # 2. Allocate static input, position, and attention mask buffers
        self.static_input_ids = next_token.clone().to(self.device)
        self.static_position_ids = torch.tensor([[cur_pos]], dtype=torch.long, device=self.device)
        self.static_attention_mask = torch.zeros((1, 1, 1, cur_pos + 1), dtype=self.model.dtype, device=self.device)

        # Warmup passes to stabilize CUDA caching allocator & memory pointers
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(device=self.device))
        with torch.cuda.stream(s):
            for _ in range(warmup_steps):
                with torch.no_grad():
                    out = self.model(
                        self.static_input_ids,
                        attention_mask=self.static_attention_mask,
                        position_ids=self.static_position_ids,
                        past_key_values=self.past_key_values,
                        use_cache=True,
                    )
                    self.past_key_values = out.past_key_values
                    self.static_position_ids += 1
                    self.static_attention_mask = torch.zeros((1, 1, 1, self.static_position_ids.item() + 1), dtype=self.model.dtype, device=self.device)
        torch.cuda.current_stream(device=self.device).wait_stream(s)

        # 3. Record static CUDA Graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            with torch.no_grad():
                out = self.model(
                    self.static_input_ids,
                    attention_mask=self.static_attention_mask,
                    position_ids=self.static_position_ids,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                )
                self.static_logits = out.logits
                self.past_key_values = out.past_key_values
        torch.cuda.current_stream(device=self.device).wait_stream(s)

    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 64) -> tuple[list[int], float]:
        """Generate new tokens using CUDA Graph replay for single-token steps.

        Returns (generated_token_ids, tokens_per_second).
        """
        self.capture(prompt_tokens)

        generated = [self.static_input_ids.item()]
        
        torch.cuda.synchronize(self.device)
        start_t = time.perf_counter()

        for _ in range(max_new_tokens - 1):
            self.static_position_ids += 1
            self.graph.replay()
            next_token_id = torch.argmax(self.static_logits[:, -1, :], dim=-1).item()
            if next_token_id in (self.tokenizer.eos_token_id, getattr(self.tokenizer, "pad_token_id", None)):
                break
            generated.append(next_token_id)
            self.static_input_ids.fill_(next_token_id)

        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start_t
        tokens_per_sec = (max_new_tokens - 1) / elapsed if elapsed > 0 else 0.0

        return generated, tokens_per_sec
