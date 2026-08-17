"""Bucketed CUDA-graph speculative decoding: keep graph replay AND speculation.

WHY THIS EXISTS
---------------
`docs/DECISIONS.md` §24 measured the fork:

    graph autoregressive (server today)   36.67 tok/s   1.135x over eager
    eager speculative                     37.55 tok/s   1.162x over eager
                                                        1.024x over GRAPH

The graph and speculation were **mutually exclusive**: the captured graph is a
fixed single-token shape, while speculative verification is a K+1 chunk and the
commit re-forward is n_acc+1 tokens. Wiring in eager speculation therefore GAVE
BACK the graph's 1.135x to buy speculation's 1.162x, netting +2.4%.

The resolution is that speculative chunk widths are **discrete and tightly
bounded**. Capture one graph per width and both compose.

    verification   always K+1
    commit         1 .. K
    -> K+1 graphs total (5 at K=4)

WHY POINTER STABILITY IS ALREADY SOLVED HERE
--------------------------------------------
Graph replay requires memory that never moves. Probed on this model, a
`StaticCache` built from the hybrid config contains:

    8  x StaticLayer          pre-allocated keys/values
    24 x LinearAttentionLayer fixed-size recurrent_states / conv_states dicts

Both halves are pointer-stable already, which makes rollback trivial and, more
importantly, graph-safe:

    attention   NOTHING TO DO. Rewind `cache_position`. The KV beyond it is stale
                but never read, and gets overwritten on the next write. No crop,
                no realloc, no invalidated graph pointers. (The eager speculative
                path used `layer.crop()`, which re-views the tensor and WOULD
                break a captured graph.)
    SSM         fixed-size states, restored by in-place `copy_` from a
                pre-allocated snapshot. Same discipline the state ring buffer
                applies, on buffers that never move.

KNOWN CAPTURE HAZARD
--------------------
fla's Triton kernels autotune by launching variants and timing them, which is host
synchronisation and illegal during capture. Every bucket is warmed BEFORE its
capture so the autotune cache is already populated. If a bucket is captured cold,
capture either fails or records a timing path.
"""

from __future__ import annotations

import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StaticCache


class BucketedSpeculativeDecoder:
    """CUDA-graph speculative decode with one captured graph per chunk width."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        draft_head,
        k: int = 4,
        max_seq_len: int = 2048,
        device: torch.device | str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.head = draft_head
        self.k = k
        self.max_seq_len = max_seq_len
        self.device = torch.device(device or next(model.parameters()).device)
        self.cache: StaticCache | None = None
        self.buckets: dict[int, dict] = {}
        self._pool = None
        self._ssm_snap: list[dict] | None = None
        self._locked = False

    # ---------------------------------------------------------------- capture
    def capture(self, prompt_tokens: torch.Tensor, warmup_steps: int = 3) -> None:
        """Capture one graph per width in 1..k+1, all sharing ONE StaticCache."""
        if self._locked:
            raise RuntimeError("re-capture attempted; buckets are single-capture")
        if self.device.type != "cuda":
            raise RuntimeError("graph capture requires a CUDA/HIP device")

        self.model.eval()
        prompt_tokens = prompt_tokens.to(self.device)
        dtype = getattr(self.model, "dtype", torch.bfloat16)
        self.cache = StaticCache(
            config=self.model.config, max_batch_size=1,
            max_cache_len=self.max_seq_len, device=self.device, dtype=dtype,
        )

        # prefill once so the cache and the autotune caches are warm
        cur = prompt_tokens.shape[1]
        with torch.no_grad():
            self.model(prompt_tokens, past_key_values=self.cache,
                       cache_position=torch.arange(0, cur, device=self.device), use_cache=True)

        # FULL-LENGTH mask, never sliced: slicing bakes a fixed mask width into the
        # graph and later positions then decode against a mask that stops short of
        # them (measured divergence at token 12 in the single-width decoder).
        self.attn_mask = torch.ones((1, self.max_seq_len), dtype=torch.long, device=self.device)

        for width in range(1, self.k + 2):
            self._capture_width(width, cur, warmup_steps)
            cur += 0  # positions are set per-replay; capture position is arbitrary

        self._alloc_ssm_snapshot()
        self._locked = True

    def _capture_width(self, width: int, start_pos: int, warmup_steps: int) -> None:
        ids = torch.zeros((1, width), dtype=torch.long, device=self.device)
        pos_ids = torch.arange(start_pos, start_pos + width, device=self.device).view(1, width)
        cache_pos = torch.arange(start_pos, start_pos + width, device=self.device)

        # WARM FIRST -- fla autotunes on first sight of a shape, and autotuning
        # inside capture is a host sync.
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(device=self.device))
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(warmup_steps):
                self.model(ids, attention_mask=self.attn_mask, position_ids=pos_ids,
                           cache_position=cache_pos, past_key_values=self.cache,
                           use_cache=True, output_hidden_states=True)
        torch.cuda.current_stream(device=self.device).wait_stream(s)

        g = torch.cuda.CUDAGraph()
        ctx = (torch.cuda.graph(g, stream=s, pool=self._pool) if self._pool
               else torch.cuda.graph(g, stream=s))
        with ctx, torch.no_grad():
            out = self.model(ids, attention_mask=self.attn_mask, position_ids=pos_ids,
                             cache_position=cache_pos, past_key_values=self.cache,
                             use_cache=True, output_hidden_states=True)
            logits = out.logits
            hidden = out.hidden_states[-1]
        torch.cuda.current_stream(device=self.device).wait_stream(s)
        if self._pool is None:
            self._pool = g.pool()

        self.buckets[width] = {"graph": g, "ids": ids, "pos_ids": pos_ids,
                               "cache_pos": cache_pos, "logits": logits, "hidden": hidden}

    # ------------------------------------------------------------- SSM state
    def _alloc_ssm_snapshot(self) -> None:
        """Pre-allocate the rollback buffers once; never allocate in the loop."""
        snap = []
        for layer in self.cache.layers:
            entry = {}
            for name in ("recurrent_states", "conv_states"):
                d = getattr(layer, name, None)
                if isinstance(d, dict):
                    entry[name] = {k: (v.clone() if isinstance(v, torch.Tensor) else None)
                                   for k, v in d.items()}
            snap.append(entry)
        self._ssm_snap = snap

    def _snapshot_ssm(self) -> None:
        for layer, entry in zip(self.cache.layers, self._ssm_snap, strict=True):
            for name, d in entry.items():
                live = getattr(layer, name)
                for k, buf in d.items():
                    if buf is not None and isinstance(live.get(k), torch.Tensor):
                        buf.copy_(live[k], non_blocking=True)

    def _restore_ssm(self) -> None:
        for layer, entry in zip(self.cache.layers, self._ssm_snap, strict=True):
            for name, d in entry.items():
                live = getattr(layer, name)
                for k, buf in d.items():
                    if buf is not None and isinstance(live.get(k), torch.Tensor):
                        live[k].copy_(buf, non_blocking=True)

    # ------------------------------------------------------------------ replay
    def _replay(self, width: int, tokens: torch.Tensor, start_pos: int):
        b = self.buckets[width]
        b["ids"].copy_(tokens.view(1, width))
        p = torch.arange(start_pos, start_pos + width, device=self.device)
        b["pos_ids"].copy_(p.view(1, width))
        b["cache_pos"].copy_(p)
        b["graph"].replay()
        return b["logits"], b["hidden"]

    # ---------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 64,
                 stop_ids: set[int] | None = None) -> tuple[list[int], float, dict]:
        """Speculative decode entirely on captured graphs."""
        assert self._locked, "capture() first"
        k = self.k
        stop_ids = stop_ids or {self.tokenizer.eos_token_id}
        prompt_tokens = prompt_tokens.to(self.device)

        self.cache.reset()
        cur = prompt_tokens.shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model(prompt_tokens, past_key_values=self.cache,
                         cache_position=torch.arange(0, cur, device=self.device),
                         use_cache=True, output_hidden_states=True)
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        hids = [out.hidden_states[-1]]
        seq = prompt_tokens
        toks = [nxt.item()]
        pos = cur
        st = {"steps": 0, "accepted": 0, "drafted": 0}
        done = toks[0] in stop_ids

        while len(toks) < max_new_tokens and not done:
            H = torch.cat(hids, dim=1)
            dcache = self.head.prefill(H, seq)
            draft = self.head.draft(H[:, -1:, :], nxt, k=k, start_pos=pos - 1, cache=dcache)

            self._snapshot_ssm()
            chunk = torch.cat([nxt, draft], dim=-1)
            logits, hidden = self._replay(k + 1, chunk, pos)
            target = torch.argmax(logits[0], -1)

            n_acc = 0
            for i in range(k):
                if draft[0, i].item() == target[i].item():
                    n_acc += 1
                else:
                    break
            st["steps"] += 1
            st["drafted"] += k
            st["accepted"] += n_acc

            if n_acc < k:
                # ROLLBACK: restore the fixed-size SSM state, and simply rewind the
                # position for attention -- StaticCache KV beyond it is stale but
                # unread and will be overwritten. No crop, so no graph pointer moves.
                self._restore_ssm()
                committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
                _, hidden = self._replay(n_acc + 1, committed, pos)
                new_h = hidden
            else:
                new_h = hidden[:, : n_acc + 1, :]
                committed = chunk

            bonus = target[n_acc].item()
            for t in draft[0, :n_acc].tolist() + [bonus]:
                if len(toks) >= max_new_tokens:
                    break
                toks.append(t)
                if t in stop_ids:
                    done = True
                    break

            hids.append(new_h.clone())
            seq = torch.cat([seq, committed], dim=-1)
            pos += n_acc + 1
            nxt = torch.tensor([[bonus]], device=self.device)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        st["tau"] = st["accepted"] / max(1, st["steps"])
        return toks[:max_new_tokens], elapsed, st
