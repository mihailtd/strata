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

import os
import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StaticCache

from runtime.state_ring_buffer import RingBufferReplayEngine
from runtime.macd_speculation_circuit_breaker import MACDSpeculationCircuitBreaker


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

        # Enforce strict architecture and hidden dimension compatibility
        base_cfg = getattr(model, "config", None)
        if base_cfg is not None and hasattr(base_cfg, "get_text_config"):
            base_cfg = base_cfg.get_text_config()
        head_hidden = getattr(draft_head, "target_hidden_size", None)
        base_hidden = getattr(base_cfg, "hidden_size", None)
        if isinstance(head_hidden, int) and isinstance(base_hidden, int) and head_hidden != base_hidden:
            from runtime.mtp_draft import IncompatibleDraftHeadError
            head_id = getattr(draft_head, "target_model_id", "unknown")
            raise IncompatibleDraftHeadError(
                f"Speculative draft head incompatible: Draft head '{head_id}' has hidden_size={head_hidden}, "
                f"but base model has hidden_size={base_hidden}. Cannot pair mismatched draft head with base model."
            )

        self.k = k
        self.max_seq_len = max_seq_len
        self.device = torch.device(device or next(model.parameters()).device)
        self.cache: StaticCache | None = None
        self.buckets: dict[int, dict] = {}
        self._pool = None
        # Opt-in ONLY for A/B against the aliasing bug; the safe default is private
        # pools. See _capture_width.
        self._share_pool = os.environ.get("SPECULATIVE_SHARE_GRAPH_POOL", "0") == "1"
        self.ring_engine = None
        self._len_counters: list[torch.Tensor] = []
        self._counter_pos = -1  # value the device counters currently hold
        self._ctr_src: torch.Tensor | None = None
        self._ctr_srcs: list[torch.Tensor] = []
        self.circuit_breaker = MACDSpeculationCircuitBreaker(
            alpha_fast=0.25,
            alpha_slow=0.08,
            disengage_threshold=1.8,
            reengage_threshold=2.2,
            enabled=True,
        )
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

        for width in range(1, self.k + 2):
            self._capture_width(width, cur, warmup_steps)
            cur += 0  # positions are set per-replay; capture position is arbitrary

        # Reset cache fresh after warmups and graph recordings
        self.cache.reset()

        ring_mode = os.environ.get("RING_BUFFER_MODE", "selective_hybrid")
        self.ring_engine = RingBufferReplayEngine(self.cache, max_depth=64, mode=ring_mode)
        
        self._collect_length_counters()
        self._locked = True

    def _capture_width(self, width: int, start_pos: int, warmup_steps: int) -> None:
        ids = torch.zeros((1, width), dtype=torch.long, device=self.device)
        pos_ids = torch.arange(start_pos, start_pos + width, device=self.device).view(1, width)
        cache_pos = torch.arange(start_pos, start_pos + width, device=self.device)

        # Dynamic 4D Causal Mask Buffer (prevents StaticCache boundary and stale slot pollution)
        min_val = torch.finfo(torch.bfloat16).min
        mask = torch.full((1, 1, width, self.max_seq_len), min_val, dtype=torch.bfloat16, device=self.device)
        for i in range(width):
            mask[:, :, i, : start_pos + i + 1] = 0.0

        # WARM FIRST -- fla autotunes on first sight of a shape, and autotuning
        # inside capture is a host sync.
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(device=self.device))
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(warmup_steps):
                self.model(ids, attention_mask=mask, position_ids=pos_ids,
                           cache_position=cache_pos, past_key_values=self.cache,
                           use_cache=True, output_hidden_states=True)
        torch.cuda.current_stream(device=self.device).wait_stream(s)

        g = torch.cuda.CUDAGraph()
        # PRIVATE POOL PER WIDTH. Sharing one pool across the bucket graphs is only
        # safe when they are replayed in capture order; this decoder replays width
        # K+1 every step and width n_acc+1 only on partial accepts, i.e. ARBITRARY
        # order. Sharing then lets one graph's intermediate activations alias
        # another's, and the corruption is silent for hundreds of steps before an
        # fla Triton kernel with a data-dependent loop is fed garbage and spins on
        # the GPU forever -- 100% utilisation, no completion, the host stuck on the
        # first .item() after replay, SIGINT ignored. Measured: wedged at
        # step 303/pos 891 (58-tok prompt) and step 244/pos 1193 (458-tok prompt),
        # ~20s of generation either way. A private pool per width costs a little
        # VRAM (each holds activations for <= K+1 tokens) and removes the aliasing.
        ctx = (torch.cuda.graph(g, stream=s, pool=self._pool)
               if (self._pool is not None and self._share_pool)
               else torch.cuda.graph(g, stream=s))
        with ctx, torch.no_grad():
            out = self.model(ids, attention_mask=mask, position_ids=pos_ids,
                             cache_position=cache_pos, past_key_values=self.cache,
                             use_cache=True, output_hidden_states=True)
            logits = out.logits
            hidden = out.hidden_states[-1]
        torch.cuda.current_stream(device=self.device).wait_stream(s)
        if self._pool is None and self._share_pool:
            self._pool = g.pool()

        self.buckets[width] = {"graph": g, "ids": ids, "pos_ids": pos_ids,
                               "cache_pos": cache_pos, "mask": mask, "logits": logits, "hidden": hidden}

    # ------------------------------------------------------------- SSM state
    # RingBufferReplayEngine handles snapshotting and rollback.

    # ------------------------------------------------------------------ replay
    def _collect_length_counters(self) -> None:
        """Grab the attention layers' DEVICE-SIDE cumulative_length tensors.

        THIS IS THE BUG THAT WEDGED THE GPU. transformers' StaticLayer keeps its
        write offset in a device tensor and advances it in place:

            cache_position = torch.arange(kv_length, device=...) + cumulative_length
            cumulative_length.add_(kv_length)

        Both lines are CAPTURED INTO THE GRAPH, so every replay advances the counter
        by `width` no matter what `cache_pos` we copy in -- the attention layers
        derive their own offset and never read ours. Two consequences:

          correctness  speculative decode replays OVERLAPPING ranges (a K+1 verify,
                       then a n_acc+1 commit at the SAME position). The counter does
                       not rewind, so committed tokens were written at drifting
                       offsets.
          stability    the counter grows without bound and runs past max_cache_len,
                       the attention kernel indexes out of range, and the GPU wedges
                       -- 100% busy, no completion, SIGINT ignored. Measured wedge
                       points match 2048/width exactly: ~350-400 replays at width 5,
                       ~1950-2000 at width 1, and step ~300 in real decode where a
                       step costs ~7.7 tokens of counter.

        The counters are `mark_static_address`-ed, so their storage never moves and
        writing them before a replay is graph-safe.
        """
        self._len_counters = []
        for layer in self.cache.layers:
            c = getattr(layer, "cumulative_length", None)
            if isinstance(c, torch.Tensor):
                self._len_counters.append(c)
        # Escape hatch for ATTRIBUTION ONLY: with pinning off the decoder is the
        # old, incorrect one (drifting KV offsets, wedges past 2048/width replays).
        # It exists so a benchmark can price the fix, never to serve traffic.
        if os.environ.get("SPECULATIVE_PIN_CACHE_LEN", "1") != "1":
            self._len_counters = []
        if self._len_counters:
            ref = self._len_counters[0]
            self._ctr_src = torch.zeros_like(ref)
            self._ctr_srcs = [self._ctr_src] * len(self._len_counters)

    def _replay(self, width: int, tokens: torch.Tensor, start_pos: int):
        b = self.buckets[width]
        b["ids"].copy_(tokens.view(1, width))
        p = torch.arange(start_pos, start_pos + width, device=self.device)
        b["pos_ids"].copy_(p.view(1, width))
        b["cache_pos"].copy_(p)

        # Dynamic causal mask update: unmask prefix up to each token in chunk
        min_val = torch.finfo(torch.bfloat16).min
        mask = b["mask"]
        for i in range(width):
            mask[:, :, i, : start_pos + i + 1] = 0.0
            mask[:, :, i, start_pos + i + 1 :] = min_val

        # Pin the captured counter to the TRUE position -- but ONLY when it has
        # actually drifted. The graph leaves the counter at start_pos + width, which
        # is exactly where the NEXT sequential replay wants it, so the common path
        # (verify, then the following verify after a full accept) needs no write at
        # all. Only a rollback rewinds, and only that pays.
        #
        # Writing all 8 counters unconditionally cost 8 tiny device writes per
        # replay and dropped a 1400-token decode to 20.99 tok/s, BELOW the ~33 tok/s
        # autoregressive graph baseline -- a correct decoder that is not worth
        # running. _foreach_fill_ collapses the rewind into one fused multi-tensor
        # op, and the tracked position removes it from the fast path entirely.
        if self._counter_pos != start_pos and self._len_counters:
            # torch 2.13+rocm7.2 has no _foreach_fill_, so stage the value in one
            # pinned scalar and fan it out with a single fused _foreach_copy_:
            # 2 kernel launches instead of one per attention layer.
            self._ctr_src.fill_(start_pos)
            torch._foreach_copy_(self._len_counters, self._ctr_srcs)
        self._counter_pos = start_pos + width
        b["graph"].replay()
        return b["logits"], b["hidden"]

    # ---------------------------------------------------------------- generate
    @torch.no_grad()
    def stream_generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 64,
        stop_ids: set[int] | None = None,
        engine=None,
        expert=None,
        gate=None,
        supervisor: Any | None = None,
    ):
        """Yields chunks of decoded token IDs list[int] as each speculative chunk is verified."""
        assert self._locked, "capture() first"
        from runtime.cuda_graph import apply_expert_state

        k = self.k
        stop_ids = stop_ids or {self.tokenizer.eos_token_id}
        prompt_tokens = prompt_tokens.to(self.device)
        swap_ms = apply_expert_state(engine, expert)

        self.cache.reset()
        if self.ring_engine is not None:
            self.ring_engine.reset()
        self.circuit_breaker.reset()
        cur = prompt_tokens.shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model(
            prompt_tokens,
            past_key_values=self.cache,
            cache_position=torch.arange(0, cur, device=self.device),
            use_cache=True,
            output_hidden_states=True,
        )
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        self._counter_pos = cur  # reset() zeroed the counters; prefill advanced to cur
        hids = [out.hidden_states[-1]]
        seq = prompt_tokens
        toks = [nxt.item()]
        pos = cur
        st = {"steps": 0, "accepted": 0, "drafted": 0}
        done = toks[0] in stop_ids

        if supervisor is not None:
            supervisor.notify_token_emitted(toks[0])

        yield [toks[0]]

        # Build the initial draft head KV cache over the full prompt hidden states.
        # (one-time O(context) cost). After this, we extend it incrementally.
        # `full_hids` tracks all backbone hidden states so we can rebuild dcache on rollback.
        full_hids = hids[0]  # shape [1, cur, H]
        dcache = self.head.prefill(full_hids, seq)

        while len(toks) < max_new_tokens and not done:
            # Check Thinking Runtime Supervisor before drafting
            if supervisor is not None and getattr(supervisor, "in_thinking_mode", False):
                # Inspect last logits and hidden state
                last_logits = out.logits[:, -1, :] if len(toks) == 1 else logits[:, -1, :]
                sup_action = supervisor.process_step(last_logits, full_hids[:, -1:, :])
                if sup_action.should_force_transition:
                    for trans_tok in sup_action.transition_token_ids:
                        trans_t = torch.tensor([[trans_tok]], device=self.device)
                        toks.append(trans_tok)
                        yield [trans_tok]
                        logits, hidden = self._replay(1, trans_t, pos)
                        full_hids = torch.cat([full_hids, hidden[:, -1:, :]], dim=1)
                        self.ring_engine.commit_on_acceptance(1)
                        seq = torch.cat([seq, trans_t], dim=-1)
                        pos += 1
                        if supervisor is not None:
                            supervisor.notify_token_emitted(trans_tok)
                    nxt = torch.tensor([[sup_action.transition_token_ids[-1]]], device=self.device)
                    continue
            run_draft, k_to_use = self.circuit_breaker.should_draft()

            # DISENGAGED (Circuit-Breaker Tripped): Raw W=1 CUDA Graph decode
            if not run_draft:
                logits, hidden = self._replay(1, nxt, pos)
                nxt_val = torch.argmax(logits[0, -1, :], -1).item()
                committed = nxt
                full_hids = torch.cat([full_hids, hidden[:, -1:, :]], dim=1)
                self.ring_engine.commit_on_acceptance(1)
                self.circuit_breaker.update_acceptance(1.0)
                st["steps"] += 1

                toks.append(nxt_val)
                yield [nxt_val]
                if nxt_val in stop_ids:
                    done = True
                    break

                seq = torch.cat([seq, committed], dim=-1)
                pos += 1
                nxt = torch.tensor([[nxt_val]], device=self.device)
                continue

            # ENGAGED or PROBE STEP: Speculative drafting & verification
            k_eff = k_to_use if k_to_use > 0 else k
            draft = self.head.draft(full_hids[:, -1:, :], nxt, k=k_eff, start_pos=pos - 1, cache=dcache, gate=gate)
            k_actual = draft.shape[1]

            # Must capture the returned slot -- rollback_on_rejection needs the EXACT
            # slot this call wrote, not the n_accepted-based guess it falls back to
            # when slot is omitted. See rollback_on_rejection's docstring.
            ckpt_slot = self.ring_engine.checkpoint(self.cache)
            chunk = torch.cat([nxt, draft], dim=-1)
            logits, hidden = self._replay(k_actual + 1, chunk, pos)
            target = torch.argmax(logits[0], -1)

            n_acc = 0
            for i in range(k_actual):
                if draft[0, i].item() == target[i].item():
                    n_acc += 1
                else:
                    break
            st["steps"] += 1
            st["drafted"] += k_actual
            st["accepted"] += n_acc

            # Update MACD circuit breaker with this step's acceptance yield
            self.circuit_breaker.update_acceptance(float(n_acc + 1))

            if n_acc < k_actual:
                # ROLLBACK: restore the fixed-size SSM state, and simply rewind the
                # position for attention -- StaticCache KV beyond it is stale but
                # unread and will be overwritten. No crop, so no graph pointer moves.
                self.ring_engine.rollback_on_rejection(self.cache, n_acc, slot=ckpt_slot)
                committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
                _, hidden = self._replay(n_acc + 1, committed, pos)
                new_h = hidden
                # dcache has k_actual stale KV entries from the rejected draft tokens.
                # DynamicCache is append-only, so we can't truncate it. Rebuild from
                # the full committed hidden state history (full_hids ++ new_h committed slice).
                # This is O(current_len) but only happens on rejections, which are rare.
                new_full = torch.cat([full_hids, new_h], dim=1)
                committed_seq = torch.cat([seq, committed], dim=-1)
                dcache = self.head.prefill(new_full, committed_seq)
                full_hids = new_full
            else:
                new_h = hidden[:, : n_acc + 1, :]
                committed = chunk
                # All K+1 drafts accepted. draft() already wrote K new KV entries to dcache
                # for the draft positions. dcache is now up-to-date through position pos+k.
                # Just extend full_hids with the new committed hidden states.
                full_hids = torch.cat([full_hids, new_h], dim=1)

            self.ring_engine.commit_on_acceptance(n_acc + 1)

            bonus = target[n_acc].item()
            new_batch = []
            for t in draft[0, :n_acc].tolist() + [bonus]:
                if len(toks) >= max_new_tokens:
                    break
                toks.append(t)
                new_batch.append(t)
                if t in stop_ids:
                    done = True
                    break

            if new_batch:
                yield new_batch

            seq = torch.cat([seq, committed], dim=-1)
            pos += n_acc + 1
            nxt = torch.tensor([[bonus]], device=self.device)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        st["elapsed"] = elapsed
        st["tau"] = st["accepted"] / max(1, st["steps"])
        st["swap_ms"] = swap_ms
        st["tok_s"] = len(toks[:max_new_tokens]) / max(1e-9, elapsed)
        st["circuit_breaker"] = self.circuit_breaker.get_summary()
        self.last_stats = st

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 64,
        stop_ids: set[int] | None = None,
        engine=None,
        expert=None,
    ) -> tuple[list[int], float, dict]:
        """Speculative decode entirely on captured graphs."""
        toks: list[int] = []
        for batch in self.stream_generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            stop_ids=stop_ids,
            engine=engine,
            expert=expert,
        ):
            toks.extend(batch)
        st = getattr(self, "last_stats", {})
        return toks[:max_new_tokens], st.get("elapsed", 0.0), st
