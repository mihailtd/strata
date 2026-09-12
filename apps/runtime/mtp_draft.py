"""EAGLE-style speculative decoding using Qwen3.5's SHIPPED MTP draft head.

WHY NOT TRAIN A HEAD
--------------------
The obvious plan is to train a 1-layer auxiliary head on Layer-18 hidden states
and distil it against the base model. That is unnecessary here: the Qwen3.5-4B
checkpoint already ships a trained multi-token-prediction head (15 tensors under
`mtp.`, `mtp_num_hidden_layers: 1` in the config), and its architecture is
exactly EAGLE:

    mtp.pre_fc_norm_hidden / pre_fc_norm_embedding   norms on each input
    mtp.fc            (2560, 5120)                   fuse(hidden, next-token embed)
    mtp.layers.0                                     ONE full-attention block
    mtp.norm                                         final norm -> tied lm_head

`transformers` never loads these weights (no MTP class in the Qwen3.5 modeling
code), so this module loads them itself. A head Qwen trained on their full
corpus beats anything distilled here in ten minutes on 5-10k sequences.

Two corrections to the "Layer-18" framing:

  * Layer 18 is `linear_attention` -- a GatedDeltaNet RECURRENT layer, not a
    "non-recurrent feed-forward boundary". The draft-side argument survives
    anyway (a forward hook READS a hidden state without stepping any state),
    but not for the stated reason. The MTP head taps the final hidden state,
    which is what it was trained against.
  * `mtp.layers.0` is full attention, so the DRAFT path has an ordinary KV
    cache and no recurrent state. Draft rollback is free. The recurrent state
    only ever advances during verification.

MEASURED FEASIBILITY ON THIS RIG (RX 7900 XTX, gfx1100, ROCm 7.2)
-----------------------------------------------------------------
Speculation pays only if verifying K tokens costs about what 1 token costs.
Measured on this model:

    K=1   33.56 ms   1.00x
    K=2   95.44 ms   2.84x     <-- jump
    K=4   94.33 ms   2.81x
    K=8   91.25 ms   2.72x

Cost jumps 2.84x from 1 to 2 tokens, then is flat. The GatedDeltaNet layers
switch from a fast single-token recurrent step to a chunked multi-token path
which, without `fla`/`causal_conv1d` (not buildable on this AMD rig), falls back
to slow PyTorch. **Break-even is ~2.8 accepted tokens per verification.** Best
case at K=8 with perfect acceptance is 8/2.72 = 2.94x; at a realistic 65%
per-token acceptance the expected yield is ~2.2 tokens = 0.8x, i.e. SLOWER than
plain decode. The claimed 1.5-2.0x assumed verification is nearly free, which is
not true here.

This is implemented and benchmarked anyway, because acceptance rate is an
empirical question and the code becomes worthwhile the moment fast linear-
attention kernels are available.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import torch
from torch import nn


def _snapshot_dir(model_id: str = "Qwen/Qwen3.5-4B") -> str:
    pat = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*")
    hits = glob.glob(pat)
    if not hits:
        raise FileNotFoundError(f"No local snapshot for {model_id}; download it first.")
    return hits[0]


def load_mtp_weights(model_id: str = "Qwen/Qwen3.5-4B") -> dict[str, torch.Tensor]:
    """Read the `mtp.*` tensors straight out of the sharded checkpoint."""
    from safetensors import safe_open

    snap = _snapshot_dir(model_id)
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        idx = json.load(f)["weight_map"]
    wanted = {k: v for k, v in idx.items() if k.startswith("mtp.")}
    if not wanted:
        raise RuntimeError(f"{model_id} ships no mtp.* weights; this head cannot be built.")

    out: dict[str, torch.Tensor] = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(os.path.join(snap, shard), framework="pt") as f:
            for k, s in wanted.items():
                if s == shard:
                    out[k[len("mtp.") :]] = f.get_tensor(k)
    return out


class IncompatibleDraftHeadError(ValueError):
    """Raised when a draft head does not match the base model architecture or dimension."""

    pass


class Qwen35MTPDraftHead(nn.Module):
    """The shipped MTP head, wired to draft tokens from a hidden state.

    One draft step is:
        e      = embed(next_token)
        fused  = fc([norm_h(h) ; norm_e(e)])
        h'     = norm(decoder_layer(fused))
        logits = lm_head(h')
    and h' feeds the next draft step, so drafting is autoregressive in the
    head's own feature space (2560 on 4B, 4096 on 9B) and never touches the base model.
    """

    def __init__(self, model: nn.Module, model_id: str = "Qwen/Qwen3.5-4B"):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
        )

        cfg = model.config.get_text_config()
        h = cfg.hidden_size
        self.hidden_size = h
        self.target_model_id = model_id
        self.target_hidden_size = h

        # The head's single block must be full attention; build a config whose
        # layer 0 is typed that way rather than inheriting the base model's
        # (layer 0 there is linear_attention).
        head_cfg = type(cfg).from_dict(cfg.to_dict())
        head_cfg.layer_types = ["full_attention"]
        head_cfg.num_hidden_layers = 1

        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        # Without this the standalone layer dispatches attention with
        # _attn_implementation=None and warns; pin it to the base model's.
        head_cfg._attn_implementation = getattr(cfg, "_attn_implementation", "sdpa") or "sdpa"
        self.layer = Qwen3_5DecoderLayer(head_cfg, layer_idx=0)
        self.norm = Qwen3_5RMSNorm(h, eps=cfg.rms_norm_eps)

        sd = load_mtp_weights(model_id)
        remapped = {k.replace("layers.0.", "layer."): v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if unexpected:
            raise RuntimeError(f"MTP weights not consumed by the head: {sorted(unexpected)[:6]}")
        if missing:
            raise RuntimeError(f"MTP head has uninitialised params: {sorted(missing)[:6]}")

        # Reuse the base model's tied embedding / output projection.
        self._embed = model.get_input_embeddings()
        self._lm_head = model.get_output_embeddings()
        # rotary lives on the text model, whose path differs between the
        # multimodal wrapper and the plain text model
        inner = model.model
        self._rotary = getattr(inner, "rotary_emb", None) or inner.language_model.rotary_emb

        dev = next(model.parameters()).device
        dt = next(model.parameters()).dtype
        self.to(device=dev, dtype=dt)

    def _fuse(self, h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        """fc expects [embedding ; hidden], NOT [hidden ; embedding].

        Determined empirically -- the order is not recoverable from shapes since
        both halves are 2560. With [hidden ; embedding] the head degenerates to
        echoing the token it was given (0% next-next-token accuracy); with
        [embedding ; hidden] it reaches 70.0% against the base model's own 76.2%
        next-token ceiling on the same text.
        """
        e = self._embed(tok)
        return self.fc(torch.cat([self.pre_fc_norm_embedding(e), self.pre_fc_norm_hidden(h)], dim=-1))

    def _run_layer(self, fused: torch.Tensor, positions: torch.Tensor, cache) -> torch.Tensor:
        pos_ids = positions.unsqueeze(0)  # (1, T) -- what the layer wants
        # Qwen3.5 is multimodal and its rotary takes mRoPE ids of shape
        # (3, B, T), not (B, T). The text model builds (4, B, T), uses [0] for
        # masks and passes [1:] to rotary. Feeding 2D ids here silently yields
        # wrong rotary embeddings and the head drafts gibberish.
        rope_ids = pos_ids.unsqueeze(0).expand(3, -1, -1)
        pe = self._rotary(fused, rope_ids)
        out = self.layer(
            fused,
            position_embeddings=pe,
            attention_mask=None,
            position_ids=pos_ids,
            past_key_values=cache,
            cache_position=positions,
        )
        return self.norm(out)

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        cache=None,
        position_ids: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Forward pass for training and sequence evaluation.

        `hidden`: (B, T, H) hidden states from base model backbone.
        `input_ids`: (B, T) next token ids to fuse with hidden states.
        Returns: (logits (B, T, V), cache) or (logits, norm_out, cache) if return_hidden=True.
        """
        B, T, _ = hidden.shape
        fused = self._fuse(hidden, input_ids)  # (B, T, H)
        positions = torch.arange(T, device=fused.device) if position_ids is None else position_ids

        norm_out = self._run_layer(fused, positions, cache)
        logits = self._lm_head(norm_out)
        if return_hidden:
            return logits, norm_out, cache
        return logits, cache

    @torch.no_grad()
    def prefill(self, hidden: torch.Tensor, input_ids: torch.Tensor):
        """Build the head's KV cache over the prompt, mirroring the main sequence.

        This is load-bearing. The head's single attention block was trained to
        attend over the whole sequence; drafting from an EMPTY cache gives it one
        token of context and it degenerates (measured: 0/6 tokens matched,
        output " loss loss loss"). EAGLE-style heads keep their own KV cache
        alongside the target model's.

        `hidden` is (1, T, H) hidden states for x_0..x_{T-1}; position t fuses
        h_t with embed(x_{t+1}), so the cache covers t = 0..T-2.
        """
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        t = input_ids.shape[1]
        if t < 2:
            return cache
        fused = self._fuse(hidden[:, : t - 1, :], input_ids[:, 1:t])
        positions = torch.arange(t - 1, device=fused.device)
        self._run_layer(fused, positions, cache)
        return cache

    @torch.no_grad()
    def draft(
        self,
        hidden: torch.Tensor,
        next_token: torch.Tensor,
        k: int = 4,
        start_pos: int = 0,
        cache=None,
        gate=None,
    ) -> torch.Tensor:
        """Autoregressively draft up to K tokens using the MTP head.

        The base model is never called here, so no recurrent state advances --
        which is the whole point on a hybrid architecture.

        If `gate` is supplied (RangeStatisticGate), evaluates single-pass candidate range
        spread on each step. If uncertainty exceeds threshold, aborts early to avoid junk drafts.
        """
        from transformers.cache_utils import DynamicCache

        if cache is None:
            cache = DynamicCache()
        drafted = []
        h, tok = hidden, next_token
        for i in range(k):
            fused = self._fuse(h, tok)
            positions = torch.arange(start_pos + i, start_pos + i + 1, device=fused.device)
            h = self._run_layer(fused, positions, cache)
            logits = self._lm_head(h)[:, -1, :]

            # Single-Pass Range Statistic & Weibull Hazard Gate (Chapters 3 & 8)
            if gate is not None and i > 0:
                should_abort, *_ = gate.should_early_exit(logits, step_idx=i)
                if should_abort:
                    break

            tok = torch.argmax(logits, dim=-1, keepdim=True)
            drafted.append(tok)

        if not drafted:
            drafted.append(tok)
        return torch.cat(drafted, dim=-1)


def snapshot_state(cache) -> list[dict] | int:
    """Deep-copy or ring-buffer checkpoint the state of every layer in a hybrid cache.

    If `cache` has `_ring_buffer` attached, delegates to zero-allocation in-place
    slot push (< 2 µs). Otherwise falls back to deep-cloning.
    """
    if hasattr(cache, "_ring_buffer"):
        return cache._ring_buffer.push(cache)

    snap = []
    for layer in cache.layers:
        entry = {}
        for name, val in layer.__dict__.items():
            if isinstance(val, torch.Tensor):
                entry[name] = val.clone()
            elif isinstance(val, list) and all(isinstance(x, torch.Tensor) for x in val):
                entry[name] = [x.clone() for x in val]
            elif isinstance(val, dict) and any(isinstance(v, torch.Tensor) for v in val.values()):
                entry[name] = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in val.items()}
        snap.append(entry)
    return snap


def restore_state(cache, snap: list[dict] | int) -> None:
    """Restore a snapshot taken by `snapshot_state`, in place."""
    if hasattr(cache, "_ring_buffer") and isinstance(snap, int):
        cache._ring_buffer.rollback(cache, slot=snap)
        return

    for layer, entry in zip(cache.layers, snap, strict=True):
        for name, val in entry.items():
            cur = getattr(layer, name, None)
            if isinstance(val, torch.Tensor) and isinstance(cur, torch.Tensor) and cur.shape == val.shape:
                cur.copy_(val)
            elif isinstance(val, list) and isinstance(cur, list) and len(val) == len(cur):
                for v, c in zip(val, cur):
                    if isinstance(v, torch.Tensor) and isinstance(c, torch.Tensor) and c.shape == v.shape:
                        c.copy_(v)
            elif isinstance(val, dict) and isinstance(cur, dict):
                for k, v in val.items():
                    if isinstance(v, torch.Tensor) and isinstance(cur.get(k), torch.Tensor):
                        cur[k].copy_(v)
                    else:
                        cur[k] = v
            else:
                setattr(layer, name, val)


def attach_state_ring_buffer(cache, max_depth: int = 8):
    """Attach a pre-allocated StateRingBuffer to a hybrid cache for zero-allocation snapshots."""
    from runtime.state_ring_buffer import StateRingBuffer

    ring = StateRingBuffer(cache, max_depth=max_depth)
    cache._ring_buffer = ring
    return ring


def state_nbytes(cache) -> int:
    total = 0
    for layer in cache.layers:
        for val in layer.__dict__.values():
            if isinstance(val, torch.Tensor):
                total += val.numel() * val.element_size()
            elif isinstance(val, list):
                for v in val:
                    if isinstance(v, torch.Tensor):
                        total += v.numel() * v.element_size()
            elif isinstance(val, dict):
                for v in val.values():
                    if isinstance(v, torch.Tensor):
                        total += v.numel() * v.element_size()
    return total


def mtp_adapter_path(domain: str, version: str = "v7") -> Path:
    """Canonical path for domain-adapted MTP micro-adapter."""
    from runtime.canon import REPO_ROOT

    return REPO_ROOT / "results" / "adapters" / f"mtp_{domain}_r64_a64_{version}"


def fold_mtp_adapter(
    head: Qwen35MTPDraftHead,
    adapter_dir: Path | str,
    pristine: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Folds an MTP micro-adapter directly into head projection weights in-place.

    Returns the pristine weight dictionary for zero-copy rollback and instant expert swapping.
    """
    adapter_dir = Path(adapter_dir)
    cfg_file = adapter_dir / "novel_adapter_config.json"
    weights_file = adapter_dir / "novel_adapter.pt"
    if not cfg_file.exists() or not weights_file.exists():
        raise FileNotFoundError(f"Missing MTP adapter files in {adapter_dir}")

    cfg = json.loads(cfg_file.read_text())
    state = torch.load(weights_file, map_location="cpu")
    scaling = cfg.get("scaling", cfg.get("alpha", 64.0) / cfg.get("rank", 64.0))

    # Save pristine baseline weights if not already provided
    if pristine is None:
        pristine = {}
        for name, param in head.named_parameters():
            if not name.startswith("_"):
                pristine[name] = param.detach().clone()
    else:
        # Restore pristine baseline before folding new adapter
        param_map = dict(head.named_parameters())
        for name, p_data in pristine.items():
            if name in param_map:
                param_map[name].data.copy_(p_data)

    param_dict = dict(head.named_parameters())
    for mod_name in cfg.get("modules", []):
        # mod_name format: "mtp.fc" or "mtp.layer.self_attn.q_proj"
        clean_name = mod_name.replace("mtp.", "") + ".weight"
        a_key = f"{mod_name}.lora_a"
        b_key = f"{mod_name}.lora_b"
        if a_key in state and b_key in state and clean_name in param_dict:
            target_param = param_dict[clean_name]
            a = state[a_key].to(device=target_param.device, dtype=target_param.dtype)
            b = state[b_key].to(device=target_param.device, dtype=target_param.dtype)
            # a is (in, r), b is (r, out) -> b.T is (out, r), a.T is (r, in)
            target_param.data.addmm_(b.T, a.T, alpha=scaling)

    return pristine
