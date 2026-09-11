r"""Activation feature matrices for the CPU statistical estimators.

Every estimator in this family (Vecchia banded precision, CLIME, GEE, copula tail
routing) consumes an [n_samples x n_features] real matrix. This module builds that
matrix two ways, and the difference between them is the single most important caveat
attached to any number they produce:

  1. `residual_stream_features` -- SURROGATE. Propagates a synthetic hidden state
     through the REAL low-rank factors of the trained v7 adapters, layer by layer:

         h_{l+1} = h_l + s * U_o V_o h_l + s * U_down V_down silu(U_up V_up h_l)

     The adapter weights are real. The base-model weights are NOT applied and the
     inputs are Gaussian, not tokens. So the cross-layer dependence it exhibits is
     produced by the residual recursion, which is exactly the structure being tested
     -- which means a "layers are locally Markov" result measured here is partly a
     property of the construction. It is a sanity fixture with real weights, not
     evidence about the base model.

  2. `capture_real_hidden_states` -- REAL, and CPU-only. Runs the base model forward
     on real domain prompts with `output_hidden_states=True`, on the CPU, and takes
     the true per-layer residual-stream deltas h_{l+1} - h_l. Slow (minutes) and
     memory-hungry (~8 GB in bf16), which is why it is opt-in -- but it is the arm
     that can actually falsify the banding premise, so the benchmarks that depend on
     cross-layer structure carry a `--real-activations` flag for it.

Neither path touches the GPU. Adapters load with map_location="cpu"; the base model
is loaded with no device_map and the device env vars are cleared by the caller's
`enforce_cpu_only()` before torch initialises anything.

FEATURE SCALE
-------------
Layer/head features are LOG magnitudes. Norms are positive and right-skewed; the
Gaussian estimators (Vecchia, CLIME, GEE) assume something roughly elliptical, and
handing them raw norms would measure skew rather than dependence. The copula
estimator is rank-based and therefore invariant to this choice, which is one of its
actual advantages -- so `expert_magnitudes` is returned untransformed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "load_experts",
    "domain_prompts",
    "residual_stream_features",
    "expert_magnitudes",
    "real_expert_response_magnitudes",
    "capture_real_hidden_states",
    "as_matrix",
    "feature_provenance",
    "CORPUS_DIRS",
]

# The financial adapter is m2_financial_* while its corpus lives in financial_planning/.
# Legacy, documented in runtime.canon -- normalised here rather than by renaming adapters.
CORPUS_DIRS: dict[str, str] = {
    "astral": "astral",
    "postgresql": "postgresql",
    "duckdb": "duckdb",
    "financial": "financial_planning",
    "python_modern": "python_modern",
    "python_web": "python_web",
}

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def load_experts(domains: list[str], version: str | None = None) -> dict[str, Any]:
    """Load trained experts onto the CPU. Torch is imported lazily and stays on CPU."""
    from runtime.canon import adapter_path
    from runtime.novel_peft import FoldableExpert

    experts: dict[str, Any] = {}
    for d in domains:
        experts[d] = FoldableExpert.from_dir(adapter_path(d, version=version), name=d)
    return experts


def domain_prompts(
    domains: list[str],
    n_per_domain: int = 8,
    seed: int = 42,
    source: str = "evaluation",
) -> list[dict[str, str]]:
    """Real domain text, deterministically sampled.

    source="evaluation"  held-out eval prompts (short: ~20-60 tokens each).
    source="training"    the v6 training corpus `text` field -- full question+answer
                         passages of several hundred tokens. Longer sequences are what an
                         activation-covariance estimate actually needs, and this text is
                         still unseen by the BASE model: only the adapters trained on it.

    Falls back to the disposition eval file for domains that have no evaluation_data.jsonl
    (python_modern, python_web), so all six domains are representable.
    """
    from runtime.canon import REPO_ROOT

    rng = np.random.default_rng(seed)
    out: list[dict[str, str]] = []
    for d in domains:
        base = REPO_ROOT / "data" / CORPUS_DIRS.get(d, d)
        if source == "training":
            candidates = [base / "training_data_v6.jsonl", base / "training_data_v4.jsonl"]
            field = "text"
        else:
            candidates = [base / "evaluation_data.jsonl", base / "evaluation_data_disposition.jsonl"]
            field = "prompt"

        corpus = next((c for c in candidates if c.exists()), None)
        if corpus is None:
            continue
        rows = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r.get(field)]
        if not rows:
            continue
        pick = rng.choice(len(rows), size=min(n_per_domain, len(rows)), replace=False)
        out.extend(
            {"domain": d, "prompt": rows[int(i)][field], "corpus": corpus.name} for i in pick
        )
    return out


def _layer_keys(expert: Any) -> dict[int, dict[str, str]]:
    """Group an expert's module keys by layer index and projection name."""
    grouped: dict[int, dict[str, str]] = {}
    for key in expert.factors:
        m = _LAYER_RE.search(key)
        if not m:
            continue
        layer = int(m.group(1))
        proj = key.split("layers.")[1].split(".", 1)[1].replace(".weight", "")
        grouped.setdefault(layer, {})[proj] = key
    return grouped


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -30, 30)))


def residual_stream_features(
    expert: Any,
    n_samples: int = 512,
    seed: int = 42,
    head_dim: int = 256,
) -> dict[str, Any]:
    """Propagate a synthetic residual stream through one expert's real factors.

    The v7 adapters carry SwiGLU MLP factors on all 32 layers and attention factors on
    every fourth layer only, so the attention branch is applied where it exists rather
    than assumed uniform -- the layer-to-layer structure this produces reflects the
    real adapter placement.

    Returns log-magnitude features:
      layer_delta    [n_samples, L]      -- per-layer contribution norm, in layer order
      head_slice     [n_samples, A*heads] -- per-head response on every attention layer
      total_response [n_samples]          -- summed contribution norm, RAW (copula input)
    """
    rng = np.random.default_rng(seed)
    grouped = _layer_keys(expert)
    layers = sorted(grouped)
    if not layers:
        raise ValueError(f"expert {getattr(expert, 'name', '?')} has no layer-indexed modules")

    scale = float(expert.scaling)

    def factor(layer: int, proj: str) -> tuple[np.ndarray, np.ndarray] | None:
        key = grouped[layer].get(proj)
        if key is None:
            return None
        u, v = expert.factors[key]
        # .cpu() is not defensive padding: FoldableExpert loads PEFT-format adapters with
        # load_peft_weights, which puts them on the default device. Under --gpu-capture the
        # device is visible, so the factors land in VRAM and .numpy() raises. The surrogate
        # is a numpy computation either way.
        return (
            u.detach().cpu().numpy().astype(np.float64),
            v.detach().cpu().numpy().astype(np.float64),
        )

    # d_model must come from a module whose INPUT is the residual stream. Two traps
    # here, both hit while building this: down_proj's input is the 9216-wide MLP
    # interior, and o_proj's input is the 4096-wide concatenated head space
    # (32 heads x 128), not the 2560-wide stream. Reading either would silently size
    # the stream wrong.
    d_model = 0
    for layer in layers:
        for proj in ("mlp.gate_proj", "mlp.up_proj", "self_attn.q_proj"):
            f = factor(layer, proj)
            if f is not None:
                d_model = int(f[1].shape[1])
                break
        if d_model:
            break
    if not d_model:
        raise ValueError("could not determine d_model from any residual-input projection")

    attn_layers = [
        ell for ell in layers
        if "self_attn.o_proj" in grouped[ell] and "self_attn.v_proj" in grouped[ell]
    ]

    h = rng.standard_normal((n_samples, d_model)) / np.sqrt(d_model)
    deltas = np.zeros((n_samples, len(layers)))
    head_blocks: list[np.ndarray] = []
    head_names: list[str] = []

    for pos, layer in enumerate(layers):
        contribution = np.zeros_like(h)

        # Attention branch, through the model's real GQA geometry:
        #   h [2560] --v_proj--> [kv_heads*head_dim] --GQA repeat--> [heads*head_dim]
        #                        --o_proj--> [2560]
        # The intermediate is the concatenated per-head space, so slicing it by head_dim
        # gives per-head activation magnitudes on the model's real head partition.
        # NOT simulated: the softmax mixing across positions. These samples are
        # independent tokens, so there is no sequence for attention to mix.
        vp = factor(layer, "self_attn.v_proj")
        o = factor(layer, "self_attn.o_proj")
        if vp is not None and o is not None:
            u_v, v_v = vp
            u_o, v_o = o
            kv = scale * ((h @ v_v.T) @ u_v.T)              # [n, kv_heads * head_dim]
            o_in = int(v_o.shape[1])
            repeat = max(o_in // max(kv.shape[1], 1), 1)
            a = np.repeat(kv.reshape(n_samples, -1, head_dim), repeat, axis=1)
            a = a.reshape(n_samples, -1)[:, :o_in]          # [n, heads * head_dim]
            contribution += scale * ((a @ v_o.T) @ u_o.T)
            per_head = np.linalg.norm(a.reshape(n_samples, o_in // head_dim, head_dim), axis=2)
            head_blocks.append(per_head)
            head_names.extend(f"L{layer}.h{hh:02d}" for hh in range(per_head.shape[1]))

        gate = factor(layer, "mlp.gate_proj")
        up = factor(layer, "mlp.up_proj")
        down = factor(layer, "mlp.down_proj")
        if gate is not None and up is not None and down is not None:
            u_g, v_g = gate
            u_u, v_u = up
            u_d, v_d = down
            g = scale * ((h @ v_g.T) @ u_g.T)
            z = _silu(g) * (scale * ((h @ v_u.T) @ u_u.T))
            contribution += scale * ((z @ v_d.T) @ u_d.T)

        deltas[:, pos] = np.linalg.norm(contribution, axis=1)
        h = h + contribution

    out: dict[str, Any] = {
        "layer_delta": np.log(np.maximum(deltas, 1e-30)),
        "total_response": deltas.sum(axis=1),
        "layers": layers,
        "attention_layers": attn_layers,
        "d_model": d_model,
        "source": "residual_stream_surrogate",
        "expert": getattr(expert, "name", "?"),
        "head_dim": head_dim,
    }
    if head_blocks:
        out["head_slice"] = np.log(np.maximum(np.hstack(head_blocks), 1e-30))
        out["head_names"] = head_names
    return out


def expert_magnitudes(
    experts: dict[str, Any],
    n_samples: int = 1024,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Per-expert response magnitude on ONE shared synthetic token stream. [n, n_experts].

    Untransformed on purpose: this feeds the rank-based copula router, which does not
    care about the marginal shape and should not be handed a pre-Gaussianised input.
    Every expert starts from the same seeded stream, so the columns are comparable.
    """
    names = sorted(experts)
    cols = [
        residual_stream_features(experts[name], n_samples=n_samples, seed=seed)["total_response"]
        for name in names
    ]
    return np.column_stack(cols), names


def real_expert_response_magnitudes(
    prompts: list[dict[str, str]],
    experts: dict[str, Any],
    model_id: str | None = None,
    max_length: int = 384,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    threads: int = 8,
    cache_path: str | None = None,
) -> dict[str, Any]:
    r"""Per-expert response magnitude on REAL tokens. The routing signal, measured exactly.

    For each module an adapter targets, this hooks the module's TRUE input during a base
    model forward and computes what that expert's delta would output on it:

        dW = s * U @ V,   module computes x @ W^T,   so delta_out = s * (x @ V^T) @ U^T

    The per-token magnitude is then accumulated across all 128 targeted modules. This is
    the quantity a router would score experts on: "how strongly would this expert respond
    to this token?"

    WHY THIS IS NOT THE SURROGATE
    -----------------------------
    `expert_magnitudes` drives the same factors with GAUSSIAN inputs, so the only structure
    it can show is whatever the factors impose. Here `x` is the real activation the base
    model produced for a real token, so the co-activation structure between experts is a
    property of the model and the prompt distribution -- which is the thing a routing
    decision actually depends on.

    EXACT, AND CHEAP
    ----------------
    ||delta_out_t||^2 = s^2 * z_t' (U'U) z_t  with z = x @ V^T, so the [T, d_out] output is
    never materialised: with G = U'U (r x r), the per-token norm costs O(T r^2) instead of
    O(T r d_out). This is an identity, not an approximation.

    WHAT IT IS NOT
    --------------
    A first-order response: the delta is evaluated on the BASE model's activations, not on
    activations produced by an already-folded expert. That is the correct signal for a
    routing decision made before folding, and it is not a measurement of a folded forward
    pass.
    """
    import numpy as _np

    if cache_path is not None:
        cached = Path(cache_path)
        if cached.exists():
            z = _np.load(cached, allow_pickle=True)
            return {
                "magnitudes": z["magnitudes"],
                "expert_names": [str(x) for x in z["expert_names"]],
                "token_domains": [str(x) for x in z["token_domains"]],
                "source": "real_expert_response",
                "model_id": str(z["model_id"]),
                "n_prompts": int(z["n_prompts"]),
                "n_tokens": int(z["n_tokens"]),
                "device": str(z["device"]),
                "from_cache": True,
            }

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from runtime.canon import CANON

    torch.set_num_threads(threads)
    model_id = model_id or CANON.BASE_MODEL
    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[dtype]

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, device_map=None)
    model.eval()
    if device != "cpu":
        model.to(device)

    names = sorted(experts)
    # per module: list of (expert_index, V [r, d_in], G = U'U [r, r], scaling)
    by_module: dict[str, list[tuple[int, Any, Any, float]]] = {}
    for ei, name in enumerate(names):
        exp = experts[name]
        scale = float(exp.scaling)
        for key, (u, v) in exp.factors.items():
            module_path = key[: -len(".weight")] if key.endswith(".weight") else key
            u32 = u.detach().to(device=device, dtype=torch.float32)
            v32 = v.detach().to(device=device, dtype=torch.float32)
            by_module.setdefault(module_path, []).append((ei, v32, u32.T @ u32, scale))

    acc: dict[str, Any] = {"buf": None}
    handles = []

    def make_hook(module_path: str):
        entries = by_module[module_path]

        def hook(_module, args, _kwargs=None):
            x = args[0] if isinstance(args, tuple) else args
            if not isinstance(x, torch.Tensor):
                return
            xf = x.detach()[0].float() if x.dim() == 3 else x.detach().float()
            if acc["buf"] is None or acc["buf"].shape[0] != xf.shape[0]:
                acc["buf"] = torch.zeros(xf.shape[0], len(names), device=xf.device, dtype=torch.float32)
            for ei, v32, g32, scale in entries:
                z = xf @ v32.T                                   # [T, r]
                q = torch.einsum("ta,ab,tb->t", z, g32, z).clamp_min(0.0)
                acc["buf"][:, ei] += scale * torch.sqrt(q)
        return hook

    matched = 0
    for module_name, module in model.named_modules():
        if module_name in by_module:
            handles.append(module.register_forward_pre_hook(make_hook(module_name)))
            matched += 1
    if matched == 0:
        for h in handles:
            h.remove()
        raise RuntimeError("no adapter-targeted modules matched the model; check key prefixes")

    rows: list[np.ndarray] = []
    token_domains: list[str] = []
    try:
        with torch.no_grad():
            for item in prompts:
                acc["buf"] = None
                enc = tok(item["prompt"], return_tensors="pt", truncation=True, max_length=max_length)
                enc = {k: t.to(device) for k, t in enc.items()}
                model(**enc)
                buf = acc["buf"]
                if buf is None or buf.shape[0] < 2:
                    continue
                rows.append(buf[1:].cpu().numpy())          # drop position 0
                token_domains.extend([item.get("domain", "?")] * (buf.shape[0] - 1))
    finally:
        for h in handles:
            h.remove()
        del model
        if device != "cpu":
            torch.cuda.empty_cache()

    magnitudes = np.vstack(rows).astype(np.float64)
    result = {
        "magnitudes": magnitudes,
        "expert_names": names,
        "token_domains": token_domains,
        "source": "real_expert_response",
        "model_id": model_id,
        "n_prompts": len(prompts),
        "n_tokens": int(magnitudes.shape[0]),
        "n_modules_hooked": matched,
        "device": device,
        "from_cache": False,
    }
    if cache_path is not None:
        cached = Path(cache_path)
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cached,
            magnitudes=magnitudes, expert_names=np.array(names),
            token_domains=np.array(token_domains), model_id=model_id,
            n_prompts=len(prompts), n_tokens=result["n_tokens"], device=device,
        )
    return result


def capture_real_hidden_states(
    prompts: list[str],
    model_id: str | None = None,
    max_length: int = 96,
    dtype: str = "bfloat16",
    threads: int = 8,
    head_dim: int | None = None,
    cache_path: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Real per-layer residual deltas and per-head magnitudes from a CPU forward pass.

    `device="cpu"` is the default and is safe while something else holds the GPU: the
    model is loaded with `device_map=None` while the device env vars are cleared, so no
    HIP context is ever created. `device="cuda:0"` runs the same capture on the GPU --
    roughly 20x faster, which is what makes a large real-token sample affordable. The
    features are identical either way up to bf16 reduction order; only the wall time
    differs.

    One row per token position (position 0 is dropped -- its residual delta is the
    embedding's own first update and is not comparable to the rest).

      layer_delta[t, l] = log || h_{l+1}[t] - h_l[t] ||   the real contribution of
                                                          layer l at token t
      head_slice[t, (l,h)] = log || o_proj input slice ||  captured by forward pre-hook
                                                          on each layer's o_proj, which
                                                          IS the concatenated per-head
                                                          attention output

    `cache_path` stores the result as .npz so repeat benchmark runs skip the forward pass.
    """
    import numpy as _np

    if cache_path is not None:
        cached = Path(cache_path)
        if cached.exists():
            z = _np.load(cached, allow_pickle=True)
            return {
                "layer_delta": z["layer_delta"],
                "head_slice": z["head_slice"],
                "head_names": [str(x) for x in z["head_names"]],
                "layers": [int(x) for x in z["layers"]],
                "source": "real_forward",
                "model_id": str(z["model_id"]),
                "dtype": str(z["dtype"]),
                "n_prompts": int(z["n_prompts"]),
                "n_tokens": int(z["n_tokens"]),
                "head_dim": int(z["head_dim"]),
                "device": str(z["device"]) if "device" in z else "unknown",
                "from_cache": True,
            }

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from runtime.canon import CANON

    torch.set_num_threads(threads)
    model_id = model_id or CANON.BASE_MODEL
    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[dtype]

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, device_map=None)
    model.eval()
    if device != "cpu":
        model.to(device)

    cfg = model.config
    inner = getattr(cfg, "text_config", cfg)
    if head_dim is None:
        head_dim = int(getattr(inner, "head_dim", 0) or 0)

    # Forward pre-hooks on o_proj: its INPUT is the concatenated per-head attention
    # output, the only place the head partition is directly observable.
    captured: dict[int, torch.Tensor] = {}
    handles = []
    o_proj_layers: list[int] = []

    def make_hook(idx: int):
        def hook(_module, args, _kwargs=None):
            x = args[0] if isinstance(args, tuple) else args
            if isinstance(x, torch.Tensor):
                captured[idx] = x.detach()[0].float().cpu()
        return hook

    decoder_layers = model.model.layers if hasattr(model.model, "layers") else []
    for idx, layer in enumerate(decoder_layers):
        o_proj = getattr(getattr(layer, "self_attn", None), "o_proj", None)
        if o_proj is not None:
            handles.append(o_proj.register_forward_pre_hook(make_hook(idx)))
            o_proj_layers.append(idx)

    layer_rows: list[np.ndarray] = []
    head_rows: list[np.ndarray] = []
    head_names: list[str] = []
    token_counts: list[int] = []

    try:
        with torch.no_grad():
            for prompt in prompts:
                captured.clear()
                enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc, output_hidden_states=True)
                hs = out.hidden_states                       # tuple length L+1, each [1, T, d]
                n_layers = len(hs) - 1

                deltas = [
                    torch.linalg.norm((hs[ell + 1] - hs[ell])[0].float(), dim=-1).cpu()
                    for ell in range(n_layers)
                ]
                layer_rows.append(torch.stack(deltas, dim=1)[1:].numpy())    # [T-1, L]

                blocks = []
                names: list[str] = []
                for idx in o_proj_layers:
                    x = captured.get(idx)
                    if x is None:
                        continue
                    hd = head_dim or x.shape[-1]
                    n_heads = max(x.shape[-1] // hd, 1)
                    per_head = torch.linalg.norm(x[1:, : n_heads * hd].reshape(-1, n_heads, hd), dim=-1)
                    blocks.append(per_head.numpy())
                    names.extend(f"L{idx}.h{hh:02d}" for hh in range(n_heads))
                if blocks:
                    head_rows.append(np.hstack(blocks))
                    head_names = names
                token_counts.append(int(enc["input_ids"].shape[1]))
    finally:
        for h in handles:
            h.remove()
        del model
        if device != "cpu":
            torch.cuda.empty_cache()

    layer_delta = np.log(np.maximum(np.vstack(layer_rows).astype(np.float64), 1e-30))
    head_slice = (
        np.log(np.maximum(np.vstack(head_rows).astype(np.float64), 1e-30)) if head_rows else np.zeros((0, 0))
    )

    result = {
        "layer_delta": layer_delta,
        "head_slice": head_slice,
        "head_names": head_names,
        "layers": list(range(layer_delta.shape[1])),
        "source": "real_forward",
        "model_id": model_id,
        "dtype": dtype,
        "device": device,
        "n_prompts": len(prompts),
        "n_tokens": int(layer_delta.shape[0]),
        "head_dim": int(head_dim or 0),
        "mean_prompt_tokens": float(np.mean(token_counts)) if token_counts else 0.0,
        "from_cache": False,
    }
    if cache_path is not None:
        cached = Path(cache_path)
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cached,
            layer_delta=layer_delta, head_slice=head_slice,
            head_names=np.array(head_names), layers=np.array(result["layers"]),
            model_id=model_id, dtype=dtype, n_prompts=len(prompts),
            n_tokens=result["n_tokens"], head_dim=result["head_dim"], device=device,
        )
    return result


def as_matrix(features: dict[str, Any], key: str = "layer_delta") -> np.ndarray:
    """Standardised feature matrix: zero mean, unit variance per column."""
    X = np.asarray(features[key], dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return X / np.maximum(sd, 1e-12)


def feature_provenance(features: dict[str, Any]) -> dict[str, Any]:
    """The provenance block every artifact must carry, so a surrogate is never read as real."""
    keys = ("source", "expert", "model_id", "dtype", "n_prompts", "n_tokens", "head_layer", "d_model")
    return {k: features[k] for k in keys if k in features}
