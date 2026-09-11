"""Does a domain-adapted MTP head draft better? Measures ACCEPTANCE, not latency.

THE QUESTION
------------
`train_mtp_adapter.py` produced five adapters for the MTP draft head itself
(`mtp.fc`, `mtp.layer.self_attn.*`, `mtp.layer.mlp.*`) at 150 steps. Nothing has
ever loaded them: `mtp_draft.py` reads bare `mtp.*` tensors straight from the
checkpoint, and every speculative benchmark builds the head un-adapted. So they
sit on disk unevaluated.

Acceptance is the right target. Net speculative speedup is governed by tau
(accepted tokens per verification) against a break-even of ~1.39 on this rig,
and tau currently sits near 2.0. Draft LATENCY is the wrong target: drafting is
only ~27% of a speculative round (4 x 3.3ms against a 35.6ms verify), so even
eliminating it entirely is bounded at ~1.14x, while acceptance moves the lever
without that ceiling.

WHAT THE PRIOR RESULT DOES AND DOES NOT SAY
-------------------------------------------
`benchmark_mtp_folding_sweep.py` reported that adapting the head made it WORSE
(21.95% -> 12.20-14.63% top-1). That result does not transfer here, for three
reasons:

  * It trained its own throwaway adapters INLINE at **30 steps**
    (`train_quick_mtp_adapter`, steps=30). It never loaded the 150-step
    adapters on disk. Those are 5x more trained and have never been measured.
  * n = 41 tokens over 4 prompts. The whole sweep spans 4 tokens.
  * It measured single-token top-1 accuracy, not accepted prefix length.
    Speculation stops at the first mismatch, so accuracy and tau are different
    quantities.

It is still evidence pointing the wrong way, and it is the reason the
un-adapted baseline is measured here in the same process, on the same prompts,
rather than compared against a remembered number.

METHOD
------
For each condition (un-adapted, or one MTP adapter folded into the head):

  1. Generate truth tokens greedily FROM THE BACKBONE. The backbone is
     identical across conditions -- only the head changes -- so the truth
     sequence is shared and acceptance is scored against the same target.
  2. One forward over [prompt + truth] for hidden states.
  3. At several offsets, prime the head's cache and draft K tokens.
  4. Score the ACCEPTED PREFIX (stop at first mismatch), which is what
     speculation actually gets, and report tau = mean accepted prefix.

Folding uses W += scaling * (lora_a @ lora_b).T, matching NovelLoraLinear's
`delta = (x @ lora_a) @ lora_b * scaling` and nn.Linear's y = x @ W.T. Scaling
is read from each adapter's own config, never hardcoded -- hardcoding alpha/rank
is what produced the retracted speculation matrix. Weights are restored from a
pristine buffer between conditions (bit-exact copy_, not subtract-the-delta).

    uv run --env-file .env scripts/runtime/speculative/mtp_head_folding/benchmark_mtp_head_adapter_acceptance.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# fla's device probe is @cache'd at import; touch CUDA before transformers.
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.mtp_draft import Qwen35MTPDraftHead  # noqa: E402
from runtime.novel_peft import set_hard_vram_cap  # noqa: E402

ADAPTERS = [
    "results/adapters/mtp_astral_sweep_r64_a16",
    "results/adapters/mtp_astral_sweep_r64_a32",
    "results/adapters/mtp_astral_sweep_r64_a64",
    "results/adapters/mtp_astral_sweep_r64_a128",
    "results/adapters/mtp_astral_lora_r64_a64",
]
BREAK_EVEN_TAU = 1.39


def load_mtp_adapter(path: Path):
    """Return (module_path -> (lora_a, lora_b), scaling) for an MTP head adapter."""
    cfg = json.loads((path / "novel_adapter_config.json").read_text())
    state = torch.load(path / "novel_adapter.pt", map_location="cpu")
    scaling = cfg.get("scaling")
    if scaling is None:  # derive rather than assume; never hardcode alpha/rank
        scaling = float(cfg["alpha"]) / float(cfg["rank"])
    factors = {}
    for key, tensor in state.items():
        if key.endswith(".lora_a"):
            mod = key[: -len(".lora_a")]
            mod = mod[len("mtp.") :] if mod.startswith("mtp.") else mod
            factors.setdefault(mod, {})["a"] = tensor
        elif key.endswith(".lora_b"):
            mod = key[: -len(".lora_b")]
            mod = mod[len("mtp.") :] if mod.startswith("mtp.") else mod
            factors.setdefault(mod, {})["b"] = tensor
    out = {}
    for mod, ab in factors.items():
        if "a" in ab and "b" in ab:
            out[mod] = (ab["a"], ab["b"])
    return out, float(scaling), cfg


def resolve(head, dotted: str):
    obj = head
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


@torch.no_grad()
def fold_into_head(head, factors, scaling, pristine):
    """W += scaling * (lora_a @ lora_b).T, in-place, from pristine weights."""
    applied = 0
    for mod, (a, b) in factors.items():
        try:
            layer = resolve(head, mod)
        except AttributeError:
            print(f"    WARNING: head has no module '{mod}' -- skipped")
            continue
        w = layer.weight
        if mod not in pristine:
            pristine[mod] = w.detach().clone()
        delta = (a.to(w.device, torch.float32) @ b.to(w.device, torch.float32)).T
        if delta.shape != w.shape:
            print(f"    WARNING: {mod} delta {tuple(delta.shape)} != weight {tuple(w.shape)} -- skipped")
            continue
        w.copy_(pristine[mod])
        w.add_((scaling * delta).to(w.dtype))
        applied += 1
    return applied


@torch.no_grad()
def restore_head(head, pristine):
    for mod, w0 in pristine.items():
        resolve(head, mod).weight.copy_(w0)


@torch.no_grad()
def greedy_truth(model, ids, n_new):
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    for _ in range(n_new - 1):
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        toks.append(nxt.item())
    return toks


@torch.no_grad()
def accepted_prefixes(model, head, ids, truth, k, offsets):
    seq = torch.cat([ids, torch.tensor([truth], device=ids.device)], dim=1)
    hidden = model(seq, output_hidden_states=True, use_cache=False).hidden_states[-1]
    T = ids.shape[1]
    rows = []
    for off in offsets:
        t = T - 1 + off
        if t + 2 + k > seq.shape[1]:
            continue
        cache = head.prefill(hidden[:, : t + 1, :], seq[:, : t + 1])
        nxt = seq[:, t + 1 : t + 2]
        draft = head.draft(hidden[:, t : t + 1, :], nxt, k=k, start_pos=t, cache=cache)
        tgt = seq[0, t + 2 : t + 2 + k]
        acc = 0
        for i in range(k):
            if draft[0, i].item() == tgt[i].item():
                acc += 1
            else:
                break
        rows.append(acc)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--truth-tokens", type=int, default=40)
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 8, 16, 24])
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/mtp_head_adapter_acceptance.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    head = Qwen35MTPDraftHead(model, args.model_name)

    qs = [
        json.loads(x)["prompt"]
        for x in (REPO_ROOT / "data/astral/evaluation_data.jsonl").read_text().splitlines()
        if x.strip()
    ][: args.n_prompts]
    prompts = [f"### Question:\n{q}\n\n### Answer:\n" for q in qs]
    print(f"{len(prompts)} astral prompts x {len(args.offsets)} offsets = "
          f"{len(prompts) * len(args.offsets)} draft events per condition, K={args.k}\n")

    # Truth sequences come from the BACKBONE, which never changes across
    # conditions, so they are computed once and shared. This also removes truth
    # generation from the per-condition cost.
    print("Generating shared truth sequences from the backbone...")
    truths = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(model.device)
        truths.append((ids, greedy_truth(model, ids, args.truth_tokens)))

    pristine: dict[str, torch.Tensor] = {}
    results = {}
    conditions = [("unadapted", None)] + [(Path(a).name, a) for a in ADAPTERS if (REPO_ROOT / a).exists()]

    for name, rel in conditions:
        if rel is None:
            restore_head(head, pristine)
            scaling, cfg = None, {}
        else:
            factors, scaling, cfg = load_mtp_adapter(REPO_ROOT / rel)
            n = fold_into_head(head, factors, scaling, pristine)
            print(f"  folded {n} modules from {name} (scaling {scaling})")

        accs = []
        for ids, truth in truths:
            accs += accepted_prefixes(model, head, ids, truth, args.k, args.offsets)
        tau = sum(accs) / len(accs)
        results[name] = {
            "tau_mean_accepted": tau,
            "accept_rate_pct": 100 * sum(accs) / (len(accs) * args.k),
            "n_draft_events": len(accs),
            "scaling": scaling,
            "alpha": cfg.get("alpha"),
            "rank": cfg.get("rank"),
            "above_break_even": tau >= BREAK_EVEN_TAU,
            "per_event_accepted": accs,
        }
        print(f"  {name:32s} tau={tau:.3f}  accept={results[name]['accept_rate_pct']:.1f}%  n={len(accs)}")

    restore_head(head, pristine)

    base = results["unadapted"]["tau_mean_accepted"]
    print("\n" + "=" * 84)
    print(f" MTP head adaptation vs draft acceptance (K={args.k}, break-even tau={BREAK_EVEN_TAU})")
    print("=" * 84)
    print(f" {'condition':32s} {'scaling':>8s} {'tau':>7s} {'vs unadapted':>14s} {'verdict':>12s}")
    for name, r in results.items():
        d = "" if name == "unadapted" else f"{r['tau_mean_accepted'] - base:+8.3f}"
        sc = "-" if r["scaling"] is None else f"{r['scaling']:.2f}"
        v = "above" if r["above_break_even"] else "BELOW break-even"
        print(f" {name:32s} {sc:>8s} {r['tau_mean_accepted']:7.3f} {d:>14s} {v:>12s}")
    best = max(results, key=lambda n: results[n]["tau_mean_accepted"])
    print(f"\n best: {best} (tau={results[best]['tau_mean_accepted']:.3f})")
    if best == "unadapted":
        print(" -> adapting the MTP head does NOT improve acceptance on this workload.")
    else:
        print(f" -> adapting improves tau by {results[best]['tau_mean_accepted'] - base:+.3f}; "
              "re-run the end-to-end speculative benchmark to convert this into tok/s.")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"k": args.k, "offsets": args.offsets, "results": results}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
