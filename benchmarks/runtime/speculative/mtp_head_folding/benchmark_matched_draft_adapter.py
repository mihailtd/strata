"""Does folding the SAME adapter into the draft head raise acceptance?

WHY THIS IS NOT THE EXPERIMENT THAT ALREADY FAILED
--------------------------------------------------
`benchmark_mtp_head_adapter_acceptance.py` measured that adapting the MTP head
DESTROYS acceptance: tau 2.456 -> 1.531-1.988, every CI excluding zero. But read
its method: *"The backbone is identical across conditions -- only the head
changes."* It tested an **adapted head against a PRISTINE backbone** — the
MISMATCHED case — and its own explanation of the failure is:

    "A draft head's only job is to perfectly mimic the main backbone. If you make
     the draft head fluent in a domain, it forms its own opinions and disagrees
     with the backbone, causing the backbone to reject the drafts."

That mechanism **predicts the opposite result when the backbone carries the same
adapter.** A domain-fluent head disagrees with a pristine backbone — but it should
AGREE with a domain-fluent one. The matched case has never been run.

That is the configuration the engine actually serves: the backbone is ALWAYS
folded with a domain expert at inference time. So the relevant question was never
"adapted head vs pristine backbone" but "does the head benefit from matching the
expert the backbone is already wearing".

WHY IT IS EVEN POSSIBLE
-----------------------
The MTP head is architecturally a Qwen decoder layer, and its module shapes are
IDENTICAL to the backbone's:

    backbone adapter  mlp.gate_proj  U(9216,8) V(8,2560)  ->  weight (9216, 2560)
    MTP head          layer.mlp.gate_proj                      weight (9216, 2560)

so backbone LoRA factors fold directly into the head with no reshaping. The head
is one layer and the backbone has 32, so the source layer is a choice; this uses
the LAST shape-compatible backbone layer, since the head consumes the final hidden
state and sits conceptually after the stack.

    ARM A   backbone folded, head PRISTINE   (what ships today)
    ARM B   backbone folded, head folded with the SAME adapter

Restores are bit-exact `copy_` from a pristine buffer, never subtract-the-delta.

    uv run --env-file .env python \
        benchmarks/runtime/speculative/mtp_head_folding/benchmark_matched_draft_adapter.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.mtp_draft import (  # noqa: E402
    Qwen35MTPDraftHead,
    restore_state,
    snapshot_state,
)
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

DOMAINS = {
    "astral": ("results/adapters/m2_astral_r8a128", "data/astral/evaluation_data.jsonl"),
    "postgresql": ("results/adapters/m2_postgresql_r8a128",
                   "data/postgresql/evaluation_data.jsonl"),
}
# head module -> the backbone module suffix whose factors match it
HEAD_MAP = {
    "layer.self_attn.q_proj": "self_attn.q_proj",
    "layer.self_attn.k_proj": "self_attn.k_proj",
    "layer.self_attn.v_proj": "self_attn.v_proj",
    "layer.self_attn.o_proj": "self_attn.o_proj",
    "layer.mlp.gate_proj": "mlp.gate_proj",
    "layer.mlp.up_proj": "mlp.up_proj",
    "layer.mlp.down_proj": "mlp.down_proj",
}


def head_linears(head) -> dict:
    return {n: m for n, m in head.named_modules() if isinstance(m, torch.nn.Linear)}


def pick_source_layer(expert: FoldableExpert, head) -> tuple[int, dict]:
    """Last backbone layer whose factors match ALL the head's linears by shape."""
    lin = head_linears(head)
    by_layer: dict[int, dict] = {}
    for key, (u, v) in expert.factors.items():
        parts = key.split(".")
        if "layers" not in parts:
            continue
        li = int(parts[parts.index("layers") + 1])
        suffix = ".".join(parts[parts.index("layers") + 2:]).replace(".weight", "")
        by_layer.setdefault(li, {})[suffix] = (u, v)

    for li in sorted(by_layer, reverse=True):
        cand = {}
        ok = True
        for hname, suffix in HEAD_MAP.items():
            if hname not in lin or suffix not in by_layer[li]:
                ok = False
                break
            u, v = by_layer[li][suffix]
            w = lin[hname].weight
            if (u.shape[0], v.shape[1]) != tuple(w.shape):
                ok = False
                break
            cand[hname] = (u, v)
        if ok and cand:
            return li, cand
    raise RuntimeError("no backbone layer shape-matches the MTP head")


def fold_head(head, cand: dict, scaling: float, pristine: dict) -> int:
    lin = head_linears(head)
    n = 0
    for hname, (u, v) in cand.items():
        w = lin[hname].weight
        if hname not in pristine:
            pristine[hname] = w.data.detach().clone()
        delta = (u.float() @ v.float()) * scaling
        w.data.copy_((pristine[hname].float() + delta).to(w.dtype))
        n += 1
    return n


def restore_head(head, pristine: dict) -> None:
    lin = head_linears(head)
    for hname, w0 in pristine.items():
        lin[hname].weight.data.copy_(w0)


@torch.no_grad()
def measure_tau(model, tok, head, prompts, n_new: int, k: int) -> dict:
    steps = accepted = 0
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        eos = tok.eos_token_id
        out = model(ids, use_cache=True, output_hidden_states=True)
        cache = out.past_key_values
        hids = [out.hidden_states[-1]]
        seq = ids
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        toks = [nxt.item()]
        pos = ids.shape[1]
        done = toks[0] == eos
        while len(toks) < n_new and not done:
            H = torch.cat(hids, dim=1)
            dcache = head.prefill(H, seq)
            draft = head.draft(H[:, -1:, :], nxt, k=k, start_pos=pos - 1, cache=dcache)
            snap = snapshot_state(cache)
            chunk = torch.cat([nxt, draft], dim=-1)
            o = model(chunk, past_key_values=cache, use_cache=True, output_hidden_states=True)
            target = torch.argmax(o.logits[0], -1)
            n_acc = 0
            for i in range(k):
                if draft[0, i].item() == target[i].item():
                    n_acc += 1
                else:
                    break
            steps += 1
            accepted += n_acc
            committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
            if n_acc < k:
                restore_state(cache, snap)
                o = model(committed, past_key_values=cache, use_cache=True,
                          output_hidden_states=True)
                new_h = o.hidden_states[-1]
            else:
                new_h = o.hidden_states[-1][:, : n_acc + 1, :]
            bonus = target[n_acc].item()
            for t in draft[0, :n_acc].tolist() + [bonus]:
                if len(toks) >= n_new:
                    break
                toks.append(t)
                if t == eos:
                    done = True
                    break
            hids.append(new_h)
            seq = torch.cat([seq, committed], dim=-1)
            pos += n_acc + 1
            nxt = torch.tensor([[bonus]], device=ids.device)
    return {"steps": steps, "accepted": accepted, "tau": accepted / max(1, steps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/matched_draft_adapter.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  MATCHED DRAFT ADAPTER — same expert in the backbone AND the head")
    print("=" * 100)
    print(f"  device={dev}  K={args.k}  tokens={args.tokens}  prompts={args.n_prompts}")
    print("  ARM A = backbone folded, head pristine (ships today)")
    print("  ARM B = backbone folded, head folded with the SAME adapter\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()
    head = Qwen35MTPDraftHead(model, args.model_name)

    report: dict = {"device": dev, "k": args.k, "domains": {}}
    print(f"  {'domain':<14}{'arm':<10}{'steps':>7}{'accepted':>10}{'tau':>9}{'delta':>9}")
    print("  " + "-" * 96)

    for dom in args.domains:
        adapter_rel, prompts_rel = DOMAINS[dom]
        expert = FoldableExpert.from_dir(REPO_ROOT / adapter_rel, dom)
        engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
        rows = [json.loads(x) for x in (REPO_ROOT / prompts_rel).read_text().splitlines()
                if x.strip()]
        prompts = [f"### Question:\n{r['prompt']}\n\n### Answer:\n"
                   for r in rows[: args.n_prompts]]

        li, cand = pick_source_layer(expert, head)
        pristine: dict = {}

        engine.activate(expert)
        measure_tau(model, tok, head, prompts[:1], 8, args.k)  # warmup

        a = measure_tau(model, tok, head, prompts, args.tokens, args.k)
        print(f"  {dom:<14}{'A pristine':<10}{a['steps']:>7}{a['accepted']:>10}{a['tau']:>9.3f}"
              f"{'--':>9}")

        n_folded = fold_head(head, cand, expert.scaling, pristine)
        b = measure_tau(model, tok, head, prompts, args.tokens, args.k)
        print(f"  {dom:<14}{'B matched':<10}{b['steps']:>7}{b['accepted']:>10}{b['tau']:>9.3f}"
              f"{b['tau'] - a['tau']:>+9.3f}   (head folded from backbone layer {li}, "
              f"{n_folded} modules, scaling {expert.scaling})")

        restore_head(head, pristine)
        engine.restore()
        report["domains"][dom] = {"arm_a_pristine": a, "arm_b_matched": b,
                                  "delta_tau": b["tau"] - a["tau"],
                                  "source_layer": li, "modules_folded": n_folded}
        del engine
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("  RESULT")
    print("=" * 100)
    deltas = [v["delta_tau"] for v in report["domains"].values()]
    mean_d = sum(deltas) / len(deltas)
    for dom, v in report["domains"].items():
        print(f"    {dom:<16} tau {v['arm_a_pristine']['tau']:.3f} -> "
              f"{v['arm_b_matched']['tau']:.3f}   ({v['delta_tau']:+.3f})")
    print(f"\n    mean delta tau: {mean_d:+.3f}")
    if mean_d > 0.05:
        print("    => MATCHING HELPS. The prior 'never adapt the head' result does not")
        print("       cover this case: it tested an adapted head against a PRISTINE")
        print("       backbone, and its own mechanism predicted exactly this.")
    elif mean_d < -0.05:
        print("    => MATCHING HURTS TOO. The prior result generalises: the head must")
        print("       stay pristine regardless of what the backbone wears.")
    else:
        print("    => NO EFFECT. Matching neither helps nor hurts; the head already")
        print("       tracks the backbone through its hidden states.")
    report["mean_delta_tau"] = mean_d

    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
