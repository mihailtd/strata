"""Do domain-adapted backbones hurt the native MTP head's draft acceptance?

THE QUESTION, MEASURED DIRECTLY
-------------------------------
The MTP head was trained against the UN-adapted backbone's hidden states. Fold a
domain adapter into the backbone and those hidden states shift. Does the head's
acceptance rate collapse?

This was previously approached by proposing an offline SVD proxy that would
*predict* acceptance from subspace overlap. That proxy is not well-posed here
(`mtp.fc` is (2560, 5120) and shares no shape group with any domain adapter, so
there is nothing to project onto), and validating it would require measuring
acceptance anyway. Measuring acceptance directly takes minutes. So: measure it.

METHOD
------
For each condition (un-adapted, or one domain adapter folded into the backbone):

  1. Generate N truth tokens greedily FROM THAT CONDITION. Acceptance must be
     scored against the model you would actually be verifying with, not against
     the un-adapted model.
  2. One forward over [prompt + truth] to obtain every hidden state.
  3. At several offsets t along the sequence, prime the head's KV cache over
     S[:t+1], then draft K tokens from (h_t, e_{S[t+1]}).
  4. The head predicts S[t+2 ...], so drafted[i] is compared against S[t+2+i].
     Count the ACCEPTED PREFIX -- speculation stops at the first mismatch, so
     trailing lucky matches do not count.

This is the same isolated draft path that measured 66.7% acceptance, not the
end-to-end speculative loop in benchmark_mtp_speculative.py, which still
diverges from plain greedy and whose numbers are therefore not usable.

Prompts span all three domains so both in-domain and out-of-domain effects are
visible: if the financial adapter only degrades drafting on astral prompts,
that is a different finding from degrading everywhere.

    uv run --env-file .env scripts/benchmark_mtp_acceptance_vs_adapter.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# fla's device probe is @cache'd and runs at import; if no GPU context exists
# yet it latches to CPU for the process and the Triton kernels never engage.
# Touch CUDA before transformers pulls fla in.
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
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

# DEFAULTS: the m2 expert set (bf16 + Liger, methodology-matched). Verify with
# `uv run python scripts/audit/audit_adapters.py`. Do NOT default to m1 (4-bit NF4)
# adapters -- every benchmark here loads a bf16 base, so an m1 adapter folds a
# correction-to-quantized-weights into unquantized ones.
# §33 RE-TEST: the original verdict ("a draft head must AGREE with its backbone")
# was measured on adapters trained with loss over the PROMPT as well as the answer
# (§32: 48.4% of every batch was question text). An adapter trained to generate
# question text is precisely one whose hidden states disagree with the backbone --
# so the measurement and the defect may be the same phenomenon. ADAPTER_SET lets
# the same instrument compare the old adapters against completion-only v3 ones.
_SETS = {
    "v2": {
        "financial": "results/adapters/m2_financial_r8a128",
        "astral": "results/adapters/m2_astral_r8a128_v2",
        "postgres": "results/adapters/m2_postgresql_r8a128_v2",
    },
    "v3": {
        "astral_v3": "results/adapters/m2_astral_r8a128_v3",
        "postgres_v3": "results/adapters/m2_postgresql_r8a128_v3",
    },
    "both": {
        "astral_v2": "results/adapters/m2_astral_r8a128_v2",
        "astral_v3": "results/adapters/m2_astral_r8a128_v3",
        "postgres_v2": "results/adapters/m2_postgresql_r8a128_v2",
        "postgres_v3": "results/adapters/m2_postgresql_r8a128_v3",
    },
}
ADAPTERS = _SETS.get(os.environ.get("ADAPTER_SET", "legacy"), {
    "financial": "results/adapters/m2_financial_r8a128",
    "astral": "results/adapters/m2_astral_r8a128",
    "postgres": "results/adapters/m2_postgresql_r8a128",
})

PROMPTS = {
    "financial": [
        "### Question:\nWhat are money scripts and how do they affect saving?\n\n### Answer:\n",
        "### Question:\nExplain sequence-of-returns risk in retirement.\n\n### Answer:\n",
        "### Question:\nHow does loss aversion affect a client in a market drop?\n\n### Answer:\n",
    ],
    "astral": [
        "### Question:\nHow do I add a dependency to a Python project?\n\n### Answer:\n",
        "### Question:\nHow do I lint and format a Python codebase?\n\n### Answer:\n",
        "### Question:\nHow do I create and manage a virtual environment?\n\n### Answer:\n",
    ],
    "postgres": [
        "### Question:\nHow do I store vector embeddings in Postgres?\n\n### Answer:\n",
        "### Question:\nWhich index type suits approximate nearest-neighbour search?\n\n### Answer:\n",
        "### Question:\nHow do I run a similarity query over embeddings?\n\n### Answer:\n",
    ],
}


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
def acceptance_for_prompt(model, head, ids, n_truth, k, offsets):
    """Accepted-prefix length at several anchors along this condition's own output."""
    truth = greedy_truth(model, ids, n_truth)
    seq = torch.cat([ids, torch.tensor([truth], device=ids.device)], dim=1)
    hidden = model(seq, output_hidden_states=True, use_cache=False).hidden_states[-1]
    T = ids.shape[1]

    rows = []
    for off in offsets:
        t = T - 1 + off  # anchor position
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
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/mtp_acceptance_vs_adapter.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    head = Qwen35MTPDraftHead(model, args.model_name)

    experts = {n: FoldableExpert.from_dir(REPO_ROOT / p, n) for n, p in ADAPTERS.items() if (REPO_ROOT / p).exists()}
    engine = WeightFoldingEngine(model, experts.values(), keep_pristine=True)
    print(f"conditions: un-adapted + {sorted(experts)}   K={args.k}   offsets={args.offsets}\n")

    results = {}
    conditions = [("none", None)] + [(n, e) for n, e in experts.items()]
    for cname, expert in conditions:
        if expert is None:
            engine.restore()
        else:
            engine.activate(expert)

        per_domain = {}
        for dom, prompts in PROMPTS.items():
            accs = []
            for p in prompts:
                ids = tok(p, return_tensors="pt").input_ids.to(model.device)
                accs += acceptance_for_prompt(model, head, ids, args.truth_tokens, args.k, args.offsets)
            per_domain[dom] = {
                "mean_accepted": sum(accs) / len(accs),
                "accept_rate_pct": 100 * sum(accs) / (len(accs) * args.k),
                "n_drafts": len(accs),
            }
        allacc = [v["mean_accepted"] for v in per_domain.values()]
        results[cname] = {"per_domain": per_domain, "overall_mean_accepted": sum(allacc) / len(allacc)}

        tag = "un-adapted" if cname == "none" else f"{cname} folded"
        line = "  ".join(f"{d}={per_domain[d]['mean_accepted']:.2f}" for d in PROMPTS)
        print(f"  {tag:20s} mean accepted/{args.k}:  {line}   overall={results[cname]['overall_mean_accepted']:.2f}")

    engine.restore()

    print("\n" + "=" * 78)
    print(f" Draft acceptance vs backbone adaptation (mean accepted prefix out of K={args.k})")
    print("=" * 78)
    print(f" {'condition':20s} {'financial':>10s} {'astral':>9s} {'postgres':>9s} {'overall':>9s} {'vs base':>9s}")
    base = results["none"]["overall_mean_accepted"]
    for cname in results:
        r = results[cname]
        pd = r["per_domain"]
        tag = "un-adapted (base)" if cname == "none" else f"{cname} folded"
        delta = "" if cname == "none" else f"{r['overall_mean_accepted'] - base:+8.2f}"
        print(
            f" {tag:20s} {pd['financial']['mean_accepted']:10.2f} {pd['astral']['mean_accepted']:9.2f} "
            f"{pd['postgres']['mean_accepted']:9.2f} {r['overall_mean_accepted']:9.2f} {delta:>9s}"
        )
    print("\n break-even for speculation on this rig is ~2.8 accepted tokens per verification")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"k": args.k, "offsets": args.offsets, "results": results}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
