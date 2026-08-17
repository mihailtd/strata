"""Does speculative decoding produce WORSE ANSWERS, not just different ones?

THE QUESTION THIS ANSWERS
-------------------------
The engine's largest measured win is speculative decoding. Every speedup it has
ever reported travels with a caveat: the speculative decoder does not always
reproduce its own verifier's token path (worst cell 75% exact). Nobody has ever
measured whether the divergent text is WORSE. Until that is measured, a 2.20x
"speedup" might be a speed/quality trade of exactly the shape that got adapter
stacking killed (-10.96pp for 10.4ms) -- and it would be shipped by default.

WHY EXACT-MATCH CANNOT ANSWER IT
-------------------------------
Greedy speculative decoding is supposed to be lossless: a draft token is
committed only when it equals argmax of the target model's logits, so in exact
arithmetic the output IS the greedy output. It is not lossless here because the
two decode paths use different kernels. Plain decode steps the GatedDeltaNet
layers with the single-token RECURRENT kernel; verification runs the multi-token
CHUNKED kernel. Those disagree in the last bits, so argmax flips on near-ties.
The measured control is decisive: with NO speculation involved, the two kernels
already produce different text on ~17% of generations. Exact-match is therefore
saturated by numerics and can never isolate the effect of speculation. The only
way to answer the question is to score the text.

THREE ARMS, BECAUSE TWO CANNOT ATTRIBUTE
----------------------------------------
Comparing autoregressive against speculative confounds two different causes. A
third arm separates them:

    A  autoregressive   single-token recurrent kernel. The reference path.
    C  forced_reject    identical speculative machinery, but EVERY draft is
                        rejected (n_acc forced to 0). Tokens are chosen by the
                        chunked verifier; no draft is ever committed.
    B  speculative      the shipped decoder, K=4, real acceptance.

    A vs C  ->  the effect of the chunked kernel alone (pure numerics)
    C vs B  ->  the effect of ACCEPTING drafts (speculation proper)
    A vs B  ->  the shipped comparison, what a user actually receives

If B and C move together away from A, the divergence is kernel arithmetic and
speculation is exonerated. If B alone degrades, speculation itself costs quality
and the gate needs a quality term.

SCORING IS THE REPO'S, DELIBERATELY
-----------------------------------
`score_question` and the domain term lists are imported from
`evaluate_folded_vs_wrapped.py` rather than redefined, and the prompt sets are
that script's canonical DOMAINS. This makes these numbers directly comparable to
every other quality number in the repo (notably the stacking result). Rubric
coverage where the eval data carries an `expects` list, else the good/bad term
ratio.

Generation is greedy, so each arm is a deterministic function of (weights,
prompt) -- confirmed by tau_spread = 0.0000 across repeats in the speculation
matrix. One pass per arm is therefore sufficient and repeats would buy nothing.
Uncertainty here is over PROMPTS, not over runs, so the CIs come from a paired
bootstrap over prompts.

256 new tokens, matching the canonical eval. The speculation matrix uses 32,
which is far too short for a rubric to be covered and cannot score quality.

    uv run --env-file .env python \
        benchmarks/runtime/speculative/speculation_quality/benchmark_speculative_quality.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

# fla's device probe is @cache'd at import; touch CUDA before transformers pulls
# it in or the process latches to a fallback for its whole lifetime.
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "runtime" / "folding"))

from evaluate_folded_vs_wrapped import DOMAINS, score_question  # noqa: E402

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

ARMS = ("autoregressive", "forced_reject", "speculative")


def load_questions(path_rel: str) -> list[dict]:
    path = REPO_ROOT / path_rel
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def clean(text: str) -> str:
    """Same truncation the canonical eval applies.

    Without it the model runs past its answer and invents a new '### Question:'
    block, which then gets scored -- an artifact that supplied 57% of base's
    financial term hits before it was fixed there.
    """
    return re.split(r"#+\s*Question", text)[0].strip()


# --- Arm A: plain autoregressive decode (single-token recurrent kernel) -------

@torch.no_grad()
def run_autoregressive(model, tokenizer, prompt: str, n_new: int) -> tuple[list[int], float]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    eos = tokenizer.eos_token_id
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model(ids, use_cache=True)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]

    while len(toks) < n_new and toks[-1] != eos:
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        toks.append(nxt.item())

    torch.cuda.synchronize()
    return toks, time.perf_counter() - t0


# --- Arms B and C: one decoder, one flag -------------------------------------

@torch.no_grad()
def run_speculative(
    model, tokenizer, head, prompt: str, n_new: int, k: int, force_reject: bool = False
) -> tuple[list[int], float, dict]:
    """EAGLE-style speculation. `force_reject` turns this into the arm-C control.

    With force_reject the drafts are still generated (so the head runs and the
    chunk is still K+1 wide, keeping the kernel path identical) but n_acc is
    pinned to 0, so every emitted token is the chunked verifier's own argmax and
    no draft is ever committed. That is the chunked kernel WITHOUT speculation.
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    eos = tokenizer.eos_token_id
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model(ids, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    hids = [out.hidden_states[-1]]
    seq = ids
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    pos = ids.shape[1]
    stats = {"steps": 0, "drafted": 0, "accepted": 0}
    # EOS must be checked per COMMITTED TOKEN, not on toks[-1] at the top of the
    # loop. A speculative step commits up to n_acc+1 tokens at once, so an EOS
    # landing mid-block leaves a non-EOS token last and generation runs on.
    # Measured cost of getting this wrong: arm B ran 119.2 tokens against arm A's
    # 46.1 on astral (2.6x), and because the good/bad term ratio rewards length,
    # that artifact alone manufactured a "+11.55pp SIGNIFICANT quality gain from
    # speculation". Arms A and C commit exactly one token per iteration and were
    # never affected, which is what made the asymmetry visible.
    done = toks[0] == eos

    while len(toks) < n_new and not done:
        H = torch.cat(hids, dim=1)
        dcache = head.prefill(H, seq)
        draft = head.draft(H[:, -1:, :], nxt, k=k, start_pos=pos - 1, cache=dcache)

        snap = snapshot_state(cache)
        chunk = torch.cat([nxt, draft], dim=-1)
        out = model(chunk, past_key_values=cache, use_cache=True, output_hidden_states=True)
        target = torch.argmax(out.logits[0], -1)

        n_acc = 0
        if not force_reject:
            for i in range(k):
                if draft[0, i].item() == target[i].item():
                    n_acc += 1
                else:
                    break

        stats["steps"] += 1
        stats["drafted"] += k
        stats["accepted"] += n_acc

        committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
        if n_acc < k:
            restore_state(cache, snap)
            out = model(committed, past_key_values=cache, use_cache=True, output_hidden_states=True)
            new_h = out.hidden_states[-1]
        else:
            new_h = out.hidden_states[-1][:, : n_acc + 1, :]

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

    torch.cuda.synchronize()
    return toks[:n_new], time.perf_counter() - t0, stats


# --- Paired statistics --------------------------------------------------------

def paired_bootstrap(deltas: list[float], n_boot: int = 20000, seed: int = 0) -> dict:
    """CI over PROMPTS. Decoding is deterministic, so prompts are the only noise."""
    if not deltas:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "mean": sum(deltas) / n,
        "ci_low": means[int(0.025 * n_boot)],
        "ci_high": means[int(0.975 * n_boot)],
        "n": n,
        "excludes_zero": not (means[int(0.025 * n_boot)] <= 0.0 <= means[int(0.975 * n_boot)]),
    }


def build_contrasts(rows: list[dict], key: str) -> dict:
    """The three contrasts that make the attribution possible.

    Reported on both the full answers and the length-matched ones, so a verbosity
    effect can never be mistaken for a quality effect (it was once).
    """
    def s(arm: str) -> list[float]:
        return [r["arms"][arm][key] for r in rows]

    def delta(lo: str, hi: str) -> list[float]:
        return [b - a for a, b in zip(s(lo), s(hi), strict=True)]

    return {
        "numerics_only__forced_reject_minus_autoregressive": paired_bootstrap(
            delta("autoregressive", "forced_reject")
        ),
        "speculation_only__speculative_minus_forced_reject": paired_bootstrap(
            delta("forced_reject", "speculative")
        ),
        "shipped__speculative_minus_autoregressive": paired_bootstrap(
            delta("autoregressive", "speculative")
        ),
    }


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument("--limit", type=int, default=0, help="cap prompts per domain (smoke tests)")
    ap.add_argument("--out", default="results/speculative_quality.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 94)
    print("  DOES SPECULATIVE DECODING PRODUCE WORSE ANSWERS?")
    print("=" * 94)
    print(f"  device={dev}  K={args.k}  max_new_tokens={args.max_new_tokens}")
    print("  arms: A=autoregressive (recurrent)  C=forced_reject (chunked, 0 accepted)  "
          "B=speculative\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading {args.model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    head = Qwen35MTPDraftHead(model, args.model_name)

    experts = {
        d: FoldableExpert.from_dir(REPO_ROOT / DOMAINS[d]["adapter"], d) for d in args.domains
    }
    engine = WeightFoldingEngine(model, list(experts.values()), keep_pristine=True)

    report: dict = {
        "device": dev,
        "k_draft": args.k,
        "max_new_tokens": args.max_new_tokens,
        "arms": list(ARMS),
        "domains": {},
    }

    for domain in args.domains:
        cfg = DOMAINS[domain]
        questions = load_questions(cfg["questions"])
        if args.limit:
            questions = questions[: args.limit]

        engine.activate(experts[domain])
        print(f"\n▶ {domain.upper()}  (expert folded, n={len(questions)} prompts)")
        print("─" * 94)

        # Warmup on the folded weights so the first timed generation is not
        # paying for lazy kernel compilation.
        _ = run_autoregressive(model, tokenizer, "### Question:\nwarm\n\n### Answer:\n", 4)
        _ = run_speculative(model, tokenizer, head, "### Question:\nwarm\n\n### Answer:\n", 4, args.k)

        rows: list[dict] = []
        tot_acc = tot_steps = 0

        for qi, q in enumerate(questions):
            prompt = f"### Question:\n{q['prompt']}\n\n### Answer:\n"
            rec: dict = {"id": q.get("id", qi), "arms": {}}

            toks_a, t_a = run_autoregressive(model, tokenizer, prompt, args.max_new_tokens)
            toks_c, t_c, _ = run_speculative(
                model, tokenizer, head, prompt, args.max_new_tokens, args.k, force_reject=True
            )
            toks_b, t_b, st = run_speculative(
                model, tokenizer, head, prompt, args.max_new_tokens, args.k
            )
            tot_acc += st["accepted"]
            tot_steps += st["steps"]

            # The good/bad term ratio rewards length: more text is more chances to
            # match a good pattern, and an answer containing NO domain term at all
            # scores 0.0 by the metric's own else-branch, so a longer arm can
            # rescue a 0.0 into a high score without being more correct. Score a
            # second time with every arm truncated to the shortest arm's token
            # count, so any surviving difference cannot be verbosity.
            n_match = min(len(toks_a), len(toks_c), len(toks_b))
            rec["n_tokens_matched"] = n_match

            for arm, toks, dt in (
                ("autoregressive", toks_a, t_a),
                ("forced_reject", toks_c, t_c),
                ("speculative", toks_b, t_b),
            ):
                text = clean(tokenizer.decode(toks, skip_special_tokens=True).strip())
                sc = score_question(q, text, cfg["good"], cfg["bad"])
                text_m = clean(
                    tokenizer.decode(toks[:n_match], skip_special_tokens=True).strip()
                )
                sc_m = score_question(q, text_m, cfg["good"], cfg["bad"])
                rec["arms"][arm] = {
                    **sc,
                    "score_pct_lenmatched": sc_m["score_pct"],
                    "n_tokens": len(toks),
                    "n_chars": len(text),
                    "gen_time_s": dt,
                    "tok_s": len(toks) / max(1e-6, dt),
                    "response": text,
                }

            rec["exact_vs_autoregressive"] = {
                "forced_reject": toks_c == toks_a,
                "speculative": toks_b == toks_a,
            }
            rec["first_divergence_vs_autoregressive"] = {
                "forced_reject": first_divergence(toks_a, toks_c),
                "speculative": first_divergence(toks_a, toks_b),
            }
            rows.append(rec)

            sa = rec["arms"]["autoregressive"]["score_pct"]
            sc_ = rec["arms"]["forced_reject"]["score_pct"]
            sb = rec["arms"]["speculative"]["score_pct"]
            print(
                f"    [{qi + 1:2d}/{len(questions)}] {str(q.get('id', qi))[:14]:<14s} "
                f"A={sa:5.1f}  C={sc_:5.1f}  B={sb:5.1f}   "
                f"exact(C,B)=({int(toks_c == toks_a)},{int(toks_b == toks_a)})"
            )

        contrasts = build_contrasts(rows, "score_pct")
        contrasts_lenmatched = build_contrasts(rows, "score_pct_lenmatched")

        def scores(arm: str, key: str = "score_pct", rs: list[dict] = rows) -> list[float]:
            return [r["arms"][arm][key] for r in rs]

        summary = {
            "n_prompts": len(rows),
            "tau_accepted_per_step": tot_acc / max(1, tot_steps),
            "arm_means": {
                arm: {
                    "score_pct": sum(scores(arm)) / len(rows),
                    "score_pct_lenmatched": sum(scores(arm, "score_pct_lenmatched")) / len(rows),
                    "mean_tokens": sum(r["arms"][arm]["n_tokens"] for r in rows) / len(rows),
                    "mean_tok_s": sum(r["arms"][arm]["tok_s"] for r in rows) / len(rows),
                }
                for arm in ARMS
            },
            "exact_vs_autoregressive_pct": {
                arm: 100.0 * sum(r["exact_vs_autoregressive"][arm] for r in rows) / len(rows)
                for arm in ("forced_reject", "speculative")
            },
            "contrasts": contrasts,
            "contrasts_lenmatched": contrasts_lenmatched,
            "rows": rows,
        }
        report["domains"][domain] = summary

        print(f"\n    τ={summary['tau_accepted_per_step']:.2f}   "
              f"exact vs A: C={summary['exact_vs_autoregressive_pct']['forced_reject']:.0f}%  "
              f"B={summary['exact_vs_autoregressive_pct']['speculative']:.0f}%")
        print("    mean tokens: " + "  ".join(
            f"{a[0].upper()}={summary['arm_means'][a]['mean_tokens']:.1f}" for a in ARMS
        ))
        for label, cs in (("full", contrasts), ("len-matched", contrasts_lenmatched)):
            print(f"      -- {label} --")
            for name, c in cs.items():
                verdict = "SIGNIFICANT" if c["excludes_zero"] else "not significant"
                print(f"      {name:<52s} {c['mean']:+6.2f}pp  "
                      f"CI [{c['ci_low']:+6.2f}, {c['ci_high']:+6.2f}]  {verdict}")

        engine.restore()

    # --- pooled across domains ------------------------------------------------
    pooled_rows = [r for d in report["domains"].values() for r in d["rows"]]

    def pooled(arm: str) -> list[float]:
        return [r["arms"][arm]["score_pct"] for r in pooled_rows]

    report["pooled"] = {
        "n_prompts": len(pooled_rows),
        "arm_means": {arm: sum(pooled(arm)) / len(pooled_rows) for arm in ARMS},
        "mean_tokens": {
            arm: sum(r["arms"][arm]["n_tokens"] for r in pooled_rows) / len(pooled_rows)
            for arm in ARMS
        },
        "contrasts": build_contrasts(pooled_rows, "score_pct"),
        "contrasts_lenmatched": build_contrasts(pooled_rows, "score_pct_lenmatched"),
        "exact_vs_autoregressive_pct": {
            arm: 100.0 * sum(r["exact_vs_autoregressive"][arm] for r in pooled_rows) / len(pooled_rows)
            for arm in ("forced_reject", "speculative")
        },
    }

    print("\n" + "=" * 94)
    print("  POOLED RESULT")
    print("=" * 94)
    print(f"  n={len(pooled_rows)} paired triples across {len(report['domains'])} domains")
    for arm in ARMS:
        print(f"    {arm:<18s} mean score {report['pooled']['arm_means'][arm]:6.2f}%   "
              f"mean tokens {report['pooled']['mean_tokens'][arm]:6.1f}")
    print()
    for label, key in (("full answers", "contrasts"), ("length-matched", "contrasts_lenmatched")):
        print(f"  -- {label} --")
        for name, c in report["pooled"][key].items():
            verdict = "SIGNIFICANT" if c["excludes_zero"] else "not significant"
            print(f"    {name:<52s} {c['mean']:+6.2f}pp  "
                  f"CI [{c['ci_low']:+6.2f}, {c['ci_high']:+6.2f}]  {verdict}")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
