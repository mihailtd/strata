"""Does eager speculative decode beat the server's CUDA-graph autoregressive path?

THE GATE THIS ANSWERS
---------------------
Every speculative number in this repo (tau 1.506, W=2 branching, matched draft
head, 1.352x compounded) was measured against **eager** autoregressive decode. But
`server.py` does not decode eagerly — it replays a captured CUDA graph
(`FoldedCudaGraphDecoder.generate_with_graph`), which exists precisely to remove
per-token kernel-launch overhead.

**Speculation cannot use that graph.** The graph is captured for a FIXED
single-token shape; speculative verification is a K+1 token chunk and the commit
re-forward is n_acc+1 tokens. Neither shape matches, so the speculative path runs
eager end to end.

So the honest question before wiring speculation into the server is:

    graph-replayed autoregressive     vs     eager speculative

If the graph is worth more than speculation, wiring speculation in would make the
server SLOWER while every benchmark still reported a speedup — because the
benchmarks used the wrong baseline. That is a multi-hour build gated on one
measurement, so measure first.

    ARM A   graph autoregressive   (what server.py does today)
    ARM B   eager autoregressive   (the baseline every spec benchmark used)
    ARM C   eager speculative      (what wiring it in would deliver)

A vs B prices the graph. C vs A is the decision.

    uv run --env-file .env python \
        benchmarks/runtime/speculative/serving_gate/benchmark_graph_vs_speculative.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.bucketed_speculative import BucketedSpeculativeDecoder  # noqa: E402
from gnn_experiment.cuda_graph import FoldedCudaGraphDecoder  # noqa: E402
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

ADAPTER = "results/adapters/m2_astral_r8a128"
PROMPTS_FILE = "data/astral/evaluation_data.jsonl"


@torch.no_grad()
def eager_autoregressive(model, tok, prompt: str, n: int) -> tuple[int, float]:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    eos = tok.eos_token_id
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    while len(toks) < n and toks[-1] != eos:
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        toks.append(nxt.item())
    torch.cuda.synchronize()
    return len(toks), time.perf_counter() - t0


@torch.no_grad()
def eager_speculative(model, tok, head, prompt: str, n: int, k: int) -> tuple[int, float, dict]:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    eos = tok.eos_token_id
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(ids, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    hids = [out.hidden_states[-1]]
    seq = ids
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    pos = ids.shape[1]
    st = {"steps": 0, "accepted": 0}
    done = toks[0] == eos
    while len(toks) < n and not done:
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
        st["steps"] += 1
        st["accepted"] += n_acc
        committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
        if n_acc < k:
            restore_state(cache, snap)
            o = model(committed, past_key_values=cache, use_cache=True, output_hidden_states=True)
            new_h = o.hidden_states[-1]
        else:
            new_h = o.hidden_states[-1][:, : n_acc + 1, :]
        bonus = target[n_acc].item()
        for t in draft[0, :n_acc].tolist() + [bonus]:
            if len(toks) >= n:
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
    return len(toks), time.perf_counter() - t0, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--n-prompts", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/graph_vs_speculative.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  SERVING GATE — graph autoregressive vs eager speculative")
    print("=" * 100)
    print(f"  device={dev}  K={args.k}  tokens={args.tokens}  "
          f"prompts={args.n_prompts}  repeats={args.repeats}\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()
    head = Qwen35MTPDraftHead(model, args.model_name)
    expert = FoldableExpert.from_dir(REPO_ROOT / ADAPTER, "astral")
    engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
    engine.activate(expert)

    rows = [json.loads(x) for x in (REPO_ROOT / PROMPTS_FILE).read_text().splitlines() if x.strip()]
    prompts = [f"### Question:\n{r['prompt']}\n\n### Answer:\n" for r in rows[: args.n_prompts]]

    print("  Capturing CUDA graph (as server.py does at boot)...")
    decoder = FoldedCudaGraphDecoder(model, tok, max_seq_len=args.max_seq_len, device=model.device)
    dummy = tok(prompts[0], return_tensors="pt").input_ids.to(model.device)
    decoder.capture(dummy)
    print(f"  captured (count={decoder.capture_count})\n")

    # warmup all three paths
    decoder.generate_with_graph(dummy, max_new_tokens=8)
    eager_autoregressive(model, tok, prompts[0], 8)
    eager_speculative(model, tok, head, prompts[0], 8, args.k)

    print("  Capturing bucketed speculative graphs (widths 1..K+1)...")
    buck = BucketedSpeculativeDecoder(model, tok, head, k=args.k,
                                      max_seq_len=args.max_seq_len)
    buck.capture(dummy)
    print(f"  captured widths {sorted(buck.buckets)}\n")
    buck.generate(dummy, max_new_tokens=8)

    res = {"graph_auto": [], "eager_auto": [], "eager_spec": [], "bucketed_spec": []}
    taus = []
    for _ in range(args.repeats):
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(model.device)
            gt, gs, _tps, _sw = decoder.generate_with_graph(ids, max_new_tokens=args.tokens)
            res["graph_auto"].append((len(gt), gs))

            n_a, s_a = eager_autoregressive(model, tok, p, args.tokens)
            res["eager_auto"].append((n_a, s_a))

            n_s, s_s, st = eager_speculative(model, tok, head, p, args.tokens, args.k)
            res["eager_spec"].append((n_s, s_s))
            taus.append(st["accepted"] / max(1, st["steps"]))

            bt, bs, bst = buck.generate(ids, max_new_tokens=args.tokens)
            res["bucketed_spec"].append((len(bt), bs))

    def tps(key):
        tot_tok = sum(n for n, _ in res[key])
        tot_s = sum(s for _, s in res[key])
        return tot_tok / max(1e-9, tot_s)

    ga, ea, es, bs_ = tps("graph_auto"), tps("eager_auto"), tps("eager_spec"), tps("bucketed_spec")
    tau = sum(taus) / len(taus)

    print("=" * 100)
    print("  RESULT")
    print("=" * 100)
    print(f"  {'arm':<34}{'tok/s':>10}{'vs eager auto':>16}{'vs GRAPH auto':>16}")
    print("  " + "-" * 96)
    print(f"  {'A  graph autoregressive (server)':<34}{ga:>10.2f}{ga / ea:>15.3f}x{'--':>16}")
    print(f"  {'B  eager autoregressive':<34}{ea:>10.2f}{'1.000x':>16}{ea / ga:>15.3f}x")
    print(f"  {'C  eager speculative':<34}{es:>10.2f}{es / ea:>15.3f}x{es / ga:>15.3f}x")
    print(f"  {'D  BUCKETED graph speculative':<34}{bs_:>10.2f}{bs_ / ea:>15.3f}x{bs_ / ga:>15.3f}x")
    print(f"\n  measured tau = {tau:.3f}")
    print(f"\n  the CUDA graph is worth {ga / ea:.3f}x over eager decode")
    print(f"  speculation is worth      {es / ea:.3f}x over eager decode")
    print()
    print(f"\n  BUCKETED vs the server's graph path: {bs_ / ga:.3f}x")
    print(f"  BUCKETED vs eager speculative      : {bs_ / es:.3f}x  "
          "(what capturing the chunk shapes recovered)")
    if bs_ > ga:
        print(f"\n  => SHIP THE BUCKETED PATH. {bs_ / ga:.3f}x over the current server,")
        print("     keeping graph replay instead of trading it away.")
    else:
        print(f"\n  => BUCKETED DOES NOT BEAT THE GRAPH PATH ({bs_ / ga:.3f}x). The chunk")
        print("     graphs did not recover enough to overcome speculation's overhead.")
    if es > ga:
        print(f"  => WIRE IT IN. Eager speculation beats the graph path by {es / ga:.3f}x.")
        print("     Every benchmark baseline was eager, and the conclusion survives")
        print("     the correct comparison.")
    else:
        print(f"  => DO NOT WIRE IT IN AS-IS. The graph path is {ga / es:.3f}x FASTER than")
        print("     eager speculation. Every speculative number in this repo used an")
        print("     eager baseline the server does not use. Speculation would have to")
        print("     either capture a graph for the K+1 chunk shape, or beat the graph")
        print("     by a wider margin than tau currently allows.")

    report = {"device": dev, "k": args.k, "tokens": args.tokens,
              "graph_auto_tps": ga, "eager_auto_tps": ea, "eager_spec_tps": es,
              "bucketed_spec_tps": bs_, "bucketed_vs_graph": bs_ / ga,
              "bucketed_vs_eager_spec": bs_ / es,
              "graph_vs_eager": ga / ea, "spec_vs_eager": es / ea,
              "spec_vs_graph": es / ga, "tau": tau}
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
