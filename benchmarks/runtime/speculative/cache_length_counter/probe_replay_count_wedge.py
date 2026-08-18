"""WHICH INGREDIENT wedges it? One replay mix per MODE.

Established: not position (all 20 positions incl. 891/1193/2000 replay fine from
a fresh cache) and not decode (wedges with no draft head, no token generation).
It dies at ~466-530 cumulative replays at a FIXED position. Modes isolate why:

    full     width K+1 + snapshot, width 2 + restore every 3rd  (the real loop)
    onlyw5   width K+1 alone: no SSM copies, no second width
    onlyw1   width 1 alone -- recurrent kernel instead of chunked
    swaponly width K+1 + snapshot_ssm, single width
    mixonly  width K+1 and width 2 mixed, no SSM copies

ORIGINAL DOC: Does a REPLAY COUNT wedge it? Fixed position, no decode, just hammer the graphs.

Device-synced markers proved the hang is inside _replay(width=5) at pos=891:
phase D (snapshot_ssm) completes on device, phase E never returns. Two candidates:

    (a) the captured width-5 graph cannot execute around pos ~891   -> POSITION
    (b) state built up over ~300 decode steps wedges a kernel       -> STATE

This replays the SAME graph at a sweep of positions from a freshly prefilled
cache, with no decode history. Hang at 891 here => (a), and we have the trigger.
Every position clean => (b), and position is a red herring.

Each replay is followed by synchronize() plus a finiteness check, so silent
corruption surfaces as bad values rather than only as a hang.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/mihai/gnn-experiment")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

T0 = time.perf_counter()


def stage(m: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)


import torch  # noqa: E402

torch.zeros(1, device="cuda"); torch.cuda.synchronize()
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gnn_experiment.bucketed_speculative import BucketedSpeculativeDecoder  # noqa: E402
from gnn_experiment.mtp_draft import Qwen35MTPDraftHead  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

MODEL, K, MAX_SEQ = "Qwen/Qwen3.5-4B", 4, 2048
ADAPTER = "results/adapters/m2_astral_r8a128"


def main() -> None:
    set_hard_vram_cap(22.0)
    stage("loading model")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    head = Qwen35MTPDraftHead(model, MODEL)
    expert = FoldableExpert.from_dir(REPO_ROOT / ADAPTER, "astral")
    engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
    engine.activate(expert)

    prompt = tok.apply_chat_template(
        [{"role": "user", "content":
          "Design a production vector search system for 100M documents. Cover the "
          "storage layer, index construction, sharding, the query path, recall/latency "
          "tradeoffs, and how you would benchmark it. Be thorough and specific."}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    stage(f"prompt={ids.shape[1]} tokens")

    dec = BucketedSpeculativeDecoder(model, tok, head, k=K, max_seq_len=MAX_SEQ)
    dec.capture(ids)
    dec.generate(ids, max_new_tokens=8)
    torch.cuda.synchronize()
    stage("captured + warm")

    dec.cache.reset()
    cur = ids.shape[1]
    with torch.no_grad():
        dec.model(ids, past_key_values=dec.cache,
                  cache_position=torch.arange(0, cur, device=dec.device),
                  use_cache=True, output_hidden_states=True)
    torch.cuda.synchronize()
    stage("prompt prefilled; NO decode history beyond this point")

    chunk = torch.full((1, K + 1), 100, dtype=torch.long, device=dec.device)
    small = torch.full((1, 2), 100, dtype=torch.long, device=dec.device)
    one = torch.full((1, 1), 100, dtype=torch.long, device=dec.device)
    stage("hammering replays at a FIXED position (891) -- position is ruled out, so")
    stage("anything that breaks here is a function of REPLAY COUNT, not sequence state.")
    stage("Mix mirrors real decode: width K+1 every iter, a rollback width every 3rd.")

    import os as _os
    import time as _t
    mode = _os.environ.get("MODE", "full")
    stage(f"MODE={mode}")

    n = 0
    t_last = _t.perf_counter()
    for i in range(1, 4001):
        if mode in ("full", "swaponly"):
            dec._snapshot_ssm()
        # onlyw1 replays the WIDTH-1 bucket. fla dispatches the fused RECURRENT
        # gated-delta kernel at width 1 and the CHUNKED kernel at width > 1, so
        # this separates "any captured graph of mine wedges" from "the chunked
        # kernel inside a graph wedges". The server's single-width decoder did
        # 1024 replays clean, which already points at the chunked path.
        if mode == "onlyw1":
            dec._replay(1, one, 891)
        else:
            dec._replay(K + 1, chunk, 891)
        n += 1
        if mode in ("full", "mixonly") and i % 3 == 0:
            if mode == "full":
                dec._restore_ssm()
            dec._replay(2, small, 891)
            n += 1
        if i % 50 == 0:
            torch.cuda.synchronize()
            now = _t.perf_counter()
            stage(f"  iter={i:5d} replays={n:6d} "
                  f"{50 / max(1e-9, now - t_last):7.1f} it/s "
                  f"alloc={torch.cuda.memory_allocated() / 2**30:.2f}G "
                  f"res={torch.cuda.memory_reserved() / 2**30:.2f}G")
            t_last = now

    torch.cuda.synchronize()
    stage(f"SURVIVED {n} replays in MODE={mode}")
    stage("ALL DONE")


if __name__ == "__main__":
    main()
