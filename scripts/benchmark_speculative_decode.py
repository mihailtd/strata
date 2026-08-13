"""Does speculative decoding break the ~17.8 tok/s decode ceiling on this rig?

Three arms on the same greedy eval prompts:
  baseline        plain greedy decode from the 4B target
  prompt_lookup   n-gram speculation from the prompt itself; no draft model,
                  no extra VRAM. Wins when the output echoes the input
                  (code editing, RAG); expected to do little here, where the
                  prompts are short questions and the answers are new text.
  draft           real speculative decoding with Qwen3.5-0.8B as assistant.
                  Tokenizer verified identical to the 4B (248,077 vocab, same
                  EOS, byte-identical encodings), which standard assisted
                  generation requires.

CORRECTNESS CHECK, not just speed: with greedy verification speculative
decoding is *exact* -- it must emit the same tokens as the baseline. Every arm
is therefore compared against the baseline string. A mismatch means the
speculation path is broken, not that it is "a bit different".

That check matters specifically here: 24 of this model's 32 layers are
`Qwen3_5GatedDeltaNet` linear-attention layers carrying a fixed-size recurrent
state rather than a growing KV cache. Rejecting a drafted token requires
rolling that state back, which is trivial for a KV cache (truncate) and not
trivial for a recurrent one. If that rollback is unimplemented the outputs will
silently diverge -- which is exactly what the comparison catches.

WHAT THIS ANSWERS THAT NOTHING ELSE DOES
----------------------------------------
The shipped MTP head reaches tau ~ 2.3-2.4 and 1.32-1.39x end to end. Whether a
real 0.8B draft model does better or worse is UNMEASURED. This is that
comparison. The 0.8B shares the hybrid layout (full_attention_interval=4).

Runs in bf16 by default. It previously hardcoded a 4-bit target, which made its
numbers an m1-regime measurement not comparable to the MTP head results; pass
--target-4bit to reproduce the old behaviour.

`fla` (flash-linear-attention 0.5.2, triton-rocm 3.7.1) is bound and active on
gfx1100, which flattened verification from ~2.84x to ~1.19x per K=4 step.
`causal_conv1d` is a SEPARATE dependency, still absent, costing ~3.8% of wall.

EXPECT THE DRAFT ARM TO FAIL. `transformers` refuses assisted generation for
stateful models ("not supported with stateful models, such as
Qwen3_5ForCausalLM") because 24 of 32 layers carry a recurrent state that
cannot be rolled back by truncation. That refusal is caught and reported as a
result, not a crash -- and it is exactly why the MTP head path exists.

    uv run --env-file .env scripts/benchmark_speculative_decode.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# fla's device probe is @cache'd at import; touch CUDA before transformers pulls
# it in or the process latches to a fallback for its whole lifetime and the
# chunked-kernel cost stays at ~2.8x instead of ~1.19x.
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from gnn_experiment.novel_peft import set_hard_vram_cap  # noqa: E402


def load_questions(path: Path, limit: int | None) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return qs[:limit] if limit else qs


def generate_once(model, tokenizer, prompt: str, max_new_tokens: int, **gen_kwargs) -> tuple[str, int, float]:
    formatted = f"### Question:\n{prompt}\n\n### Answer:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **gen_kwargs,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new = out[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(new, skip_special_tokens=True), int(new.shape[0]), dt


def run_arm(name, model, tokenizer, questions, max_new_tokens, baseline_texts=None, **gen_kwargs):
    print(f"\n=== {name} ===")
    texts, toks, times = [], [], []
    for i, q in enumerate(questions, 1):
        try:
            txt, n, dt = generate_once(model, tokenizer, q["prompt"], max_new_tokens, **gen_kwargs)
        except Exception as e:  # noqa: BLE001 -- report and continue; an unsupported path is a result
            print(f"  [{i}/{len(questions)}] FAILED: {type(e).__name__}: {str(e)[:160]}")
            return None
        texts.append(txt)
        toks.append(n)
        times.append(dt)
        print(f"  [{i}/{len(questions)}] {n:4d} tok in {dt:6.2f}s = {n / dt:6.2f} tok/s")
    # warm = drop first (kernel/algorithm autotuning on first-seen shapes)
    warm_t, warm_n = sum(times[1:]) or times[0], sum(toks[1:]) or toks[0]
    res = {
        "tok_s_all": sum(toks) / sum(times),
        "tok_s_warm": warm_n / warm_t,
        "total_tokens": sum(toks),
        "total_time": sum(times),
        "per_q_tok_s": [n / t for n, t in zip(toks, times, strict=True)],
        "texts": texts,
    }
    print(f"  -> {res['tok_s_all']:.2f} tok/s (all)   {res['tok_s_warm']:.2f} tok/s (warm)")
    if baseline_texts is not None:
        exact = sum(1 for a, b in zip(texts, baseline_texts, strict=True) if a == b)
        res["exact_match"] = exact / len(texts)
        flag = "OK" if exact == len(texts) else "*** DIVERGENCE ***"
        print(f"  -> exact match vs baseline: {exact}/{len(texts)}  {flag}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--draft-name", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--questions", default="data/astral/evaluation_data.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=8, help="questions to run (8 keeps a full sweep ~10 min)")
    ap.add_argument("--lookup-ngram", type=int, default=3)
    ap.add_argument("--assistant-tokens", type=int, default=5)
    ap.add_argument("--draft-4bit", action="store_true", help="load draft in 4-bit instead of bf16")
    ap.add_argument("--target-4bit", action="store_true",
                    help="load target in 4-bit (m1 regime). Default bf16, matching every other "
                         "current benchmark -- 4-bit numbers are not comparable to them.")
    ap.add_argument("--vram-cap-gb", type=float, default=20.0)
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dt = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    qpath = Path(args.questions)
    if not qpath.is_absolute():
        qpath = REPO_ROOT / qpath
    questions = load_questions(qpath, args.limit)
    print(f"{len(questions)} questions, max_new_tokens={args.max_new_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    assert tokenizer is not None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dt, bnb_4bit_quant_type="nf4")
    tgt_kwargs = {"quantization_config": bnb} if args.target_4bit else {"dtype": dt}
    print(f"loading target {args.model_name} ({'4-bit' if args.target_4bit else 'bf16'})")
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
        **tgt_kwargs,
    )
    target.eval()

    # warm-up so the first timed question is not paying autotuning
    generate_once(target, tokenizer, questions[0]["prompt"], 16)

    results = {}
    base = run_arm("baseline (plain greedy)", target, tokenizer, questions, args.max_new_tokens)
    results["baseline"] = base

    results["prompt_lookup"] = run_arm(
        f"prompt-lookup (n={args.lookup_ngram}, no draft model)",
        target,
        tokenizer,
        questions,
        args.max_new_tokens,
        baseline_texts=base["texts"],
        prompt_lookup_num_tokens=args.lookup_ngram,
    )

    print(f"\nloading draft {args.draft_name} ({'4-bit' if args.draft_4bit else 'bf16'})")
    draft_kwargs = {"quantization_config": bnb} if args.draft_4bit else {"dtype": dt}
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft_name,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
        **draft_kwargs,
    )
    draft.eval()
    if torch.cuda.is_available():
        print(f"  VRAM after both models: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    results["draft"] = run_arm(
        f"speculative (draft={args.draft_name}, k={args.assistant_tokens})",
        target,
        tokenizer,
        questions,
        args.max_new_tokens,
        baseline_texts=base["texts"],
        assistant_model=draft,
        num_assistant_tokens=args.assistant_tokens,
    )

    print("\n=== SUMMARY (warm tok/s) ===")
    b = base["tok_s_warm"]
    print(f"  {'arm':44s} {'tok/s':>8s} {'speedup':>9s} {'exact':>7s}")
    for k, r in results.items():
        if r is None:
            print(f"  {k:44s} {'FAILED':>8s}")
            continue
        em = r.get("exact_match")
        print(
            f"  {k:44s} {r['tok_s_warm']:8.2f} {r['tok_s_warm'] / b:8.2f}x {('n/a' if em is None else f'{em:.0%}'):>7s}"
        )
    out = REPO_ROOT / "results" / "speculative_decode_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    slim = {k: ({kk: vv for kk, vv in v.items() if kk != "texts"} if v else None) for k, v in results.items()}
    out.write_text(json.dumps(slim, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
