import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# fla's device probe is @cache'd at import. This CUDA touch MUST stay ABOVE the
# transformers import -- if an import sorter hoists transformers above it, the
# process latches to the slow fallback and the chunked-kernel cost silently
# returns to ~2.8x. The noqa: E402 markers below exist to keep that ordering.
torch.zeros(1, device="cuda")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import set_hard_vram_cap  # noqa: E402


def measure_prefill_decode(model, input_ids, max_new_tokens=1024):
    # 1. Isolate Prefill
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    prefill_time = t1 - t0

    # 2. Isolate Decode
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    
    generated_tokens = [next_token]
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated_tokens.append(next_token)

    torch.cuda.synchronize()
    t3 = time.perf_counter()
    decode_time = t3 - t2
    
    return prefill_time, decode_time

def load_real_text(tokenizer, target_tokens, device):
    # Load some real text from evaluation data
    data_path = REPO_ROOT / "data" / "astral" / "evaluation_data.jsonl"
    text = ""
    if data_path.exists():
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    text += json.loads(line).get("prompt", "") + "\n\n"
    else:
        text = "Hello world! This is a fallback text. " * 1000

    # Tokenize
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    # Repeat if not enough tokens
    while input_ids.shape[1] < target_tokens:
        input_ids = torch.cat([input_ids, input_ids], dim=1)
        
    return input_ids[:, :target_tokens].to(device)

def main():
    parser = argparse.ArgumentParser(description="Measure Prefill vs Decode Share")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--vram-cap-gb", type=float, default=22.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    # 2. Hard VRAM cap
    set_hard_vram_cap(args.vram_cap_gb)

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()

    context_lengths = [2048, 4096, 8192]
    print(f"Target Output Length: {args.max_new_tokens} tokens")
    
    results = {}
    
    for ctx_len in context_lengths:
        print(f"\n--- Context Length: {ctx_len} ---")
        
        # Use real tokenized text
        input_ids = load_real_text(tokenizer, ctx_len, "cuda:0")
        
        print("Per-shape warmup...")
        # 3. Per-shape warmup BEFORE timed region
        measure_prefill_decode(model, input_ids, max_new_tokens=2)
        
        prefill_times = []
        decode_times = []
        
        print(f"Running {args.repeats} repeats...")
        for rep in range(args.repeats):
            pt, dt = measure_prefill_decode(model, input_ids, max_new_tokens=args.max_new_tokens)
            prefill_times.append(pt)
            decode_times.append(dt)
            print(f"  Rep {rep+1}: prefill={pt:.3f}s, decode={dt:.3f}s")
            
        # 4. Report median
        prefill_median = float(np.median(prefill_times))
        decode_median = float(np.median(decode_times))
        total_time = prefill_median + decode_median
        
        prefill_pct = (prefill_median / total_time) * 100
        decode_pct = (decode_median / total_time) * 100
        
        print(f"Median Prefill time ({ctx_len} tokens): {prefill_median:.3f} s ({prefill_pct:.1f}%)")
        print(f"Median Decode time  ({args.max_new_tokens} tokens): {decode_median:.3f} s ({decode_pct:.1f}%)")
        print(f"Median Total time   : {total_time:.3f} s")
        
        results[ctx_len] = {
            "prefill_s": prefill_median,
            "decode_s": decode_median,
            "total_s": total_time,
            "prefill_pct": prefill_pct,
            "decode_pct": decode_pct
        }

    out_path = REPO_ROOT / "results/prefill_share_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
