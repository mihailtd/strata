import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

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

def main():
    parser = argparse.ArgumentParser(description="Measure Prefill vs Decode Share")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()

    context_lengths = [2048, 4096, 8192]
    
    print(f"Target Output Length: {args.max_new_tokens} tokens")
    
    # Warmup
    print("Warming up GPU...")
    dummy_input = torch.randint(0, 1000, (1, 128)).to("cuda:0")
    measure_prefill_decode(model, dummy_input, max_new_tokens=10)
    
    results = {}
    
    for ctx_len in context_lengths:
        print(f"\n--- Context Length: {ctx_len} ---")
        # generate random tokens for prompt
        input_ids = torch.randint(0, 1000, (1, ctx_len)).to("cuda:0")
        
        prefill_time, decode_time = measure_prefill_decode(model, input_ids, max_new_tokens=args.max_new_tokens)
        
        total_time = prefill_time + decode_time
        prefill_pct = (prefill_time / total_time) * 100
        decode_pct = (decode_time / total_time) * 100
        
        print(f"Prefill time ({ctx_len} tokens): {prefill_time:.3f} s ({prefill_pct:.1f}%)")
        print(f"Decode time  ({args.max_new_tokens} tokens): {decode_time:.3f} s ({decode_pct:.1f}%)")
        print(f"Total time   : {total_time:.3f} s")
        
        results[ctx_len] = {
            "prefill_s": prefill_time,
            "decode_s": decode_time,
            "total_s": total_time,
            "prefill_pct": prefill_pct,
            "decode_pct": decode_pct
        }

    out_path = Path("results/prefill_share_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
