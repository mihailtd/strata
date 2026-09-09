"""Profile Native27BEngine.generate() breakdown."""

import time
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runtime.native_27b_engine import Native27BEngine

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Loading engine...")
engine = Native27BEngine(num_layers=64, device=str(device))
engine.load_from_cache()

prompt = list(range(100, 132))  # 32 tokens
print(f"\nProfiling generate with {len(prompt)} prompt tokens, max_new_tokens=32...")

# Profile forward_prompt alone
t0 = time.perf_counter()
state_dict = engine.init_kv_caches(1, 128)
logits, state_dict = engine.forward_prompt(prompt, state_dict)
torch.cuda.synchronize()
t_prefill = (time.perf_counter() - t0) * 1000.0
print(f"forward_prompt latency: {t_prefill:.2f} ms")

# Profile forward_token decode alone
curr = int(torch.argmax(logits[0, :]).item())
t0 = time.perf_counter()
for pos in range(len(prompt), len(prompt) + 32):
    logits, state_dict = engine.forward_token(curr, state_dict, pos=pos, use_graph=True)
    curr = int(torch.argmax(logits[0, :]).item())
torch.cuda.synchronize()
t_decode = (time.perf_counter() - t0) * 1000.0
print(f"32 decode tokens latency: {t_decode:.2f} ms ({t_decode / 32:.2f} ms/token = {32 / (t_decode / 1000.0):.2f} tok/s)")

# Profile full generate()
t0 = time.perf_counter()
output = engine.generate(prompt, max_new_tokens=32)
torch.cuda.synchronize()
t_full = (time.perf_counter() - t0) * 1000.0
print(f"full generate() latency: {t_full:.2f} ms ({len(output) / (t_full / 1000.0):.2f} tok/s)")
