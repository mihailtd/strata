import time
import torch
from runtime.native_27b_engine import Native27BEngine

print("[Test Verify Latency] Loading 64-layer engine...")
engine = Native27BEngine(num_layers=64, device="cuda:0")
engine.load_from_cache(force_convert=False)

state = engine.init_kv_caches(batch_size=1, max_seq_len=128)
pos = 10

# S = 1 (Single token)
t0 = time.perf_counter()
for _ in range(10):
    logits1, s_dict = engine.forward_token(100, state, pos=pos, use_graph=True)
torch.cuda.synchronize()
dt_single = (time.perf_counter() - t0) * 1000 / 10
print(f"Single-token forward_token (HIP Graph): {dt_single:.2f} ms ({1000/dt_single:.2f} tok/s)")

# S = 2 (Candidate pair verification)
cand = [100, 101]
t0 = time.perf_counter()
for _ in range(10):
    logits2, history = engine.forward_verify(cand, state, pos=pos)
torch.cuda.synchronize()
dt_verify2 = (time.perf_counter() - t0) * 1000 / 10
print(f"forward_verify (S=2) latency: {dt_verify2:.2f} ms")
print(f"Verify S=2 throughput: {2 * 1000 / dt_verify2:.2f} tok/s")
print(f"Ratio of Verify(2) to Single(1): {dt_verify2 / dt_single:.2f}x")
