import time
import torch
from transformers import AutoTokenizer
from pathlib import Path
from runtime.native_27b_engine import Native27BEngine, SSMChunkVerifyGraph

print("[Test K=2 Verify] Loading engine...")
engine = Native27BEngine(num_layers=64, device="cuda:0")
engine.load_from_cache(force_convert=False)

# Capture K=2 verify graphs
print("[Test K=2 Verify] Capturing K=2 verify graphs...")
t0 = time.perf_counter()
verify_graphs_k2 = []
chunk_count = engine.num_layers // 4
for k in range(chunk_count):
    ssm_layers = [engine.layers[4 * k], engine.layers[4 * k + 1], engine.layers[4 * k + 2]]
    v_chunk = SSMChunkVerifyGraph(ssm_layers, k=2, device=engine.device)
    verify_graphs_k2.append(v_chunk)
print(f"[Test K=2 Verify] Captured 16 K=2 verify graphs in {(time.perf_counter() - t0)*1000:.1f}ms!")

# Test replay latency of 16 K=2 chunks
x = torch.randn((1, 2, 5120), dtype=torch.bfloat16, device="cuda:0")
init_ssms = [torch.zeros((48, 128, 128), dtype=torch.bfloat16, device="cuda:0") for _ in range(3)]
init_convs = [torch.zeros((10240, 3), dtype=torch.bfloat16, device="cuda:0") for _ in range(3)]

# Warmup
for k in range(chunk_count):
    verify_graphs_k2[k].replay(x, init_ssms, init_convs)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(20):
    curr = x
    for k in range(chunk_count):
        curr, _, _ = verify_graphs_k2[k].replay(curr, init_ssms, init_convs)
torch.cuda.synchronize()
dt_ms = (time.perf_counter() - t0) * 1000 / 20
print(f"[Test K=2 Verify] 48-layer SSM verification (K=2) latency: {dt_ms:.2f} ms!")
