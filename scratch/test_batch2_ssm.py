import time
import torch
from runtime.native_27b_engine import Native27BEngine

print("[Test Batch 2 SSM] Loading 1 layer...")
engine = Native27BEngine(num_layers=4, device="cuda:0")
engine.load_from_cache(force_convert=False)
layer = engine.layers[0]

# S = 1 (Single token)
x1 = torch.randn((1, 1, 5120), dtype=torch.bfloat16, device="cuda:0")
ssm1 = torch.zeros((48, 128, 128), dtype=torch.bfloat16, device="cuda:0")
conv1 = torch.zeros((10240, 3), dtype=torch.bfloat16, device="cuda:0")

# Warmup
for _ in range(5):
    layer(x1, ssm1, conv1)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(100):
    layer(x1, ssm1, conv1)
torch.cuda.synchronize()
dt_s1 = (time.perf_counter() - t0) * 1000 / 100
print(f"Single-token (S=1) layer latency: {dt_s1:.3f} ms")

# S = 2 (Batched 2 tokens)
x2 = torch.randn((1, 2, 5120), dtype=torch.bfloat16, device="cuda:0")
ssm2 = torch.zeros((48, 128, 128), dtype=torch.bfloat16, device="cuda:0")
conv2 = torch.zeros((10240, 3), dtype=torch.bfloat16, device="cuda:0")

for _ in range(5):
    layer(x2, ssm2, conv2)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(100):
    layer(x2, ssm2, conv2)
torch.cuda.synchronize()
dt_s2 = (time.perf_counter() - t0) * 1000 / 100
print(f"Batched 2-token (S=2) layer latency: {dt_s2:.3f} ms")
print(f"Overhead of 2 tokens vs 1 token: {(dt_s2 - dt_s1):.3f} ms (Ratio: {dt_s2/dt_s1:.2f}x)")
