"""Test if Qwen35FullAttentionBlock can be captured in a torch.cuda.CUDAGraph with PreallocatedKVCache."""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime.native_27b_engine import Qwen35FullAttentionBlock, PreallocatedKVCache

dev = torch.device("cuda:0")
block = Qwen35FullAttentionBlock(3, device=dev)
# Load real weights
l3_dict = torch.load("models/qwen3.8-27b-triton/layer_3.pt", map_location=dev, weights_only=False)
block.load_weights(l3_dict)

kv = PreallocatedKVCache(1, 4, 256, 128, mode="bf16", device=dev)
static_x = torch.zeros((1, 5120), dtype=torch.bfloat16, device=dev)
static_out = torch.zeros((1, 5120), dtype=torch.bfloat16, device=dev)
static_cos = torch.ones((1, 1, 1, 32), dtype=torch.bfloat16, device=dev)
static_sin = torch.zeros((1, 1, 1, 32), dtype=torch.bfloat16, device=dev)

def forward_step():
    out, _ = block(static_x, kv_cache=kv, cos_sin=(static_cos, static_sin))
    static_out.copy_(out)

# Warmup on side stream
stream = torch.cuda.Stream(device=dev)
stream.wait_stream(torch.cuda.current_stream(device=dev))
with torch.cuda.stream(stream):
    for _ in range(3):
        forward_step()
torch.cuda.current_stream(device=dev).wait_stream(stream)

kv.reset()

try:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        forward_step()
    print(">>> Attention Block CUDAGraph Capture SUCCESSFUL! <<<")
    
    # Test replay
    static_x.copy_(torch.randn((1, 5120), dtype=torch.bfloat16, device=dev))
    graph.replay()
    torch.cuda.synchronize(dev)
    print(">>> Attention Block CUDAGraph Replay SUCCESSFUL! Output shape:", static_out.shape)
except Exception as e:
    print(">>> Attention Block Capture FAILED:", e)
