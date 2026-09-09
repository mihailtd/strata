"""Test dynamic position indexing and attention masking inside a CUDAGraph."""

import torch
import torch.nn.functional as F

dev = torch.device("cuda:0")
max_len = 16
num_heads = 4
head_dim = 256

# Preallocated static buffers
k_cache = torch.zeros((1, num_heads, max_len, head_dim), dtype=torch.bfloat16, device=dev)
v_cache = torch.zeros((1, num_heads, max_len, head_dim), dtype=torch.bfloat16, device=dev)
static_mask = torch.full((1, 1, 1, max_len), -1e4, dtype=torch.bfloat16, device=dev)
static_pos = torch.zeros((1,), dtype=torch.long, device=dev)

static_q = torch.randn((1, num_heads, 1, head_dim), dtype=torch.bfloat16, device=dev)
static_k_new = torch.randn((1, num_heads, 1, head_dim), dtype=torch.bfloat16, device=dev)
static_v_new = torch.randn((1, num_heads, 1, head_dim), dtype=torch.bfloat16, device=dev)
static_out = torch.zeros((1, num_heads, 1, head_dim), dtype=torch.bfloat16, device=dev)

def forward_step():
    # Dynamic in-place write via GPU index tensor
    k_cache.index_copy_(2, static_pos, static_k_new)
    v_cache.index_copy_(2, static_pos, static_v_new)
    
    # SDPA with static shape and dynamic mask
    attn = F.scaled_dot_product_attention(static_q, k_cache, v_cache, attn_mask=static_mask)
    static_out.copy_(attn)

# Warmup
stream = torch.cuda.Stream(device=dev)
stream.wait_stream(torch.cuda.current_stream(device=dev))
with torch.cuda.stream(stream):
    for _ in range(3):
        forward_step()
torch.cuda.current_stream(device=dev).wait_stream(stream)

k_cache.zero_()
v_cache.zero_()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    forward_step()

print("Graph captured successfully.")

# Test stepping through positions 0, 1, 2
for p in range(3):
    static_pos.fill_(p)
    static_mask[0, 0, 0, p] = 0.0
    static_k_new.fill_(float(p + 1))
    static_v_new.fill_(float(p + 1))
    graph.replay()
    torch.cuda.synchronize(dev)
    print(f"Step {p}: k_cache[:, :, {p}, 0] =", k_cache[0, 0, p, 0].item())

assert k_cache[0, 0, 0, 0].item() == 1.0
assert k_cache[0, 0, 1, 0].item() == 2.0
assert k_cache[0, 0, 2, 0].item() == 3.0
print(">>> DYNAMIC KV CACHE INDEXING INSIDE CUDAGRAPH VERIFIED! <<<")
