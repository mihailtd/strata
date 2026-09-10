import time
import torch
from transformers import AutoTokenizer
from pathlib import Path
from runtime.native_27b_engine import Native27BEngine, PreallocatedKVCache
import sys
sys.path.insert(0, ".")
from scratch.test_mtp_block import Qwen35MTPBlock

print("[Test MTP Accuracy] Loading 27B model...")
engine = Native27BEngine(num_layers=64, device="cuda:0")
engine.load_from_cache(force_convert=False)

print("[Test MTP Accuracy] Loading blk.64 (MTP block)...")
mtp_dict = torch.load("models/qwen3.8-27b-triton/layer_64.pt", map_location="cuda:0")
mtp = Qwen35MTPBlock(device="cuda:0")
mtp.load_weights(mtp_dict)

snap = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*"))[0]
tokenizer = AutoTokenizer.from_pretrained(str(snap))

prompt = (
    "<|im_start|>system\nYou are a high-performance Astral Python specialist.<|im_end|>\n"
    "<|im_start|>user\nConfigure a modern pyproject.toml workspace for UV with dependencies torch and triton.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n"
    "```toml\n[project]\nname = \"gnn-experiment\"\nversion = \"0.1.0\"\ndependencies = [\n"
)
tokens = tokenizer.encode(prompt)
print(f"Prompt length: {len(tokens)} tokens")

# Initialize engine KV cache
state = engine.init_kv_caches(batch_size=1, max_seq_len=len(tokens) + 64)
logits, state = engine.forward_prompt(tokens, state)
curr_token = int(torch.argmax(logits[0]).item())
print(f"Token after prefill: {curr_token} ({repr(tokenizer.decode([curr_token]))})")

# Dedicated KV cache for MTP
mtp_kv = PreallocatedKVCache(num_heads=4, head_dim=256, max_seq_len=len(tokens) + 64, device="cuda:0")

# Run prefill through MTP KV cache (MTP needs past context keys/values)
# In MTP, during prefill, tokens are processed to populate mtp_kv
print("\nEvaluating MTP next-token accuracy on 20 subsequent decode steps:")
matches = 0
total = 0

pos = len(tokens)
for step in range(20):
    # 1. Target model step (predicts y_{t+1})
    # We run forward_token to get next token and layer 63 hidden state
    # Let's see: how do we get layer 63 output?
    # In engine, layer 63 is the last layer before output_norm!
    # Let's inspect engine.forward_token
    cos_sin = engine._get_cos_sin(1, offset=pos)
    
    # Run through all 64 layers manually to capture h_t from layer 63
    x = engine.token_embd[curr_token : curr_token + 1].view(1, 1, -1)
    curr = x
    for i, layer in enumerate(engine.layers):
        if i % 4 == 3:
            curr, _ = layer(curr, state[f"kv_{i}"], cos_sin)
        else:
            curr, state[f"ssm_{i}"], state[f"conv_{i}"] = layer(curr, state[f"ssm_{i}"], state[f"conv_{i}"])
    
    h_t = curr # Output of layer 63 before output_norm!
    x_norm = engine.output_norm(curr)
    logits_target = engine.lm_head(x_norm)
    y_next = torch.argmax(logits_target[0, -1]).item()
    
    # 2. MTP block predicts d_{t+2} using [h_t, Token_Embd(y_next)]
    tok_emb_next = engine.token_embd[y_next : y_next + 1].view(1, 1, -1)
    mtp_cos_sin = engine._get_cos_sin(1, offset=pos)
    mtp_out = mtp(h_t, tok_emb_next, pos=pos, kv_cache=mtp_kv, cos_sin=mtp_cos_sin)
    logits_mtp = engine.lm_head(mtp_out)
    d_next2 = torch.argmax(logits_mtp[0, -1]).item()
    
    # 3. Now run the target model for the NEXT step to find actual y_{t+2}
    pos += 1
    curr_token = y_next
    cos_sin2 = engine._get_cos_sin(1, offset=pos)
    x2 = engine.token_embd[curr_token : curr_token + 1].view(1, 1, -1)
    curr2 = x2
    for i, layer in enumerate(engine.layers):
        if i % 4 == 3:
            curr2, _ = layer(curr2, state[f"kv_{i}"], cos_sin2)
        else:
            curr2, state[f"ssm_{i}"], state[f"conv_{i}"] = layer(curr2, state[f"ssm_{i}"], state[f"conv_{i}"])
    y_actual2 = torch.argmax(engine.lm_head(engine.output_norm(curr2))[0, -1]).item()
    
    is_match = (d_next2 == y_actual2)
    if is_match:
        matches += 1
    total += 1
    
    print(f"Step {step+1:02d}: Target(t+1)={repr(tokenizer.decode([y_next])):<12} | MTP(t+2)={repr(tokenizer.decode([d_next2])):<12} | Actual(t+2)={repr(tokenizer.decode([y_actual2])):<12} | {'MATCH [100%]' if is_match else 'MISS'}")
    curr_token = y_actual2
    pos += 1

acc = matches / total * 100
print(f"\n[Test MTP Accuracy] Acceptance Rate: {matches}/{total} ({acc:.1f}%)!")
