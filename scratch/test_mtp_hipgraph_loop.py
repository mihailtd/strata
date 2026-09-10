import time
import torch
from transformers import AutoTokenizer
from pathlib import Path
from runtime.native_27b_engine import Native27BEngine, PreallocatedKVCache
import sys
sys.path.insert(0, ".")
from scratch.test_mtp_block import Qwen35MTPBlock

print("[MTP HIP Graph Test] Loading engine...")
engine = Native27BEngine(num_layers=64, device="cuda:0")
engine.load_from_cache(force_convert=False)

print("[MTP HIP Graph Test] Loading MTP blk.64...")
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
prompt_tokens = tokenizer.encode(prompt)
MAX_NEW_TOKENS = 40

# 1. Greedy with HIP Graph
print("\n--- Running Greedy Baseline with HIP Graph ---")
torch.cuda.synchronize()
t0 = time.perf_counter()
greedy_tokens = engine.generate(prompt_tokens, max_new_tokens=MAX_NEW_TOKENS, use_hip_graph=True)
torch.cuda.synchronize()
t_greedy = time.perf_counter() - t0
greedy_tok_s = len(greedy_tokens) / t_greedy
print(f"Greedy (HIP Graph): {len(greedy_tokens)} tokens in {t_greedy:.2f}s -> {greedy_tok_s:.2f} tok/s")

# 2. Custom forward_token that returns hidden state h_t before output_norm
def forward_token_h(engine, token_id, state_dict, pos, use_graph=True):
    cos_sin = engine._get_cos_sin(1, offset=pos)
    if engine.token_embd is not None and token_id < engine.token_embd.shape[0]:
        x = engine.token_embd[token_id : token_id + 1, :].to(engine.device)
    else:
        x = torch.zeros((1, 5120), dtype=torch.bfloat16, device=engine.device)

    chunk_count = engine.num_layers // 4
    if use_graph and engine.hip_graph_captured and len(engine.ssm_graphs) == chunk_count:
        for k in range(chunk_count):
            x = engine.ssm_graphs[k].replay(x)
            attn_layer = engine.layers[4 * k + 3]
            kv_s = state_dict[f"kv_{4 * k + 3}"]
            x, _ = attn_layer(x, kv_cache=kv_s, cos_sin=cos_sin)
    else:
        for i, layer in enumerate(engine.layers):
            if i % 4 == 3:
                x, _ = layer(x, kv_cache=state_dict[f"kv_{i}"], cos_sin=cos_sin)
            else:
                x, state_dict[f"ssm_{i}"], state_dict[f"conv_{i}"] = layer(x, ssm_state=state_dict[f"ssm_{i}"], conv_state=state_dict[f"conv_{i}"])

    h_t = x # Output of layer 63 before output_norm
    x_final = engine.output_norm(x)
    logits = engine.lm_head(x_final)
    return logits, h_t


def generate_mtp_hip_speculative(prompt_ids, max_new_tokens=40):
    state = engine.init_kv_caches(batch_size=1, max_seq_len=len(prompt_ids) + max_new_tokens + 32)
    mtp_kv = PreallocatedKVCache(num_heads=4, head_dim=256, max_seq_len=len(prompt_ids) + max_new_tokens + 32, device="cuda:0")

    # Prefill
    logits, state = engine.forward_prompt(prompt_ids, state)
    curr_token = int(torch.argmax(logits[0]).item())
    generated = [curr_token]
    if curr_token in engine.STOP_TOKEN_IDS or max_new_tokens <= 1:
        return generated, 0, 0

    engine.sync_states_to_graphs(state)
    pos = len(prompt_ids)

    # Initial target step on curr_token
    logits_0, h_curr = forward_token_h(engine, curr_token, state, pos=pos, use_graph=True)
    verified_token = int(torch.argmax(logits_0[0, -1]).item())
    generated.append(verified_token)
    pos += 1

    accepted_count = 0
    drafted_count = 0

    while len(generated) < max_new_tokens and verified_token not in engine.STOP_TOKEN_IDS:
        # Step A: MTP drafts candidate token for next position (takes ~1.0 ms)
        tok_emb = engine.token_embd[verified_token : verified_token + 1].view(1, 1, -1)
        mtp_cos_sin = engine._get_cos_sin(1, offset=pos)
        mtp_out = mtp(h_curr.view(1, 1, -1), tok_emb, pos=pos, kv_cache=mtp_kv, cos_sin=mtp_cos_sin)
        d_cand = int(torch.argmax(engine.lm_head(mtp_out)[0, -1]).item())
        drafted_count += 1

        # Step B: Target model evaluates verified_token (using pre-compiled HIP Graph)
        logits_true, h_verified = forward_token_h(engine, verified_token, state, pos=pos, use_graph=True)
        y_true = int(torch.argmax(logits_true[0, -1]).item())

        if d_cand == y_true:
            # ACCEPTED!
            accepted_count += 1
            generated.append(y_true)
            if y_true in engine.STOP_TOKEN_IDS or len(generated) >= max_new_tokens:
                break
            
            # Step C: Evaluate y_true immediately with target model
            pos += 1
            logits_bonus, h_bonus = forward_token_h(engine, y_true, state, pos=pos, use_graph=True)
            y_bonus = int(torch.argmax(logits_bonus[0, -1]).item())
            generated.append(y_bonus)
            pos += 1
            verified_token = y_bonus
            h_curr = h_bonus
        else:
            # REJECTED!
            generated.append(y_true)
            verified_token = y_true
            h_curr = h_verified
            pos += 1

    return generated, accepted_count, drafted_count

print("\n--- Running MTP Speculative Decode with HIP Graph ---")
torch.cuda.synchronize()
t0 = time.perf_counter()
spec_tokens, accepted, drafted = generate_mtp_hip_speculative(prompt_tokens, max_new_tokens=MAX_NEW_TOKENS)
torch.cuda.synchronize()
t_spec = time.perf_counter() - t0
spec_tok_s = len(spec_tokens) / t_spec
acc_rate = (accepted / drafted * 100) if drafted > 0 else 0.0

print(f"MTP Spec (HIP Graph): {len(spec_tokens)} tokens in {t_spec:.2f}s -> {spec_tok_s:.2f} tok/s")
print(f"Acceptance Rate: {accepted}/{drafted} ({acc_rate:.1f}%)")
print(f"MTP text:\n{repr(tokenizer.decode(spec_tokens))}")

is_exact = (greedy_tokens == spec_tokens)
print(f"\nBit-Exact Equivalence: {'PASSED (100% BIT-EXACT)' if is_exact else 'FAILED'}")
