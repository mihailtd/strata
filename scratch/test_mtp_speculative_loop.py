import time
import torch
from transformers import AutoTokenizer
from pathlib import Path
from runtime.native_27b_engine import Native27BEngine, PreallocatedKVCache
import sys
sys.path.insert(0, ".")
from scratch.test_mtp_block import Qwen35MTPBlock

print("[MTP Spec Loop] Loading 27B model...")
engine = Native27BEngine(num_layers=64, device="cuda:0")
engine.load_from_cache(force_convert=False)

print("[MTP Spec Loop] Loading blk.64 (MTP block)...")
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

# 1. Greedy Baseline
print("\n--- Running Greedy Baseline ---")
torch.cuda.synchronize()
t0 = time.perf_counter()
greedy_tokens = engine.generate(prompt_tokens, max_new_tokens=MAX_NEW_TOKENS, use_hip_graph=True)
torch.cuda.synchronize()
t_greedy = time.perf_counter() - t0
greedy_tok_s = len(greedy_tokens) / t_greedy
print(f"Greedy: {len(greedy_tokens)} tokens in {t_greedy:.2f}s -> {greedy_tok_s:.2f} tok/s")
print(f"Greedy text:\n{repr(tokenizer.decode(greedy_tokens))[:150]}")

# 2. MTP Speculative Decode Function
def generate_mtp_speculative(prompt_ids, max_new_tokens=40):
    state = engine.init_kv_caches(batch_size=1, max_seq_len=len(prompt_ids) + max_new_tokens + 16)
    mtp_kv = PreallocatedKVCache(num_heads=4, head_dim=256, max_seq_len=len(prompt_ids) + max_new_tokens + 16, device="cuda:0")
    
    # Prefill
    logits, state = engine.forward_prompt(prompt_ids, state)
    curr_token = int(torch.argmax(logits[0]).item())
    
    generated = [curr_token]
    if curr_token in engine.STOP_TOKEN_IDS or max_new_tokens <= 1:
        return generated
    
    pos = len(prompt_ids)
    
    # Also get h_last from prompt for MTP
    # We can get h_last by running 1 token forward with curr_token, or forward_prompt can return h_last
    # Let's run forward_token on curr_token to get first h
    accepted_count = 0
    drafted_count = 0
    
    while len(generated) < max_new_tokens:
        # Step A: Run target model on curr_token (capturing h and next target prediction)
        cos_sin = engine._get_cos_sin(1, offset=pos)
        x = engine.token_embd[curr_token : curr_token + 1].view(1, 1, -1)
        curr = x
        for i, layer in enumerate(engine.layers):
            if i % 4 == 3:
                curr, _ = layer(curr, state[f"kv_{i}"], cos_sin)
            else:
                curr, state[f"ssm_{i}"], state[f"conv_{i}"] = layer(curr, state[f"ssm_{i}"], state[f"conv_{i}"])
        
        h_t = curr # Layer 63 hidden state
        target_logits = engine.lm_head(engine.output_norm(curr))
        y_next = int(torch.argmax(target_logits[0, -1]).item())
        generated.append(y_next)
        if y_next in engine.STOP_TOKEN_IDS or len(generated) >= max_new_tokens:
            break
        pos += 1
        
        # Step B: MTP drafts candidate d for pos+1 using [h_t, Token_Embd(y_next)]
        tok_emb = engine.token_embd[y_next : y_next + 1].view(1, 1, -1)
        mtp_cos_sin = engine._get_cos_sin(1, offset=pos)
        mtp_out = mtp(h_t, tok_emb, pos=pos, kv_cache=mtp_kv, cos_sin=mtp_cos_sin)
        d_cand = int(torch.argmax(engine.lm_head(mtp_out)[0, -1]).item())
        drafted_count += 1
        
        # Step C: Speculatively test candidate d_cand at pos+1
        # Save state snapshot for rollback if rejected
        saved_ssm = {f"ssm_{i}": state[f"ssm_{i}"].clone() for i in range(len(engine.layers)) if i % 4 != 3}
        saved_conv = {f"conv_{i}": state[f"conv_{i}"].clone() for i in range(len(engine.layers)) if i % 4 != 3}
        saved_mtp_len = mtp_kv.current_len
        saved_attn_lens = {f"kv_{i}": state[f"kv_{i}"].current_len for i in range(len(engine.layers)) if i % 4 == 3}
        
        cos_sin_cand = engine._get_cos_sin(1, offset=pos)
        x_cand = engine.token_embd[y_next : y_next + 1].view(1, 1, -1)
        curr_cand = x_cand
        for i, layer in enumerate(engine.layers):
            if i % 4 == 3:
                curr_cand, _ = layer(curr_cand, state[f"kv_{i}"], cos_sin_cand)
            else:
                curr_cand, state[f"ssm_{i}"], state[f"conv_{i}"] = layer(curr_cand, state[f"ssm_{i}"], state[f"conv_{i}"])
        
        target_cand_logits = engine.lm_head(engine.output_norm(curr_cand))
        actual_cand = int(torch.argmax(target_cand_logits[0, -1]).item())
        
        if actual_cand == d_cand:
            # ACCEPTED!
            accepted_count += 1
            generated.append(d_cand)
            pos += 1
            curr_token = d_cand
            if d_cand in engine.STOP_TOKEN_IDS:
                break
        else:
            # REJECTED! Rollback state to y_next
            for k_ssm, v_ssm in saved_ssm.items():
                state[k_ssm].copy_(v_ssm)
            for k_conv, v_conv in saved_conv.items():
                state[k_conv].copy_(v_conv)
            for k_attn, l_attn in saved_attn_lens.items():
                state[k_attn].current_len = l_attn
            mtp_kv.current_len = saved_mtp_len
            curr_token = actual_cand # True next token from target model!
            generated.append(actual_cand)
            pos += 1
            if actual_cand in engine.STOP_TOKEN_IDS:
                break
                
    return generated, accepted_count, drafted_count

print("\n--- Running MTP Speculative Decode ---")
torch.cuda.synchronize()
t0 = time.perf_counter()
spec_tokens, accepted, drafted = generate_mtp_speculative(prompt_tokens, max_new_tokens=MAX_NEW_TOKENS)
torch.cuda.synchronize()
t_spec = time.perf_counter() - t0
spec_tok_s = len(spec_tokens) / t_spec
acc_rate = (accepted / drafted * 100) if drafted > 0 else 0.0

print(f"MTP Spec: {len(spec_tokens)} tokens in {t_spec:.2f}s -> {spec_tok_s:.2f} tok/s")
print(f"Acceptance Rate: {accepted}/{drafted} ({acc_rate:.1f}%)")
print(f"MTP Spec text:\n{repr(tokenizer.decode(spec_tokens))[:150]}")

# Verify Bit-Exact Equivalence
is_exact = (greedy_tokens == spec_tokens)
print(f"\nBit-Exact Equivalence to Greedy: {'PASSED (100% BIT-EXACT)' if is_exact else 'FAILED'}")
if not is_exact:
    for idx, (g, s) in enumerate(zip(greedy_tokens, spec_tokens)):
        if g != s:
            print(f"First mismatch at token {idx}: greedy={g} ({repr(tokenizer.decode([g]))}) vs spec={s} ({repr(tokenizer.decode([s]))})")
            break
