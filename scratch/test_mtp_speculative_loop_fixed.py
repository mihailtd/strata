import time
import torch
from transformers import AutoTokenizer
from pathlib import Path
from runtime.native_27b_engine import Native27BEngine, PreallocatedKVCache
import sys
sys.path.insert(0, ".")
from scratch.test_mtp_block import Qwen35MTPBlock

print("[MTP Spec Loop Fixed] Loading 27B model...")
engine = Native27BEngine(num_layers=64, device="cuda:0")
engine.load_from_cache(force_convert=False)

print("[MTP Spec Loop Fixed] Loading blk.64 (MTP block)...")
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
print(f"Greedy text:\n{repr(tokenizer.decode(greedy_tokens))}")

# 2. MTP Speculative Decode Function (Mathematical Equivalence Guaranteed)
def generate_mtp_speculative_exact(prompt_ids, max_new_tokens=40):
    state = engine.init_kv_caches(batch_size=1, max_seq_len=len(prompt_ids) + max_new_tokens + 32)
    mtp_kv = PreallocatedKVCache(num_heads=4, head_dim=256, max_seq_len=len(prompt_ids) + max_new_tokens + 32, device="cuda:0")
    
    # 1. Prefill
    logits, state = engine.forward_prompt(prompt_ids, state)
    curr_token = int(torch.argmax(logits[0]).item())
    
    generated = [curr_token]
    if curr_token in engine.STOP_TOKEN_IDS or max_new_tokens <= 1:
        return generated, 0, 0
    
    pos = len(prompt_ids)
    accepted_count = 0
    drafted_count = 0
    
    # Run curr_token through target model to get h_0 and initial prediction
    cos_sin = engine._get_cos_sin(1, offset=pos)
    x = engine.token_embd[curr_token : curr_token + 1].view(1, 1, -1)
    curr = x
    for i, layer in enumerate(engine.layers):
        if i % 4 == 3:
            curr, _ = layer(curr, state[f"kv_{i}"], cos_sin)
        else:
            curr, state[f"ssm_{i}"], state[f"conv_{i}"] = layer(curr, state[f"ssm_{i}"], state[f"conv_{i}"])
    
    h_curr = curr
    target_logits = engine.lm_head(engine.output_norm(curr))
    verified_token = int(torch.argmax(target_logits[0, -1]).item())
    generated.append(verified_token)
    pos += 1
    
    while len(generated) < max_new_tokens and verified_token not in engine.STOP_TOKEN_IDS:
        # Step 1: MTP drafts candidate d for pos+1 using [h_curr, Token_Embd(verified_token)]
        tok_emb = engine.token_embd[verified_token : verified_token + 1].view(1, 1, -1)
        mtp_cos_sin = engine._get_cos_sin(1, offset=pos)
        mtp_out = mtp(h_curr, tok_emb, pos=pos, kv_cache=mtp_kv, cos_sin=mtp_cos_sin)
        d_cand = int(torch.argmax(engine.lm_head(mtp_out)[0, -1]).item())
        drafted_count += 1
        
        # Step 2: Target model evaluates verified_token (this MUST happen in greedy anyway!)
        cos_sin = engine._get_cos_sin(1, offset=pos)
        x = engine.token_embd[verified_token : verified_token + 1].view(1, 1, -1)
        curr = x
        for i, layer in enumerate(engine.layers):
            if i % 4 == 3:
                curr, _ = layer(curr, state[f"kv_{i}"], cos_sin)
            else:
                curr, state[f"ssm_{i}"], state[f"conv_{i}"] = layer(curr, state[f"ssm_{i}"], state[f"conv_{i}"])
        
        h_verified = curr
        target_logits = engine.lm_head(engine.output_norm(curr))
        y_true = int(torch.argmax(target_logits[0, -1]).item())
        
        # Check: Did MTP correctly predict y_true?
        if d_cand == y_true:
            # ACCEPT!
            accepted_count += 1
            generated.append(y_true)
            if y_true in engine.STOP_TOKEN_IDS or len(generated) >= max_new_tokens:
                break
            
            # Since d_cand was accepted, y_true is already verified!
            # Now evaluate y_true with target model for next token
            pos += 1
            cos_sin2 = engine._get_cos_sin(1, offset=pos)
            x2 = engine.token_embd[y_true : y_true + 1].view(1, 1, -1)
            curr2 = x2
            for i, layer in enumerate(engine.layers):
                if i % 4 == 3:
                    curr2, _ = layer(curr2, state[f"kv_{i}"], cos_sin2)
                else:
                    curr2, state[f"ssm_{i}"], state[f"conv_{i}"] = layer(curr2, state[f"ssm_{i}"], state[f"conv_{i}"])
            
            h_curr = curr2
            target_logits2 = engine.lm_head(engine.output_norm(curr2))
            verified_token = int(torch.argmax(target_logits2[0, -1]).item())
            generated.append(verified_token)
            pos += 1
        else:
            # REJECT! MTP was wrong.
            # But y_true was produced by running verified_token through target model!
            # So state is ALREADY updated with verified_token!
            # y_true is the ground truth next token!
            generated.append(y_true)
            verified_token = y_true
            h_curr = h_verified
            pos += 1
            
    return generated, accepted_count, drafted_count

print("\n--- Running MTP Speculative Decode Exact ---")
torch.cuda.synchronize()
t0 = time.perf_counter()
spec_tokens, accepted, drafted = generate_mtp_speculative_exact(prompt_tokens, max_new_tokens=MAX_NEW_TOKENS)
torch.cuda.synchronize()
t_spec = time.perf_counter() - t0
spec_tok_s = len(spec_tokens) / t_spec
acc_rate = (accepted / drafted * 100) if drafted > 0 else 0.0

print(f"MTP Spec: {len(spec_tokens)} tokens in {t_spec:.2f}s -> {spec_tok_s:.2f} tok/s")
print(f"Acceptance Rate: {accepted}/{drafted} ({acc_rate:.1f}%)")
print(f"MTP Spec text:\n{repr(tokenizer.decode(spec_tokens))}")

# Verify Bit-Exact Equivalence
is_exact = (greedy_tokens == spec_tokens)
print(f"\nBit-Exact Equivalence to Greedy: {'PASSED (100% BIT-EXACT)' if is_exact else 'FAILED'}")
if not is_exact:
    print(f"Greedy len: {len(greedy_tokens)} vs Spec len: {len(spec_tokens)}")
    for idx, (g, s) in enumerate(zip(greedy_tokens, spec_tokens)):
        if g != s:
            print(f"First mismatch at token {idx}: greedy={g} ({repr(tokenizer.decode([g]))}) vs spec={s} ({repr(tokenizer.decode([s]))})")
            break
