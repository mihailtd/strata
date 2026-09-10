import time
import torch
from transformers import AutoTokenizer
from pathlib import Path
from runtime.native_27b_engine import Native27BEngine, PreallocatedKVCache, apply_rotary_emb
import sys
sys.path.insert(0, ".")
from scratch.test_mtp_block import Qwen35MTPBlock

print("[MTP Parallel Verify] Loading 27B model...")
engine = Native27BEngine(num_layers=64, device="cuda:0")
engine.load_from_cache(force_convert=False)

print("[MTP Parallel Verify] Loading blk.64 (MTP block)...")
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

# 2. Parallel 2-Token Verification Pass
def forward_verify_2(engine, tokens_2, state, pos):
    """Executes parallel forward pass for 2 candidate tokens [x1, x2] returning logits and intermediate states."""
    s = 2
    x = engine.token_embd[tokens_2, :].unsqueeze(0).to(engine.device) # (1, 2, 5120)
    cos_sin = engine._get_cos_sin(seq_len=s, offset=pos)
    
    tot_len = pos + s
    mask = torch.zeros((1, 1, s, tot_len), dtype=torch.bool, device=engine.device)
    mask[0, 0, 0, : pos + 1] = True
    mask[0, 0, 1, : pos + 2] = True
    
    history_ssm = {}
    history_conv = {}
    
    curr = x
    b, seq_len, _ = curr.shape
    for i, layer in enumerate(engine.layers):
        if i % 4 == 3:
            # Attention block
            x_norm = layer.attn_norm(curr)
            q_full = layer.attn_q(x_norm) if layer.attn_q else x_norm
            query_states, gate = torch.chunk(q_full.view(b, s, 24, 256 * 2), 2, dim=-1)
            gate = gate.reshape(b, s, -1)
            q = layer.attn_q_norm(query_states).transpose(1, 2)
            k_proj = layer.attn_k_norm(layer.attn_k(x_norm).view(b, s, 4, 256)).transpose(1, 2)
            v_proj = layer.attn_v(x_norm).view(b, s, 4, 256).transpose(1, 2)
            cos, sin = cos_sin
            q = apply_rotary_emb(q, cos, sin)
            k_proj = apply_rotary_emb(k_proj, cos, sin)
            k_all, v_all = state[f"kv_{i}"].update(k_proj, v_proj)
            k_rep = k_all.repeat_interleave(6, dim=1)
            v_rep = v_all.repeat_interleave(6, dim=1)
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask)
            attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1) * torch.sigmoid(gate)
            curr = curr + (layer.attn_output(attn_out) if layer.attn_output else attn_out)
            x_ffn_norm = layer.post_attention_norm(curr)
            mlp_out = layer.ffn_down(torch.nn.functional.silu(layer.ffn_gate(x_ffn_norm)) * layer.ffn_up(x_ffn_norm))
            curr = curr + mlp_out
        else:
            # SSM Block
            s_hist = []
            c_hist = []
            curr_step = curr
            # Step token 0
            out0, ssm0, conv0 = layer(curr_step[:, 0:1, :], state[f"ssm_{i}"], state[f"conv_{i}"])
            s_hist.append(ssm0.clone())
            c_hist.append(conv0.clone())
            # Step token 1
            out1, ssm1, conv1 = layer(curr_step[:, 1:2, :], ssm0, conv0)
            s_hist.append(ssm1.clone())
            c_hist.append(conv1.clone())
            
            curr = torch.cat([out0, out1], dim=1)
            history_ssm[f"ssm_{i}"] = s_hist
            history_conv[f"conv_{i}"] = c_hist
            state[f"ssm_{i}"] = ssm1
            state[f"conv_{i}"] = conv1

    x_norm = engine.output_norm(curr)
    logits = engine.lm_head(x_norm) # (1, 2, vocab_size)
    h_layer63 = curr # (1, 2, 5120)
    return logits[0], h_layer63[0], history_ssm, history_conv


def generate_mtp_parallel_spec(prompt_ids, max_new_tokens=40):
    state = engine.init_kv_caches(batch_size=1, max_seq_len=len(prompt_ids) + max_new_tokens + 32)
    mtp_kv = PreallocatedKVCache(num_heads=4, head_dim=256, max_seq_len=len(prompt_ids) + max_new_tokens + 32, device="cuda:0")
    
    # 1. Prefill
    logits, state = engine.forward_prompt(prompt_ids, state)
    t1 = int(torch.argmax(logits[0]).item())
    generated = [t1]
    if t1 in engine.STOP_TOKEN_IDS or max_new_tokens <= 1:
        return generated, 0, 0
    
    pos = len(prompt_ids)
    accepted_count = 0
    drafted_count = 0
    
    # Run t1 through single step to get initial h
    cos_sin = engine._get_cos_sin(1, offset=pos)
    x = engine.token_embd[t1 : t1 + 1].view(1, 1, -1)
    curr = x
    for i, layer in enumerate(engine.layers):
        if i % 4 == 3:
            curr, _ = layer(curr, state[f"kv_{i}"], cos_sin)
        else:
            curr, state[f"ssm_{i}"], state[f"conv_{i}"] = layer(curr, state[f"ssm_{i}"], state[f"conv_{i}"])
    
    h_last = curr[0, 0] # (5120,)
    target_logits = engine.lm_head(engine.output_norm(curr))
    verified_token = int(torch.argmax(target_logits[0, -1]).item())
    generated.append(verified_token)
    pos += 1
    
    while len(generated) < max_new_tokens and verified_token not in engine.STOP_TOKEN_IDS:
        # Step A: MTP drafts candidate d for pos+1 using [h_last, Token_Embd(verified_token)]
        tok_emb = engine.token_embd[verified_token : verified_token + 1].view(1, 1, -1)
        mtp_cos_sin = engine._get_cos_sin(1, offset=pos)
        mtp_out = mtp(h_last.view(1, 1, -1), tok_emb, pos=pos, kv_cache=mtp_kv, cos_sin=mtp_cos_sin)
        d_cand = int(torch.argmax(engine.lm_head(mtp_out)[0, -1]).item())
        drafted_count += 1
        
        # Step B: Parallel verify [verified_token, d_cand]
        # In this 2-token pass:
        # token 0: verified_token -> predicts y_true
        # token 1: d_cand -> predicts bonus token y_bonus
        cand_pair = [verified_token, d_cand]
        logits_2, h_2, hist_ssm, hist_conv = forward_verify_2(engine, cand_pair, state, pos=pos)
        
        y_true = int(torch.argmax(logits_2[0]).item())
        
        if y_true == d_cand:
            # ACCEPTED!
            accepted_count += 1
            generated.append(d_cand)
            if d_cand in engine.STOP_TOKEN_IDS or len(generated) >= max_new_tokens:
                break
                
            y_bonus = int(torch.argmax(logits_2[1]).item())
            generated.append(y_bonus)
            pos += 2
            verified_token = y_bonus
            h_last = h_2[1]
        else:
            # REJECTED! Rollback index 1 state to index 0 (token verified_token was valid, d_cand was wrong)
            for k_ssm, s_list in hist_ssm.items():
                state[k_ssm].copy_(s_list[0])
            for k_conv, c_list in hist_conv.items():
                state[k_conv].copy_(c_list[0])
            for i in range(len(engine.layers)):
                if i % 4 == 3:
                    state[f"kv_{i}"].current_len -= 1 # roll back 1 token
            
            generated.append(y_true)
            pos += 1
            verified_token = y_true
            h_last = h_2[0]

    return generated, accepted_count, drafted_count

print("\n--- Running MTP Parallel Speculative Decode ---")
torch.cuda.synchronize()
t0 = time.perf_counter()
spec_tokens, accepted, drafted = generate_mtp_parallel_spec(prompt_tokens, max_new_tokens=MAX_NEW_TOKENS)
torch.cuda.synchronize()
t_spec = time.perf_counter() - t0
spec_tok_s = len(spec_tokens) / t_spec
acc_rate = (accepted / drafted * 100) if drafted > 0 else 0.0

print(f"MTP Parallel Spec: {len(spec_tokens)} tokens in {t_spec:.2f}s -> {spec_tok_s:.2f} tok/s")
print(f"Acceptance Rate: {accepted}/{drafted} ({acc_rate:.1f}%)")
print(f"MTP Parallel Spec text:\n{repr(tokenizer.decode(spec_tokens))}")

is_exact = (greedy_tokens == spec_tokens)
print(f"\nBit-Exact Equivalence: {'PASSED (100% BIT-EXACT)' if is_exact else 'FAILED'}")
if not is_exact:
    for idx, (g, s) in enumerate(zip(greedy_tokens, spec_tokens)):
        if g != s:
            print(f"First mismatch at token {idx}: greedy={g} ({repr(tokenizer.decode([g]))}) vs spec={s} ({repr(tokenizer.decode([s]))})")
            break
