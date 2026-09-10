import time
import torch
from transformers import AutoTokenizer
from pathlib import Path
from runtime.native_27b_engine import Native27BEngine, PreallocatedKVCache
import sys
sys.path.insert(0, ".")
from scratch.test_mtp_block import Qwen35MTPBlock

print("[Test MTP Integrated] Loading engine...")
engine = Native27BEngine(num_layers=64, device="cuda:0")
engine.load_from_cache(force_convert=False)

print("[Test MTP Integrated] Loading blk.64...")
mtp_dict = torch.load("models/qwen3.8-27b-triton/layer_64.pt", map_location="cuda:0")
engine.mtp_layer = Qwen35MTPBlock(device="cuda:0")
engine.mtp_layer.load_weights(mtp_dict)

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

# Monkey-patch forward_token to return hidden state
orig_forward_token = engine.forward_token

def forward_token_with_hidden(self, token_id, state_dict=None, pos=0, use_graph=True, return_hidden=False):
    state_dict = state_dict or {}
    cos_sin = self._get_cos_sin(1, offset=pos)
    if self.token_embd is not None and token_id < self.token_embd.shape[0]:
        x = self.token_embd[token_id : token_id + 1, :].to(self.device)
    else:
        x = torch.zeros((1, 5120), dtype=torch.bfloat16, device=self.device)

    new_states = {}
    can_use_graphs = use_graph and self.hip_graph_captured and len(self.ssm_graphs) == (self.num_layers // 4)
    if can_use_graphs:
        chunk_count = self.num_layers // 4
        for k in range(chunk_count):
            x = self.ssm_graphs[k].replay(x)
            for j in range(3):
                layer_idx = 4 * k + j
                new_states[f"ssm_{layer_idx}"] = self.ssm_graphs[k].ssm_states[j]
                new_states[f"conv_{layer_idx}"] = self.ssm_graphs[k].conv_states[j]

            attn_idx = 4 * k + 3
            attn_layer = self.layers[attn_idx]
            kv_s = state_dict.get(f"kv_{attn_idx}")
            x, new_kv_s = attn_layer(x, kv_cache=kv_s, cos_sin=cos_sin)
            new_states[f"kv_{attn_idx}"] = new_kv_s
    else:
        for i, layer in enumerate(self.layers):
            if i % 4 == 3:
                x, new_kv_s = layer(x, kv_cache=state_dict.get(f"kv_{i}"), cos_sin=cos_sin)
                new_states[f"kv_{i}"] = new_kv_s
            else:
                x, new_ssm_s, new_conv_s = layer(x, ssm_state=state_dict.get(f"ssm_{i}"), conv_state=state_dict.get(f"conv_{i}"))
                new_states[f"ssm_{i}"] = new_ssm_s
                new_states[f"conv_{i}"] = new_conv_s

    h_out = x
    x_final = self.output_norm(x)
    logits = self.lm_head(x_final)
    if return_hidden:
        return logits, new_states, h_out
    return logits, new_states

import types
engine.forward_token = types.MethodType(forward_token_with_hidden, engine)

# 1. Greedy Baseline
print("\n--- Running Greedy Baseline (HIP Graph) ---")
torch.cuda.synchronize()
t0 = time.perf_counter()
greedy_tokens = engine.generate(prompt_tokens, max_new_tokens=MAX_NEW_TOKENS, use_hip_graph=True)
torch.cuda.synchronize()
t_greedy = time.perf_counter() - t0
greedy_tok_s = len(greedy_tokens) / t_greedy
print(f"Greedy: {len(greedy_tokens)} tokens in {t_greedy:.2f}s -> {greedy_tok_s:.2f} tok/s")
print(f"Greedy text:\n{repr(tokenizer.decode(greedy_tokens))}")

# 2. Speculative Decode MTP
def generate_speculative_mtp(engine, prompt_ids, max_new_tokens=40, use_hip_graph=True):
    state_dict = engine.init_kv_caches(
        batch_size=1,
        max_seq_len=len(prompt_ids) + max_new_tokens + 32,
    )
    mtp_kv = PreallocatedKVCache(
        num_heads=4,
        head_dim=256,
        max_seq_len=len(prompt_ids) + max_new_tokens + 32,
        device=engine.device,
    )

    if use_hip_graph and engine.hip_graph_captured:
        engine.reset_hip_graphs()

    # 1. Prefill
    logits, state_dict = engine.forward_prompt(prompt_ids, state_dict)
    next_token = int(torch.argmax(logits[0, :]).item())
    generated = [next_token]
    if next_token in engine.STOP_TOKEN_IDS or max_new_tokens <= 1:
        return generated, 0, 0

    curr_token = next_token
    engine.sync_states_to_graphs(state_dict)
    pos = len(prompt_ids)

    # Initial target step
    logits_0, state_dict, h_curr = engine.forward_token(
        curr_token, state_dict, pos=pos, use_graph=use_hip_graph, return_hidden=True
    )
    verified_token = int(torch.argmax(logits_0[0, :]).item())
    generated.append(verified_token)
    if verified_token in engine.STOP_TOKEN_IDS:
        return generated, 0, 0
    pos += 1

    accepted_count = 0
    drafted_count = 0

    while len(generated) < max_new_tokens and verified_token not in engine.STOP_TOKEN_IDS:
        # Step A: MTP neural draft (1.0 ms)
        tok_emb = engine.token_embd[verified_token : verified_token + 1].view(1, 1, -1)
        mtp_cos_sin = engine._get_cos_sin(1, offset=pos)
        mtp_out = engine.mtp_layer(h_curr.view(1, 1, -1), tok_emb, pos=pos, kv_cache=mtp_kv, cos_sin=mtp_cos_sin)
        d_cand = int(torch.argmax(engine.lm_head(mtp_out)[0, -1]).item())
        drafted_count += 1

        # Step B: Target model evaluates verified_token
        logits_true, state_dict, h_verified = engine.forward_token(
            verified_token, state_dict, pos=pos, use_graph=use_hip_graph, return_hidden=True
        )
        y_true = int(torch.argmax(logits_true[0, :]).item())

        if d_cand == y_true:
            # Candidate ACCEPTED!
            accepted_count += 1
            generated.append(y_true)
            if y_true in engine.STOP_TOKEN_IDS or len(generated) >= max_new_tokens:
                break

            # Step C: Evaluate y_true immediately with target model for next token
            pos += 1
            logits_bonus, state_dict, h_bonus = engine.forward_token(
                y_true, state_dict, pos=pos, use_graph=use_hip_graph, return_hidden=True
            )
            y_bonus = int(torch.argmax(logits_bonus[0, :]).item())
            generated.append(y_bonus)
            pos += 1
            verified_token = y_bonus
            h_curr = h_bonus
        else:
            # Candidate REJECTED!
            generated.append(y_true)
            verified_token = y_true
            h_curr = h_verified
            pos += 1

    return generated, accepted_count, drafted_count

print("\n--- Running MTP Speculative Decode Integrated ---")
torch.cuda.synchronize()
t0 = time.perf_counter()
spec_tokens, accepted, drafted = generate_speculative_mtp(engine, prompt_tokens, max_new_tokens=MAX_NEW_TOKENS, use_hip_graph=True)
torch.cuda.synchronize()
t_spec = time.perf_counter() - t0
spec_tok_s = len(spec_tokens) / t_spec
acc_rate = (accepted / drafted * 100) if drafted > 0 else 0.0

print(f"MTP Spec: {len(spec_tokens)} tokens in {t_spec:.2f}s -> {spec_tok_s:.2f} tok/s")
print(f"Acceptance Rate: {accepted}/{drafted} ({acc_rate:.1f}%)")
print(f"MTP text:\n{repr(tokenizer.decode(spec_tokens))}")

is_exact = (greedy_tokens == spec_tokens)
print(f"\nBit-Exact Equivalence to Greedy: {'PASSED (100% BIT-EXACT)' if is_exact else 'FAILED'}")
