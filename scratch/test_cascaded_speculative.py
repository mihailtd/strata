import time
import torch
from pathlib import Path
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from runtime.native_27b_engine import Native27BEngine
from runtime.state_handoff_27b import get_27b_tokenizer
from runtime.state_handoff_27b import StateHandoffSession, AgentTurn

print("=== Benchmarking Cascaded N-Gram + MTP Speculative Decode ===")
engine = Native27BEngine(num_layers=64, device="cuda:0")
engine.load_from_cache(force_convert=False)

tokenizer = get_27b_tokenizer()

prompt = (
    "<|im_start|>user\n"
    "Write a Python class UserDatabase with methods: create_user(name, email), "
    "get_user(user_id), delete_user(user_id), and list_users(). Include docstrings and type hints.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
prompt_tokens = tokenizer.encode(prompt)
MAX_TOKENS = 64

# Warmup pass
print("\nWarming up...")
_ = engine.generate(prompt_tokens[:16], max_new_tokens=4, use_hip_graph=True)
if torch.cuda.is_available():
    torch.cuda.synchronize()

# 1. Native Greedy Baseline
print("\n--- 1. Native Greedy Baseline ---")
if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.perf_counter()
greedy_tokens = engine.generate(prompt_tokens, max_new_tokens=MAX_TOKENS, use_hip_graph=True)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_greedy = time.perf_counter() - t0
tok_s_greedy = len(greedy_tokens) / t_greedy
print(f"Greedy: {len(greedy_tokens)} tokens in {t_greedy:.3f}s -> {tok_s_greedy:.2f} tok/s")

# 2. Cascaded N-Gram + MTP on Native Engine
print("\n--- 2. Cascaded N-Gram + MTP Speculative Decode (Native27BEngine) ---")
if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.perf_counter()
spec_tokens = engine.generate_speculative(
    prompt_tokens,
    max_new_tokens=MAX_TOKENS,
    use_hip_graph=True,
    draft_k=3,
    draft_n=5,
    min_n=3,
    use_mtp=True,
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_spec = time.perf_counter() - t0
tok_s_spec = len(spec_tokens) / t_spec
print(f"Speculative: {len(spec_tokens)} tokens in {t_spec:.3f}s -> {tok_s_spec:.2f} tok/s")
print(f"Speedup vs Greedy: {tok_s_spec / tok_s_greedy:.2f}x")

# 3. StateHandoffSession Multi-Turn with Cascaded Speculative
print("\n--- 3. StateHandoffSession Multi-Turn Test ---")
session = StateHandoffSession(engine=engine, clone_on_handoff=True)

turn1 = AgentTurn(
    agent_id="architect",
    role="Architect",
    instruction="Define a Pydantic model for User with id: int, name: str, email: str.",
    max_new_tokens=MAX_TOKENS,
    use_speculative=True,
)
r1 = session.execute_turn(turn1)
print(f"Turn 1 ({r1.agent_id}): {r1.tokens_generated} tokens in {r1.decode_ms:.1f}ms ({r1.tok_per_sec:.2f} tok/s), prefill={r1.prefill_ms:.1f}ms")

turn2 = AgentTurn(
    agent_id="engineer",
    role="Engineer",
    instruction="Write a function to create a new user and validate email format.",
    max_new_tokens=MAX_TOKENS,
    use_speculative=True,
)
r2 = session.execute_turn(turn2)
print(f"Turn 2 ({r2.agent_id}): {r2.tokens_generated} tokens in {r2.decode_ms:.1f}ms ({r2.tok_per_sec:.2f} tok/s), handoff={r2.handoff_ms:.3f}ms, avoided={r2.tokens_avoided}")

print("\n=== Benchmark Complete ===")
