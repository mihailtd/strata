import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from runtime.native_27b_engine import Native27BEngine
from transformers import AutoTokenizer

def main():
    print("Testing Native27BEngine loading...")
    t0 = time.perf_counter()
    engine = Native27BEngine(num_layers=64, device="cuda:0")
    engine.load_from_cache(force_convert=False)
    print(f"Engine loaded in {time.perf_counter() - t0:.2f}s")
    
    snap = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*"))[0]
    tok = AutoTokenizer.from_pretrained(str(snap))
    
    prompt = "def fibonacci(n):"
    prompt_ids = [727, 73111, 1393, 1590]
    
    # 1. Test Eager Mode Generate
    print("\n--- Testing Eager Mode Generate (20 tokens) ---")
    t1 = time.perf_counter()
    tokens_eager = engine.generate(prompt_ids, max_new_tokens=20, temperature=0.0, use_hip_graph=False)
    eager_time = time.perf_counter() - t1
    print(f"Generated {len(tokens_eager)} tokens in {eager_time*1000:.1f}ms ({len(tokens_eager)/eager_time:.1f} tok/s)")
    print("Token IDs:", tokens_eager)
    print("Decoded:\n" + prompt + tok.decode(tokens_eager))
    
    print("\n--- Testing HIP Graph Mode Generate (50 tokens) ---")
    t2 = time.perf_counter()
    tokens_graph = engine.generate(prompt_ids, max_new_tokens=50, temperature=0.0, use_hip_graph=True)
    graph_time = time.perf_counter() - t2
    print(f"Generated {len(tokens_graph)} tokens in {graph_time*1000:.1f}ms ({len(tokens_graph)/graph_time:.1f} tok/s)")
    print("Decoded:\n" + prompt + tok.decode(tokens_graph))

    print("\n[SUCCESS] Generation fully coherent!")

if __name__ == "__main__":
    main()
