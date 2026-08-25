import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from runtime.state_ring_buffer import RingBufferReplayEngine

def test_chunk_capture_and_rollback():
    model_id = "Qwen/Qwen3.5-9B"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    prompt = "Add ruff and ty as dev dependencies, then format and lint the whole codebase."
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    input_ids = tokenizer.encode(formatted, return_tensors="pt").cuda()
    cur = input_ids.shape[1]

    # Create StaticCache
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=2048, device="cuda:0", dtype=torch.bfloat16)

    # Allocate buffers and capture graphs for widths 1, 2, 3 WITHOUT attention_mask
    buckets = {}
    for width in [1, 2, 3]:
        ids = torch.zeros((1, width), dtype=torch.long, device="cuda:0")
        pos_ids = torch.zeros((1, width), dtype=torch.long, device="cuda:0")
        cache_pos = torch.zeros((width,), dtype=torch.long, device="cuda:0")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                model(ids, position_ids=pos_ids, cache_position=cache_pos, past_key_values=cache, use_cache=True, output_hidden_states=True)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s), torch.no_grad():
            out = model(ids, position_ids=pos_ids, cache_position=cache_pos, past_key_values=cache, use_cache=True, output_hidden_states=True)
            logits = out.logits
            hidden = out.hidden_states[-1]
        torch.cuda.current_stream().wait_stream(s)

        buckets[width] = {
            "graph": g,
            "ids": ids,
            "pos_ids": pos_ids,
            "cache_pos": cache_pos,
            "logits": logits,
            "hidden": hidden,
        }

    ring_engine = RingBufferReplayEngine(cache, max_depth=64, mode="selective_hybrid")

    # RESET cache fresh for inference
    cache.reset()
    with torch.no_grad():
        out_prefill = model(
            input_ids,
            past_key_values=cache,
            cache_position=torch.arange(0, cur, device="cuda:0"),
            use_cache=True,
            output_hidden_states=True,
        )
    nxt = torch.argmax(out_prefill.logits[:, -1, :], -1, keepdim=True)

    len_counters = [getattr(l, "cumulative_length", None) for l in cache.layers if isinstance(getattr(l, "cumulative_length", None), torch.Tensor)]
    ctr_src = torch.zeros_like(len_counters[0])
    ctr_srcs = [ctr_src] * len(len_counters)

    def replay(width, tokens, start_pos):
        b = buckets[width]
        b["ids"].copy_(tokens.view(1, width))
        p = torch.arange(start_pos, start_pos + width, device="cuda:0")
        b["pos_ids"].copy_(p.view(1, width))
        b["cache_pos"].copy_(p)
        ctr_src.fill_(start_pos)
        torch._foreach_copy_(len_counters, ctr_srcs)
        b["graph"].replay()
        return b["logits"], b["hidden"]

    # Test replay of width 1 for 10 steps sequentially
    toks = [nxt.item()]
    pos = cur
    for step in range(15):
        logits, _ = replay(1, nxt, pos)
        nxt = torch.argmax(logits[0, -1, :], -1, keepdim=True)
        toks.append(nxt.item())
        pos += 1

    print("Sequential replay tokens (15 steps):")
    print(repr(tokenizer.decode(toks)))

if __name__ == "__main__":
    test_chunk_capture_and_rollback()
