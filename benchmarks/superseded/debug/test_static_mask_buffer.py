import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

def test_static_mask_buffer():
    model_id = "Qwen/Qwen3.5-9B"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    prompt = "Add ruff and ty as dev dependencies, then format and lint the whole codebase."
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    input_ids = tokenizer.encode(formatted, return_tensors="pt").cuda()
    cur = input_ids.shape[1]
    N = 40

    # 1. Eager baseline for N steps
    cache_eager = StaticCache(config=model.config, max_batch_size=1, max_cache_len=2048, device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        out_eager = model(input_ids, past_key_values=cache_eager, cache_position=torch.arange(0, cur, device="cuda:0"), use_cache=True)
    
    eager_toks = []
    nxt = torch.argmax(out_eager.logits[:, -1, :], -1, keepdim=True)
    eager_toks.append(nxt.item())
    pos = cur
    for _ in range(N - 1):
        with torch.no_grad():
            out = model(nxt, past_key_values=cache_eager, cache_position=torch.tensor([pos], device="cuda:0"), use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        eager_toks.append(nxt.item())
        pos += 1

    print("Eager generated text:\n", repr(tokenizer.decode(eager_toks)))

    # 2. Graph capture with static 4D mask buffer
    cache_graph = StaticCache(config=model.config, max_batch_size=1, max_cache_len=2048, device="cuda:0", dtype=torch.bfloat16)
    ids_buf = torch.zeros((1, 1), dtype=torch.long, device="cuda:0")
    pos_ids_buf = torch.zeros((1, 1), dtype=torch.long, device="cuda:0")
    cache_pos_buf = torch.zeros((1,), dtype=torch.long, device="cuda:0")
    
    # Static 4D mask buffer
    min_val = torch.finfo(torch.bfloat16).min
    mask_buf = torch.full((1, 1, 1, 2048), min_val, dtype=torch.bfloat16, device="cuda:0")
    mask_buf[:, :, :, :cur] = 0.0

    # Capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            model(ids_buf, attention_mask=mask_buf, position_ids=pos_ids_buf, cache_position=cache_pos_buf, past_key_values=cache_graph, use_cache=True)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s), torch.no_grad():
        out_g = model(ids_buf, attention_mask=mask_buf, position_ids=pos_ids_buf, cache_position=cache_pos_buf, past_key_values=cache_graph, use_cache=True)
        g_logits = out_g.logits
    torch.cuda.current_stream().wait_stream(s)

    # Clean prefill
    cache_graph.reset()
    with torch.no_grad():
        out_graph = model(input_ids, past_key_values=cache_graph, cache_position=torch.arange(0, cur, device="cuda:0"), use_cache=True)

    len_counters = [getattr(l, "cumulative_length", None) for l in cache_graph.layers if isinstance(getattr(l, "cumulative_length", None), torch.Tensor)]
    ctr_src = torch.zeros_like(len_counters[0])
    ctr_srcs = [ctr_src] * len(len_counters)

    # Replay N steps
    graph_toks = []
    nxt = torch.argmax(out_graph.logits[:, -1, :], -1, keepdim=True)
    graph_toks.append(nxt.item())
    pos = cur
    for step in range(N - 1):
        ids_buf.copy_(nxt)
        pos_ids_buf.copy_(torch.tensor([[pos]], device="cuda:0"))
        cache_pos_buf.copy_(torch.tensor([pos], device="cuda:0"))
        
        # Update mask: unmask up to pos + 1
        mask_buf[:, :, :, :pos + 1] = 0.0
        mask_buf[:, :, :, pos + 1:] = min_val
        
        ctr_src.fill_(pos)
        torch._foreach_copy_(len_counters, ctr_srcs)
        g.replay()
        nxt = torch.argmax(g_logits[0, -1, :], -1, keepdim=True)
        graph_toks.append(nxt.item())
        pos += 1

    print("\nGraph generated text:\n", repr(tokenizer.decode(graph_toks)))
    print(f"BIT-EXACT MATCH across all {N} tokens: {eager_toks == graph_toks}")

if __name__ == "__main__":
    test_static_mask_buffer()
