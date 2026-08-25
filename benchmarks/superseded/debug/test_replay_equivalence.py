import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

def test_qwen_rotary_and_replay():
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

    # Run 1: Normal eager forward pass
    cache_eager = StaticCache(config=model.config, max_batch_size=1, max_cache_len=2048, device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        out_eager = model(
            input_ids,
            past_key_values=cache_eager,
            cache_position=torch.arange(0, cur, device="cuda:0"),
            use_cache=True,
            output_hidden_states=True,
        )
    next_tok_eager = torch.argmax(out_eager.logits[:, -1, :], -1)
    print("Eager next token id:", next_tok_eager.item(), repr(tokenizer.decode([next_tok_eager.item()])))

    # Step 1: Eager step 1 token
    tok_in = next_tok_eager.view(1, 1)
    with torch.no_grad():
        out_step1 = model(
            tok_in,
            past_key_values=cache_eager,
            cache_position=torch.tensor([cur], device="cuda:0"),
            use_cache=True,
        )
    next_step1_tok = torch.argmax(out_step1.logits[:, -1, :], -1)
    print("Eager step 1 next token:", next_step1_tok.item(), repr(tokenizer.decode([next_step1_tok.item()])))

    # Step 2: Now test with CUDA Graph replay
    cache_graph = StaticCache(config=model.config, max_batch_size=1, max_cache_len=2048, device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        out_graph_prefill = model(
            input_ids,
            past_key_values=cache_graph,
            cache_position=torch.arange(0, cur, device="cuda:0"),
            use_cache=True,
            output_hidden_states=True,
        )
    next_tok_graph = torch.argmax(out_graph_prefill.logits[:, -1, :], -1)

    # Capture width 1
    ids_buf = torch.zeros((1, 1), dtype=torch.long, device="cuda:0")
    pos_ids_buf = torch.zeros((1, 1), dtype=torch.long, device="cuda:0")
    cache_pos_buf = torch.zeros((1,), dtype=torch.long, device="cuda:0")
    attn_mask = torch.ones((1, 2048), dtype=torch.long, device="cuda:0")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            model(ids_buf, attention_mask=attn_mask, position_ids=pos_ids_buf, cache_position=cache_pos_buf, past_key_values=cache_graph, use_cache=True)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s), torch.no_grad():
        out_g = model(ids_buf, attention_mask=attn_mask, position_ids=pos_ids_buf, cache_position=cache_pos_buf, past_key_values=cache_graph, use_cache=True)
        g_logits = out_g.logits
    torch.cuda.current_stream().wait_stream(s)

    # Replay with tok_in at pos cur
    ids_buf.copy_(tok_in)
    pos_ids_buf.copy_(torch.tensor([[cur]], device="cuda:0"))
    cache_pos_buf.copy_(torch.tensor([cur], device="cuda:0"))
    
    # Grab cumulative_length counters
    len_counters = [getattr(l, "cumulative_length", None) for l in cache_graph.layers if isinstance(getattr(l, "cumulative_length", None), torch.Tensor)]
    for c in len_counters:
        c.fill_(cur)

    g.replay()
    next_graph_tok = torch.argmax(g_logits[0, -1, :], -1)
    print("Graph step 1 next token:", next_graph_tok.item(), repr(tokenizer.decode([next_graph_tok.item()])))

    print("MATCH:", next_step1_tok.item() == next_graph_tok.item())

if __name__ == "__main__":
    test_qwen_rotary_and_replay()
