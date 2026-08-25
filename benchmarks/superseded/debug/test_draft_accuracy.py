import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.mtp_draft import Qwen35MTPDraftHead
from runtime.novel_peft import FoldableExpert, WeightFoldingEngine
from runtime.server import format_prompt, ChatMessage
from runtime.canon import REPO_ROOT

def test_draft_head_accuracy():
    model_id = "Qwen/Qwen3.5-9B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    expert = FoldableExpert.from_dir(REPO_ROOT / "results" / "adapters" / "m2_astral_r8a128_v7_9b", "astral")
    engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
    draft_head = Qwen35MTPDraftHead(model, model_id)
    draft_head.eval()
    engine.register_draft_head(draft_head)
    engine.activate(expert)

    prompt = "Add ruff and ty as dev dependencies, then format and lint the whole codebase."
    prompt_str = format_prompt([ChatMessage(role="user", content=prompt)], thinking_effort="off")
    input_ids = tokenizer.encode(prompt_str, return_tensors="pt").cuda()

    # Get eager hidden states and target tokens from base model
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
        hids = out.hidden_states[-1]
        base_logits = out.logits

    # Test Draft Head forward pass
    with torch.no_grad():
        draft_logits, _ = draft_head(hids[:, :-1, :], input_ids[:, 1:])
    
    # Compare draft head predictions with target tokens input_ids[:, 2:]
    target_tokens = input_ids[:, 2:]
    draft_preds = torch.argmax(draft_logits[:, :-1, :], dim=-1)

    matches = (draft_preds[0] == target_tokens[0]).sum().item()
    total = target_tokens.shape[1]
    acc = matches / total
    print(f"Draft head next-next-token match rate on prompt: {matches}/{total} ({acc:.1%})")

    # Now let's test autoregressive drafting for 4 steps
    dcache = draft_head.prefill(hids, input_ids)
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    print("Base next token:", nxt.item(), repr(tokenizer.decode([nxt.item()])))

    draft_toks = draft_head.draft(hids[:, -1:, :], nxt, k=4, start_pos=input_ids.shape[1] - 1, cache=dcache)
    print("Drafted 4 tokens:", draft_toks.tolist())
    print("Drafted text:", repr(tokenizer.decode(draft_toks[0].tolist())))

    # Let's see what base model actually predicts for next 4 tokens
    base_next_4 = []
    cur_in = input_ids
    for _ in range(4):
        with torch.no_grad():
            o = model(cur_in)
        t = torch.argmax(o.logits[:, -1, :], -1, keepdim=True)
        base_next_4.append(t.item())
        cur_in = torch.cat([cur_in, t], dim=1)
    print("Base model actual next 4 tokens:", base_next_4)
    print("Base model text:", repr(tokenizer.decode(base_next_4)))

if __name__ == "__main__":
    test_draft_head_accuracy()
