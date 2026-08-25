import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.mtp_draft import Qwen35MTPDraftHead
from runtime.server import format_prompt, ChatMessage

def test_fuse_order():
    model_id = "Qwen/Qwen3.5-9B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    prompt = "Add ruff and ty as dev dependencies, then format and lint the whole codebase."
    prompt_str = format_prompt([ChatMessage(role="user", content=prompt)], thinking_effort="off")
    input_ids = tokenizer.encode(prompt_str, return_tensors="pt").cuda()

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
        hids = out.hidden_states[-1]

    target_tokens = input_ids[:, 2:]

    # Test 1: Pristine base MTP head with [embedding ; hidden]
    head1 = Qwen35MTPDraftHead(model, model_id).eval()
    with torch.no_grad():
        draft_logits1, _ = head1(hids[:, :-1, :], input_ids[:, 1:])
    p1 = torch.argmax(draft_logits1[:, :-1, :], dim=-1)
    acc1 = (p1[0] == target_tokens[0]).sum().item() / target_tokens.shape[1]
    print(f"Order [embedding ; hidden] match rate: {acc1:.1%}")

    # Test 2: Pristine base MTP head with [hidden ; embedding]
    def _fuse_swap(self, h, tok):
        e = self._embed(tok)
        return self.fc(torch.cat([self.pre_fc_norm_hidden(h), self.pre_fc_norm_embedding(e)], dim=-1))
    
    head1._fuse = _fuse_swap.__get__(head1, Qwen35MTPDraftHead)
    with torch.no_grad():
        # manual forward with swapped fuse
        fused = head1._fuse(hids[:, :-2, :], input_ids[:, 1:-1])
        positions = torch.arange(input_ids.shape[1] - 2, device=fused.device)
        h_out = head1._run_layer(fused, positions, None)
        logits2 = head1._lm_head(h_out)
    p2 = torch.argmax(logits2, dim=-1)
    acc2 = (p2[0] == target_tokens[0]).sum().item() / target_tokens.shape[1]
    print(f"Order [hidden ; embedding] match rate: {acc2:.1%}")

if __name__ == "__main__":
    test_fuse_order()
