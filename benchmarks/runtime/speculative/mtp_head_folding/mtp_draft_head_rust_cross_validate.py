"""§137 cross-validation, step 2 of 2: runs the REAL, shipped Python
`Qwen35MTPDraftHead` on the EXACT real hidden-state bytes the Rust port
(`apps/runtime-next/src/mtp_draft.rs`) extracted from its own real forward
pass, and dumps the real drafted token ids for the Rust side's own
decisive test to compare against.

Deliberately does NOT re-run the Python backbone's own forward pass to
get a hidden state -- that would re-test backbone fidelity (already
covered by many existing decisive tests), not the draft-head PORT
specifically. Loading the Rust-computed hidden bytes directly isolates
the one real variable this cross-validation exists to check: does the
Rust port's own fuse/layer/norm/lm_head math agree with the real,
original Python implementation, given the identical real input.

    uv run --env-file .env benchmarks/runtime/speculative/mtp_head_folding/mtp_draft_head_rust_cross_validate.py
"""

import json
import sys

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM  # noqa: E402

from runtime.canon import REPO_ROOT  # noqa: E402

sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.mtp_draft import Qwen35MTPDraftHead  # noqa: E402
from runtime.novel_peft import set_hard_vram_cap  # noqa: E402


def bf16_bytes_to_tensor(path, n_elems, device):
    raw = open(path, "rb").read()
    assert len(raw) == n_elems * 2, f"expected {n_elems * 2} bytes, got {len(raw)}"
    u16 = torch.frombuffer(bytearray(raw), dtype=torch.int16)
    # bf16 IS its own 16-bit storage format -- reinterpret the raw little-
    # endian u16 bytes Rust wrote directly as bf16, no numeric conversion.
    return u16.view(torch.bfloat16).to(device=device).clone()


def main():
    scratch = "/tmp/claude-1000/-home-mihai-Projects-gnn-experiment/62408538-8f70-4fd0-a0ab-8c9b65ca71dd/scratchpad"
    meta = json.loads(open(f"{scratch}/mtp_xval_meta.json").read())
    print(f"real meta from Rust: {meta}")

    set_hard_vram_cap(22.0)
    model_id = "Qwen/Qwen3.5-4B"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    model.eval()
    head = Qwen35MTPDraftHead(model, model_id)

    device = next(model.parameters()).device
    hidden_size = meta["hidden_size"]
    hidden_flat = bf16_bytes_to_tensor(f"{scratch}/mtp_xval_hidden.bin", hidden_size, device)
    hidden = hidden_flat.view(1, 1, hidden_size)  # (B=1, T=1, H) -- head.draft's own expected shape

    next_token = torch.tensor([[meta["next_token"]]], device=device, dtype=torch.long)
    k = meta["k"]
    start_pos = meta["position"]

    with torch.no_grad():
        drafted = head.draft(hidden, next_token, k=k, start_pos=start_pos, cache=None)

    drafted_ids = drafted[0].tolist()
    print(f"real Python drafted token ids (k={k}, start_pos={start_pos}): {drafted_ids}")

    out = {"drafted_ids": drafted_ids, "k": k, "start_pos": start_pos, "next_token": meta["next_token"]}
    out_path = f"{scratch}/mtp_xval_python_reference.json"
    open(out_path, "w").write(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
