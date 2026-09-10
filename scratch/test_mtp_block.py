import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from runtime.native_27b_engine import RMSNorm, W4A16Linear, apply_rotary_emb, PreallocatedKVCache

class Qwen35MTPBlock(nn.Module):
    """Native Qwen 3.5 / 3.8 27B Neural Multi-Token Prediction (MTP) Layer (blk.64)."""

    def __init__(self, device: str = "cuda:0"):
        super().__init__()
        self.device = torch.device(device)

        # MTP state fusion norms and projection
        self.hnorm = RMSNorm(5120, device=self.device)
        self.enorm = RMSNorm(5120, device=self.device)
        self.eh_proj: Optional[W4A16Linear] = None

        # Transformer Attention Block
        self.attn_norm = RMSNorm(5120, device=self.device)
        self.post_attention_norm = RMSNorm(5120, device=self.device)
        self.attn_q_norm = RMSNorm(256, device=self.device)
        self.attn_k_norm = RMSNorm(256, device=self.device)

        self.attn_q: Optional[W4A16Linear] = None
        self.attn_k: Optional[W4A16Linear] = None
        self.attn_v: Optional[W4A16Linear] = None
        self.attn_output: Optional[W4A16Linear] = None

        # SwiGLU MLP
        self.ffn_gate: Optional[W4A16Linear] = None
        self.ffn_up: Optional[W4A16Linear] = None
        self.ffn_down: Optional[W4A16Linear] = None

        # Final head norm before shared LM head
        self.shared_head_norm = RMSNorm(5120, device=self.device)

    def load_weights(self, layer_dict: dict) -> None:
        """Loads all 15 MTP tensors unpacked from layer_64.pt."""
        with torch.no_grad():
            self.hnorm.weight.data.copy_(layer_dict["nextn.hnorm.weight"]["weight"].to(self.device).to(torch.bfloat16))
            self.enorm.weight.data.copy_(layer_dict["nextn.enorm.weight"]["weight"].to(self.device).to(torch.bfloat16))
            self.attn_norm.weight.data.copy_(layer_dict["attn_norm.weight"]["weight"].to(self.device).to(torch.bfloat16))
            self.post_attention_norm.weight.data.copy_(layer_dict["post_attention_norm.weight"]["weight"].to(self.device).to(torch.bfloat16))
            self.attn_q_norm.weight.data.copy_(layer_dict["attn_q_norm.weight"]["weight"].to(self.device).to(torch.bfloat16))
            self.attn_k_norm.weight.data.copy_(layer_dict["attn_k_norm.weight"]["weight"].to(self.device).to(torch.bfloat16))
            self.shared_head_norm.weight.data.copy_(layer_dict["nextn.shared_head_norm.weight"]["weight"].to(self.device).to(torch.bfloat16))

        # W4A16 Linear Projections
        projs = {
            "eh_proj": "nextn.eh_proj.weight",
            "attn_q": "attn_q.weight",
            "attn_k": "attn_k.weight",
            "attn_v": "attn_v.weight",
            "attn_output": "attn_output.weight",
            "ffn_gate": "ffn_gate.weight",
            "ffn_up": "ffn_up.weight",
            "ffn_down": "ffn_down.weight",
        }
        for attr, key in projs.items():
            if key in layer_dict:
                entry = layer_dict[key]
                lin = W4A16Linear.from_packed(
                    qweight=entry["qweight"],
                    scales=entry["scales"],
                    group_size=128,
                    device=self.device,
                )
                setattr(self, attr, lin)

    def forward(
        self,
        h: torch.Tensor,
        tok_emb: torch.Tensor,
        pos: int,
        kv_cache: PreallocatedKVCache,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Executes forward pass of MTP block for candidate token prediction.
        
        Args:
            h: Target model output hidden state (b, 1, 5120) before output_norm
            tok_emb: Token embedding of predicted token (b, 1, 5120)
            pos: Current sequence position
            kv_cache: Dedicated MTP KV cache
            cos_sin: RoPE frequencies (cos, sin)
        Returns:
            Hidden state before lm_head (b, 1, 5120)
        """
        b, s, d = h.shape
        # 1. State Fusion
        h_norm = self.hnorm(h)
        e_norm = self.enorm(tok_emb)
        concat = torch.cat([e_norm, h_norm], dim=-1)  # (b, s, 10240)
        cur = self.eh_proj(concat)                     # (b, s, 5120)
        inp_sa = cur

        # 2. Attention Block
        cur_norm = self.attn_norm(cur)
        q_full = self.attn_q(cur_norm)                 # (b, s, 12288)
        query_states, gate = torch.chunk(q_full.view(b, s, 24, 256 * 2), 2, dim=-1)
        gate = gate.reshape(b, s, -1)
        q = self.attn_q_norm(query_states).transpose(1, 2)  # (b, 24, s, 256)
        k = self.attn_k_norm(self.attn_k(cur_norm).view(b, s, 4, 256)).transpose(1, 2)
        v = self.attn_v(cur_norm).view(b, s, 4, 256).transpose(1, 2)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        k_all, v_all = kv_cache.update(k, v)
        k_rep = k_all.repeat_interleave(6, dim=1)
        v_rep = v_all.repeat_interleave(6, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1) * torch.sigmoid(gate)
        cur = inp_sa + (self.attn_output(attn_out) if self.attn_output else attn_out)

        # 3. SwiGLU MLP Block
        ffn_res = cur
        cur_ffn_norm = self.post_attention_norm(cur)
        mlp_out = self.ffn_down(F.silu(self.ffn_gate(cur_ffn_norm)) * self.ffn_up(cur_ffn_norm))
        cur = ffn_res + mlp_out

        # 4. Head Norm
        return self.shared_head_norm(cur)


# Test Loading & Forward Pass
print("[Test MTP] Loading layer_64.pt...")
t0 = time.perf_counter()
layer_dict = torch.load("models/qwen3.8-27b-triton/layer_64.pt", map_location="cuda:0")
mtp = Qwen35MTPBlock(device="cuda:0")
mtp.load_weights(layer_dict)
print(f"[Test MTP] Loaded weights in {(time.perf_counter() - t0)*1000:.1f}ms!")

# Test Dummy Forward
h = torch.randn((1, 1, 5120), dtype=torch.bfloat16, device="cuda:0")
tok_emb = torch.randn((1, 1, 5120), dtype=torch.bfloat16, device="cuda:0")
kv_cache = PreallocatedKVCache(num_heads=4, head_dim=256, max_seq_len=64, device="cuda:0")

# Dummy cos, sin for RoPE
cos = torch.ones((1, 1, 1, 64), dtype=torch.bfloat16, device="cuda:0")
sin = torch.zeros((1, 1, 1, 64), dtype=torch.bfloat16, device="cuda:0")

# Warmup
for _ in range(3):
    out = mtp(h, tok_emb, pos=0, kv_cache=kv_cache, cos_sin=(cos, sin))
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(50):
    kv_cache.current_len = 0
    out = mtp(h, tok_emb, pos=0, kv_cache=kv_cache, cos_sin=(cos, sin))
torch.cuda.synchronize()
dt_ms = (time.perf_counter() - t0) * 1000 / 50
print(f"[Test MTP] Forward execution time: {dt_ms:.2f} ms per step!")
print(f"[Test MTP] Output shape: {out.shape}, dtype: {out.dtype}")
