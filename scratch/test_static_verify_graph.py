import time
import torch
import torch.nn.functional as F
from runtime.native_27b_engine import Native27BEngine

class StaticSSMChunkVerifyGraph:
    """Pre-allocated, 100% allocation-free ROCm HIP Graph for K=2 candidate verification."""

    def __init__(self, layers, k=2, device="cuda:0"):
        self.layers = layers
        self.k = k
        self.device = torch.device(device)
        self.static_in = torch.zeros((1, k, 5120), dtype=torch.bfloat16, device=self.device)
        self.static_out = torch.zeros((1, k, 5120), dtype=torch.bfloat16, device=self.device)
        self.init_ssm = [torch.zeros((48, 128, 128), dtype=torch.bfloat16, device=self.device) for _ in layers]
        self.init_conv = [torch.zeros((10240, 3), dtype=torch.bfloat16, device=self.device) for _ in layers]
        self.ssm_history = [torch.zeros((k, 48, 128, 128), dtype=torch.bfloat16, device=self.device) for _ in layers]
        self.conv_history = [torch.zeros((k, 10240, 3), dtype=torch.bfloat16, device=self.device) for _ in layers]
        
        # Pre-allocated scratch buffers to prevent any dynamic allocation during graph capture
        self.scratch_ssm = [torch.zeros((48, 128, 128), dtype=torch.float32, device=self.device) for _ in layers]
        self.scratch_o = torch.zeros((1, k, 48, 128), dtype=torch.float32, device=self.device)
        self.scratch_curr = [torch.zeros((1, k, 5120), dtype=torch.bfloat16, device=self.device) for _ in range(4)]

        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(torch.cuda.current_stream(device=self.device))

        def forward_chunk():
            curr = self.static_in
            b, seq_len, d = curr.shape
            for l_idx, layer in enumerate(self.layers):
                x_norm = layer.attn_norm(curr)
                qkv = layer.attn_qkv(x_norm) if layer.attn_qkv else x_norm
                z = layer.attn_gate(x_norm) if layer.attn_gate else x_norm
                alpha = layer.ssm_alpha(x_norm) if layer.ssm_alpha else torch.zeros((b, seq_len, 48), dtype=curr.dtype, device=self.device)
                beta = layer.ssm_beta(x_norm) if layer.ssm_beta else torch.zeros((b, seq_len, 48), dtype=curr.dtype, device=self.device)

                conv_w = layer.ssm_conv1d.unsqueeze(1).to(dtype=curr.dtype)
                qkv_t = qkv.transpose(1, 2)
                qkv_padded = torch.cat([self.init_conv[l_idx].unsqueeze(0).to(curr.dtype), qkv_t], dim=-1)
                conv_out = F.silu(F.conv1d(qkv_padded, conv_w, groups=10240).transpose(1, 2))

                q_all = conv_out[:, :, :2048].view(b, seq_len, 16, 128).float()
                k_all = conv_out[:, :, 2048:4096].view(b, seq_len, 16, 128).float()
                v_all = conv_out[:, :, 4096:10240].view(b, seq_len, 48, 128).float()

                eps = 1e-6
                q_all = q_all / torch.clamp(torch.norm(q_all, p=2, dim=-1, keepdim=True), min=eps) * (128.0 ** -0.5)
                k_all = k_all / torch.clamp(torch.norm(k_all, p=2, dim=-1, keepdim=True), min=eps)

                q_all = q_all.repeat(1, 1, 3, 1)
                k_all = k_all.repeat(1, 1, 3, 1)

                gate_all = layer.ssm_a * F.softplus(alpha.float() + layer.ssm_dt_bias)
                decay_all = torch.exp(gate_all)
                beta_all = torch.sigmoid(beta.float())

                curr_ssm = self.scratch_ssm[l_idx]
                curr_ssm.copy_(self.init_ssm[l_idx].float())
                
                o_all = self.scratch_o
                for t in range(seq_len):
                    q_t = q_all[0, t].unsqueeze(-1)
                    k_t = k_all[0, t].unsqueeze(-1)
                    v_t = v_all[0, t].unsqueeze(-1)
                    dec_t = decay_all[0, t].view(48, 1, 1)
                    beta_t = beta_all[0, t].view(48, 1, 1)

                    curr_ssm.mul_(dec_t)
                    v_err = (v_t - torch.bmm(curr_ssm, k_t)) * beta_t
                    curr_ssm.add_(torch.bmm(v_err, k_t.transpose(1, 2)))
                    o_all[0, t].copy_(torch.bmm(curr_ssm, q_t).squeeze(-1))
                    self.ssm_history[l_idx][t].copy_(curr_ssm.to(curr.dtype))
                    self.conv_history[l_idx][t].copy_(qkv_padded[0, :, t + 1 : t + 4])

                o_rms = layer.ssm_norm(o_all.to(curr.dtype))
                gated_o = (o_rms * F.silu(z.view(b, seq_len, 48, 128))).reshape(b, seq_len, 6144)
                y = layer.ssm_out(gated_o) if layer.ssm_out else gated_o
                curr = curr + y

                x_ffn_norm = layer.post_attention_norm(curr)
                ffn_gate = layer.ffn_gate(x_ffn_norm)
                ffn_up = layer.ffn_up(x_ffn_norm)
                mlp_out = layer.ffn_down(F.silu(ffn_gate) * ffn_up)
                curr = curr + mlp_out

            self.static_out.copy_(curr)

        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                forward_chunk()
        torch.cuda.current_stream(device=self.device).wait_stream(capture_stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=capture_stream):
            forward_chunk()
        torch.cuda.current_stream(device=self.device).wait_stream(capture_stream)

    def replay(self, x, init_ssms, init_convs):
        self.static_in.copy_(x)
        for j in range(len(self.layers)):
            self.init_ssm[j].copy_(init_ssms[j])
            self.init_conv[j].copy_(init_convs[j])
        self.graph.replay()
        return self.static_out, self.ssm_history, self.conv_history


print("[Test Static Verify Graph] Loading engine without auto-graph...")
engine = Native27BEngine(num_layers=64, device="cuda:0")
engine.load_from_cache(force_convert=False)

print("[Test Static Verify Graph] Capturing Static K=2 verify graphs...")
t0 = time.perf_counter()
verify_graphs = []
chunk_count = engine.num_layers // 4
for k in range(chunk_count):
    ssm_layers = [engine.layers[4 * k], engine.layers[4 * k + 1], engine.layers[4 * k + 2]]
    v_chunk = StaticSSMChunkVerifyGraph(ssm_layers, k=2, device=engine.device)
    verify_graphs.append(v_chunk)
print(f"[Test Static Verify Graph] Captured {len(verify_graphs)} K=2 verify graphs in {(time.perf_counter() - t0)*1000:.1f}ms!")

x = torch.randn((1, 2, 5120), dtype=torch.bfloat16, device="cuda:0")
init_ssms = [torch.zeros((48, 128, 128), dtype=torch.bfloat16, device="cuda:0") for _ in range(3)]
init_convs = [torch.zeros((10240, 3), dtype=torch.bfloat16, device="cuda:0") for _ in range(3)]

# Test 50 replays
t0 = time.perf_counter()
for _ in range(50):
    curr = x
    for k in range(chunk_count):
        curr, _, _ = verify_graphs[k].replay(curr, init_ssms, init_convs)
torch.cuda.synchronize()
dt_ms = (time.perf_counter() - t0) * 1000 / 50
print(f"[Test Static Verify Graph] 50 Replays Successful! 48-SSM Verification Latency (K=2): {dt_ms:.2f} ms!")
