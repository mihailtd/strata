import torch
from runtime.canon import adapter_path, DOMAINS
from runtime.novel_peft import FoldableExpert

experts = {
    d: FoldableExpert.from_dir(adapter_path(d, "v6"), name=d)
    for d in ["astral", "duckdb", "financial", "postgresql", "python_modern", "python_web"]
}

print(f"{'Domain':<15} | {'Total ||dW||_F':>15} | {'Mean factor ||U||_F':>20} | {'Mean factor ||V||_F':>20}")
print("-" * 78)
for d, exp in experts.items():
    total_dw_norm = 0.0
    u_norms = []
    v_norms = []
    for k, (u, v) in exp.factors.items():
        dw = exp.scaling * (u @ v)
        total_dw_norm += float(torch.norm(dw, p="fro") ** 2)
        u_norms.append(float(torch.norm(u, p="fro")))
        v_norms.append(float(torch.norm(v, p="fro")))
    total_dw = total_dw_norm ** 0.5
    print(f"{d:<15} | {total_dw:>15.4f} | {sum(u_norms)/len(u_norms):>20.4f} | {sum(v_norms)/len(v_norms):>20.4f}")
