"""Test Autograd Backprop for LoRA Adapters on top of W4A16 Quantized Base."""

import torch
import torch.nn as nn
from runtime.w4a16_loader import W4A16Linear
from runtime.triton_w4a16 import quantize_and_pack_w4


class TrainableW4A16Linear(nn.Module):
    """W4A16 Linear layer with trainable LoRA adapter branch."""

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 128.0, group_size: int = 128, device: str = "cuda:0"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / r

        # Frozen W4A16 Base weights
        self.w4_layer = W4A16Linear(in_features, out_features, group_size=group_size, device=device)
        self.w4_layer.qweight.random_(-10000, 10000)
        self.w4_layer.scales.fill_(0.05)
        self.w4_layer.qweight.requires_grad = False
        self.w4_layer.scales.requires_grad = False

        # Trainable LoRA parameters
        self.lora_a = nn.Parameter(torch.randn((in_features, r), dtype=torch.bfloat16, device=device) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros((r, out_features), dtype=torch.bfloat16, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base frozen pass (detached from graph)
        with torch.no_grad():
            base_out = self.w4_layer(x)

        # Trainable LoRA branch
        lora_out = (x @ self.lora_a) @ self.lora_b * self.scaling
        return base_out + lora_out


def test_lora_train():
    device = "cuda:0"
    print("Testing Trainable W4A16 + LoRA Backpropagation on RDNA3 GPU...")

    layer = TrainableW4A16Linear(5120, 17408, r=8, alpha=128.0, device=device)
    optimizer = torch.optim.AdamW([layer.lora_a, layer.lora_b], lr=1e-3)

    x = torch.randn((2, 5120), dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        base_init = layer.w4_layer(x)
    target = base_init + torch.randn((2, 17408), dtype=torch.bfloat16, device=device) * 5.0

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        out = layer(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Step {step+1}: Loss = {loss.item():.6f}")

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
    print("✅ LoRA Backprop over W4A16 Base Verified Successfully!")


if __name__ == "__main__":
    test_lora_train()
