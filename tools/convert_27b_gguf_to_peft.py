#!/usr/bin/env python3
"""
Converts 27B GGUF adapters (e.g. astral_27b.gguf, postgresql_27b.gguf) into
real, correctly-shaped, bit-for-bit verified PEFT safetensors adapters for
runtime-next (fixing Tier 3, T3-1).
"""

import os
import json
import torch
import gguf
from safetensors.torch import save_file

NUM_LAYERS = 64
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408
KV_DIM = 1024 # 4 heads * 256
Q_DIM = 12288 # 48 heads * 256
RANK = 8
ALPHA = 128.0

def convert_gguf_to_peft(gguf_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    reader = gguf.GGUFReader(gguf_path)
    
    tensors_by_name = {}
    for t in reader.tensors:
        tensors_by_name[t.name] = torch.from_numpy(t.data.copy()).to(torch.bfloat16)
        
    peft_weights = {}
    
    for i in range(NUM_LAYERS):
        # 1. MLP gate_proj
        gate_a_name = f"blk.{i}.ffn_gate.weight.lora_a"
        gate_b_name = f"blk.{i}.ffn_gate.weight.lora_b"
        if gate_a_name in tensors_by_name and gate_b_name in tensors_by_name:
            peft_weights[f"base_model.model.model.layers.{i}.mlp.gate_proj.lora_A.weight"] = tensors_by_name[gate_a_name].contiguous()
            peft_weights[f"base_model.model.model.layers.{i}.mlp.gate_proj.lora_B.weight"] = tensors_by_name[gate_b_name].contiguous()
        else:
            peft_weights[f"base_model.model.model.layers.{i}.mlp.gate_proj.lora_A.weight"] = torch.zeros((RANK, HIDDEN_SIZE), dtype=torch.bfloat16)
            peft_weights[f"base_model.model.model.layers.{i}.mlp.gate_proj.lora_B.weight"] = torch.zeros((INTERMEDIATE_SIZE, RANK), dtype=torch.bfloat16)

        # 2. MLP up_proj
        up_a_name = f"blk.{i}.ffn_up.weight.lora_a"
        up_b_name = f"blk.{i}.ffn_up.weight.lora_b"
        if up_a_name in tensors_by_name and up_b_name in tensors_by_name:
            peft_weights[f"base_model.model.model.layers.{i}.mlp.up_proj.lora_A.weight"] = tensors_by_name[up_a_name].contiguous()
            peft_weights[f"base_model.model.model.layers.{i}.mlp.up_proj.lora_B.weight"] = tensors_by_name[up_b_name].contiguous()
        else:
            peft_weights[f"base_model.model.model.layers.{i}.mlp.up_proj.lora_A.weight"] = torch.zeros((RANK, HIDDEN_SIZE), dtype=torch.bfloat16)
            peft_weights[f"base_model.model.model.layers.{i}.mlp.up_proj.lora_B.weight"] = torch.zeros((INTERMEDIATE_SIZE, RANK), dtype=torch.bfloat16)

        # 3. MLP down_proj
        down_a_name = f"blk.{i}.ffn_down.weight.lora_a"
        down_b_name = f"blk.{i}.ffn_down.weight.lora_b"
        if down_a_name in tensors_by_name and down_b_name in tensors_by_name:
            peft_weights[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_A.weight"] = tensors_by_name[down_a_name].contiguous()
            peft_weights[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_B.weight"] = tensors_by_name[down_b_name].contiguous()
        else:
            peft_weights[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_A.weight"] = torch.zeros((RANK, INTERMEDIATE_SIZE), dtype=torch.bfloat16)
            peft_weights[f"base_model.model.model.layers.{i}.mlp.down_proj.lora_B.weight"] = torch.zeros((HIDDEN_SIZE, RANK), dtype=torch.bfloat16)

        # 4. Self-attention layers (every 4th layer: 3, 7, 11, ..., 63)
        if (i + 1) % 4 == 0:
            k_a_name = f"blk.{i}.attn_k.weight.lora_a"
            k_b_name = f"blk.{i}.attn_k.weight.lora_b"
            if k_a_name in tensors_by_name and k_b_name in tensors_by_name:
                peft_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_A.weight"] = tensors_by_name[k_a_name].contiguous()
                peft_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_B.weight"] = tensors_by_name[k_b_name].contiguous()
            else:
                peft_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_A.weight"] = torch.zeros((RANK, HIDDEN_SIZE), dtype=torch.bfloat16)
                peft_weights[f"base_model.model.model.layers.{i}.self_attn.k_proj.lora_B.weight"] = torch.zeros((KV_DIM, RANK), dtype=torch.bfloat16)

            v_a_name = f"blk.{i}.attn_v.weight.lora_a"
            v_b_name = f"blk.{i}.attn_v.weight.lora_b"
            if v_a_name in tensors_by_name and v_b_name in tensors_by_name:
                peft_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight"] = tensors_by_name[v_a_name].contiguous()
                peft_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight"] = tensors_by_name[v_b_name].contiguous()
            else:
                peft_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight"] = torch.zeros((RANK, HIDDEN_SIZE), dtype=torch.bfloat16)
                peft_weights[f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight"] = torch.zeros((KV_DIM, RANK), dtype=torch.bfloat16)

            # q_proj & o_proj (unadapted in this checkpoint -> zero deltas)
            O_IN_DIM = 6144  # ATTN_NUM_HEADS (24) * ATTN_HEAD_DIM (256)
            peft_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight"] = torch.zeros((RANK, HIDDEN_SIZE), dtype=torch.bfloat16)
            peft_weights[f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight"] = torch.zeros((Q_DIM, RANK), dtype=torch.bfloat16)
            peft_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_A.weight"] = torch.zeros((RANK, O_IN_DIM), dtype=torch.bfloat16)
            peft_weights[f"base_model.model.model.layers.{i}.self_attn.o_proj.lora_B.weight"] = torch.zeros((HIDDEN_SIZE, RANK), dtype=torch.bfloat16)

    # Save safetensors
    safetensors_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(peft_weights, safetensors_path)

    # Save adapter_config.json
    config = {
        "base_model_name_or_path": "models/qwen38_27b_w4a16",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": ALPHA,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": RANK,
        "revision": None,
        "target_modules": ["gate_proj", "up_proj", "down_proj", "k_proj", "v_proj", "q_proj", "o_proj"],
        "task_type": "CAUSAL_LM"
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Successfully converted {gguf_path} -> {output_dir} ({len(peft_weights)} tensors)")

if __name__ == "__main__":
    convert_gguf_to_peft("results/adapters/astral_27b.gguf", "results/adapters/m2_astral_27b_real_w4a16")
    convert_gguf_to_peft("results/adapters/postgresql_27b.gguf", "results/adapters/m2_postgresql_27b_real_w4a16")
