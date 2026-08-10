"""Registry of PEFT method configs, keyed by name, for the benchmark harness.

Each factory takes the common knobs (rank, alpha, target_modules) and returns
a ready-to-use peft config. `qlora` is not a distinct peft config — it's a
LoRA config paired with 4-bit base model loading, handled in bench.py via
the `quantize` flag.
"""

from peft import (
    AdaLoraConfig,
    IA3Config,
    LoraConfig,
    PrefixTuningConfig,
    VeraConfig,
)

METHODS = {
    "lora": lambda r, alpha, targets, total_steps: LoraConfig(
        r=r, lora_alpha=alpha, target_modules=targets, task_type="CAUSAL_LM"
    ),
    "dora": lambda r, alpha, targets, total_steps: LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=targets,
        use_dora=True,
        task_type="CAUSAL_LM",
    ),
    "pissa": lambda r, alpha, targets, total_steps: LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=targets,
        init_lora_weights="pissa",
        task_type="CAUSAL_LM",
    ),
    "qlora": lambda r, alpha, targets, total_steps: LoraConfig(
        r=r, lora_alpha=alpha, target_modules=targets, task_type="CAUSAL_LM"
    ),
    "adalora": lambda r, alpha, targets, total_steps: AdaLoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=targets,
        total_step=total_steps,
        task_type="CAUSAL_LM",
    ),
    "vera": lambda r, alpha, targets, total_steps: VeraConfig(
        r=r, target_modules=targets, task_type="CAUSAL_LM"
    ),
    "ia3": lambda r, alpha, targets, total_steps: IA3Config(
        target_modules=targets,
        feedforward_modules=targets[-1:],
        task_type="CAUSAL_LM",
    ),
    "prefix_tuning": lambda r, alpha, targets, total_steps: PrefixTuningConfig(
        num_virtual_tokens=r, task_type="CAUSAL_LM"
    ),
}

# Methods that need the base model loaded in 4-bit (bitsandbytes) rather than
# full/half precision.
QUANTIZED_METHODS = {"qlora"}


def build_config(method: str, r: int, alpha: int, target_modules: list[str], total_steps: int):
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Options: {sorted(METHODS)}")
    return METHODS[method](r, alpha, target_modules, total_steps)
