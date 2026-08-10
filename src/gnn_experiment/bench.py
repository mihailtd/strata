"""Core benchmarking routine: fine-tune one (model, PEFT method) pair for a
handful of steps and report time / VRAM / trainable-param metrics.

This is intentionally short-horizon (a few dozen steps) — it's meant to
compare technique overhead and convergence *trend*, not to produce a
production-quality adapter.
"""

import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from gnn_experiment.peft_methods import QUANTIZED_METHODS, build_config


def _compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_model(model_name: str, method: str):
    dtype = _compute_dtype()
    if method in QUANTIZED_METHODS:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map={"": 0}
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map={"": 0}
        )
    return model


def _load_dataset(tokenizer, dataset_name: str, text_field: str, max_length: int, n_examples: int):
    if Path(dataset_name).exists():
        ds = load_dataset("json", data_files=dataset_name, split=f"train[:{n_examples}]")
    else:
        ds = load_dataset(dataset_name, split=f"train[:{n_examples}]")

    def tokenize(batch):
        return tokenizer(
            batch[text_field], truncation=True, max_length=max_length, padding="max_length"
        )

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


def run_one(
    model_name: str,
    method: str,
    dataset_name: str,
    text_field: str,
    r: int,
    alpha: int,
    target_modules: list[str],
    max_steps: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    n_examples: int,
    output_dir: str,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(model_name, method)
    peft_config = build_config(method, r, alpha, target_modules, max_steps)
    model = get_peft_model(model, peft_config)

    trainable, total = model.get_nb_trainable_parameters()

    train_ds = _load_dataset(tokenizer, dataset_name, text_field, max_length, n_examples)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=_compute_dtype() == torch.bfloat16,
        fp16=_compute_dtype() == torch.float16,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start

    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "method": method,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
        "wall_time_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "final_loss": result.training_loss,
    }
