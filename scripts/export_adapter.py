"""Export fine-tuned Astral micro-probe LoRA adapter to disk with MLflow tracking.

Trains/saves the LoRA adapter to `results/adapters/astral_qwen3.5_micro`.
Logs exact fine-tuning metrics (wall time, peak VRAM, final loss, model name) to MLflow.
"""

import argparse
import time
from pathlib import Path
import torch
import mlflow

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from gnn_experiment.micro_probe.dataset import load_astral_micro_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


def export_adapter(
    model_name: str = "Qwen/Qwen3.5-4B",
    out_dir: str = "results/adapters/astral_qwen3.5_micro",
    train_steps: int = 150,
    experiment_name: str = "astral_fine_tuning",
):
    out_path = REPO_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    mlflow_db = REPO_ROOT / "mlruns.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment(experiment_name)

    print(f"Loading exact target model: {model_name} for adapter export...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    print("Loading 1,000+ Astral docs dataset for fast fine-tuning...")
    dataset = load_astral_micro_dataset()

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=str(REPO_ROOT / "results" / "tmp_export"),
        per_device_train_batch_size=2,
        max_steps=train_steps,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
    )

    trainer = Trainer(model=model, args=train_args, train_dataset=tokenized_ds, data_collator=collator)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    print(f"Executing {train_steps}-step fast training pass on {model_name}...")
    train_result = trainer.train()

    wall_time_s = time.perf_counter() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    print(f"Saving trained LoRA adapter to {out_path}...")
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    # Log explicit fine-tuning run to MLflow
    with mlflow.start_run(run_name=f"finetune_{model_name.replace('/', '_')}_{train_steps}steps"):
        mlflow.log_params({
            "model_name": model_name,
            "train_steps": train_steps,
            "batch_size": 2,
            "learning_rate": 2e-4,
            "r": 8,
            "alpha": 16,
            "dataset_size": len(dataset),
        })
        mlflow.log_metrics({
            "wall_time_s": wall_time_s,
            "peak_vram_gb": peak_vram_gb,
            "final_loss": train_result.training_loss,
            "steps_per_sec": train_steps / max(0.001, wall_time_s),
        })

    print(f"Fine-tuning completed in {wall_time_s:.2f}s (Peak VRAM: {peak_vram_gb:.2f} GB, Final Loss: {train_result.training_loss:.4f})")
    print(f"MLflow fine-tuning run logged under experiment '{experiment_name}'!")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", default="results/adapters/astral_qwen3.5_micro")
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()
    export_adapter(model_name=args.model_name, out_dir=args.out, train_steps=args.steps)
