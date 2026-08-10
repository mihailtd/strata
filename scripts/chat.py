"""Interactive terminal chat with a base model, optionally with a trained
LoRA adapter loaded on top.

    uv run --env-file .env scripts/chat.py
    uv run --env-file .env scripts/chat.py --model Qwen/Qwen3.5-2B-Instruct
    uv run --env-file .env scripts/chat.py --adapter results/adapters/lora-run1

--env-file .env sets LD_PRELOAD (required for torch to see the GPU on WSL,
see scripts/check_gpu.py) and HF_TOKEN.

Type 'exit' or Ctrl+C to quit, 'reset' to clear conversation history.
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Instruct")
    p.add_argument("--adapter", help="path to a trained LoRA adapter directory")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible to torch, running on CPU (will be slow).")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map={"": 0} if device == "cuda" else None
    )

    if args.adapter:
        print(f"Loading adapter {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    history = []
    print("\nReady. Type 'exit' to quit, 'reset' to clear history.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "reset":
            history = []
            print("(history cleared)")
            continue

        history.append({"role": "user", "content": user_input})
        inputs = tokenizer.apply_chat_template(
            history, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        print("model> ", end="")
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        print()

        reply = tokenizer.decode(output[0][inputs.shape[1] :], skip_special_tokens=True)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
