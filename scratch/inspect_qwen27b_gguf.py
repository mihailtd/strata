"""Inspect Qwen3.8:27b GGUF metadata, architecture, and tensor layout."""

from pathlib import Path
import gguf

GGUF_PATH = Path("/var/lib/ollama/blobs/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d")

print(f"Reading GGUF file: {GGUF_PATH} ({GGUF_PATH.stat().st_size / 1e9:.2f} GB)...")
reader = gguf.GGUFReader(GGUF_PATH)

print(f"Arch: {reader.fields.get('general.architecture')}")
print(f"Name: {reader.fields.get('general.name')}")
print(f"Quant: {reader.fields.get('general.file_type')}")
print(f"Total Tensors: {len(reader.tensors)}")

# Print sample tensor metadata
print("\nSample Tensors:")
for i, tensor in enumerate(reader.tensors[:15]):
    print(f"  [{i}] {tensor.name}: shape={tensor.shape}, type={tensor.tensor_type}")

# Print all fields
print("\nArchitecture Fields:")
for k, v in reader.fields.items():
    if not k.startswith("tokenizer"):
        val_str = str(v.parts[-1].tobytes().decode('utf-8', errors='ignore')) if hasattr(v, 'parts') else str(v)
        print(f"  {k}: {val_str[:60]}")

