# 🛠️ Operational CLI Tools & Training Pipelines

This directory contains the production training pipelines, dataset curation utilities, domain evaluation harnesses, and server launcher scripts.

> **Note**: For empirical performance benchmarks, hardware scaling suites, and mathematical proofs, see the top-level [`benchmarks/`](../benchmarks/) directory.

---

## 📂 Tooling Index by Functional Area

### 1. 🎓 Fine-Tuning & Adapter Training
* **[`train_expert_CURRENT_m2.py`](train_expert_CURRENT_m2.py)**: **Authoritative SFT Trainer (M2)**. Trains domain-specialized LoRA experts directly in native `bfloat16` with fused Liger kernels (cross-entropy, RMSNorm, SwiGLU) and completion-only loss masking (`label=-100`).
  ```bash
  uv run --env-file .env python scripts/train_expert_CURRENT_m2.py --domain postgresql
  ```
* **[`train_mtp_adapter.py`](train_mtp_adapter.py)**: Specialized trainer for multi-token speculative draft heads.
* **[`export_adapter.py`](export_adapter.py)** & **[`finetune_novel_adapter.py`](finetune_novel_adapter.py)**: Legacy training and export scripts.

---

### 2. 📊 Dataset Generation & Synthetic Curation
* **[`build_financial_planning_dataset.py`](build_financial_planning_dataset.py)**: Generates high-quality synthetic training pairs grounded in canonical financial planning concepts.
* **[`clean_financial_training_data.py`](clean_financial_training_data.py)**: Cleans, formats, and deduplicates financial training data.
* **[`add_opinionated_sft_data.py`](add_opinionated_sft_data.py)**: Injects opinionated modern tooling practices into domain training data.
* **[`run_datagen.py`](run_datagen.py)**: Automated synthetic data generation pipeline runner.

---

### 3. 📐 SVD Basis Extraction & Projection
* **[`extract_svd_basis.py`](extract_svd_basis.py)**: Extracts principal orthonormal basis matrices ($U, \Sigma, V^T$) from model weight layers.
* **[`project_adapter_to_basis.py`](project_adapter_to_basis.py)**: Projects low-rank LoRA adapter matrices onto pre-computed shared basis banks.

---

### 4. 🎯 Domain Evaluation & Integrity Auditing
* **[`audit_adapters.py`](audit_adapters.py)**: Audits SHA-256 content hashes, parameter ranks, and metadata integrity across all adapters in `results/adapters/`.
  ```bash
  uv run python scripts/audit_adapters.py
  ```
* **[`evaluate_astral_models.py`](evaluate_astral_models.py)**: Evaluates Astral domain tool adherence (`uv`/`ruff`/`ty` vs `pip`/`flake8`).
* **[`evaluate_postgres_adapter.py`](evaluate_postgres_adapter.py)**: Evaluates PostgreSQL vector adherence (`pgvector`/`hnsw` vs `pinecone`/`weaviate`).
* **[`evaluate_novel_adapter.py`](evaluate_novel_adapter.py)**: Generalized multi-variant evaluation runner.

---

### 5. 🚀 Production Serving & Interactive Chat
* **[`run_openai_api_server.py`](run_openai_api_server.py)**: Launches the FastAPI OpenAI-compatible REST server (`/v1/chat/completions`) with resident factor preloading, in-place low-rank weight folding, and pointer-stable CUDA Graph replay.
  ```bash
  uv run --env-file .env python scripts/run_openai_api_server.py --port 8000
  ```
* **[`test_openai_api_server.py`](test_openai_api_server.py)**: End-to-end integration test suite verifying dynamic multi-expert swapping over the OpenAI API.
* **[`chat.py`](chat.py)**: Interactive terminal CLI for multi-turn conversations with on-the-fly expert switching.
* **[`check_gpu.py`](check_gpu.py)**: Diagnostic script verifying ROCm GPU detection, driver availability, and PyTorch context.
* **[`install_rocm_wsl.sh`](install_rocm_wsl.sh)**: Environment setup script configuring WSL2 ROCm driver paths and environment variables.
