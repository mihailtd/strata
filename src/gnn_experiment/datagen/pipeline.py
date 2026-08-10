"""Orchestrates: fetch -> chunk -> generate question -> verify -> generate
answer -> filter -> write dataset, with MLflow tracking throughout.
"""

import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from datasets import Dataset

from gnn_experiment.datagen import prompts
from gnn_experiment.datagen.chunk import chunk_all
from gnn_experiment.datagen.fetch import fetch_all
from gnn_experiment.datagen.llm_client import LocalLLMClient
from gnn_experiment.datagen.schema import Chunk, QARecord

REFUSAL_PATTERNS = ("i cannot", "i can't", "as an ai", "i'm not able to")


def _render_text_fallback(question: str, answer: str) -> str:
    return f"### Question:\n{question}\n\n### Answer:\n{answer}"


def _is_clean_yes(verifier_output: str) -> bool:
    return verifier_output.strip().upper().startswith("YES")


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(p in lowered for p in REFUSAL_PATTERNS)


def process_chunk(
    chunk: Chunk,
    client: LocalLLMClient,
    temperature: float,
    max_tokens: dict,
    chunk_index: int,
    verify_client: LocalLLMClient | None = None,
    nudge: str = "",
) -> tuple[QARecord | None, dict]:
    """Runs the 3-stage flow for one chunk. Returns (record_or_None, log_info).

    max_tokens is per-stage: {"question_gen": int, "verify": int, "answer_gen": int}.
    verify_client defaults to `client` (self-verification) if not given.
    """
    verify_client = verify_client or client
    log_info: dict = {
        "chunk_source": chunk.source_path,
        "heading_path": chunk.heading_path,
    }

    q_result = client.chat(
        prompts.question_gen_messages(chunk, nudge),
        temperature,
        max_tokens=max_tokens["question_gen"],
        span_name="question_gen",
    )
    question = q_result.text.strip()
    log_info["question_gen"] = {
        "prompt_messages": prompts.question_gen_messages(chunk, nudge),
        "response_text": question,
        "latency_s": q_result.latency_s,
    }
    mlflow.log_metric("latency.question_gen_s", q_result.latency_s, step=chunk_index)

    if not question or len(question) > 500:
        log_info["rejected_at"] = "question_gen"
        return None, log_info

    v_result = verify_client.chat(
        prompts.verify_messages(chunk, question),
        temperature=0.0,
        max_tokens=max_tokens["verify"],
        span_name="verify",
    )
    verify_output = v_result.text.strip()
    log_info["verify"] = {
        "prompt_messages": prompts.verify_messages(chunk, question),
        "response_text": verify_output,
        "latency_s": v_result.latency_s,
    }
    mlflow.log_metric("latency.verify_s", v_result.latency_s, step=chunk_index)

    if not _is_clean_yes(verify_output):
        log_info["rejected_at"] = "verify"
        return None, log_info

    a_result = client.chat(
        prompts.answer_gen_messages(chunk, question, nudge),
        temperature,
        max_tokens=max_tokens["answer_gen"],
        span_name="answer_gen",
    )
    answer = a_result.text.strip()
    log_info["answer_gen"] = {
        "prompt_messages": prompts.answer_gen_messages(chunk, question, nudge),
        "response_text": answer,
        "latency_s": a_result.latency_s,
    }
    mlflow.log_metric("latency.answer_gen_s", a_result.latency_s, step=chunk_index)

    if len(answer) < 20 or _looks_like_refusal(answer):
        log_info["rejected_at"] = "answer_gen"
        return None, log_info

    record = QARecord(
        chunk=chunk,
        question=question,
        answer=answer,
        question_gen_version=prompts.QUESTION_GEN_VERSION,
        answer_gen_version=prompts.ANSWER_GEN_VERSION,
    )
    return record, log_info


def run_pipeline(
    config: dict,
    limit: int | None = None,
    refresh_docs: bool = False,
    dry_run: bool = False,
) -> Path:
    llm_cfg = config["llm"]
    client = LocalLLMClient(base_url=llm_cfg["base_url"], model=llm_cfg["model"])

    if not client.ping():
        raise RuntimeError(
            f"llama-server not reachable at {llm_cfg['base_url']} — "
            "start it first (see serving/README.md)."
        )

    verify_base_url = llm_cfg.get("verify_base_url")
    verify_model = llm_cfg.get("verify_model")
    if verify_base_url or verify_model:
        resolved_verify_base_url = verify_base_url or llm_cfg["base_url"]
        resolved_verify_model = verify_model or llm_cfg["model"]
        verify_client = LocalLLMClient(
            base_url=resolved_verify_base_url, model=resolved_verify_model
        )
        if not verify_client.ping():
            raise RuntimeError(
                f"verify_base_url {resolved_verify_base_url} not reachable"
            )
        print(
            f"Verifier: independent model ({resolved_verify_model} @ {resolved_verify_base_url})"
        )
    else:
        verify_client = client
        print(
            f"Verifier: same model as generator ({llm_cfg['model']}) — self-verification, see README caveat"
        )

    nudge = config.get("extraction_nudge", "") or ""

    docs_cache_dir = Path(config["docs_cache_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching docs for: {config['repos']}")
    docs_roots = fetch_all(config["repos"], docs_cache_dir, refresh=refresh_docs)

    print("Chunking...")
    chunks = chunk_all(
        docs_roots, config["chunk_size_chars"], config["min_chunk_chars"]
    )
    total_available = len(chunks)
    print(f"Got {total_available} chunks total.")
    if limit:
        chunks = chunks[:limit]
        print(f"Limiting to first {limit} chunks.")

    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "astral-expert-datagen"))

    run_type = "dry_run" if dry_run else ("partial" if limit else "full")
    model_short = Path(
        llm_cfg["model"]
    ).stem  # "models/Qwen3.5-4B-Q8_0.gguf" -> "Qwen3.5-4B-Q8_0"
    run_name = f"astral-sft-{run_type}-{model_short}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    records: list[QARecord] = []
    stage_logs = {"question_gen": [], "verify": [], "answer_gen": []}
    verifier_pass, verifier_fail = 0, 0

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "model": llm_cfg["model"],
                "verifier_model": verify_model or llm_cfg["model"],
                "self_verified": str(verify_client is client),
                "run_type": run_type,  # "dry_run" | "partial" | "full" — filterable in the MLflow UI
            }
        )
        mlflow.log_params(
            {
                "model": llm_cfg["model"],
                "verifier_model": verify_model or llm_cfg["model"],
                "verifier_base_url": verify_base_url or llm_cfg["base_url"],
                "extraction_nudge": nudge or "(none)",
                "question_gen_version": prompts.QUESTION_GEN_VERSION,
                "verify_version": prompts.VERIFY_VERSION,
                "answer_gen_version": prompts.ANSWER_GEN_VERSION,
                "chunk_size_chars": config["chunk_size_chars"],
                "min_chunk_chars": config["min_chunk_chars"],
                "repos": ",".join(config["repos"]),
                "limit": limit or "none",
                "total_chunks": len(chunks),
            }
        )

        max_tokens = llm_cfg["max_tokens"]
        seen_questions: set[str] = set()
        chunk_wall_times: list[float] = []
        for i, chunk in enumerate(chunks):
            print(
                f"[{i + 1}/{len(chunks)}] {chunk.source_path} ({' > '.join(chunk.heading_path) or '-'})"
            )
            chunk_start = time.perf_counter()
            try:
                record, log_info = process_chunk(
                    chunk,
                    client,
                    llm_cfg["temperature"],
                    max_tokens,
                    i,
                    verify_client=verify_client,
                    nudge=nudge,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            finally:
                chunk_wall_times.append(time.perf_counter() - chunk_start)

            for stage in ("question_gen", "verify", "answer_gen"):
                if stage in log_info:
                    stage_logs[stage].append(
                        {"chunk_source": log_info["chunk_source"], **log_info[stage]}
                    )

            if record is None:
                verifier_fail += 1
                print(f"  rejected at: {log_info.get('rejected_at')}")
                continue

            dedup_key = record.question.strip().casefold()
            if dedup_key in seen_questions:
                verifier_fail += 1
                continue
            seen_questions.add(dedup_key)

            verifier_pass += 1
            records.append(record)

        mlflow.log_metric("chunks_processed", len(chunks))
        mlflow.log_metric("verifier_pass_count", verifier_pass)
        mlflow.log_metric("verifier_fail_count", verifier_fail)
        mlflow.log_metric("qa_pairs_written", len(records))

        if chunk_wall_times:
            avg_chunk_s = sum(chunk_wall_times) / len(chunk_wall_times)
            mlflow.log_metric("avg_chunk_wall_s", avg_chunk_s)
            if limit and limit < total_available:
                eta_s = avg_chunk_s * total_available
                mlflow.log_metric("estimated_full_run_s", eta_s)
                print(
                    f"\n=== ESTIMATE (based on {len(chunk_wall_times)} sampled chunks) ===\n"
                    f"Avg time per chunk: {avg_chunk_s:.2f}s\n"
                    f"Total chunks available: {total_available}\n"
                    f"Estimated time for full run: {eta_s / 60:.1f} min ({eta_s / 3600:.2f} hr)\n"
                )

        output_records = [
            r.to_dict(_render_text_fallback(r.question, r.answer)) for r in records
        ]
        output_jsonl = output_dir / "astral_expert_sft.jsonl"
        with open(output_jsonl, "w") as f:
            for rec in output_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(output_records)} records to {output_jsonl}")

        hf_dataset_dir = output_dir / "hf_dataset"
        Dataset.from_list(output_records).save_to_disk(str(hf_dataset_dir))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for stage, entries in stage_logs.items():
                stage_file = tmp_path / f"{stage}.jsonl"
                with open(stage_file, "w") as f:
                    for entry in entries:
                        f.write(json.dumps(entry) + "\n")
                mlflow.log_artifact(str(stage_file), "prompts_and_outputs")
        mlflow.log_artifact(str(output_jsonl), "dataset")

    if llm_cfg.get("unload_after_run", True):
        if client.shutdown_server():
            print(
                f"Unloaded model from VRAM (stopped llama-server on port {client._client.base_url.port})."
            )
        if verify_client is not client and verify_client.shutdown_server():
            print(
                f"Unloaded verifier model from VRAM (stopped llama-server on port {verify_client._client.base_url.port})."
            )

    client.close()
    if verify_client is not client:
        verify_client.close()
    return output_jsonl
