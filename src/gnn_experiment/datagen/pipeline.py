"""Orchestrates: fetch -> chunk -> generate question -> verify -> generate
answer -> filter -> write dataset, with MLflow tracking throughout.

Chunks are processed concurrently (up to `max_concurrency` in flight at
once, matching llama-server's parallel slots -- see LocalLLMClient.achat)
rather than one blocking request at a time. Accepted records are appended
to the output JSONL as soon as they're produced, not batched in memory
until the end -- a run that gets interrupted partway through (killed,
timed out, crashed) still leaves everything generated so far on disk
instead of losing it. Per-chunk timing is aggregated in-process and logged
to MLflow as a single summary at the end of the run, not one call per
chunk -- hundreds of small tracking-DB writes interleaved with concurrent
requests is itself a bottleneck.
"""

import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset

from gnn_experiment.datagen import prompts
from gnn_experiment.datagen.chunk import chunk_all
from gnn_experiment.datagen.epub import chunk_all_epubs
from gnn_experiment.datagen.fetch import fetch_all
from gnn_experiment.datagen.llm_client import LocalLLMClient
from gnn_experiment.datagen.schema import Chunk, QARecord
from gnn_experiment.utils.logger import log_benchmark_metric

REFUSAL_PATTERNS = ("i cannot", "i can't", "as an ai", "i'm not able to")


def _render_text_fallback(question: str, answer: str) -> str:
    return f"### Question:\n{question}\n\n### Answer:\n{answer}"


def _is_clean_yes(verifier_output: str) -> bool:
    return verifier_output.strip().upper().startswith("YES")


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(p in lowered for p in REFUSAL_PATTERNS)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct * (len(ordered) - 1))))
    return ordered[idx]


def _stage_stats(stage: str, latencies: list[float], ttfts: list[float]) -> dict[str, float]:
    if not latencies:
        return {}
    return {
        f"latency.{stage}_mean_s": statistics.mean(latencies),
        f"latency.{stage}_median_s": statistics.median(latencies),
        f"latency.{stage}_p90_s": _percentile(latencies, 0.9),
        f"latency.{stage}_max_s": max(latencies),
        f"latency.{stage}_ttft_mean_s": statistics.mean(ttfts),
        f"latency.{stage}_ttft_p90_s": _percentile(ttfts, 0.9),
    }


async def process_chunk(
    chunk: Chunk,
    client: LocalLLMClient,
    temperature: float,
    max_tokens: dict,
    chunk_index: int,
    verify_client: LocalLLMClient | None = None,
    nudge: str = "",
    domain_description: str = prompts.DEFAULT_DOMAIN_DESCRIPTION,
    source_label: str = prompts.DEFAULT_SOURCE_LABEL,
) -> tuple[QARecord | None, dict]:
    """Runs the 3-stage flow for one chunk. Returns (record_or_None, log_info).

    max_tokens is per-stage: {"question_gen": int, "verify": int, "answer_gen": int}.
    verify_client defaults to `client` (self-verification) if not given.
    domain_description/source_label default to this project's original Astral-docs
    wording (see prompts.py) -- override when the corpus is a different domain/kind.

    Stages within one chunk are inherently sequential (verify needs the
    generated question, answer_gen needs the question that passed verify) --
    the concurrency this pipeline exploits is *across* chunks, each running
    its own independent 3-stage chain (see _process_all_chunks).
    """
    verify_client = verify_client or client
    log_info: dict = {
        "chunk_index": chunk_index,
        "chunk_source": chunk.source_path,
        "heading_path": chunk.heading_path,
    }

    q_messages = prompts.question_gen_messages(
        chunk, nudge, domain_description=domain_description, source_label=source_label
    )
    q_result = await client.achat(
        q_messages,
        temperature,
        max_tokens=max_tokens["question_gen"],
        span_name="question_gen",
    )
    question = q_result.text.strip()
    log_info["question_gen"] = {
        "prompt_messages": q_messages,
        "response_text": question,
        "latency_s": q_result.latency_s,
        "ttft_s": q_result.ttft_s,
    }

    if not question or len(question) > 500:
        log_info["rejected_at"] = "question_gen"
        return None, log_info

    v_messages = prompts.verify_messages(chunk, question, source_label=source_label)
    v_result = await verify_client.achat(
        v_messages,
        temperature=0.0,
        max_tokens=max_tokens["verify"],
        span_name="verify",
    )
    verify_output = v_result.text.strip()
    log_info["verify"] = {
        "prompt_messages": v_messages,
        "response_text": verify_output,
        "latency_s": v_result.latency_s,
        "ttft_s": v_result.ttft_s,
    }

    if not _is_clean_yes(verify_output):
        log_info["rejected_at"] = "verify"
        return None, log_info

    a_messages = prompts.answer_gen_messages(
        chunk, question, nudge, domain_description=domain_description, source_label=source_label
    )
    a_result = await client.achat(
        a_messages,
        temperature,
        max_tokens=max_tokens["answer_gen"],
        span_name="answer_gen",
    )
    answer = a_result.text.strip()
    log_info["answer_gen"] = {
        "prompt_messages": a_messages,
        "response_text": answer,
        "latency_s": a_result.latency_s,
        "ttft_s": a_result.ttft_s,
    }

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


async def _process_all_chunks(
    chunks: list[Chunk],
    client: LocalLLMClient,
    verify_client: LocalLLMClient,
    temperature: float,
    max_tokens: dict,
    nudge: str,
    domain_description: str,
    source_label: str,
    output_jsonl: Path,
    max_concurrency: int,
) -> dict:
    """Runs process_chunk for every chunk with up to max_concurrency in
    flight at once. Writes each accepted record to output_jsonl as soon as
    it's produced rather than batching in memory until the end."""
    sem = asyncio.Semaphore(max_concurrency)
    records: list[QARecord] = []
    stage_logs: dict[str, list[dict]] = {"question_gen": [], "verify": [], "answer_gen": []}
    seen_questions: set[str] = set()
    verifier_pass = 0
    verifier_fail = 0
    chunk_wall_times: list[float] = []
    latencies: dict[str, list[float]] = {s: [] for s in ("question_gen", "verify", "answer_gen")}
    ttfts: dict[str, list[float]] = {s: [] for s in ("question_gen", "verify", "answer_gen")}
    done_count = 0
    total = len(chunks)

    with open(output_jsonl, "w") as f_out:

        async def worker(i: int, chunk: Chunk) -> None:
            nonlocal verifier_pass, verifier_fail, done_count
            async with sem:
                chunk_start = time.perf_counter()
                try:
                    record, log_info = await process_chunk(
                        chunk,
                        client,
                        temperature,
                        max_tokens,
                        i,
                        verify_client=verify_client,
                        nudge=nudge,
                        domain_description=domain_description,
                        source_label=source_label,
                    )
                except Exception as e:
                    done_count += 1
                    print(f"[{done_count}/{total}] {chunk.source_path}: ERROR {e}")
                    return
                finally:
                    chunk_wall_times.append(time.perf_counter() - chunk_start)

                # Everything from here on is synchronous (no `await`), so it
                # runs atomically between asyncio task-switch points -- safe
                # to mutate the shared dicts/lists/file above without a lock
                # even with several workers finishing around the same time.
                for stage in ("question_gen", "verify", "answer_gen"):
                    if stage in log_info:
                        stage_logs[stage].append({"chunk_source": log_info["chunk_source"], **log_info[stage]})
                        latencies[stage].append(log_info[stage]["latency_s"])
                        ttfts[stage].append(log_info[stage]["ttft_s"])

                done_count += 1
                if record is None:
                    verifier_fail += 1
                    print(f"[{done_count}/{total}] {chunk.source_path}: rejected at {log_info.get('rejected_at')}")
                    return

                dedup_key = record.question.strip().casefold()
                if dedup_key in seen_questions:
                    verifier_fail += 1
                    print(f"[{done_count}/{total}] {chunk.source_path}: duplicate question, skipped")
                    return
                seen_questions.add(dedup_key)

                verifier_pass += 1
                records.append(record)
                rec_dict = record.to_dict(_render_text_fallback(record.question, record.answer))
                f_out.write(json.dumps(rec_dict) + "\n")
                f_out.flush()
                print(f"[{done_count}/{total}] {chunk.source_path}: wrote QA pair ({len(records)} so far)")

        await asyncio.gather(*(worker(i, chunk) for i, chunk in enumerate(chunks)))

    await client.aclose()
    if verify_client is not client:
        await verify_client.aclose()

    stats: dict[str, float] = {}
    for stage in ("question_gen", "verify", "answer_gen"):
        stats.update(_stage_stats(stage, latencies[stage], ttfts[stage]))

    return {
        "records": records,
        "stage_logs": stage_logs,
        "verifier_pass": verifier_pass,
        "verifier_fail": verifier_fail,
        "chunk_wall_times": chunk_wall_times,
        "latency_stats": stats,
    }


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
            f"llama-server not reachable at {llm_cfg['base_url']} — start it first (see serving/README.md)."
        )

    verify_base_url = llm_cfg.get("verify_base_url")
    verify_model = llm_cfg.get("verify_model")
    if verify_base_url or verify_model:
        resolved_verify_base_url = verify_base_url or llm_cfg["base_url"]
        resolved_verify_model = verify_model or llm_cfg["model"]
        verify_client = LocalLLMClient(base_url=resolved_verify_base_url, model=resolved_verify_model)
        if not verify_client.ping():
            raise RuntimeError(f"verify_base_url {resolved_verify_base_url} not reachable")
        print(f"Verifier: independent model ({resolved_verify_model} @ {resolved_verify_base_url})")
    else:
        verify_client = client
        print(f"Verifier: same model as generator ({llm_cfg['model']}) — self-verification, see README caveat")

    nudge = config.get("extraction_nudge", "") or ""
    domain_description = config.get("domain_description", prompts.DEFAULT_DOMAIN_DESCRIPTION)
    source_label = config.get("source_label", prompts.DEFAULT_SOURCE_LABEL)
    # Matches llama-server's total_slots (check `curl localhost:8080/props`) --
    # higher doesn't help throughput since the server can't run more than
    # that many generations in parallel anyway, it'd just queue.
    max_concurrency = config.get("max_concurrency", 4)

    repos = config.get("repos", [])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    if repos:
        docs_cache_dir = Path(config["docs_cache_dir"])
        print(f"Fetching docs for: {repos}")
        docs_roots = fetch_all(repos, docs_cache_dir, refresh=refresh_docs)
        print("Chunking docs...")
        chunks.extend(chunk_all(docs_roots, config["chunk_size_chars"], config["min_chunk_chars"]))

    epub_sources = config.get("epub_sources", {}) or {}
    if epub_sources:
        print(f"Chunking EPUB sources: {list(epub_sources)}")
        epub_paths = {name: Path(p) for name, p in epub_sources.items()}
        chunks.extend(chunk_all_epubs(epub_paths, config["chunk_size_chars"], config["min_chunk_chars"]))

    if not chunks:
        raise ValueError("No chunks produced -- set 'repos' and/or 'epub_sources' in the config.")
    total_available = len(chunks)
    print(f"Got {total_available} chunks total.")
    if limit:
        chunks = chunks[:limit]
        print(f"Limiting to first {limit} chunks.")

    run_type = "dry_run" if dry_run else ("partial" if limit else "full")
    model_short = Path(llm_cfg["model"]).stem  # "models/Qwen3.5-4B-Q8_0.gguf" -> "Qwen3.5-4B-Q8_0"
    run_name = f"astral-sft-{run_type}-{model_short}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    output_jsonl = output_dir / "training_data.jsonl"

    result = asyncio.run(
        _process_all_chunks(
            chunks,
            client,
            verify_client,
            llm_cfg["temperature"],
            llm_cfg["max_tokens"],
            nudge,
            domain_description,
            source_label,
            output_jsonl,
            max_concurrency,
        )
    )
    records: list[QARecord] = result["records"]
    stage_logs = result["stage_logs"]
    verifier_pass = result["verifier_pass"]
    verifier_fail = result["verifier_fail"]
    chunk_wall_times = result["chunk_wall_times"]

    summary_metrics: dict[str, float] = {
        "chunks_processed": len(chunks),
        "verifier_pass_count": verifier_pass,
        "verifier_fail_count": verifier_fail,
        "qa_pairs_written": len(records),
        **result["latency_stats"],
    }
    if chunk_wall_times:
        avg_chunk_s = sum(chunk_wall_times) / len(chunk_wall_times)
        summary_metrics["avg_chunk_wall_s"] = avg_chunk_s
        if limit and limit < total_available:
            eta_s = avg_chunk_s * total_available
            summary_metrics["estimated_full_run_s"] = eta_s
            print(
                f"\n=== ESTIMATE (based on {len(chunk_wall_times)} sampled chunks) ===\n"
                f"Avg time per chunk: {avg_chunk_s:.2f}s\n"
                f"Total chunks available: {total_available}\n"
                f"Estimated time for full run: {eta_s / 60:.1f} min ({eta_s / 3600:.2f} hr)\n"
            )

    log_benchmark_metric(
        {
            "run_name": run_name,
            "run_type": run_type,
            "model": llm_cfg["model"],
            "verifier_model": verify_model or llm_cfg["model"],
            "verifier_base_url": verify_base_url or llm_cfg["base_url"],
            "extraction_nudge": nudge or "(none)",
            "question_gen_version": prompts.QUESTION_GEN_VERSION,
            "verify_version": prompts.VERIFY_VERSION,
            "answer_gen_version": prompts.ANSWER_GEN_VERSION,
            "chunk_size_chars": config["chunk_size_chars"],
            "min_chunk_chars": config["min_chunk_chars"],
            "repos": ",".join(repos) or "(none)",
            "epub_sources": ",".join(epub_sources) or "(none)",
            "domain_description": domain_description,
            "source_label": source_label,
            "limit": limit or "none",
            "total_chunks": len(chunks),
            "max_concurrency": max_concurrency,
            **summary_metrics,
        },
        filepath="results/datagen_runs.jsonl",
    )

    output_records = [r.to_dict(_render_text_fallback(r.question, r.answer)) for r in records]
    print(f"Wrote {len(output_records)} records to {output_jsonl}")

    hf_dataset_dir = output_dir / "hf_dataset"
    Dataset.from_list(output_records).save_to_disk(str(hf_dataset_dir))

    prompts_dir = output_dir / "stage_logs"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for stage, entries in stage_logs.items():
        stage_file = prompts_dir / f"{stage}.jsonl"
        with open(stage_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    if llm_cfg.get("unload_after_run", True):
        if client.shutdown_server():
            print(f"Unloaded model from VRAM (stopped llama-server on port {client._client.base_url.port}).")
        if verify_client is not client and verify_client.shutdown_server():
            port = verify_client._client.base_url.port
            print(f"Unloaded verifier model from VRAM (stopped llama-server on port {port}).")

    client.close()
    if verify_client is not client:
        verify_client.close()
    return output_jsonl
