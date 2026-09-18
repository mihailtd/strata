"""Dataset generation and data loaders for the System One Decision Engine.

Provides multi-task data ingestion and generation across:
1. Domain Choice: DSH specialist adapter routing (postgresql, python_web, duckdb, astral, python_modern, financial_planning)
2. Tool Choice: Dynamic tool selection given function signatures and user instructions
3. Noul Gate: Boolean readiness, test failure triage, and execution validation
4. Score Rubric: Continuous quality metrics (0-100) for code patches, coverage, and diff cleanliness
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from .models import (
    HARNESS_DOMAIN_DESCRIPTIONS,
    HARNESS_DOMAINS,
)

# -----------------------------------------------------------------------------
# Data Samples and Schema Definitions
# -----------------------------------------------------------------------------

@dataclass
class DecisionSample:
    """A single multi-task training or evaluation sample."""

    prompt: str
    task_type: str  # "choice", "noul", "score", or "multi"
    # Choice fields
    candidate_texts: list[str] | None = None
    candidate_labels: list[str] | None = None
    choice_target: int | None = None
    # Noul fields (boolean binary target in {0.0, 1.0})
    noul_target: float | None = None
    # Score fields (continuous target in [0.0, 100.0])
    score_target: float | None = None


# -----------------------------------------------------------------------------
# Domain Routing Seeds & Generators
# -----------------------------------------------------------------------------

DOMAIN_TEMPLATES: dict[str, list[str]] = {
    "postgresql": [
        "Write a SQL migration adding a pgvector embedding column to table '{table}' with 1536 dimensions.",
        "Create an HNSW index on '{table}.embedding' using cosine distance operator <=> in PostgreSQL 17.",
        "Tune an asyncpg connection pool with max_size={size} and statement cache for concurrent queries.",
        "Optimize slow query on '{table}' using EXPLAIN ANALYZE, composite indexes, and vacuuming.",
        "Implement a partitioned time-series table in PostgreSQL with pg_partman and automated retention.",
        "Write an async query using asyncpg.fetch() to select nearest neighbors with vector <=> $1 LIMIT 10.",
        "Fix deadlocks in concurrent transaction updates on PostgreSQL row-level locking.",
        "Configure pgvector ivfflat and hnsw build parameters m=16 ef_construction=64 for fast retrieval.",
    ],
    "python_web": [
        "Create a FastAPI application using an @asynccontextmanager lifespan handler to initialize DB connections.",
        "Define a Pydantic v2 BaseSchema with ConfigDict(from_attributes=True, populate_by_name=True).",
        "Implement an authenticated FastAPI dependency yielding current user from bearer JWT token.",
        "Write an async endpoint @app.post('/items') validating request body with Pydantic model.",
        "Add Starlette middleware for structured JSON logging with request ID and latency headers.",
        "Implement custom exception handlers in FastAPI returning standardized Problem Details RFC 7807.",
        "Set up streaming server-sent events (SSE) endpoint using Starlette EventSourceResponse.",
        "Configure CORS, rate limiting, and gzip compression middleware in FastAPI router.",
    ],
    "duckdb": [
        "Run an out-of-core analytical query over 50GB Parquet files using DuckDB Python API.",
        "Write a DuckDB SQL query utilizing the native QUALIFY clause with ROW_NUMBER() window function.",
        "Export DuckDB query results directly into Snappy-compressed Parquet with COPY ... TO 's3://bucket/data.parquet'.",
        "Perform columnar percentile calculation and aggregations across partitioned Parquet dataset.",
        "Attach an in-memory DuckDB connection and query multiple parquet files using glob pattern '*.parquet'.",
        "Use DuckDB vectorized window analytics for rolling 30-day cumulative sums and moving averages.",
        "Compare query execution plans between PostgreSQL and DuckDB for vectorized aggregation.",
        "Register an Apache Arrow table inside DuckDB without zero-copy data transfer overhead.",
    ],
    "astral": [
        "Configure a multi-package Python monorepo workspace using Astral uv and pyproject.toml.",
        "Update [tool.ruff.lint] in pyproject.toml to enable rules I, B, UP, SIM, and strict type checking.",
        "Resolve conflicting dependencies using 'uv lock --upgrade-package' and generate reproducible lockfile.",
        "Configure uv pip compile to generate hashes and pinned constraints for production deployment.",
        "Run ruff format and ruff check with automatic fix options in CI pipeline.",
        "Set up virtual environment using 'uv venv --python 3.12' and sync pinned packages.",
        "Structure pyproject.toml workspace members with path dependencies and version overrides.",
        "Speed up Docker build by multi-stage caching of uv cache and pre-built wheels.",
    ],
    "python_modern": [
        "Refactor legacy typing.TypeVar and Generic[T] to modern Python 3.12 PEP 695 syntax: class Container[T]: ...",
        "Write a generic function def process_items[T: (int, str)](items: Sequence[T]) -> list[T]: using PEP 695.",
        "Define a type alias type Point[T] = tuple[T, T] with PEP 695 type parameter syntax.",
        "Use structural pattern matching match/case with class guards and type narrowing.",
        "Implement generic protocols and runtime-checkable protocols with PEP 695 generics.",
        "Convert legacy NamedTuple to dataclass with slots=True and kw_only=True.",
        "Use itertools.batched and new Python 3.12 standard library features.",
        "Implement custom generic mapping types with PEP 695 syntax and strict static analysis.",
    ],
    "financial_planning": [
        "Calculate historical and parametric Value at Risk (VaR) at 95% and 99% confidence intervals using numpy.",
        "Compute Conditional Value at Risk (CVaR / Expected Shortfall) from asset return distributions.",
        "Run a 10,000-path Monte Carlo wealth simulation modeling portfolio drawdowns and retirement ruin probability.",
        "Optimize mean-variance portfolio weights using quadratic programming subject to target return constraints.",
        "Calculate Sharpe ratio, Sortino ratio, and maximum drawdown for historical equities portfolio.",
        "Compute asset correlation matrix and Cholesky decomposition for multivariate return generation.",
        "Model lognormal asset prices with geometric Brownian motion and stochastic volatility.",
        "Implement tax-loss harvesting and dynamic asset rebalancing algorithm for taxable accounts.",
    ],
}

TOOL_CHOICE_SEEDS: list[dict[str, Any]] = [
    {
        "prompt": "Find all occurrences of 'hipblasGemmStridedBatchedEx' across the C++ and Rust codebase.",
        "tools": ["grep_search", "view_file", "run_command", "replace_file_content"],
        "target": "grep_search",
    },
    {
        "prompt": "Read the first 50 lines of src/model.rs to inspect the struct definition.",
        "tools": ["grep_search", "view_file", "run_command", "replace_file_content"],
        "target": "view_file",
    },
    {
        "prompt": "Execute cargo test --test-threads=1 to verify HIP kernel changes.",
        "tools": ["grep_search", "view_file", "run_command", "replace_file_content"],
        "target": "run_command",
    },
    {
        "prompt": "Update line 140 of w4a16_gemm_prefill.hip to change the LDS tile boundary.",
        "tools": ["grep_search", "view_file", "run_command", "replace_file_content"],
        "target": "replace_file_content",
    },
    {
        "prompt": "List the files in the directory apps/runtime-next/src/kernels to see what HIP files exist.",
        "tools": ["list_dir", "view_file", "run_command", "replace_file_content"],
        "target": "list_dir",
    },
    {
        "prompt": "Search the web for the latest ROCm 7.2 release notes and compiler flags.",
        "tools": ["search_web", "view_file", "run_command", "grep_search"],
        "target": "search_web",
    },
]

NOUL_BOOLEAN_SEEDS: list[dict[str, Any]] = [
    {
        "prompt": "Test Output: 'test result: ok. 70 passed; 0 failed; 41 ignored'. Did all executed tests pass?",
        "target": 1.0,
    },
    {
        "prompt": "Compiler Output: 'error[E0425]: cannot find function `fused_qkv_prep` in this scope'. Is build clean?",
        "target": 0.0,
    },
    {
        "prompt": "Runtime Log: 'hipErrorOutOfMemory: out of memory on device (24GB exceeded)'. Can allocation succeed?",
        "target": 0.0,
    },
    {
        "prompt": "Git Status: 'nothing to commit, working tree clean'. Are there uncommitted changes?",
        "target": 0.0,
    },
    {
        "prompt": "Verification Log: 'Linter passed: 0 errors, 0 warnings found in 45 files'. Did linter succeed?",
        "target": 1.0,
    },
    {
        "prompt": "Agent Loop: 'Tried same fix on model.rs 3 times with identical error output'. Has agent stalled?",
        "target": 1.0,
    },
    {
        "prompt": "Agent Loop: 'New file created, 5 new unit tests passing'. Has agent stalled?",
        "target": 0.0,
    },
    {
        "prompt": "Database connection: 'asyncpg.exceptions.CannotConnectNowError: connection refused'. Is DB ready?",
        "target": 0.0,
    },
]

SCORE_RUBRIC_SEEDS: list[dict[str, Any]] = [
    {
        "prompt": "Diff: Added full unit tests, comprehensive docstrings, zero compiler warnings, 100% test pass rate.",
        "score": 98.0,
    },
    {
        "prompt": "Diff: Fixed bug cleanly, but omitted comments and test coverage for edge cases.",
        "score": 72.0,
    },
    {
        "prompt": "Diff: Hardcoded dummy mock data returning fake torch.randn tensors, violating zero-mock rule.",
        "score": 10.0,
    },
    {
        "prompt": "Diff: Incomplete stub function with 'pass' statement causing compile errors.",
        "score": 25.0,
    },
    {
        "prompt": "Diff: Optimized LDS double-buffering kernel, measured 2.7x speedup, passing all 70 unit tests.",
        "score": 99.0,
    },
    {
        "prompt": "Diff: Refactored function names with no functional change or performance improvement.",
        "score": 60.0,
    },
]


def generate_synthetic_samples(
    num_samples: int = 1200,
    seed: int = 42,
) -> list[DecisionSample]:
    """Generate a balanced multi-task synthetic dataset for training and evaluation."""
    rng = random.Random(seed)
    samples: list[DecisionSample] = []

    # 1. DSH Domain Routing Samples (Choice over fixed 6 domains)
    domains = HARNESS_DOMAINS
    samples_per_domain = num_samples // (4 * len(domains))

    tables = ["users", "vector_store", "documents", "embeddings", "sessions", "analytics"]
    sizes = [10, 20, 50, 100]

    for domain_idx, domain in enumerate(domains):
        templates = DOMAIN_TEMPLATES[domain]
        for _ in range(samples_per_domain):
            tmpl = rng.choice(templates)
            prompt = tmpl.format(
                table=rng.choice(tables),
                size=rng.choice(sizes),
            )
            samples.append(
                DecisionSample(
                    prompt=prompt,
                    task_type="choice",
                    candidate_labels=domains,
                    candidate_texts=[HARNESS_DOMAIN_DESCRIPTIONS[d] for d in domains],
                    choice_target=domain_idx,
                )
            )

    # 2. Dynamic Tool Choice Samples
    num_tool_samples = num_samples // 4
    for _ in range(num_tool_samples):
        seed_item = rng.choice(TOOL_CHOICE_SEEDS)
        tools = list(seed_item["tools"])
        rng.shuffle(tools)
        target_idx = tools.index(seed_item["target"])
        samples.append(
            DecisionSample(
                prompt=seed_item["prompt"],
                task_type="choice",
                candidate_labels=tools,
                candidate_texts=[f"Tool command: {t}" for t in tools],
                choice_target=target_idx,
            )
        )

    # 3. Boolean Readiness / Verification Samples (Noul)
    num_noul_samples = num_samples // 4
    for _ in range(num_noul_samples):
        seed_item = rng.choice(NOUL_BOOLEAN_SEEDS)
        samples.append(
            DecisionSample(
                prompt=seed_item["prompt"],
                task_type="noul",
                noul_target=seed_item["target"],
            )
        )

    # 4. Continuous Quality Rubrics (Score)
    num_score_samples = num_samples // 4
    for _ in range(num_score_samples):
        seed_item = rng.choice(SCORE_RUBRIC_SEEDS)
        # Add slight jitter to prevent discrete overfitting
        noise = rng.uniform(-2.0, 2.0)
        target_score = max(0.0, min(100.0, seed_item["score"] + noise))
        samples.append(
            DecisionSample(
                prompt=seed_item["prompt"],
                task_type="score",
                score_target=target_score,
            )
        )

    rng.shuffle(samples)
    return samples


# -----------------------------------------------------------------------------
# PyTorch Dataset & Collate Function
# -----------------------------------------------------------------------------

class DecisionDataset(Dataset):
    """PyTorch Dataset for multi-task Decision Engine training and evaluation."""

    def __init__(
        self,
        samples: list[DecisionSample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        fixed_candidates: list[str] | None = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fixed_candidates = fixed_candidates or HARNESS_DOMAINS

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        return {
            "prompt": item.prompt,
            "task_type": item.task_type,
            "candidate_texts": item.candidate_texts,
            "candidate_labels": item.candidate_labels,
            "choice_target": item.choice_target,
            "noul_target": item.noul_target,
            "score_target": item.score_target,
        }


def make_decision_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
):
    """Factory for custom collate function handling dynamic candidates and multi-task tensors."""

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompts = [b["prompt"] for b in batch]
        task_types = [b["task_type"] for b in batch]

        tokenized = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        choice_targets = []
        noul_targets = []
        score_targets = []

        for b in batch:
            c_target = b["choice_target"] if b["choice_target"] is not None else -100
            n_target = b["noul_target"] if b["noul_target"] is not None else -100.0
            s_target = b["score_target"] if b["score_target"] is not None else -100.0

            choice_targets.append(c_target)
            noul_targets.append(n_target)
            score_targets.append(s_target)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "task_types": task_types,
            "choice_targets": torch.tensor(choice_targets, dtype=torch.long),
            "noul_targets": torch.tensor(noul_targets, dtype=torch.float32),
            "score_targets": torch.tensor(score_targets, dtype=torch.float32),
            "raw_batch": batch,
        }

    return collate_fn


def create_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    num_samples: int = 1200,
    val_split: float = 0.2,
    seed: int = 42,
    max_length: int = 256,
) -> tuple[DataLoader, DataLoader]:
    """Generate dataset and create PyTorch training and validation DataLoaders."""
    all_samples = generate_synthetic_samples(num_samples=num_samples, seed=seed)

    split_idx = int(len(all_samples) * (1.0 - val_split))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    train_dataset = DecisionDataset(train_samples, tokenizer, max_length=max_length)
    val_dataset = DecisionDataset(val_samples, tokenizer, max_length=max_length)

    collate_fn = make_decision_collate_fn(tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
