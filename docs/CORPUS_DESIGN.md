# Corpus design — what these datasets teach, and why they are shaped this way

> Read before adding a domain, adding records, or designing an eval set.
> Companion to [`WHY_EXPERTS.md`](WHY_EXPERTS.md) (why disposition matters) and
> `DECISIONS.md` §44 (why the previous disposition instrument was retired).

---

## The one-line rule

**A corpus teaches a FORM, not just facts.** Whatever shape 90% of its answers
take is the shape the adapter will emit, regardless of what the question asked for.

That sentence is not a principle we adopted in advance. It is the conclusion of an
audit, and it cost us an adapter.

---

## The audit that produced this document

`scripts/corpus/audit_corpora.py`, run against the v4 corpora:

| domain | records | runnable artifact | dup answers | effective unique |
| :--- | ---: | ---: | ---: | ---: |
| astral | 1433 | **13.7%** | 24.9% | ~1076 |
| postgresql | 1385 | 83.3% | 33.1% | ~926 |
| duckdb | 1622 | 95.9% | 51.0% | ~794 |
| financial | 1628 | n/a (prose domain) | **79.9%** | **~328** |

Then the astral corpus split by provenance:

```
745 doc-scraped from Astral's own docs   -> 22.0% of answers contain a command
688 hand-written template generators     ->  0.0% of answers contain a command
```

And of those 688, **~88% were generic Python / FastAPI** — `func_lru_cache`,
`fastapi_crud_router`, `py_match_case`, `asyncpg_pool`. Nothing to do with Astral
tooling.

So when the "astral expert" was asked to add a dependency with uv, it wrote a
`pyproject.toml` and an essay — while **base, with no adapter at all**, emitted
`uv add fastapi==3.0.0 / uv lock / uv add ruff ty`.

The adapter behaved exactly as trained. We had benchmarked it against a task its
corpus never contained.

---

## Four rules that follow

### 1. Teach the form you want emitted

If you want commands, the answers must **be** commands. Prose *about* commands
teaches explanation. Target: **≥40% of answers contain a runnable artifact** for
the domain (`audit_corpora.py` reports this per corpus).

### 2. Situation → choice, not question → restatement

The old shape teaches capability, because the answer is already in the question:

> *Q: Convert a legacy `SERIAL` primary key to `GENERATED ALWAYS AS IDENTITY` and
> explain the advantage.*

Nothing is chosen. The disposition shape is:

> **situation** (names no tool) → **right approach, as code** → **`**Not X**` — why not**

The rejection half is what carries the bias. *"Use pgvector"* teaches a fact.
*"Use pgvector, **not** a separate vector database, because filtering against your
own rows is one index scan here and a two-system dance anywhere else"* teaches a
**preference** — and a preference is what survives into a situation the corpus never
showed.

### 3. Hold out instances, not constructs

Restated from `WHY_EXPERTS.md` because it is the rule most often applied backwards:

> **Reserve CONSTRUCTS when they are peripheral to the domain.
> Reserve INSTANCES when they ARE the domain.**

`uv` has ~25 commands and they are enumerable. An expert that has never seen
`uv sync` is broken, not general. So the command surface is **saturated
deliberately**, and the holdout moves to:

```
trained:   every command, every flag, every construct
held out:  packages, question phrasings, situational contexts, whole SITUATIONS
```

This is only "cheating" if the result is then reported as generalisation. Report it
as **reliability on a known surface** and the claim matches the measurement.

Applying the construct rule to a central construct fails one of two ways, and we hit
the first before noticing: a **silent no-op** (family names don't match, 0 records
drop) or a **destroyed corpus** (reserving DuckDB's constructs drops 90.3% of it).

### 4. Templated generation duplicates by default — measure it every time

Every generator written for this repo produced a duplication defect on its first
build:

```
financial (original)                  79.9% duplicate answers
duckdb (original)                     51.0% duplicate answers
build_astral_commands.py   1st build  55.0% duplicate questions
build_disposition_corpus.py 1st build 73.4% duplicate questions
```

The cause is always the same: N situations × M phrasings cannot fill K records when
K ≫ N·M. The fix is **situational context** — a pool of realistic qualifiers
appended to the question, which multiplies the space and also reduces template
overfitting. Both builders went to **>90% unique** after adding one.

Check `unique questions` in the build report. If it is under ~85%, the corpus is
smaller than it claims.

---

## Current corpora

| domain | base | + disposition | + commands | eval | teaches |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `astral` | 831 | — | **836** | 8 | uv / ruff / ty **commands** |
| `postgresql` | 1385 | **442** | — | 8 | antipatterns + modern patterns |
| `duckdb` | 1622 | **390** | — | 7 | reach for DuckDB on file work |
| `python_modern` | 301 | **520** | — | 8 | a **style** of writing Python |
| `python_web` | 301 | **468** | — | 8 | FastAPI over Django/Flask + practice |
| `financial` | 1628 | — | — | — | ⚠️ 79.9% duplicated, needs regeneration |

`python_modern` and `python_web` were **split out of astral** — see
`scripts/corpus/split_astral_domain.py`. `data/astral/training_data_v4_unsplit.jsonl`
preserves the 1433-record original, because `m2_astral_r8a128_v4` was trained on it
and an adapter you cannot rebuild is one you cannot trust a benchmark against.

⚠️ **None of the new files are merged into `training_data_v4.jsonl`.** The merge
ratio is a per-domain decision: postgres already has 405 good antipattern records, so
adding 442 more is a different call from astral, which has almost none.

---

## What each domain is biased toward

**`astral`** — `uv add` / `--dev` / `lock` / `sync --frozen` / `run` / PEP 723 /
`python pin` / `uvx` / `build` / `tree` / `export`, `ruff check --fix`,
`ruff format`, `ty check`. Explicitly **not** pip, poetry, pipenv, pyenv, virtualenv,
black, isort, flake8.

The highest-value families are `uvcmd_add_dev_judgment` (the request says
"dependencies"; a linter belongs in a dev group anyway) and `uvcmd_migrate` (off
`pip` + `requirements.txt` — the answer base reaches for unprompted).

**`postgresql`** — both halves. Antipatterns: `NOT IN` with NULLs, non-sargable
`date(col)`, check-then-insert races, `SERIAL`, EAV, polling queues, random-v4 UUID
keys, naive `timestamp`. Modern patterns: pgvector + HNSW, JSONB + GIN, generated
`tsvector` instead of a search cluster, range partitioning + BRIN,
`count(*) FILTER`, `CREATE INDEX CONCURRENTLY`, covering `INCLUDE` indexes, advisory
locks, batched `ctid` migrations.

**`duckdb`** — reach for it on file and analytical work: `read_parquet` with
pushdown over glob+concat, streaming a 60GB CSV over `chunksize`, `EXCLUDE` /
`COLUMNS` / `QUALIFY` / `GROUP BY ALL` / `PIVOT` / `ASOF JOIN` / `USING SAMPLE`,
`ATTACH … TYPE POSTGRES` over an export step, `.pl()` over the pandas round-trip,
`httpfs` for S3, `SUMMARIZE` over `df.describe()`.

**`python_modern`** — style, not syntax: `pathlib` over `os.path`, frozen slotted
dataclasses over dicts, `match` over if-chains, `@cache` over hand-rolled memos,
`TaskGroup` over bare `gather`, `StrEnum` over string constants, `zip(strict=True)`,
`datetime.now(UTC)` over `utcnow()`, generators over materialised lists, the
mutable-default trap, no bare `except`.

**`python_web`** — FastAPI over Django/Flask **with the reason**, Pydantic models
over hand validation, `response_model` so internal fields cannot leak, `Depends`
over globals, `lifespan` over deprecated `on_event`, `HTTPException` over
200-with-error-body, no `requests.get()` inside `async def`, asyncpg over psycopg2,
`APIRouter` splits, `StreamingResponse`, ASGI transport in tests,
`dependency_overrides` over monkeypatching, cursor pagination, `BaseSettings`.

---

## Eval sets

`scripts/corpus/build_disposition_evals.py` — **held-out situations, trained
constructs**. Every item carries both directions:

```json
{"id": "pg_e1", "category": "vector",
 "prompt": "We store product descriptions and want 'find me similar products'
            without standing up new infrastructure.",
 "expects": ["\\bpgvector\\b|\\bvector\\b", "<=>|<->|hnsw|ivfflat"],
 "avoid":   ["\\bpinecone\\b", "\\bweaviate\\b", "\\bfaiss\\b", "\\belasticsearch\\b"]}
```

`avoid` is the important half. **"Solved it correctly with the wrong tool" is scored
as a success by every execution-gated benchmark in this repo** — it is the failure
the experts exist to prevent, and it is invisible without an explicit avoid list.

The build **fails** rather than emitting a prompt that names its own tool. A prompt
containing "pgvector" measures capability, not disposition.

### Scoring these needs a new scorer

The existing keyword scorers read `expects` and ignore `avoid`. That is exactly why
`DECISIONS.md` §44 retired the old disposition benchmark: it scored an answer as
**NATIVE = 1.0** that invented an invalid `[tool.uv] sources = [...]` schema,
recommended git-installing Ruff, and never mentioned `uv.lock` on a
reproducible-builds question.

A scorer worth trusting must **parse what comes out**, not grep it:

- does the emitted TOML load?
- is `tool.uv.sources` a table rather than a list?
- does the declared build backend match `requires`?
- does the SQL parse against the target dialect?
- does `avoid` fire? — that outcome is worse than silence and must be counted apart

---

## Still outstanding

1. **`financial` needs regeneration**, not deduplication — 79.9% duplicate answers,
   effectively a ~330-record corpus padded 5×. It is also the expert that costs
   **−36.67pp** on other domains while being worth **+0.83pp** on its own.
2. **`family_of()` cannot label doc-scraped records.** 745 astral (52%) and 399
   postgres (29%) records carry `source_path`, not `source`, so they collapse to
   `"?"` and `reserve_eval_constructs.py` can never reserve them. Three lines.
3. **Merge ratios undecided** for every domain.
4. **`python_modern` / `python_web` have no adapters** and no entry in the
   benchmarks. They are corpora with eval sets; they are not experts yet.
