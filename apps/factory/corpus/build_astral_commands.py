"""Build the uv / ruff / ty COMMAND corpus -- answers that ARE commands.

WHY THIS EXISTS
---------------
The audit found the astral corpus taught explanation, not emission:

    745 doc-scraped records (Astral's own docs)  -> 22.0% contain a runnable command
    688 generated records                        ->  0.0% contain a runnable command

The adapter behaved exactly as trained. Asked to add a dependency with uv, it wrote
a `pyproject.toml` and an essay. Base -- with no adapter at all -- emitted
`uv add fastapi==3.0.0 / uv lock / uv add ruff ty`.

So the gap is not knowledge. Base already knows the command surface. The gap is
DISPOSITION: told "this project uses uv", base gets it right; told nothing, base was
8/8 wrong (pip, venv, pyenv). Disposition is trained by frequency and consistency,
not by novel constructs.

THE HOLDOUT AXIS -- read before adding an eval set
--------------------------------------------------
Do NOT hold out commands. `uv` has ~25 of them and they are enumerable; an expert
that has never seen `uv sync` is broken, not "generalising". Holding out constructs
is for OPEN spaces. This is a CLOSED surface, so the corpus SATURATES it deliberately
and the holdout moves to instances:

    trained:  every command, every flag
    held out: packages, question phrasings, clause combinations, judgment cases

That is the same rule as docs/WHY_EXPERTS.md -- reserve constructs when peripheral,
reserve instances when central -- and it is only "cheating" if the result is then
reported as generalisation. It is not. It is reliability on a known surface.

    uv run python scripts/corpus/build_astral_commands.py           # report
    uv run python scripts/corpus/build_astral_commands.py --write   # emit corpus + eval
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

from runtime_common.canon import REPO_ROOT

ANSWER_MARKER = "\n\n### Answer:\n"

# --------------------------------------------------------------------------
# INSTANCE POOLS. Split train/eval so the eval set uses packages and phrasings
# the corpus never contained -- while both use the SAME command surface.
# --------------------------------------------------------------------------
PKG_TRAIN = ["requests", "httpx", "pydantic", "sqlalchemy", "polars", "rich",
             "typer", "structlog", "orjson", "tenacity", "attrs", "click",
             "jinja2", "pyyaml", "boto3", "redis", "celery", "pillow"]
PKG_EVAL = ["msgspec", "cattrs", "anyio", "watchfiles", "uvloop", "granian"]

RUNTIME_DEPS = ["fastapi", "starlette", "uvicorn", "django", "flask", "aiohttp"]
DEV_TOOLS = ["ruff", "ty", "pytest", "mypy", "pytest-cov", "hypothesis", "pre-commit"]

PYVER_TRAIN = ["3.11", "3.12", "3.13"]
PYVER_EVAL = ["3.14"]

# Question phrasings. The eval templates never appear in training, so a model that
# memorised the template rather than the task scores badly -- which is the real
# overfitting risk with generated data.
PHRASE_TRAIN = [
    "How do I {task}?",
    "What's the command to {task}?",
    "I need to {task}. What do I run?",
    "Show me how to {task}.",
    "{task} -- what's the right command?",
    "Working in a uv project. How do I {task}?",
    "We use uv here. How would you {task}?",
    "Quick one: how do I {task}?",
    "What is the correct way to {task}?",
    "Need to {task}. Command please.",
]
PHRASE_EVAL = [
    "Can you tell me the command to {task}?",
    "My teammate asked me to {task}. What should I type?",
]

# Situational context. Without this the package-invariant tasks ("produce a
# lockfile") repeat verbatim on every iteration.
CTX_TRAIN = [
    "", "", 
    " on a FastAPI service",
    " in a monorepo package",
    " for a CLI tool we ship to users",
    " in a data-pipeline project",
    " before cutting a release",
    " on a fresh clone",
    " in our GitHub Actions job",
    " for a library other teams depend on",
    " in a Docker build",
    " on a colleague's machine that keeps drifting",
]
CTX_EVAL = [
    " in an Airflow DAG repo",
    " for a Jupyter-based analysis project",
]


def rec(family: str, question: str, answer: str) -> dict:
    return {
        "text": f"### Question:\n{question}{ANSWER_MARKER}{answer}",
        "messages": [{"role": "user", "content": question},
                     {"role": "assistant", "content": answer}],
        "meta": {"family": family, "tool": "uv", "gen": "astral_commands_v1"},
    }


def build(rng: random.Random, pkgs: list[str], pyvers: list[str],
          phrases: list[str], n_per: int, ctxs: list[str]) -> list[dict]:
    out: list[dict] = []

    def add(fam: str, task: str, ans: str) -> None:
        q = rng.choice(phrases).format(task=task + rng.choice(ctxs))
        out.append(rec(fam, q, ans))

    for _ in range(n_per):
        p, q = rng.sample(pkgs, 2)
        rt = rng.choice(RUNTIME_DEPS)
        dev = rng.choice(DEV_TOOLS)
        dev2 = rng.choice([d for d in DEV_TOOLS if d != dev])
        ver = rng.choice(pyvers)

        add("uvcmd_add", f"add {p} to my project",
            f"```bash\nuv add {p}\n```\n\nThis updates `pyproject.toml` and `uv.lock` "
            f"in one step -- no manual editing of either file.")

        add("uvcmd_add_pinned", f"add {rt} pinned to an exact version",
            f"```bash\nuv add \"{rt}==0.115.0\"\n```\n\nUse `==` when you need the exact "
            f"build. For a library, prefer a range like `\"{rt}>=0.115,<1.0\"` so "
            f"consumers can resolve.")

        # THE JUDGMENT CASE. The request says "dependencies"; the correct action is a
        # dev group, because a linter is not a runtime dependency. Base and the v4
        # expert both failed this.
        add("uvcmd_add_dev_judgment", f"add {dev} and {dev2} as dependencies",
            f"```bash\nuv add --dev {dev} {dev2}\n```\n\n`{dev}` and `{dev2}` are "
            f"development tools, not runtime requirements, so they belong in the dev "
            f"group even though the request said \"dependencies\" -- shipping them "
            f"would bloat the production install.")

        add("uvcmd_add_extras", f"add {rt} with its optional extras",
            f"```bash\nuv add \"{rt}[standard]\"\n```")

        add("uvcmd_remove", f"drop {q} from the project",
            f"```bash\nuv remove {q}\n```\n\nThis also prunes it from `uv.lock`.")

        add("uvcmd_lock", "produce a lockfile for reproducible builds",
            "```bash\nuv lock\n```\n\nCommit `uv.lock` to version control. It pins every "
            "direct and transitive dependency, and it is what makes the build "
            "reproducible -- not cache clearing.")

        add("uvcmd_lock_upgrade", f"upgrade only {p} and leave everything else pinned",
            f"```bash\nuv lock --upgrade-package {p}\n```\n\nTo refresh everything "
            f"instead, use `uv lock --upgrade`.")

        add("uvcmd_sync_ci", "install exactly what the lockfile says, in CI",
            "```bash\nuv sync --frozen\n```\n\n`--frozen` fails if `uv.lock` is out of "
            "date rather than silently updating it, which is what you want in CI.")

        add("uvcmd_sync_prod", "install without dev tooling for a production image",
            "```bash\nuv sync --no-dev\n```")

        add("uvcmd_run", "run my test suite inside the project environment",
            "```bash\nuv run pytest\n```\n\n`uv run` resolves and activates the "
            "environment for you -- no `source .venv/bin/activate` needed.")

        add("uvcmd_run_script", f"run a standalone script that needs {p}, without "
            f"installing it into the project",
            f"```bash\nuv run --with {p} script.py\n```\n\nOr declare it inline with "
            f"PEP 723 and just `uv run script.py`:\n\n```python\n# /// script\n"
            f"# dependencies = [\"{p}\"]\n# ///\n```")

        add("uvcmd_python", f"use Python {ver} when my system has something older",
            f"```bash\nuv python install {ver}\nuv python pin {ver}\n```\n\n`pin` writes "
            f"`.python-version` so everyone on the project gets the same interpreter.")

        add("uvcmd_init", "start a new project with a src layout",
            f"```bash\nuv init --lib myproject\ncd myproject\nuv python pin {ver}\n```")

        add("uvcmd_tool", f"run {dev} without adding it to the project",
            f"```bash\nuvx {dev}\n```\n\nOr install it once for repeated use: "
            f"`uv tool install {dev}`.")

        add("uvcmd_build", "build a wheel and sdist",
            "```bash\nuv build\n```\n\nArtifacts land in `dist/`. Publish with "
            "`uv publish`.")

        add("uvcmd_tree", f"find out why {q} is being installed",
            f"```bash\nuv tree --package {q}\n```\n\nShows which dependency pulls it in.")

        add("uvcmd_export", "export the lockfile for a tool that only reads requirements",
            "```bash\nuv export --format requirements.txt > requirements.txt\n```\n\n"
            "The lockfile stays the source of truth; the export is a build artifact.")

        # migration -- the highest-value disposition record, since the wrong answer
        # is the one the base model reaches for unprompted
        add("uvcmd_migrate", "move a project off pip and requirements.txt",
            f"```bash\nuv init\nuv add -r requirements.txt\nuv lock\n```\n\nThen delete "
            f"`requirements.txt` and commit `uv.lock`. Do not keep both -- two sources "
            f"of truth for dependencies is how environments drift.")

        add("ruffcmd_check", "find unused imports and undefined names across the repo",
            "```bash\nruff check .\n```\n\nAdd `--fix` to apply the safe fixes.")

        add("ruffcmd_format", "format the codebase and sort imports",
            "```bash\nruff format .\nruff check --select I --fix .\n```\n\n`ruff format` "
            "handles layout; import sorting is the `I` rule set.")

        add("tycmd_check", "type-check the project",
            "```bash\nty check\n```")

        # multi-clause, in the shape real requests arrive in
        add("uvcmd_multi", f"set up {rt} pinned, add {dev} and {dev2} for development, "
            f"and lock it all",
            f"```bash\nuv add \"{rt}==0.115.0\"\nuv add --dev {dev} {dev2}\nuv lock\n```\n\n"
            f"`{dev}` and `{dev2}` go in the dev group -- they are tooling, not runtime "
            f"dependencies. `uv add` already updates the lockfile; the explicit "
            f"`uv lock` is only needed if you edited `pyproject.toml` by hand.")

    return out


CMD = re.compile(r"\b(uv (add|lock|sync|run|init|venv|python|build|tool|export|remove|tree|publish)"
                 r"|uvx|ruff (check|format)|ty check)\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--n-per", type=int, default=38,
                    help="instances per family (23 families -> ~n*23 records)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train = build(rng, PKG_TRAIN, PYVER_TRAIN, PHRASE_TRAIN, args.n_per, CTX_TRAIN)
    # eval: unseen packages, unseen interpreter, unseen phrasings, same commands
    ev = build(random.Random(args.seed + 1), PKG_EVAL, PYVER_EVAL, PHRASE_EVAL, 2, CTX_EVAL)

    def report(name: str, rows: list[dict]) -> None:
        fams = Counter(r["meta"]["family"] for r in rows)
        cmds = sum(bool(CMD.search(r["messages"][1]["content"])) for r in rows)
        qs = {r["messages"][0]["content"] for r in rows}
        print(f"\n  {name}: {len(rows)} records, {len(fams)} families")
        print(f"    runnable-command answers : {cmds*100.0/len(rows):5.1f}%")
        print(f"    unique questions         : {len(qs)*100.0/len(rows):5.1f}%")

    print("=" * 78)
    print(" uv / ruff / ty COMMAND corpus")
    print("=" * 78)
    report("train", train)
    report("eval ", ev)

    tq = {r["messages"][0]["content"] for r in train}
    eq = {r["messages"][0]["content"] for r in ev}
    print(f"\n  train/eval question overlap: {len(tq & eq)}  (must be 0)")
    tp = set(PKG_TRAIN) & set(PKG_EVAL)
    print(f"  train/eval package overlap : {len(tp)}  (must be 0)")

    if not args.write:
        print("\n  (dry run -- pass --write)")
        return

    out = REPO_ROOT / "apps/factory/data/astral/training_data_commands.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in train) + "\n")
    print(f"\n  WROTE {out}  ({len(train)} records)")
    ev_out = REPO_ROOT / "data/astral/evaluation_data_commands.jsonl"
    ev_out.write_text("\n".join(json.dumps(r) for r in ev) + "\n")
    print(f"  WROTE {ev_out}  ({len(ev)} records)")
    print("\n  NOT merged into training_data_v4.jsonl yet -- merge deliberately, then")
    print("  re-run scripts/corpus/audit_corpora.py to confirm the command rate moved.")


if __name__ == "__main__":
    main()
