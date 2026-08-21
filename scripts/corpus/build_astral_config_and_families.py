"""Astral: disambiguate the blending command families, and teach the config surface.

WHAT THIS FIXES -- measured, not guessed
----------------------------------------
benchmarks/factory/agentic/attribution, v6 astral solo, asked to add ruff and ty
as dev dependencies:

    uv tool --dev ruff ty          <- emitted
    uv add --dev                   82 records in the corpus
    uv tool                        52 records
    uv tool --dev                   0 records  -- the emitted combination

The adapter recited the `uvcmd_add_dev_judgment` rationale WORD FOR WORD while
attaching it to a verb from a different family. On the other prompt it emitted
`uv check`, which is not a uv subcommand at all (2 corpus occurrences, both prose).

That is FAMILY BLENDING, the open item in docs/CORPUS_DESIGN.md: two families
share a rationale and a flag, never co-occur in a single record, and the adapter
interpolates between them. Frequency alone cannot fix it -- adding more
`uv add --dev` records also adds more `--dev` next to more rationale.

The fix is CONTRASTIVE records: both verbs in the SAME answer, with the boundary
stated. A model cannot blend two things it has been shown side by side.

SECOND GAP: the config surface. astral v5 has 1610 records and:

    [tool.ruff] 9   line-length 7   select= 3   target-version 2   [tool.uv] 68
    dependency-groups 6   required-version 0

Nine records is not a taught surface. Every one of them is doc-scraped prose
answering "how does Ruff decide ..." -- none EMIT a config block for a stated
need, which is the form the question actually asks for.

    uv run python scripts/corpus/build_astral_config_and_families.py
    uv run python scripts/corpus/build_astral_config_and_families.py --write
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

from runtime.canon import REPO_ROOT

ANSWER_MARKER = "\n\n### Answer:\n"
GEN = "astral_config_families_v1"

DEV_TOOLS = ["ruff", "ty", "pytest", "mypy", "pre-commit", "coverage", "hypothesis"]
CLI_TOOLS = ["ruff", "black", "httpie", "cookiecutter", "pgcli", "yt-dlp", "poetry"]
PY_VERS = ["3.11", "3.12", "3.13"]
LINE_LENS = [88, 100, 110, 120]
RULE_SETS = [
    ('["E", "F", "I", "UP", "B"]', "pycodestyle, pyflakes, isort, pyupgrade and bugbear"),
    ('["E", "F", "I", "N", "SIM"]', "the core set plus naming and simplification"),
    ('["ALL"]', "everything, then narrow with `ignore`"),
    ('["E", "F", "I", "ANN", "RUF"]', "the core set plus annotations and Ruff-native rules"),
]

# ANSWER-side variation. The first build of this file scored 89.8% unique
# QUESTIONS and 14.9% unique ANSWERS -- five families had a fully static answer, so
# 34 byte-identical copies each. merge_domain_corpora.py dedups on the normalised
# answer, so those would have collapsed to one and the corpus would have been a
# third of its claimed size. Question variety is not corpus variety.
PKGS = ["internal-sdk", "acme-client", "billing-core", "telemetry-sdk", "shared-schemas",
        "auth-lib", "ingest-tools", "pricing-engine"]
MODS = ["myapp", "service", "core", "platform", "api", "pipeline", "worker"]
SRC_DIRS = ["src", "lib", "packages"]
TEST_GLOBS = ["tests/**", "tests/**/*.py", "test/**"]
LEGACY = ["legacy", "vendor", "thirdparty", "compat", "old_api"]
GIT_HOSTS = ["git.internal", "github.com/acme", "git.corp.example"]
TAGS = ["v2.3.1", "v1.9.0", "v0.14.2", "v3.0.0-rc2"]

def pick(rng, seq):
    return rng.choice(seq)


# Normalised dedup (the one merge_domain_corpora.py uses) strips DIGITS, so a family
# that varies only a line-length or a version number collapses to one record. Variation
# has to be LEXICAL to survive. These are the rationale phrasings that provide it.
ALT_ADD_TOOL = [
    ("writes `{p}` into `pyproject.toml` and `uv.lock`, so a colleague who runs "
     "`uv sync` gets the identical version",
     "puts it on your PATH in its own environment and touches neither file — nothing "
     "is pinned for anyone else"),
    ("records `{p}` as a project dependency, so the lockfile pins it and CI resolves "
     "the same build every time",
     "installs it user-wide in an isolated venv; the project never learns it exists"),
    ("makes `{p}` part of what `uv sync` reconstructs for every contributor",
     "gives you a personal copy on PATH that no lockfile describes"),
]
ALT_UVX = [
    "fetches `{p}` into a throwaway environment and runs it",
    "resolves `{p}` on the fly, runs it, and discards the environment afterwards",
    "builds an ephemeral environment for `{p}`, runs the command, then throws it away",
]
ALT_PIN = [
    "`uv python pin` writes `.python-version`, which `uv run` and `uv sync` both honour",
    "`.python-version` is what `uv python pin` produces, and every `uv run` reads it",
    "the pin lands in `.python-version`; `uv run` and `uv sync` pick it up with no flags",
]
ALT_MIGRATE = [
    "Ruff's formatter is Black-compatible by design, so keeping `line-length` where "
    "Black had it makes the first reformat a near no-op instead of a diff across every file",
    "because the formatter matches Black's output, holding `line-length` steady keeps "
    "the migration commit reviewable instead of touching every file",
    "keep `line-length` at whatever Black used — the formatter is Black-compatible, and "
    "changing both at once buries the real diff",
]

# Situational qualifiers. N situations x M phrasings cannot fill K records when
# K >> N*M -- every generator written for this repo duplicated on its first build
# until one of these pools was added (docs/CORPUS_DESIGN.md rule 4).
CTX = [
    "", " in a monorepo", " for a library we publish to PyPI", " in CI",
    " for a service that ships in a slim Docker image", " on a team of six",
    " in a repo that still has a requirements.txt", " for an internal CLI tool",
    " where the previous setup used Black and Flake8", " in a fresh project",
    " for a package with a src/ layout", " that a contractor will hand back to us",
]
PHRASE = [
    "{task}?", "How do I {task}?", "{task} -- what's the right way?",
    "Need to {task}. Show me.", "What should I do to {task}?",
    "{task}. Config please.", "Can you {task}?",
]


def rec(family: str, question: str, answer: str) -> dict:
    return {
        "text": f"### Question:\n{question}{ANSWER_MARKER}{answer}",
        "messages": [{"role": "user", "content": question},
                     {"role": "assistant", "content": answer}],
        "meta": {"family": family, "tool": "uv", "gen": GEN},
    }


def build(rng: random.Random, n_per: int) -> list[dict]:
    out: list[dict] = []

    def add(fam: str, task: str, ans: str) -> None:
        q = rng.choice(PHRASE).format(task=task + rng.choice(CTX))
        out.append(rec(fam, q[0].upper() + q[1:], ans))

    for _ in range(n_per):
        dev = rng.choice(DEV_TOOLS)
        dev2 = rng.choice([d for d in DEV_TOOLS if d != dev])
        cli = rng.choice(CLI_TOOLS)
        ver = rng.choice(PY_VERS)
        ll = rng.choice(LINE_LENS)
        rules, rules_desc = rng.choice(RULE_SETS)

        # ---------------------------------------------------------------- FAMILY
        # BOUNDARY. Both verbs in one answer. This is the defect-specific record.
        add("uvfam_add_vs_tool",
            f"decide between `uv add --dev {cli}` and `uv tool install {cli}`",
            f"They do different jobs, so pick by who needs it:\n\n"
            f"```bash\n# a tool THIS project's contributors run -> project dev group\n"
            f"uv add --dev {cli}\n\n"
            f"# a tool YOU run anywhere, isolated from any project -> user-wide\n"
            f"uv tool install {cli}\n```\n\n"
            f"`uv add --dev` {rng.choice(ALT_ADD_TOOL)[0].format(p=cli)}. "
            f"`uv tool install` {rng.choice(ALT_ADD_TOOL)[1]}.\n\n"
            f"There is no `--dev` flag on `uv tool`, and no `install` subcommand on "
            f"`uv add`. If you want it reproducible for the team, it is `uv add --dev`.")

        add("uvfam_add_vs_tool_run",
            f"run {cli} once without installing it",
            f"```bash\nuvx {cli}\n```\n\n"
            f"`uvx` is `uv tool run` — it {rng.choice(ALT_UVX).format(p=cli)}. "
            f"Nothing is added to `pyproject.toml`, nothing lands on your PATH.\n\n"
            f"Use `uv add --dev {cli}` instead when the whole team needs it pinned, and "
            f"`uv tool install {cli}` when you want it permanently on your own PATH.")

        add("uvfam_dev_group",
            f"add {dev} and {dev2} as dev dependencies",
            f"```bash\nuv add --dev {dev} {dev2}\n```\n\n"
            f"`--dev` puts them in the `dev` dependency group in `pyproject.toml`, so "
            f"`uv sync` installs them for contributors and `uv sync --no-dev` leaves "
            f"them out of the production image.\n\n"
            f"Not `uv tool install` — that is user-wide and would not be pinned in "
            f"`uv.lock` for anyone else.")

        # ---------------------------------------------------------------- THE
        # NON-COMMAND the adapter invented. Naming the wrong form explicitly is the
        # only thing that removes it; frequency of the right form did not.
        tgt = rng.choice(SRC_DIRS + ["."] + [f"{m}/" for m in MODS])
        wrong = rng.choice(["uv check", "uv lint", "uv ruff", "uv format"])
        add("uvfam_no_such_command",
            f"run the linter over {'the project' if tgt == '.' else tgt}",
            f"```bash\nruff check {tgt}\n```\n\n"
            f"To apply the fixes it can make automatically:\n\n"
            f"```bash\nruff check --fix {tgt}\n```\n\n"
            f"There is no `{wrong}` — `uv` manages environments and dependencies, "
            f"`ruff` does the linting. Use `uv run ruff check {tgt}` if you want it to "
            f"run inside the project environment rather than off your PATH.")

        fmt_tgt = rng.choice(SRC_DIRS + ["."])
        layout = rng.choice(["quotes, line breaks, trailing commas",
                             "indentation, string quoting and line wrapping",
                             "spacing, quote style and where lines break"])
        lint_kind = rng.choice(["unused imports, unsorted imports, outdated syntax",
                                "dead imports, import order and legacy syntax",
                                "unused names, import ordering and deprecated idioms"])
        add("uvfam_format_vs_check",
            f"format {fmt_tgt} and fix what can be fixed automatically",
            f"```bash\nruff format {fmt_tgt}\nruff check --fix {fmt_tgt}\n```\n\n"
            f"`ruff format` rewrites layout — {layout}. `ruff check --fix` is the "
            f"linter, and it repairs a different class of problem: {lint_kind}.\n\n"
            f"Run `format` first; a reformatted file can change which lines the linter "
            f"flags.")

        # ---------------------------------------------------------------- CONFIG.
        # These EMIT a block for a stated need. The nine existing [tool.ruff] records
        # are all prose explaining how Ruff resolves config -- a different form.
        add("cfg_ruff_base",
            f"set up ruff in pyproject.toml with a {ll}-character line limit targeting "
            f"Python {ver}",
            f"```toml\n[tool.ruff]\nline-length = {ll}\ntarget-version = \"py{ver.replace('.', '')}\"\n\n"
            f"[tool.ruff.lint]\nselect = {rules}\n\n[tool.ruff.format]\n"
            f"quote-style = \"double\"\n```\n\n"
            f"{rng.choice([f'`select` here turns on {rules_desc}', f'that `select` list is {rules_desc}', f'the selected set covers {rules_desc}'])}. "
            f"{rng.choice(['`target-version` is what lets pyupgrade rules know which syntax is safe to rewrite to', '`target-version` tells the pyupgrade rules which syntax they are allowed to modernise to', 'without `target-version` the upgrade rules cannot tell which syntax your runtime accepts'])}.")

        tglob = rng.choice(TEST_GLOBS)
        drop = rng.choice([
            ('"D",      # docstring rules -- not enforcing these yet',
             '"COM812", # trailing-comma, conflicts with the formatter'),
            ('"ANN",    # annotations -- rolling out gradually',
             '"ISC001", # implicit string concat, conflicts with the formatter'),
            ('"DOC",    # docstring content rules, too noisy for now',
             '"E501",   # line length, the formatter already owns this'),
        ])
        tign = rng.choice(['"S101", "PLR2004"  # assert and magic numbers are fine in tests',
                           '"S101", "ANN201"   # asserts and untyped test helpers are fine',
                           '"PLR2004", "SLF001"  # magic numbers and private access in tests'])
        add("cfg_ruff_ignore",
            f"keep ruff's broad rule set but silence the rules that fight {tglob}",
            f"```toml\n[tool.ruff.lint]\nselect = [\"ALL\"]\nignore = [\n"
            f"    {drop[0]}\n    {drop[1]}\n]\n\n"
            f"[tool.ruff.lint.per-file-ignores]\n\"{tglob}\" = [{tign}]\n```\n\n"
            f"`per-file-ignores` is the part worth reaching for — it keeps the strict "
            f"set everywhere else instead of globally disabling a rule because one "
            f"directory trips it.")

        sd = rng.choice(SRC_DIRS)
        mod = rng.choice(MODS)
        add("cfg_ruff_isort",
            f"make ruff treat `{mod}` as first-party when sorting imports",
            f"```toml\n[tool.ruff]\nsrc = [\"{sd}\", \"tests\"]\n\n"
            f"[tool.ruff.lint]\nselect = [\"I\"]\n\n"
            f"[tool.ruff.lint.isort]\nknown-first-party = [\"{mod}\"]\n"
            f"combine-as-imports = true\n```\n\n"
            f"Without `src`, a `{sd}/` layout makes Ruff read `{mod}` as third-party "
            f"and sort it into the wrong block.")

        add("cfg_dep_groups",
            f"split our dev tooling into separate groups so CI installs only what it needs",
            f"```toml\n[dependency-groups]\ndev = [\"{dev}\", \"{dev2}\"]\n"
            f"test = [\"pytest\", \"pytest-cov\"]\ndocs = [\"mkdocs\", \"mkdocs-material\"]\n```\n\n"
            f"```bash\nuv sync --group test          # CI test job\n"
            f"uv sync --no-dev                # production image\n```\n\n"
            f"`[dependency-groups]` is the PEP 735 standard table, so it is not "
            f"uv-specific — other tools read it too.")

        add("cfg_uv_pin",
            f"make everyone on the team build against Python {ver} and the same uv",
            f"```bash\nuv python pin {ver}\n```\n\n```toml\n[project]\n"
            f"requires-python = \">={ver}\"\n\n[tool.uv]\nrequired-version = \">=0.5.0\"\n```\n\n"
            f"{rng.choice(ALT_PIN)}. `requires-python` is the constraint the resolver "
            f"uses; `required-version` fails fast when someone's uv is too old to read "
            f"the lockfile format.")

        pkg = rng.choice(PKGS)
        host = rng.choice(GIT_HOSTS)
        tag = rng.choice(TAGS)
        add("cfg_uv_source",
            f"point `{pkg}` at our internal git repo without breaking the lockfile",
            f"```toml\n[project]\ndependencies = [\"{pkg}\"]\n\n"
            f"[tool.uv.sources]\n{pkg} = {{ git = \"https://{host}/{pkg}\", "
            f"tag = \"{tag}\" }}\n```\n\n"
            f"`[tool.uv.sources]` is a TABLE keyed by package name, not a list. "
            f"`{pkg}` still appears in `[project.dependencies]` — the source table only "
            f"overrides WHERE it is fetched from, so the metadata stays standard for "
            f"anything that is not uv.")

        add("cfg_ty",
            f"turn on type checking without failing the build on the {rng.choice(LEGACY)} package",
            f"```toml\n[tool.ty.src]\ninclude = [\"{rng.choice(SRC_DIRS)}\"]\n\n"
            f"[tool.ty.rules]\npossibly-unbound-attribute = \"warn\"\n\n"
            f"[[tool.ty.overrides]]\ninclude = [\"src/{rng.choice(LEGACY)}/**\"]\n\n"
            f"[tool.ty.overrides.rules]\nunresolved-attribute = \"ignore\"\n```\n\n"
            f"```bash\nuv run ty check\n```\n\n"
            f"`[[tool.ty.overrides]]` is what keeps the strict setting on new code while "
            f"the legacy tree stays quiet — better than lowering the global rule.")

        legacy_set = rng.choice([
            ("black isort flake8", "black, isort and flake8"),
            ("black flake8 pyupgrade autoflake", "black, flake8, pyupgrade and autoflake"),
            ("flake8 isort pydocstyle", "flake8, isort and pydocstyle"),
            ("black isort flake8 bandit", "black, isort, flake8 and bandit"),
        ])
        add("cfg_migrate",
            f"replace {legacy_set[1]} with ruff in one pass",
            f"```bash\nuv remove --dev {legacy_set[0]}\nuv add --dev ruff\n"
            f"ruff check --fix {rng.choice(SRC_DIRS + ['.'])}\nruff format .\n```\n\n"
            f"```toml\n[tool.ruff]\nline-length = {ll}          # what the old setup used\n\n"
            f"[tool.ruff.lint]\nselect = {rules}\n```\n\n"
            f"{rng.choice(ALT_MIGRATE)}.")

    return out


def report(rows: list[dict]) -> None:
    qs = [r["messages"][0]["content"] for r in rows]
    ans = [r["messages"][1]["content"] for r in rows]
    def norm(s: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9\s]", " ", re.sub(r"\d+", "0", s.lower())).split())
    uq = len(set(qs)) / max(1, len(qs))
    ua = len(set(norm(a) for a in ans)) / max(1, len(ans))
    runnable = sum(bool(re.search(r"```(bash|toml)", a)) for a in ans) / max(1, len(ans))
    print(f"  records          {len(rows)}")
    print(f"  unique questions {uq:6.1%}   (under ~85% means the corpus is smaller than it claims)")
    print(f"  unique answers   {ua:6.1%}")
    print(f"  emits bash/toml  {runnable:6.1%}")
    print("\n  families:")
    for f, c in Counter(r["meta"]["family"] for r in rows).most_common():
        print(f"    {f:28s} {c:5d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--n-per", type=int, default=34, help="instances per family")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = build(random.Random(args.seed), args.n_per)

    # Dedup on the SAME normalised key merge_domain_corpora.py uses, so the count
    # printed here is the count that survives the merge. A builder that reports 442
    # and contributes 66 is how a corpus ends up a third of its claimed size.
    seen, uniq = set(), []
    for r in rows:
        k = re.sub(r"[^a-z0-9\s]", " ", re.sub(r"\d+", "0", r["messages"][1]["content"].lower()))
        k = " ".join(k.split())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    print(f"  generated {len(rows)}, {len(uniq)} survive answer-dedup "
          f"({len(uniq)/len(rows):.1%})\n")
    rows = uniq
    print("=" * 78)
    print(" ASTRAL: command-family boundaries + config emission")
    print("=" * 78)
    report(rows)

    if args.write:
        out = REPO_ROOT / "data/astral/training_data_config_families.jsonl"
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"\n  WROTE {out.relative_to(REPO_ROOT)}")
    else:
        print("\n  (dry run -- pass --write)")


if __name__ == "__main__":
    main()
