"""
===============================================================================

   ####    ####     #   #   ###   ######    ######  ####  #   #   ####  #   #
   #   #  #    #    ##  #  #   #    #         #    #    # #   #  #      #   #
   #   #  #    #    # # #  #   #    #         #    #    # #   #  #      #####
   #   #  #    #    #  ##  #   #    #         #    #    # #   #  #      #   #
   ####    ####     #   #   ###     #         #     ####   ###    ####  #   #

===============================================================================
        CANONICAL BENCHMARK CONSTANTS -- THE LATEST STATE-OF-THE-ART VALUES
===============================================================================

DO NOT CHANGE ANY VALUE IN THIS FILE UNLESS YOU INTEND TO CHANGE IT GLOBALLY,
FOR EVERY BENCHMARK, FOREVER. Changing a number here silently reinterprets every
result the repo has ever produced.

DO NOT pass these as CLI flags. DO NOT accept them as function arguments with a
different default. DO NOT copy the literal into a new script. Import them.

-------------------------------------------------------------------------------
WHY THIS FILE EXISTS -- the actual incidents, so nobody repeats them
-------------------------------------------------------------------------------
Every one of these was a real, measured, wrong conclusion caused by a constant
that lived in the wrong place:

  * max_new_tokens=64   manufactured a +20.6% "speculative decoding win" that was
                        really 22% SLOWER at realistic output length.
  * max_new_tokens=448  produced a p=0.0018 Python win. At 2048 the same base
                        model went 0.292 -> 0.917. The entire effect was the cap.
  * max_new_tokens=192  ran the whole 3-way alpha-scaling matrix. The default was
                        192 while the benchmark it duplicated defaulted to 2048.
  * max_new_tokens=768  had the base model "run out of budget and fail to
                        compile", which was then read as a robustness win.
  * adapter _v2         was still wired into two live benchmarks eight hours after
                        v4 landed, so the completion-only-loss fix, contamination
                        removal, and retention mixing were all silently bypassed.
  * unrecorded config   results JSON carried no cap, so a 192-token run was
                        indistinguishable from a 2048-token run after the fact.

The pattern is always the same: a number small enough to look harmless, sitting
somewhere a reviewer does not look, quietly deciding the result.

-------------------------------------------------------------------------------
HOW TO USE
-------------------------------------------------------------------------------
    from gnn_experiment.canon import CANON, adapter_path

    MAX_NEW_TOKENS = CANON.MAX_NEW_TOKENS       # 2048. Not 64. Not 192. Not 768.
    ast = adapter_path("astral")                # -> results/adapters/m2_astral_r8a128_v4

    results["config"] = CANON.stamp()           # ALWAYS stamp the artifact

Run `uv run python scripts/audit/check_canon.py` to fail the build on any violation.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class _Canon:
    # -------------------------------------------------------------------------
    # ADAPTER VERSION -- LATEST IS v4.
    # v4 = clean corpora (reserved eval constructs removed AND verified),
    #      completion-only loss, retention mixing.
    # v2/v3 are LEGACY. They predate the completion-only-loss fix (48.4% of every
    # v2 batch was prompt tokens) and the contamination removal. Never benchmark
    # against them except as an explicit, labelled ablation.
    # -------------------------------------------------------------------------
    ADAPTER_VERSION: str = "v4"

    # -------------------------------------------------------------------------
    # DECODE BUDGET -- 2048. TWO THOUSAND FORTY EIGHT.
    # Not 64. Not 192. Not 448. Not 768. A short cap does not "save time", it
    # fabricates results: it truncates the verbose model mid-answer and hands the
    # win to whichever model happens to be terser. Terseness is a real effect we
    # measure SEPARATELY, by counting tokens -- it must never be smuggled in as
    # a correctness signal.
    # -------------------------------------------------------------------------
    MAX_NEW_TOKENS: int = 2048

    # -------------------------------------------------------------------------
    # Everything else that must not drift between benchmarks.
    # -------------------------------------------------------------------------
    BASE_MODEL: str = "Qwen/Qwen3.5-4B"
    VRAM_CAP_GB: float = 22.0
    GREEDY: bool = True          # do_sample=False. Determinism is how we detect
                                 # that an "unchanged" condition really is one.
    LORA_RANK: int = 8
    LORA_ALPHA: int = 128

    def stamp(self) -> dict:
        """Config block to embed in EVERY results artifact.

        A results file that does not say what produced it cannot be compared to
        any other results file. This is not optional bookkeeping -- it is the
        thing that would have caught the 192-token matrix immediately.
        """
        return asdict(self)


CANON = _Canon()

# Domains that have a canonical expert. Keep in sync with results/adapters/.
DOMAINS = ("astral", "postgresql", "duckdb", "financial")


def adapter_path(domain: str, version: str | None = None) -> Path:
    """Canonical adapter path for a domain. Defaults to CANON.ADAPTER_VERSION.

    Pass `version` ONLY for a deliberate, labelled ablation against a legacy
    adapter -- never to work around a missing file.
    """
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    v = version or CANON.ADAPTER_VERSION
    p = REPO_ROOT / "results" / "adapters" / f"m2_{domain}_r8a128_{v}"
    if not p.exists():
        raise FileNotFoundError(
            f"canonical adapter missing: {p}\n"
            f"Do NOT silently fall back to an older version -- train it or fix the path."
        )
    return p
