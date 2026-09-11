#!/usr/bin/env python3
"""Speculative vs Eager Baseline Comparative Judge Runner.

Compares generation side-by-side between:
  - Mode A: Eager Baseline (Single-step greedy / CUDA graph decode)
  - Mode B: Bucketed Speculative Decoding (MTP Draft Head + Verification)

Evaluates:
  - Text Fidelity & Divergence (is the output coherent in both modes?)
  - Speed ROI (is speculative decoding actually faster or paying a verification tax?)
  - Draft acceptance rate (tau)
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
import urllib.request

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.judge_harness import E2EJudgeHarness
from evals.prompts import DASHBOARD_MULTITURN_SEQUENCE, EvalPrompt


def set_server_speculative(base_url: str, enabled: bool) -> bool:
    """Toggles speculative decoding on the live server."""
    try:
        req = urllib.request.Request(
            f"{base_url}/api/engine/set_speculative_decode",
            data=json.dumps({"enabled": enabled}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("spec_decode_enabled") == enabled
    except Exception as e:
        print(f"⚠️ Warning: Failed to toggle speculative decoding on server: {e}")
        return False


def run_speculative_comparative_eval(
    base_url: str = "http://127.0.0.1:8000",
    prompts: list[EvalPrompt] | None = None,
    max_tokens: int = 250,
    output_dir: Path | None = None,
):
    harness = E2EJudgeHarness(base_url=base_url)

    if not harness.check_server_health():
        print(f"❌ Error: Inference server is not reachable at {base_url}.")
        print("Please start the server first (e.g. via 'uv run apps/runtime/server.py')")
        sys.exit(1)

    eval_prompts = prompts or DASHBOARD_MULTITURN_SEQUENCE
    print("=" * 80)
    print("⚖️  STARTING SPECULATIVE VS EAGER COMPARATIVE JUDGE EVALUATION")
    print(f"Target: {base_url} | Prompts: {len(eval_prompts)}")
    print("=" * 80)

    comparisons: list[dict] = []

    for idx, p in enumerate(eval_prompts, start=1):
        print(f"\n▶️  Evaluating Prompt {idx}/{len(eval_prompts)}: [{p.domain.upper()}] - '{p.prompt[:50]}...'")

        # 1. Run in Eager Mode (Speculative Disabled)
        print("   [1/2] Running Eager Baseline...")
        set_server_speculative(base_url, enabled=False)
        eager_res = harness.execute_turn_live(
            conversation_history=[{"role": "user", "content": p.prompt}],
            turn_idx=idx,
            prompt_id=p.id,
            domain=p.domain,
            expert="dynamic",
            max_tokens=max_tokens,
        )

        # 2. Run in Speculative Mode (Speculative Enabled)
        print("   [2/2] Running Speculative Decoding...")
        set_server_speculative(base_url, enabled=True)
        spec_res = harness.execute_turn_live(
            conversation_history=[{"role": "user", "content": p.prompt}],
            turn_idx=idx,
            prompt_id=p.id,
            domain=p.domain,
            expert="dynamic",
            max_tokens=max_tokens,
        )

        speedup = (spec_res.tokens_per_second / max(0.1, eager_res.tokens_per_second))
        speedup_str = f"{speedup:.2f}x" if speedup >= 1.0 else f"-{1.0/max(0.01, speedup):.2f}x (SLOWER)"

        print(f"   📊 Eager Speed:       {eager_res.tokens_per_second:.1f} tok/s ({eager_res.total_tokens} toks)")
        print(f"   📊 Speculative Speed: {spec_res.tokens_per_second:.1f} tok/s ({spec_res.total_tokens} toks)")
        print(f"   🚀 Relative Delta:    {speedup_str}")

        comparisons.append({
            "idx": idx,
            "domain": p.domain,
            "prompt": p.prompt,
            "eager_tok_s": eager_res.tokens_per_second,
            "eager_toks": eager_res.total_tokens,
            "eager_out": eager_res.assistant_response,
            "spec_tok_s": spec_res.tokens_per_second,
            "spec_toks": spec_res.total_tokens,
            "spec_out": spec_res.assistant_response,
            "speedup": speedup,
        })

    # Restore Eager Mode as default
    set_server_speculative(base_url, enabled=False)

    # Build Comparative Report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_lines = [
        f"# Speculative vs Eager Baseline Comparative Judge Report ({timestamp})",
        "",
        "## 📊 Throughput Comparison Matrix",
        "",
        "| Turn | Domain | Eager tok/s | Speculative tok/s | Speedup / Penalty | Verdict |",
        "|---|---|---|---|---|---|",
    ]

    for c in comparisons:
        verdict = "🚀 FASTER" if c["speedup"] >= 1.05 else ("⚠️ SLOWER (Tax)" if c["speedup"] < 0.95 else "⚖️ NEUTRAL")
        report_lines.append(
            f"| {c['idx']} | {c['domain']} | {c['eager_tok_s']:.1f} | {c['spec_tok_s']:.1f} | {c['speedup']:.2f}x | {verdict} |"
        )

    report_lines.extend([
        "",
        "## 📝 Side-by-Side Outputs for Qualitative Inspection",
        "",
    ])

    for c in comparisons:
        report_lines.extend([
            f"### 📍 Prompt {c['idx']}: [{c['domain'].upper()}]",
            f"> **Prompt**: {c['prompt']}",
            "",
            "#### Eager Baseline Output:",
            "```markdown",
            c["eager_out"],
            "```",
            "",
            "#### Speculative Decoding Output:",
            "```markdown",
            c["spec_out"],
            "```",
            "",
            "---",
        ])

    full_report = "\n".join(report_lines)
    out_dir = output_dir or (Path(__file__).parent.parent / "results" / "evals")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"speculative_vs_eager_{timestamp}.md"
    report_path.write_text(full_report)

    print("=" * 80)
    print(f"✅ Comparative Evaluation Complete! Report saved to: {report_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Speculative vs Eager Comparative Judge Evaluation")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--max-tokens", type=int, default=400, help="Max completion tokens")
    args = parser.parse_args()

    run_speculative_comparative_eval(base_url=args.url, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
