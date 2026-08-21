"""Adapter registry: what every adapter IS, and whether it changed under you.

WHY THIS EXISTS
---------------
Adapter provenance has silently invalidated results in this repo more than once:

  * The in-domain speculation matrix described its three experts as "Stock LoRA
    r=8 alpha=128". All three were id_kron (rank_total 64, scaling 2.0). The
    headline finding was retracted.
  * The stacking benchmark described "3 bfloat16-trained experts". Two of the
    three came from export_adapter.py, which sets load_in_4bit=True.
  * `ctl_lora_fin_a128` has been overwritten three times in one day (4-bit ->
    bf16 -> bf16+Liger). The README's stacking table describes the second
    version; the file on disk is the third.

None of that is detectable by reading a benchmark script, because the adapter
path stays the same while its contents change. This tool makes it detectable.

WHAT IT DOES
------------
`--write`  records every adapter's identity into ADAPTER_MANIFEST.json (repo root,
           tracked in git; results/ is gitignored):
           content hash, architecture, rank_total, alpha, effective scaling,
           regime (bf16 / 4-bit NF4), trainer, and mtime.

(default)  re-scans and DIFFS against the manifest, reporting:
             CHANGED  content hash differs -> any result citing it is now stale
             NEW      never recorded
             MISSING  recorded but gone

REGIME INFERENCE
----------------
Only adapters written by train/train_expert.py carry regime.json. For the
rest the regime is inferred and labelled as such:

    checkpoints/ subdir present -> bf16 trainer (train_financial_adapter.py /
                                   train_stock_lora_bf16.py write SFTConfig
                                   output_dir into the adapter dir)
    no checkpoints/             -> export_adapter.py / finetune_novel_adapter.py,
                                   both of which set load_in_4bit=True and point
                                   output_dir at /tmp

Inference is a fallback, not a fact. Anything marked `inferred` should be
retrained through the canonical trainer rather than trusted.

    uv run python scripts/audit/audit_adapters.py --write   # record current state
    uv run python scripts/audit/audit_adapters.py           # check for drift
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
ADAPTER_ROOT = REPO_ROOT / "results" / "adapters"
# Tracked in git, unlike results/ which is gitignored -- the manifest is only
# useful if it survives across commits and machines.
MANIFEST = REPO_ROOT / "ADAPTER_MANIFEST.json"

# Adapters each benchmark currently depends on. Keeps "which one is current for
# this role" answerable without grepping every script.
ROLES = {
    "stacking/folding: astral": "m2_astral_r8a128",
    "stacking/folding: postgresql": "m2_postgresql_r8a128",
    "stacking/folding: financial": "m2_financial_r8a128",
    "speculation matrix: astral": "ctl_lora_r8_a128",
    "speculation matrix: postgresql": "ctl_lora_pg_a128",
    "speculation matrix: financial": "ctl_lora_fin_a128",
}


def weight_hash(d: Path) -> str:
    """Hash only weight payloads, so tokenizer/README churn is not flagged."""
    h = hashlib.sha256()
    for name in sorted(["adapter_model.safetensors", "novel_adapter.pt"]):
        f = d / name
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def describe(d: Path) -> dict:
    info = {"name": d.name}
    cfg = {}
    for n in ("adapter_config.json", "novel_adapter_config.json"):
        if (d / n).exists():
            cfg = json.loads((d / n).read_text())
            break

    if "lora_alpha" in cfg and "rank_in" not in cfg:
        arch, rt, alpha = "lora", cfg.get("r"), cfg.get("lora_alpha")
    else:
        arch = str(cfg.get("variant") or cfg.get("kind") or cfg.get("adapter_type") or "unknown")
        ri, ro = cfg.get("rank_in"), cfg.get("rank_out")
        rt = (ri * ro) if (ri and ro) else (cfg.get("rank") or cfg.get("r"))
        alpha = cfg.get("alpha") or cfg.get("lora_alpha")

    info["architecture"] = arch
    info["rank_total"] = rt
    info["alpha"] = alpha
    info["scaling"] = (alpha / rt) if (alpha and rt) else None

    reg = d / "regime.json"
    if reg.exists():
        r = json.loads(reg.read_text())
        info["regime"] = r.get("precision", "bfloat16")
        info["quantization"] = r.get("quantization")
        info["liger"] = r.get("liger_fused_kernels")
        info["trained_by"] = r.get("trained_by")
        info["methodology"] = r.get("methodology")
        info["regime_source"] = "recorded"
    else:
        bf16 = (d / "checkpoints").exists()
        info["regime"] = "bfloat16" if bf16 else "4bit-nf4"
        info["quantization"] = None if bf16 else "nf4"
        info["liger"] = None
        info["trained_by"] = None
        info["methodology"] = None
        info["regime_source"] = "inferred"

    info["weight_hash"] = weight_hash(d)
    stamp = None
    for n in ("adapter_model.safetensors", "novel_adapter.pt"):
        if (d / n).exists():
            stamp = datetime.fromtimestamp((d / n).stat().st_mtime).isoformat(timespec="seconds")
            break
    info["modified"] = stamp
    return info


def scan() -> dict:
    return {
        d.name: describe(d)
        for d in sorted(ADAPTER_ROOT.glob("*"))
        if d.is_dir() and ((d / "adapter_config.json").exists() or (d / "novel_adapter_config.json").exists())
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="record current state as the manifest")
    ap.add_argument("--quiet", action="store_true", help="only report problems")
    args = ap.parse_args()

    cur = scan()

    if args.write:
        MANIFEST.write_text(json.dumps({"recorded": datetime.now().isoformat(timespec="seconds"),
                                        "adapters": cur}, indent=2))
        print(f"Recorded {len(cur)} adapters -> {MANIFEST}")

    inferred = [a for a in cur.values() if a["regime_source"] == "inferred"]
    four_bit = [a for a in cur.values() if a["regime"] == "4bit-nf4"]

    if not args.quiet:
        print(f"\n{len(cur)} adapters: "
              f"{len(cur) - len(four_bit)} bf16, {len(four_bit)} 4-bit NF4, "
              f"{len(inferred)} with inferred (unrecorded) regime\n")
        print(f"  {'adapter':<34} {'arch':<10} {'rank':>5} {'scal':>7} {'regime':<10} {'src':<9} {'hash'}")
        for a in cur.values():
            sc = f"{a['scaling']:.2f}" if a["scaling"] else "?"
            print(f"  {a['name']:<34} {str(a['architecture'])[:10]:<10} {str(a['rank_total']):>5} "
                  f"{sc:>7} {a['regime']:<10} {a['regime_source']:<9} {a['weight_hash']}")

    print("\nCURRENT ADAPTER PER BENCHMARK ROLE")
    for role, name in ROLES.items():
        a = cur.get(name)
        if a is None:
            print(f"  {role:<32} {name:<28} MISSING")
        else:
            warn = "  <-- 4-BIT, mixed regime" if a["regime"] == "4bit-nf4" else ""
            print(f"  {role:<32} {name:<28} {a['regime']:<10} {a['weight_hash']}{warn}")

    if not MANIFEST.exists():
        print("\nNo manifest yet. Run with --write to record the current state.")
        return

    old = json.loads(MANIFEST.read_text())["adapters"]
    changed = [n for n in cur if n in old and cur[n]["weight_hash"] != old[n]["weight_hash"]]
    new = [n for n in cur if n not in old]
    gone = [n for n in old if n not in cur]

    print("\nDRIFT SINCE MANIFEST")
    if not (changed or new or gone):
        print("  none -- every adapter is byte-identical to its recorded state")
    for n in changed:
        print(f"  CHANGED  {n}: {old[n]['weight_hash']} -> {cur[n]['weight_hash']} "
              f"({old[n].get('modified')} -> {cur[n].get('modified')})")
        print("           any published result citing this adapter is now stale")
    for n in new:
        print(f"  NEW      {n} ({cur[n]['regime']})")
    for n in gone:
        print(f"  MISSING  {n} (was {old[n]['weight_hash']})")


if __name__ == "__main__":
    main()
