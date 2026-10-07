"""Assemble corpus v7 from the reasoning-augmented v6 records.

corpus v7 = corpus v6 + a model-written `thinking` field per record
(`add_thinking.py`). This step only promotes `training_data_v6_thinking.jsonl`
to `training_data_v7.jsonl` and writes `corpus_v7_manifest.json` beside it, so
the adapter changelog can state exactly what each domain's corpus contains:
source count, kept, rejected and why, and trace-length statistics.

Records whose trace failed validation are NOT given an empty think block --
`<think>\\n\\n</think>` is the "answer without reasoning" habit v9 exists to
remove -- they are left out, and the manifest counts them.

    python3 apps/factory/corpus/assemble_corpus_v7.py agentic_coding python_modern ...
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def assemble(domain: str) -> dict:
    d = DATA / domain
    src, kept_f, rej_f = d / "training_data_v6.jsonl", d / "training_data_v6_thinking.jsonl", \
        d / "training_data_v6_thinking_rejected.jsonl"
    if not kept_f.exists():
        raise SystemExit(f"{domain}: {kept_f.name} missing -- run add_thinking.py first")
    n_src = sum(1 for line in src.read_text().splitlines() if line.strip())
    kept = [json.loads(line) for line in kept_f.read_text().splitlines() if line.strip()]
    rejected = [json.loads(line) for line in rej_f.read_text().splitlines() if line.strip()] if rej_f.exists() else []
    if len(kept) + len(rejected) != n_src:
        raise SystemExit(f"{domain}: kept {len(kept)} + rejected {len(rejected)} != source {n_src} "
                         "-- generation incomplete, refusing to promote a partial corpus")
    out = d / "training_data_v7.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in kept) + "\n", encoding="utf-8")
    words = [r["thinking_meta"]["words"] for r in kept]
    return {
        "domain": domain,
        "source": str(src.relative_to(DATA.parents[2])),
        "output": str(out.relative_to(DATA.parents[2])),
        "source_records": n_src,
        "kept": len(kept),
        "rejected": len(rejected),
        "rejected_reasons": dict(Counter(r["reason"].split(" (")[0] for r in rejected)),
        "thinking_words": {"min": min(words), "median": statistics.median(words), "max": max(words)},
        "method": kept[0]["thinking_meta"]["method"] if kept else None,
    }


def main(domains: list[str]) -> int:
    manifest_path = DATA / "corpus_v7_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "corpus": "v7",
        "definition": "corpus v6 + one model-written <think> trace per record (add_thinking.py); "
                      "teacher = Qwen3.5-9B via runtime-next, enable_thinking=false, greedy, "
                      "shown question + reference answer, validated (no code, no answer leak, 40-320 words)",
        "domains": {},
    }
    for dom in domains:
        entry = assemble(dom)
        manifest["domains"][dom] = entry
        print(f"{dom:20s} source {entry['source_records']:5d}  kept {entry['kept']:5d}  "
              f"rejected {entry['rejected']:3d} {entry['rejected_reasons']}  words {entry['thinking_words']}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
