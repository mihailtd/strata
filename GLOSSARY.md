# GLOSSARY — what our names mean, and which ones already had names

Companion to `NOVELTY.md` (what is actually novel), `CURRENT.md` (what is live)
and `SYSTEM.md` (the box).

## The rule

**If an established name exists, use it. Reserve invented names for things we
actually invented.**

An invented name for an existing technique costs twice: it hides prior art from
us (so we can't read what others already learned), and it reads as a novelty
claim to anyone else. `NOVELTY.md` already warns that overclaiming and
underclaiming are both failures — naming is where that starts.

When adopting an established name, say what our variant changes. "Adapter
merging, in-place with a restore path" is precise and claims exactly the right
amount. "Weight Folding" claims more and explains less.

**Confidence labels used below**, because "this is really X" is itself a claim:

| label | means |
| :--- | :--- |
| **CONFIRMED** | verified against source in this repo's own dependencies, or already named in our docs |
| **SAME FAMILY** | same mechanism class, not identical — cite by reference, don't rename outright |
| **UNVERIFIED** | resembles a published method; nobody has checked the math against the paper |

---

## 1. Established name exists — use it

| our name | established name | status | action |
| :--- | :--- | :--- | :--- |
| "SVD-Guided Subspace Initialization", "smarter adapter init" | **PiSSA** (Meng et al. 2024) | **CONFIRMED** — `IDEAS.md:57` already names it, and peft 0.20.0 ships it as `init_lora_weights="pissa"` | Call it PiSSA. It is a config flag, not a project. See [`PISSA_ASSESSMENT.md`](benchmarks/factory/geometry/preflight_svd_probe/PISSA_ASSESSMENT.md) |
| "Draw-Call Batching" (`cuda_graph.py` title) | **kernel-launch overhead**, addressed by **CUDA Graphs** | **CONFIRMED** | Rename. There are no draw calls in LLM decode — the term is borrowed from rendering and actively misleads. The real mechanism is CPU launch cost, which is what CUDA Graph capture removes |
| "In-Place Weight Folding", "folding" | **adapter merging** (`peft.merge_and_unload`) | **CONFIRMED** | Keep "folding" as shorthand — it is used in 16 files — but define it once as *"adapter merging, performed in-place against a pristine buffer so it is reversible"*. The merging is standard; the in-place-and-restorable part is ours |
| "Pristine State Buffer" | **master weights** (mixed-precision training discipline) | **CONFIRMED** — `NOVELTY.md` already credits this lineage | Keep the name, keep the credit line. The application to runtime expert swapping is the contribution, not the idea of holding an unmutated reference copy |
| "Times-Above-Chance Metric" | **lift** (statistics / data mining); *fold enrichment* in bioinformatics | **CONFIRMED** | Keep ours — "times above chance" is self-documenting and the chance floor is the whole point — but note the synonym so readers connect it to standard practice |
| "τ", "tau" | **acceptance rate** / mean accepted draft tokens per step (speculative-decoding literature) | **CONFIRMED** | Fine as-is; define it once per document on first use |
| "MTP" | **Multi-Token Prediction** | **CONFIRMED** — the checkpoint's own tensor names | Correct already |

## 2. Same family as something published — cite it, don't claim it

| our name | closest established method | status | action |
| :--- | :--- | :--- | :--- |
| `master_basis` / `MasterBasisBank` | **VeRA** (Kopiczko et al. 2024) — peft ships `peft/tuners/vera` | **SAME FAMILY.** Verified from peft source: VeRA freezes shared `vera_A`/`vera_B` and trains only per-layer vectors `vera_lambda_b` (out_features) and `vera_lambda_d` (r). Ours freezes a shared basis bank and trains **scalar** coefficients — a more aggressive variant of the same idea (1,024 params vs VeRA's ~2,816/layer) | Do not present as novel. Document as *"VeRA-family: shared frozen basis, tiny per-task trainable payload; ours trains scalars rather than vectors."* Note VeRA is available in peft if a controlled comparison is ever wanted |
| `id_kron` | **Kronecker adapter family** — KronA, LoKr (peft ships `peft/tuners/lokr`) | **UNVERIFIED.** The structure is `I_{r1} ⊗ w_a`: the input is split into `r1` blocks and one `(in_sub, r2)` down-projection is shared across blocks. That is a Kronecker product with an identity factor, hence the name — but whether it is equivalent to a published variant has not been checked | Keep `id_kron` (the name is honest and descriptive). **Fix the framing:** `TODO.md` tabulates `id_kron` against "PEFT LoKr" as if unrelated, when both are Kronecker-family. Say so, and mark the equivalence question open |
| `mbproj_k32` | post-hoc SVD projection / low-rank distillation of a trained adapter | **UNVERIFIED** | Distinct from PiSSA — PiSSA initialises from `W0`'s SVD *before* training; this projects a *trained* `dW` onto a basis *after*. Keep them clearly separated; they get confused because both say "SVD" |
| "Factor Standby Residency" | multi-adapter VRAM residency / adapter pooling (cf. S-LoRA) | **UNVERIFIED** | Describe plainly as "how many adapters fit resident in VRAM". The branded phrase adds nothing |
| `VelocityGate`, "velocity LoRA" | dynamic layer skipping / early exit / Mixture-of-Depths | **UNVERIFIED** | Superseded and retired anyway; leave the name in the superseded record, do not revive it |

## 3. Genuinely ours — invented names are correct here

No established name is known for these, and each is a specific combination rather
than a general technique. Keep the names; keep the `NOVELTY.md` tiering honest.

| name | what it is |
| :--- | :--- |
| **Zero-Recapture In-Place Swapping** | Mutating folded weights at pointer-stable addresses so a captured CUDA Graph replays across expert swaps with `capture_count == 1`. The pieces are standard; this specific invariant is ours |
| **Precision absorption calibration** | The measured constant (0.22 × bf16 eps) and its 2.3% stability over a 16× range. `NOVELTY.md` already corrects the earlier "Universal Precision Absorption Law" — the 1/x *form* is elementary floating point; the calibrated constant is the contribution |
| **VRAM State Router** (SLA-bounded cluster scheduling) | Scheduling requests by target expert under a deadline bound. Note this replaced the retired "APSP VRAM State Router" — see [`benchmarks/superseded/apsp_floyd_warshall/`](benchmarks/superseded/apsp_floyd_warshall/) |
| **m1 / m2** methodology ids | Internal training-regime versioning. Purely ours, and working well |

---

## 4. Internal collisions — one thing, several names

These are our own inconsistencies, not prior-art problems. Pick the left column.

| use this | instead of | note |
| :--- | :--- | :--- |
| **expert** | "micro-expert", "domain expert", "adapter" (when used interchangeably) | ⚠️ **"expert" collides with Mixture-of-Experts**, where an expert is an FFN branch chosen by a gate. Ours is a LoRA adapter merged into the backbone. This matters because we also discuss MTP and MoE-adjacent work — define it on first use in any document that mentions both |
| **pristine buffer** / `W0` | "Pristine State Buffer", "master weights", "base weights" | Use `W0` in math, "pristine buffer" in prose |
| **VRAM State Router** | "APSP Router", "APSP VRAM State Router", "the Brain" | APSP is retired; do not reintroduce |
| **speculation router** | "Dynamic Speculation Router", "Production Dynamic Speculation Router" | The config key is `speculation_router_config`; match it |
| **acceptance rate (τ)** | "Mean τ", "accept %", "draft acceptance" | `accept_pct` and `τ` are *different quantities* in our own JSON (`accepted/drafted` vs `accepted/steps`) — never use them as synonyms |
| **weight folding** | "Weight Folding", "In-Place Weight Folding", "IMB folding" | Lowercase in prose; it is a mechanism, not a product |

⚠️ The `accept_pct` vs `τ` collision is the one most likely to cause a real
error: `τ = accepted/steps` (can exceed 1) and `accept_pct = accepted/drafted`
(cannot). The production gate keys on **τ**, not on accept%.

---

## 5. Before inventing a name

1. Search peft's implemented methods: `ls .venv/lib/python*/site-packages/peft/tuners/`
   — it currently ships ~40, including `vera`, `lokr`, `loha`, `boft`, `hra`,
   `vblora`, `adalora`, `ia3`, `oft`, `poly`. If it's there, it has a name.
2. Check `init_lora_weights`' accepted literals for initialisation schemes —
   `pissa`, `olora`, `eva`, `corda`, `loftq`, `orthogonal`, `mica` are all built in.
3. Check `IDEAS.md` — it already catalogues DoRA, PiSSA, AdaLoRA, VeRA, QLoRA,
   IA³, Houlsby, prefix tuning.
4. Only if all three come back empty, name it — and record here why it needed a
   new name.
