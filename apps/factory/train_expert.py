"""CURRENT training methodology (m2). Use this for all new expert training.

METHODOLOGY VERSIONING
----------------------
The filename carries `CURRENT` and the methodology id. Exactly one trainer at a
time is CURRENT. When the methodology changes:

    1. rename this file to `train_expert_m2.py` (drop CURRENT) and move it to
       scripts/superseded/
    2. create `train_expert_CURRENT_m3.py` with METHODOLOGY = "m3"
    3. adapters trained under it are named `m3_<domain>_r<rank>a<alpha>`

so the methodology is readable from both the script name and every adapter it
produces, and results trained under different methodologies can never be
silently compared.

    m1  4-bit NF4 base, no fused kernels   (export_adapter.py,
                                            finetune_novel_adapter.py -- both
                                            now legacy; adapters learn a
                                            correction to quantized weights and
                                            were then folded into bf16 ones)
    m2  bf16 base + Liger fused kernels    THIS FILE -- fused_linear_cross_entropy
                                            (no logit materialisation at 248320
                                            vocab), rms_norm, swiglu; rope off

Hyperparameters fixed across every domain so experts stay comparable: r=8,
alpha=128 (scaling 16), 7 projections, 150 steps, batch 2, grad-accum 2,
lr 2e-4, cosine, max_length 512.

WHY THIS EXISTS
---------------
`export_adapter.py` trains against a **4-bit NF4** base (`load_in_4bit=True`),
but every folding/speculation/stacking benchmark loads a **bf16** base. Adapters
produced there learn a correction to quantized weights and are then folded into
unquantized ones. That seam is measurable: the financial expert moved from
-5.00pp (4-bit-trained) to +4.17pp (bf16-trained) on its own domain, with a data
change that altered zero rubric terms.

`train_financial_adapter.py` trains in bf16 but is hardcoded to the financial
dataset. This is that script generalised, so astral and postgres can be brought
into the same regime instead of being silently mixed into a bf16 stack.

Hyperparameters are held identical to the controlled experts so results stay
comparable: r=8, alpha=128 (scaling 16), 7 projections, 150 steps, batch 2,
grad-accum 2, lr 2e-4, cosine schedule, max_length 512.

    uv run --env-file .env scripts/train_stock_lora_bf16.py --domain astral
    uv run --env-file .env scripts/train_stock_lora_bf16.py --domain postgresql
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Force GPU 0 exclusive device isolation before PyTorch/ROCm runtime initializes
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")

import _bootstrap  # noqa: E402,F401  -- adds apps/ to sys.path for the shared
import torch
from peft import LoraConfig, get_peft_model
from runtime import training_db  # noqa: E402

# apps/runtime modules below (novel_peft, training_db). These stay in
# apps/runtime (heavily used by the live serving engine too) and are
# consumed as source here, not as a package dependency -- see
# apps/factory/pyproject.toml's comment on why (torch version conflict).
from runtime.novel_peft import set_hard_vram_cap  # noqa: E402
from runtime_common.canon import CANON, REPO_ROOT  # noqa: E402
from runtime_common.gpu_preflight import ensure_gpu_exclusive  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

METHODOLOGY = "m2"  # bf16 + Liger fused kernels; see docstring

# domain -> (training data, output adapter dir). Adapter names carry the
DOMAINS = {
    "astral": ("apps/factory/data/astral/training_data_v4.jsonl", "results/adapters/m2_astral_r8a128_v4"),
    "postgresql": ("apps/factory/data/postgresql/training_data_v4.jsonl", "results/adapters/m2_postgresql_r8a128_v4"),
    "financial_planning": (
        "apps/factory/data/financial_planning/training_data_v3.jsonl",
        "results/adapters/m2_financial_r8a128_v4",
    ),
    "duckdb": ("apps/factory/data/duckdb/training_data_v4.jsonl", "results/adapters/m2_duckdb_r8a128_v4"),
    # Merged corpora. NOTE the step counts below -- a merged corpus trained for
    # the same 150 steps as a solo one gives each domain a FRACTION of the
    # exposure (merged_all: 14% of an epoch vs solo's 43%), which would make
    # merging look bad for reasons that have nothing to do with merging.
    "merged_sql": ("apps/factory/data/merged_sql/training_data_v4.jsonl", "results/adapters/m2_merged_sql_r8a128_v4"),
    "merged_all": ("apps/factory/data/merged_all/training_data_v4.jsonl", "results/adapters/m2_merged_all_r8a128_v4"),
}

# v6 = corpus v5 + geometric stopping. NOT v5: adapter v5/v5b/v5c are the failed
# L_inert experiments and corpus v5 is the merged disposition corpus -- two
# unrelated meanings. Adapter version != corpus version from here on; CHANGELOG.md
# records which corpus each adapter used.
# data dir != adapter stem for financial: the corpus lives in
# data/financial_planning/ but every adapter since v1 is m2_financial_*. Keep the
# stem canonical so adapter_path("financial") resolves.
_V6_STEM = {"financial_planning": "financial"}
_GEN_DOMAINS = ("astral", "postgresql", "duckdb", "financial_planning", "python_modern", "python_web")


def _gen_table(corpus_ver: str, adapter_ver: str) -> dict[str, tuple[str, str]]:
    return {
        d: (
            f"apps/factory/data/{d}/training_data_{corpus_ver}.jsonl",
            f"results/adapters/m2_{_V6_STEM.get(d, d)}_r8a128_{adapter_ver}",
        )
        for d in _GEN_DOMAINS
    }


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ CORPUS VERSION != ADAPTER VERSION. They are OFF BY ONE and always have been.║
# ║                                                                             ║
# ║   adapter v6  <- corpus v5     adapter v7  <- corpus v6                     ║
# ║                                                                             ║
# ║ Adapter v5 is the failed L_inert line whose weights were deleted; corpus v5 ║
# ║ is the merged disposition corpus. Unrelated things, same number (CHANGELOG). ║
# ║                                                                             ║
# ║ This table used to be written inline as                                     ║
# ║     training_data_v6.jsonl -> results/adapters/*_v6                         ║
# ║ which was correct only while training_data_v6.jsonl did not exist. The       ║
# ║ moment the round-2 corpora were written, `--v6` started reading the NEW      ║
# ║ corpus and writing over the OLD adapter: m2_python_modern_r8a128_v6 was      ║
# ║ silently replaced, destroying the baseline its 22.40 activation-scale and    ║
# ║ its dataclass-attribution result were measured on. Derive both tables from   ║
# ║ one function so the pairing cannot drift again.                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝
DOMAINS_V6 = _gen_table("v5", "v6")
DOMAINS_V7 = _gen_table("v6", "v7")

FAIR_STEPS = {"merged_sql": 326, "merged_all": 481}


ANSWER_MARKER = "\n\n### Answer:\n"


def load_dataset_records(path: Path, completion_only: bool = True):
    """Load records as PROMPT/COMPLETION pairs so loss can skip the prompt.

    PROMPT-TARGET SEPARATION. Training on the full string makes the adapter learn
    to GENERATE the corpus's questions, not just answer them. Measured on the v2
    corpora: answer-only median 124 tokens against full-text median 167, so ~26%
    of the gradient signal was teaching question text -- and far more on the
    original book-derived corpora, whose questions are long. That is a direct
    mechanism for the narrowing the held-out benchmark measured (experts -0.233
    vs base on constructs absent from their corpora).

    Splitting on the answer marker gives TRL an explicit prompt/completion pair;
    `completion_only_loss=True` then masks the prompt tokens to -100.
    """
    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "messages" in d:
                u = d["messages"][0]["content"]
                a = d["messages"][1]["content"]
            elif "text" in d and ANSWER_MARKER in d["text"]:
                head, a = d["text"].split(ANSWER_MARKER, 1)
                u = head[len("### Question:\n") :] if head.startswith("### Question:\n") else head
            else:
                # no recoverable split -- keep it, but it cannot be masked
                records.append({"text": d.get("text", "")})
                continue
            if completion_only:
                records.append({"prompt": f"### Question:\n{u}{ANSWER_MARKER}", "completion": a})
            else:
                records.append({"text": f"### Question:\n{u}{ANSWER_MARKER}{a}"})
    return records


class GoldilocksStoppingCallback(TrainerCallback):
    """Stop when the adapter's geometry reaches the Goldilocks band, not at a
    step count someone guessed.

    THE IDEA
    --------
    docs/THE_FACTORY_FINE_TUNING_AND_GEOMETRY.md Ch.4 defines the operating window
    on ||dW||/||W||, a quantity that STARTS AT ZERO (lora_B is zero-initialised) and
    grows as training proceeds:

        < 0.035   precision floor    -- bf16 mantissa truncation, merge err > 5%
        0.035-0.100  Goldilocks      -- merge err < 2.5%, peak measured quality
        > 0.150   retention ceiling  -- base representations overwritten

    So steps are only the vehicle. Training until the geometry lands in the band
    makes the stopping point independent of corpus size, duplication rate and
    learning rate -- those change how FAST you arrive, not where.

    This removes a real confound. At the fixed 150-step default the corpora were
    getting wildly different amounts of training:

        astral v5      0.37 epochs      python_modern v5   0.88 (3.24 effective)
        postgresql v5  0.44             python_web v5      0.99 (2.96 effective)

    and every cross-domain comparison silently inherited that spread.

    IT VALIDATES AGAINST WHAT WE ALREADY HAVE
    -----------------------------------------
    v4 recorded 0.0758 (postgres) and 0.0754 (astral) at 150 steps -- both
    mid-band. 150 was right for those two corpora by luck, not design.

    ⚠️ HONEST LIMIT 1 -- the band was calibrated along a different axis
    ------------------------------------------------------------------
    The V-curve that places peak quality at ||dW||/||W|| ~= 0.045 was measured by
    SCALING ALPHA on a fixed trained adapter: direction constant, magnitude varied.
    Reaching the same ratio by training longer also changes the DIRECTION. The
    scalar is the same; the path is not. Treat the band as a well-grounded stopping
    heuristic, not as a proven quality optimum along the training-length axis --
    that transfer is worth measuring once rather than assuming.

    ⚠️ HONEST LIMIT 2 -- MEASURED: this is a calibrated step count, not an
    adaptive one
    ---------------------------------------------------------------------
    From the six v6 geometry_trace.json files actually produced:

        domain          stop step   final ||dW||/||W||   ratio @100
        astral            140            0.0982            0.0806
        duckdb            135            0.1005            0.0857
        financial         140            0.1008            0.0838
        postgresql        135            0.1013            0.0855
        python_modern     130            0.0960            0.0843
        python_web        140            0.0989            0.0823

    All six fired on the HARD CEILING; the plateau criterion never triggered once
    (rel_growth at stop was ~0.019 everywhere). And the trajectories are nearly
    identical -- 6% spread at step 100 across six corpora of different size,
    duplication rate and form.

    So the claim above that this "makes the stopping point independent of corpus
    size, duplication rate and learning rate" is only half earned. ||dW||/||W||
    barely responds to corpus properties, so there was less confound to remove
    than advertised: in practice the rule lands every domain at ~135 steps. It is
    still better than a guessed 150 -- it TARGETS a measured band and it would
    self-correct if rank, alpha or LR changed -- but do not describe it as
    per-corpus adaptive. It is not.

    ⚠️ WHY NOT A RIEMANNIAN STOPPING SIGNAL (the obvious next idea)
    --------------------------------------------------------------
    d_R on weight Gramians cannot do this job: DECISIONS.md §59 measured adapter
    subspaces as fully disjoint (k = 2r) in all 128 weight matrices, with the SAME
    domain trained twice landing FARTHER apart than two different domains. The
    subspace is set by lora_A's init, not by the data.

    The version with teeth is d_R on ACTIVATION covariance, Sigma_h = E[h h^T]
    base vs current -- genuine n < p, so real Ledoit-Wolf applies
    (ledoit_wolf_from_samples), and it is invariant to the RMSNorm rescalings that
    distort a Frobenius meter. It would also catch what the plateau criterion
    structurally cannot: an adapter whose NORM has saturated while its DIRECTION
    is still rotating reads as converged here. That costs a forward pass per check
    and has not been run.
    """

    def __init__(
        self,
        model,
        alpha: int,
        rank: int,
        target: float | None,
        hard_ceiling: float = 0.100,
        every: int = 10,
        plateau: float | None = None,
        floor: float = 0.035,
        run_id: str | None = None,
    ):
        self.model, self.alpha, self.rank = model, alpha, rank
        self.target, self.hard_ceiling, self.every = target, hard_ceiling, every
        # floor: never call it "converged" below the precision floor, where bf16
        # truncation dominates and merge error exceeds 5%.
        self.plateau, self.floor = plateau, floor
        self.run_id = run_id
        self.stop_reason: str | None = None
        self.trace: list[dict] = []

    @torch.no_grad()
    def _ratio(self) -> float:
        num = den = 0.0
        for mod in self.model.modules():
            A, B, W = (getattr(mod, "lora_A", None), getattr(mod, "lora_B", None), getattr(mod, "base_layer", None))
            if A is None or B is None or W is None:
                continue
            try:
                a = A["default"].weight.detach().float()
                b = B["default"].weight.detach().float()
                w = W.weight.detach().float()
            except Exception:
                continue
            num += float(((b @ a) * (self.alpha / self.rank)).norm() ** 2)
            den += float(w.norm() ** 2)
        return (num**0.5) / max(1e-30, den**0.5)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every:
            return control
        r = self._ratio()
        prev = self.trace[-1]["dw_over_w"] if self.trace else 0.0
        rel = (r - prev) / r if r > 0 else 1.0
        self.trace.append(
            {
                "step": int(state.global_step),
                "dw_over_w": round(r, 6),
                "rel_growth": round(rel, 5),
                "merge_err_pct": round(0.167 / max(1e-9, r), 3),
            }
        )
        if self.run_id:
            training_db.record_step(
                run_id=self.run_id,
                step=int(state.global_step),
                dw_over_w=round(r, 6),
                rel_growth=round(rel, 5),
                merge_err_pct=round(0.167 / max(1e-9, r), 3),
            )
        if state.global_step % (self.every * 5) == 0:
            print(
                f"  [geometry] step {state.global_step:4d}  |dW|/|W|={r:.4f}  "
                f"rel_growth={rel * 100:.1f}%  merge_err~{0.167 / max(1e-9, r):.2f}%",
                flush=True,
            )

        if r >= self.hard_ceiling:
            print(
                f"  [geometry] STOP step {state.global_step}: |dW|/|W|={r:.4f} "
                f"reached band ceiling {self.hard_ceiling} "
                f"(merge_err~{0.167 / max(1e-9, r):.2f}%)",
                flush=True,
            )
            self.stop_reason = "hard_ceiling"
            control.should_training_stop = True
        elif self.plateau and len(self.trace) >= 4 and rel < self.plateau and r >= self.floor:
            print(
                f"  [geometry] STOP step {state.global_step}: converged -- "
                f"rel_growth {rel * 100:.2f}% < {self.plateau * 100:.1f}% at "
                f"|dW|/|W|={r:.4f} (in band, merge_err~"
                f"{0.167 / max(1e-9, r):.2f}%)",
                flush=True,
            )
            self.stop_reason = "plateau"
            control.should_training_stop = True
        elif self.target and r >= self.target:
            print(
                f"  [geometry] STOP step {state.global_step}: |dW|/|W|={r:.4f} >= fixed target {self.target}",
                flush=True,
            )
            self.stop_reason = "target_reached"
            control.should_training_stop = True
        return control


class TelemetryCallback(TrainerCallback):
    """Logs training loss and accuracy directly to the SQLite training ledger."""

    def __init__(self, run_id: str):
        self.run_id = run_id

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        step = int(state.global_step)
        loss = float(logs.get("loss", 0.0)) if "loss" in logs else None
        grad_norm = float(logs.get("grad_norm", 0.0)) if "grad_norm" in logs else None
        lr = float(logs.get("learning_rate", 0.0)) if "learning_rate" in logs else None
        acc = float(logs.get("mean_token_accuracy", 0.0)) if "mean_token_accuracy" in logs else None
        entropy = float(logs.get("entropy", 0.0)) if "entropy" in logs else None

        training_db.record_step(
            run_id=self.run_id,
            step=step,
            loss=loss,
            grad_norm=grad_norm,
            learning_rate=lr,
            token_accuracy=acc,
            entropy=entropy,
        )
        return control


class InertiaSFTTrainer(SFTTrainer):
    """SFTTrainer with an activation-space inertia penalty on OUT-OF-DOMAIN tokens.

    WHAT CHANGED AND WHY (the v5 post-mortem)
    -----------------------------------------
    The first implementation masked on `labels == -100`. With
    `completion_only_loss=True` on a single-domain corpus, those are the IN-DOMAIN
    PROMPT tokens. No out-of-domain token ever entered the penalty, so the adapter
    could not possibly learn *when* to be quiet -- only *how much* to be quiet.

    The probe measured exactly that outcome. v5 vs v4 activation energy:

        astral 0.844   postgresql 0.837   duckdb 0.842
        financial 0.849   general 0.846          <- spread of 1.4%

    A clean uniform 0.843x scaling on every domain, with ASR unchanged at 0.96.
    That is a lower effective alpha wearing a different name, and the 2048-token
    stacking benchmark showed lowering alpha is dilution: astral retained 90.3% of
    its gain at alpha=128 and 57.7% at alpha/sqrt(3).

    The mechanism was never the problem -- 15.7% movement with 1.4% spread is a
    penalty working exactly as written. It was aimed at the wrong tokens.

    THE FIX -- selectivity needs OPPOSING forces on DIFFERENT tokens
    ---------------------------------------------------------------
    Penalizing energy on in-domain prompt tokens gives the optimizer one
    instruction ("smaller") with nothing pulling the other way on those same
    tokens, so it shrinks everything. Selectivity requires a tug-of-war:

        SFT loss   on IN-DOMAIN completions  -> pushes delta UP   (it must fit)
        L_inert    on OUT-OF-DOMAIN replay   -> pushes delta DOWN (it must not fire)

    Those act on disjoint token sets, so the only way to satisfy both is to become
    input-selective. That is the whole idea, and it is what the previous version
    structurally could not express.

    Replay comes from the OTHER domains' corpora -- training astral replays
    postgresql / duckdb / financial. No new data is needed.

    TWO DETAILS THAT MATTER
    -----------------------
    1. The penalty is on ||delta|| / ||h||, not raw ||delta||^2, and the
       denominator is DETACHED. Optimize what you measure: ASR is defined on the
       ratio. And without detaching, the optimizer can shrink the ratio by
       inflating ||h|| instead of quieting the adapter -- a degenerate solution
       that would look like success in the loss and fail in the probe.
    2. Raw ||delta||^2 summed over layers is dominated by whichever layers carry
       the largest activations, so the gradient chases magnitude rather than
       relevance. That is a second, independent reason the first version came out
       uniform.
    """

    def __init__(self, *args, lambda_inert: float = 0.0, replay_batches: list | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_inert = lambda_inert
        self.replay_batches = replay_batches or []
        self._replay_i = 0
        self._mode = "train"
        self._energy = {"train": [], "replay": []}
        self.inert_log = []
        self._register_inertia_hooks()

    def _register_inertia_hooks(self):
        for _name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B") and "default" in module.lora_A:

                def make_hook(mod):
                    def forward_hook(m, inp, out):
                        if not (self.model.training and self.lambda_inert > 0):
                            return
                        h = inp[0]
                        A = mod.lora_A["default"].weight.to(h.dtype)
                        B = mod.lora_B["default"].weight.to(h.dtype)
                        scale = mod.scaling["default"]
                        delta = torch.matmul(torch.matmul(h, A.t()), B.t()) * scale
                        # detached denominator -- see docstring note 1
                        hn = h.detach().float().norm(dim=-1).clamp_min(1e-6)
                        # ALWAYS (batch, seq), never state-dependent. Gradient
                        # checkpointing re-runs this during backward; if the hook
                        # branches on mutable trainer state it saves a different
                        # number of tensors on the two passes and torch raises
                        # CheckpointError. A forward hook must be a pure function
                        # of its inputs. Padding is masked in compute_loss instead.
                        self._energy[self._mode].append(delta.float().norm(dim=-1) / hn)

                    return forward_hook

                module.register_forward_hook(make_hook(module))

    def _trunk(self):
        """The transformer trunk, without the LM head.

        Replay only needs the LoRA hooks to fire. Computing a ~150k-way vocab
        projection for tokens whose logits are discarded would dominate both the
        step time and the activation memory.
        """
        m = self.model
        for _ in range(4):
            nxt = getattr(m, "model", None) or getattr(m, "base_model", None)
            if nxt is None:
                break
            m = nxt
            if hasattr(m, "layers"):
                return m
        return None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._energy["train"].clear()
        self._energy["replay"].clear()
        self._mode = "train"

        if num_items_in_batch is not None:
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        else:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        if self.lambda_inert > 0 and self.replay_batches:
            batch = self.replay_batches[self._replay_i % len(self.replay_batches)]
            self._replay_i += 1
            ids = batch["input_ids"].to(self.model.device)
            attn = batch["attention_mask"].to(self.model.device)

            self._mode = "replay"
            trunk = self._trunk()
            if trunk is not None:
                trunk(input_ids=ids, attention_mask=attn)
            else:
                self.model(input_ids=ids, attention_mask=attn)
            self._mode = "train"

            if self._energy["replay"]:
                # PAD POSITIONS MUST NOT COUNT. Silencing <pad> is free -- it costs
                # the SFT loss nothing -- so an unmasked penalty optimises padding
                # and nothing else. That is exactly what v5b did: training ASR rose
                # to 1.073 while the probe measured 0.965, unchanged from v4.
                keep = attn.bool()
                per_layer = torch.stack(
                    [(e[keep] ** 2).mean() for e in self._energy["replay"] if e.shape == attn.shape]
                )
                l_inert = per_layer.mean()
                loss = loss + self.lambda_inert * l_inert

                # Log ASR live. Discovering after a 1h run that selectivity never
                # moved is the failure this avoids -- it should be visible by step 20.
                with torch.no_grad():
                    e_out = (
                        torch.stack([e[keep].mean() for e in self._energy["replay"] if e.shape == attn.shape])
                        .mean()
                        .item()
                    )
                    tmask = inputs.get("attention_mask")
                    tr = [
                        e[tmask.bool()].mean()
                        for e in self._energy["train"]
                        if tmask is not None and e.shape == tmask.shape
                    ]
                    e_in = torch.stack(tr).mean().item() if tr else 0.0
                    # lora_B is initialised to ZERO, so delta = B(Ah) is exactly 0
                    # at step 0 and both energies are 0.0. That also means L_inert
                    # has zero gradient at init (d/dB ||BAh||^2 = 0 when B = 0) --
                    # the SFT loss necessarily moves first, and the penalty only
                    # engages once B leaves zero. Expected, not a fault.
                    asr = (e_in / e_out) if e_out > 0 else float("nan")
                    self.inert_log.append(
                        {
                            "step": int(self.state.global_step),
                            "e_in": e_in,
                            "e_out": e_out,
                            "asr": asr,
                            "l_inert": float(l_inert),
                        }
                    )
                    if self.state.global_step % 10 == 0 and e_out > 0:
                        print(
                            f"  [inert] step {self.state.global_step:4d}  "
                            f"e_in={e_in:.4f} e_out={e_out:.4f}  ASR={asr:.3f}x",
                            flush=True,
                        )

        return (loss, outputs) if return_outputs else loss


def build_replay_batches(domain: str, tokenizer, n_batches: int = 64, batch_size: int = 2, max_len: int = 512) -> list:
    """Out-of-domain replay batches drawn from the OTHER domains' corpora.

    This is the token set L_inert is computed on. It must NOT overlap the training
    domain -- that was the entire defect in the first version.
    """
    import random

    texts = []
    for other, (rel, _out) in DOMAINS.items():
        if other == domain:
            continue
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"  [replay] SKIP {other}: {rel} missing")
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        for r in rows:
            if "messages" in r:
                t = f"### Question:\n{r['messages'][0]['content']}{ANSWER_MARKER}{r['messages'][1]['content']}"
            else:
                t = r.get("text", "")
            if t:
                texts.append(t)
        print(f"  [replay] {other}: {len(rows)} records")
    if not texts:
        raise RuntimeError(
            f"no replay text for domain {domain!r} -- L_inert would silently do "
            f"nothing, which is exactly the v5 failure. Refusing to train."
        )
    random.Random(0).shuffle(texts)
    batches = []
    for i in range(n_batches):
        chunk = texts[i * batch_size : (i + 1) * batch_size]
        if len(chunk) < batch_size:
            break
        # padding="longest", NOT "max_length". Corpus records run ~120-170 tokens;
        # padding every one to 512 made ~70% of each replay batch PAD, and since
        # the penalty ran over every position the adapter learned to be silent on
        # <pad>. That is free -- it costs the SFT loss nothing -- so training ASR
        # rose to 1.073 while the probe measured 0.965, unchanged from v4.
        # The attention mask below is the real fix; this just avoids the waste.
        enc = tokenizer(chunk, return_tensors="pt", padding="longest", truncation=True, max_length=max_len)
        batches.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    print(
        f"  [replay] built {len(batches)} out-of-domain batches from {len(texts)} records (domain {domain!r} EXCLUDED)"
    )
    return batches


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # union of both tables: --v6 swaps DOMAINS for DOMAINS_V6, but argparse
    # validates BEFORE that happens, so restricting to the v4 keys rejected
    # python_modern/python_web with exit 2 before the model ever loaded.
    ap.add_argument("--domain", required=True, choices=sorted(set(DOMAINS) | set(DOMAINS_V6) | set(DOMAINS_V7)))
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument(
        "--init-lora-weights",
        default="true",
        help=(
            "peft init scheme: 'true' (stock LoRA, B=0 so dW=0 at init), 'pissa', "
            "'pissa_niter_<N>', 'olora', 'eva', 'gaussian'. NOTE: pissa/olora MUTATE "
            "the base weights (W_res = W0 - scaling*B0@A0), so the trained adapter is a "
            "delta on W_res, NOT on pristine W0. This engine folds onto a pristine W0 "
            "buffer, so such adapters are converted back to standard LoRA at save time "
            "(peft's path_initial_model_for_weight_conversion). That conversion emits "
            "rank 2r, not r -- see the printed warning."
        ),
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="override the domain's training file (e.g. a corpus revision). Recorded in "
        "regime.json so an adapter never loses track of what it was trained on.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="SAFETY CAP when --stop-at-dw-over-w is set, not a target. "
        "A fixed step count gave the v5 corpora 0.37-0.99 epochs "
        "depending on size -- a confound in every cross-domain "
        "comparison. Prefer the geometric stop.",
    )
    ap.add_argument(
        "--v4",
        action="store_true",
        help="explicit opt-in to LEGACY v4 corpus/adapter version. "
        "The default is CANON.ADAPTER_VERSION (currently "
        f"{CANON.ADAPTER_VERSION!r}) -- pass this only for a "
        "deliberate, labelled ablation against the legacy version.",
    )
    ap.add_argument(
        "--v6",
        action="store_true",
        help="explicit opt-in to LEGACY adapter v6 <- corpus v5 "
        "(disposition + command data, deduplicated) -> "
        "results/adapters/*_v6. The default is CANON.ADAPTER_VERSION "
        f"(currently {CANON.ADAPTER_VERSION!r}).",
    )
    ap.add_argument(
        "--v7",
        action="store_true",
        help="adapter v7 <- corpus v6 (round-2 rebuild: astral command "
        "families, postgres asyncpg, duckdb analytics, python "
        "capability records) -> results/adapters/*_v7. This is "
        "already the default when CANON.ADAPTER_VERSION == 'v7' -- "
        "pass explicitly only to be unambiguous in a script.",
    )
    ap.add_argument(
        "--stop-at-dw-over-w",
        type=float,
        default=None,
        help="Stop when ||dW||/||W|| reaches this. 0.075 matches what the "
        "v4 adapters landed on (0.0754-0.0758) and sits mid-band with "
        "~2.2%% predicted merge error. The Goldilocks band is "
        "[0.035, 0.100]; below it bf16 truncates the delta, above "
        "0.150 base representations are overwritten. See "
        "docs/THE_FACTORY_FINE_TUNING_AND_GEOMETRY.md Ch.4.",
    )
    ap.add_argument(
        "--stop-at-plateau",
        type=float,
        default=None,
        help="Stop when relative growth (increment/current) falls below "
        "this, i.e. training has converged -- 0.02 is a reasonable "
        "start. This is the ON-THE-FLY criterion: it adapts per "
        "corpus, where a fixed |dW|/|W| target does not. Combined "
        "with the 0.100 band ceiling, whichever fires first wins.",
    )
    ap.add_argument("--geometry-every", type=int, default=10, help="steps between ||dW||/||W|| checks")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--no-liger", action="store_true", help="disable Liger fused kernels (A/B baseline)")
    ap.add_argument(
        "--no-completion-only",
        action="store_true",
        help="train on the FULL sequence (prompt+answer). The pre-2026-08-18 behaviour, kept only as an A/B baseline.",
    )
    ap.add_argument("--out", default=None, help="override the default output dir")
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed. Until this existed the trainer seeded NOTHING: "
        "lora_A's init drew from an unseeded global RNG, so every run "
        "landed in a different rank-8 subspace and no adapter in this "
        "repo was reproducible. Two runs of one identical config "
        "measured 22.40 and 15.94 on activation scale and stopped at "
        "step 130 vs 150.",
    )
    ap.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="loss logging interval. Use 1 to capture a per-step convergence curve; "
        "the default 10 gives only 15 points on a 150-step run, which is too "
        "coarse to locate a plateau or to drive any early-stopping rule.",
    )
    ap.add_argument(
        "--loss-curve-out",
        default=None,
        help="dump the full per-step loss history to this JSON path",
    )
    ap.add_argument(
        "--lambda-inert",
        type=float,
        default=0.0,
        help="Activation-space inertia penalty weight (Selective Silence). Penalizes "
        "||BAh||/||h|| on OUT-OF-DOMAIN replay tokens drawn from the other "
        "domains' corpora. 0 = off. Try 0.05 first and watch the [inert] ASR "
        "line: it must RISE. If e_in falls as fast as e_out, lambda is too high "
        "and you are just shrinking alpha again.",
    )
    ap.add_argument(
        "--inert-replay-batches", type=int, default=64, help="how many out-of-domain replay batches to pre-tokenize"
    )
    ap.add_argument(
        "--inert-replay-len",
        type=int,
        default=512,
        help="replay sequence length. Short is fine -- this measures "
        "whether the adapter FIRES, not whether it answers well.",
    )
    ap.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Explicitly toggle gradient checkpointing (default: False). Pinning avoids hidden TRL/PEFT defaults.",
    )
    ap.add_argument("--max-length", type=int, default=512, help="maximum sequence length (default: 512)")
    ap.add_argument("--batch-size", type=int, default=2, help="per-device training batch size (default: 2)")
    ap.add_argument("--grad-accum", type=int, default=2, help="gradient accumulation steps (default: 2)")
    ap.add_argument(
        "--qlora", action="store_true", help="use 4-bit NF4 base model for training 9B/27B models on 24GB VRAM"
    )
    args = ap.parse_args()

    # Seed BEFORE anything constructs a tensor. peft builds lora_A with kaiming init
    # off the global RNG the moment get_peft_model() runs, so seeding after that point
    # would not make the subspace reproducible.
    from transformers import set_seed

    set_seed(args.seed)

    # Version selection: default to CANON.ADAPTER_VERSION (the single source of
    # truth -- runtime_common/canon.py, "DO NOT CHANGE... changing a number here
    # silently reinterprets every result the repo has ever produced"), never a
    # hardcoded local default. --v4/--v6/--v7 remain explicit opt-ins for a
    # deliberate, labelled ablation against a non-canonical version.
    _version_flags = [v for v in ("v4", "v6", "v7") if getattr(args, v)]
    if len(_version_flags) > 1:
        raise SystemExit(f"  pass at most one of --v4/--v6/--v7 (got {', '.join('--' + v for v in _version_flags)})")
    version = _version_flags[0] if _version_flags else CANON.ADAPTER_VERSION
    _version_tables = {"v4": DOMAINS, "v6": DOMAINS_V6, "v7": DOMAINS_V7}
    if version not in _version_tables:
        raise SystemExit(
            f"  CANON.ADAPTER_VERSION={CANON.ADAPTER_VERSION!r} has no "
            f"matching table in this script; pass --v4/--v6/--v7 explicitly "
            f"or add a DOMAINS_{version.upper()} table."
        )
    table = _version_tables[version]
    if args.domain not in table:
        raise SystemExit(f"domain {args.domain!r} not available in {version} table: {sorted(table)}")
    data_rel, out_rel = table[args.domain]
    if args.domain in FAIR_STEPS and args.max_steps == 150:
        args.max_steps = FAIR_STEPS[args.domain]
        print(
            f"  [fair-steps] {args.domain}: 150 -> {args.max_steps} steps so each "
            f"merged domain gets the same exposure a solo adapter gets"
        )
    if args.dataset:
        data_rel = args.dataset
    dataset_path = REPO_ROOT / data_rel
    out_dir = Path(args.out) if args.out else REPO_ROOT / out_rel
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # PRE-FLIGHT EXCLUSIVITY GUARD: Fail fast if another job is holding VRAM
    ensure_gpu_exclusive()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(
        f"TRAINING {'QLORA (4-bit base)' if args.qlora else 'bf16 STOCK LORA'} [{args.domain}] r={args.rank} alpha={args.alpha} on {dev}"
    )
    print(f"  data: {dataset_path}")
    print(f"  out:  {out_dir}")
    print("=" * 88)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Liger fused Triton kernels. MUST be applied BEFORE from_pretrained: with
    # model=None the patcher rebinds at class level, so only models constructed
    # afterwards pick up the RMSNorm/SwiGLU swaps. Reversing these two lines
    # makes the patch silently do nothing.
    #
    # rope is left at its default False: liger raises NotImplementedError for
    # Qwen3.5 ("not available"), because of the hybrid Gated DeltaNet/attention
    # mix. There is no per-layer detection -- it is a blanket opt-out.
    #
    # Verified equivalent: Liger's RMSNorm uses offset=1.0 + casting_mode
    # "gemma", matching stock Qwen3_5RMSNorm's output * (1.0 + weight.float())
    # then cast. Loss trajectories match a non-Liger run (1.941->0.859 vs
    # 1.943->0.868).
    liger_applied = False
    if not args.no_liger and not args.qlora:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

        apply_liger_kernel_to_qwen3_5()
        liger_applied = True
        print("Liger fused kernels applied (fused_linear_cross_entropy, rms_norm, swiglu; rope=off)")

    if args.qlora:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Loading base model in NF4 4-bit QLoRA precision...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    else:
        # bf16, NOT load_in_4bit -- this is the whole point of the script
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )

    # peft wants the bool True for stock LoRA and a string for every other scheme.
    init_scheme: object = True if args.init_lora_weights.lower() == "true" else args.init_lora_weights
    mutates_base = args.init_lora_weights.lower().startswith(("pissa", "olora"))

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=init_scheme,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # PiSSA/OLoRA subtract their init from the base weights, so the adapter that
    # comes out is a delta on W_res, not on W0. Folding it onto this engine's
    # pristine W0 buffer would add the principal component twice. peft can convert
    # back to an equivalent standard LoRA if it is handed the INITIAL adapter, so
    # stash that before a single gradient step touches it.
    init_adapter_dir = None
    if mutates_base:
        init_adapter_dir = out_dir / "_pissa_init"
        init_adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(init_adapter_dir))
        print(
            f"  [{args.init_lora_weights}] base weights MUTATED (W_res = W0 - scaling*B0@A0).\n"
            f"  Initial adapter stashed -> {init_adapter_dir}\n"
            "  It will be converted back to a pristine-W0 LoRA at save time; expect rank 2r."
        )

    from datasets import Dataset

    completion_only = not args.no_completion_only
    records = load_dataset_records(dataset_path, completion_only=completion_only)
    n_split = sum(1 for r in records if "prompt" in r)
    print(f"Loaded {len(records)} records from {dataset_path}")
    print(
        f"  prompt/completion split: {n_split}/{len(records)} "
        f"({'completion-only loss ON' if completion_only else 'FULL-SEQUENCE loss'})"
    )
    if completion_only and n_split < len(records):
        print(
            f"  WARNING: {len(records) - n_split} records had no '### Answer:' marker "
            "and will train on the full sequence"
        )
    train_dataset = Dataset.from_list(records)

    sft_config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        seed=args.seed,
        data_seed=args.seed,
        completion_only_loss=completion_only,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="no",
        report_to="none",
    )
    replay_batches = []
    if args.lambda_inert > 0:
        print(f"  L_inert ACTIVE lambda={args.lambda_inert} -- building out-of-domain replay")
        replay_batches = build_replay_batches(
            args.domain, tokenizer, n_batches=args.inert_replay_batches, max_len=args.inert_replay_len
        )

    run_id = training_db.start_run(
        domain=args.domain,
        dataset_path=str(dataset_path),
        n_records=len(records),
        rank=args.rank,
        alpha=args.alpha,
        lr=args.lr,
        max_steps=args.max_steps,
        target_dw_w=args.stop_at_dw_over_w if args.stop_at_dw_over_w is not None else 0.071,
        adapter_version=version,
        config={
            "completion_only": completion_only,
            "liger": liger_applied,
            "gradient_checkpointing": args.gradient_checkpointing,
            "lambda_inert": args.lambda_inert,
        },
    )

    trainer = InertiaSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        lambda_inert=args.lambda_inert,
        replay_batches=replay_batches,
    )
    trainer.add_callback(TelemetryCallback(run_id=run_id))

    geom_cb = None
    if args.stop_at_dw_over_w or args.stop_at_plateau:
        geom_cb = GoldilocksStoppingCallback(
            model,
            args.alpha,
            args.rank,
            args.stop_at_dw_over_w,
            every=args.geometry_every,
            plateau=args.stop_at_plateau,
            run_id=run_id,
        )
        trainer.add_callback(geom_cb)
        print(
            f"  [geometry] geometric stop ACTIVE: train until |dW|/|W| >= "
            f"{args.stop_at_dw_over_w} (safety cap {args.max_steps} steps)"
        )

    # VERIFY THE MASK IS REAL, do not assume it.
    # §13 in docs/DECISIONS.md records an entire training run whose
    # orthogonality penalty silently evaluated to 0.000000 because a name did not
    # match. A masking flag that quietly does nothing fails the same way: training
    # looks fine, loss drops, and the defect only shows up as behaviour later.
    mask_report = {"checked": False}
    if completion_only:
        batch = next(iter(trainer.get_train_dataloader()))
        labels = batch["labels"]
        n_masked = int((labels == -100).sum())
        n_total = int(labels.numel())
        # a prompt-masked batch must have SOME masked and SOME unmasked positions
        assert n_masked > 0, (
            "completion_only_loss=True but NO labels are -100 -- the prompt is not "
            "being masked and the adapter is still training on question text"
        )
        assert n_masked < n_total, "every label masked -- nothing left to learn from"
        mask_report = {
            "checked": True,
            "masked_frac": round(n_masked / n_total, 4),
            "masked": n_masked,
            "total": n_total,
        }
        print(
            f"  MASK VERIFIED: {n_masked}/{n_total} label positions are -100 "
            f"({n_masked / n_total:.1%} of the batch is prompt, excluded from loss)"
        )

    # POST-TRAIN GEOMETRY GATE (pre-flight SVD probe + times-above-chance).
    # Both probes compare TWO adapters, so they can only run once this one exists.
    # Recorded, not enforced: high overlap with a sibling expert predicts that the
    # two will interfere, and near-chance overlap is what makes stacking safe.
    def geometry_report(new_dir: Path) -> dict:
        try:
            sys.path.insert(0, str(REPO_ROOT / "benchmarks/factory/geometry/preflight_svd_probe"))
            from probe_subspace_overlap import evaluate_subspace_overlap
        except Exception as ex:
            return {"error": f"probe unavailable: {type(ex).__name__}: {ex}"}

        def summarise(a, b, k):
            res = evaluate_subspace_overlap(a, b, k=k)
            if not res:
                return float("nan"), float("nan"), float("nan")
            ret = sum(m["retained_energy_pct"] for m in res.values()) / len(res)
            fl = sum(m["random_floor_pct"] for m in res.values()) / len(res)
            return ret, fl, ret / max(1e-30, fl)

        out = {}
        for sib in sorted((REPO_ROOT / "results/adapters").glob("m2_*")):
            if sib.resolve() == new_dir.resolve() or not (sib / "adapter_model.safetensors").exists():
                continue
            try:
                ret, floor, ratio = summarise(new_dir, sib, 32)
                out[sib.name] = {
                    "retained_pct": round(ret, 3),
                    "random_floor_pct": round(floor, 3),
                    "times_above_chance": round(ratio, 3),
                }
            except Exception as ex:
                out[sib.name] = {"error": f"{type(ex).__name__}"}
        return out

    # PRECISION FLOOR, measured not assumed (see apps/factory/calibrate_expert_alpha.py).
    # alpha alone is meaningless; the hardware responds to the perturbation
    # magnitude |dW|/|W|, and dW = (alpha/r) * B@A depends on what B@A actually
    # LEARNED -- unknowable before training. Two adapters trained with identical
    # hyperparameters measured 0.0750 and 0.0741. Recording it per adapter makes
    # the lower bound of the admissible alpha window computable for free, since
    # |dW|/|W| is exactly linear in alpha.
    def precision_report() -> dict:
        try:
            num = den = 0.0
            pairs = 0
            for mod in model.modules():
                A = getattr(mod, "lora_A", None)
                B = getattr(mod, "lora_B", None)
                W = getattr(mod, "base_layer", None)
                if A is None or B is None or W is None:
                    continue
                try:
                    a = A["default"].weight.detach().float()
                    b = B["default"].weight.detach().float()
                    w = W.weight.detach().float()
                except Exception:
                    continue
                dW = (b @ a) * (args.alpha / args.rank)
                num += float(dW.norm() ** 2)
                den += float(w.norm() ** 2)
                pairs += 1
            if pairs == 0:
                return {"error": "no lora/base pairs found"}
            ratio = (num**0.5) / max(1e-30, den**0.5)
            # law fitted across the 5-point alpha sweep: err_pct * |dW|/|W| ~= 0.167
            return {
                "pairs": pairs,
                "dw_over_w": round(ratio, 6),
                "predicted_merge_err_pct": round(0.167 / max(1e-9, ratio), 4),
                "note": "|dW|/|W| is linear in alpha; divide/multiply to re-derive",
            }
        except Exception as ex:
            return {"error": f"{type(ex).__name__}: {ex}"}

    print(f"Starting SFT training ({args.max_steps} steps)...")
    train_result = trainer.train()

    stopped_step = int(trainer.state.global_step)
    final_loss = None
    final_token_acc = None
    for item in reversed(trainer.state.log_history):
        if "loss" in item and final_loss is None:
            try:
                final_loss = float(item["loss"])
            except Exception:
                pass
        if "mean_token_accuracy" in item and final_token_acc is None:
            try:
                final_token_acc = float(item["mean_token_accuracy"])
            except Exception:
                pass
        if final_loss is not None and final_token_acc is not None:
            break

    prec = precision_report()
    final_dw_w = prec.get("dw_over_w") if isinstance(prec, dict) else None
    pred_merge_err = prec.get("predicted_merge_err_pct") if isinstance(prec, dict) else None
    stop_reason = (
        geom_cb.stop_reason
        if (geom_cb and geom_cb.stop_reason)
        else ("max_steps" if stopped_step >= args.max_steps else "completed")
    )
    runtime_sec = (
        float(train_result.metrics.get("train_runtime", 0.0))
        if hasattr(train_result, "metrics") and "train_runtime" in train_result.metrics
        else None
    )

    training_db.finish_run(
        run_id=run_id,
        status="completed",
        stopped_at_step=stopped_step,
        stop_reason=stop_reason,
        final_loss=final_loss,
        final_token_acc=final_token_acc,
        final_dw_w=final_dw_w,
        predicted_merge_err=pred_merge_err,
        runtime_seconds=runtime_sec,
    )

    if args.loss_curve_out:
        curve_path = Path(args.loss_curve_out)
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        effective_batch = 2 * 2  # per_device_train_batch_size * gradient_accumulation_steps
        curve_path.write_text(
            json.dumps(
                {
                    "domain": args.domain,
                    "methodology": METHODOLOGY,
                    "init_lora_weights": args.init_lora_weights,
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "max_steps": args.max_steps,
                    "logging_steps": args.logging_steps,
                    "lr": args.lr,
                    "lr_scheduler_type": "cosine",
                    "warmup_ratio": 0.03,
                    "n_records": len(records),
                    "effective_batch": effective_batch,
                    "epochs_seen": args.max_steps * effective_batch / max(1, len(records)),
                    "log_history": trainer.state.log_history,
                },
                indent=2,
            )
        )
        print(f"Loss curve -> {curve_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if init_adapter_dir is not None:
        # Emits dW = scaling*(B_trained@A_trained - B0@A0) refactorised as a plain
        # LoRA on pristine W0 -- which is what WeightFoldingEngine requires. The
        # refactorisation of a difference of two rank-r products is rank 2r.
        model.save_pretrained(str(out_dir), path_initial_model_for_weight_conversion=str(init_adapter_dir))
    else:
        model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # The ASR trace lives NEXT TO THE ADAPTER. An adapter trained with L_inert is
    # not interchangeable with one trained without it, and "did selectivity
    # actually move?" must be answerable from the artifact rather than from a log
    # that scrolled away.
    if geom_cb is not None and geom_cb.trace:
        # The trace lives next to the adapter: "what geometry did this stop at, and
        # after how many steps" must be answerable from the artifact, not from a
        # log that scrolled away.
        (out_dir / "geometry_trace.json").write_text(
            json.dumps(
                {
                    "target_dw_over_w": args.stop_at_dw_over_w,
                    "stopped_at_step": geom_cb.trace[-1]["step"],
                    "final": geom_cb.trace[-1],
                    "goldilocks_band": [0.035, 0.100],
                    "trace": geom_cb.trace,
                },
                indent=2,
            )
        )
        f = geom_cb.trace[-1]
        print(
            f"  [geometry] FINAL |dW|/|W|={f['dw_over_w']} at step {f['step']} "
            f"(merge_err~{f['merge_err_pct']}%) -> geometry_trace.json"
        )

    if trainer.inert_log:
        (out_dir / "inert_trace.json").write_text(
            json.dumps(
                {
                    "lambda_inert": args.lambda_inert,
                    "replay_domains": [d for d in DOMAINS if d != args.domain],
                    "replay_batches": len(replay_batches),
                    "trace": trainer.inert_log,
                    "final": trainer.inert_log[-1],
                },
                indent=2,
            )
        )
        f = trainer.inert_log[-1]
        print(
            f"  [inert] FINAL e_in={f['e_in']:.4f} e_out={f['e_out']:.4f} "
            f"ASR={f['asr']:.3f}x  -> {out_dir / 'inert_trace.json'}"
        )
        if not (f["asr"] > 1.0):
            print(
                "  [inert] WARNING: ASR did not exceed 1.0 -- the adapter is NOT "
                "selective. Check whether e_in fell alongside e_out (lambda too "
                "high, you are shrinking alpha) before trusting this adapter."
            )
    # Record the regime in the adapter itself. 0 of 69 existing adapters do this,
    # so provenance was previously recoverable only from directory-layout side
    # effects (export_adapter.py leaves no checkpoints/ subdir; this one does).
    (out_dir / "regime.json").write_text(
        json.dumps(
            {
                "methodology": METHODOLOGY,
                "precision": "bfloat16",
                "quantization": None,
                "liger_fused_kernels": liger_applied,
                "gradient_checkpointing": args.gradient_checkpointing,
                "completion_only_loss": completion_only,
                "prompt_mask_verified": mask_report,
                "subspace_geometry": geometry_report(out_dir),
                "merge_precision": precision_report(),
                "trained_by": "apps/factory/train_expert.py",
                "seed": args.seed,
                "domain": args.domain,
                "rank": args.rank,
                "alpha": args.alpha,
                "scaling": args.alpha / args.rank,
                "init_lora_weights": args.init_lora_weights,
                "base_weights_mutated_during_training": mutates_base,
                "converted_to_pristine_w0_lora": init_adapter_dir is not None,
                "max_steps": args.max_steps,
                "lr": args.lr,
                "dataset": data_rel,
                "n_records": len(records),
            },
            indent=2,
        )
    )
    print(f"Saved adapter + regime.json to {out_dir}")


if __name__ == "__main__":
    main()
