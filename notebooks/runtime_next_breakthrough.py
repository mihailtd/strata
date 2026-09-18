import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    app_title="Beating llama.cpp from Scratch on Consumer AMD",
)


@app.cell
def _():
    import json
    import pathlib

    import altair as alt
    import marimo as mo
    import polars as pl

    return alt, json, mo, pathlib, pl


@app.cell
def _(mo):
    mo.md(r"""
    # Beating llama.cpp from Scratch

    ### Engineering a Native Rust + HIP Inference Engine for Qwen 3.5: Prefill Batching, Network Stream Flushing, and 4-Way Split-KV Attention on Consumer AMD Hardware
    """)
    return


@app.cell
def _(mo):
    stat_tput = mo.stat(
        value="83.0 tok/s",
        label="Decode Throughput",
        caption="+16.4% vs llama.cpp (71.3) | +15.3% vs Ollama (72.0)",
        direction="increase",
        bordered=True,
    )
    stat_ttft = mo.stat(
        value="48.5 ms",
        label="Time To First Token",
        caption="-59.9% vs llama.cpp (121.0 ms) | -72.0% vs Ollama (173.2 ms)",
        direction="decrease",
        bordered=True,
    )
    stat_launches = mo.stat(
        value="3,300 → 280",
        label="Prefill Kernel Launches",
        caption="-91.5% host CPU driver dispatch overhead",
        direction="decrease",
        bordered=True,
    )
    stat_framing = mo.stat(
        value="< 0.3%",
        label="Per-Token SSE Framing Cost",
        caption="Streaming flush overhead under 0.3%",
        bordered=True,
    )
    stat_precision = mo.stat(
        value="BF16",
        label="Weight Precision",
        caption="Unquantized baseline on all three engines",
        bordered=True,
    )
    mo.hstack([stat_tput, stat_ttft, stat_launches, stat_framing, stat_precision], justify="space-between", gap=1)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **Executive Summary** — This article details the implementation and optimization of a bare-metal
            inference runtime (`runtime-next`) written in Rust and AMD HIP for the **Qwen 3.5** hybrid architecture,
            running on a single consumer GPU (**AMD Radeon RX 7900 XTX 24GB**).

            We benchmarked head-to-head against **llama.cpp** (`llama-server`) and **Ollama** (`ollama serve`)
            under identical conditions: independent HTTP daemons on localhost, OpenAI-compatible streaming chat
            endpoints over TCP sockets, and unquantized BF16 weights. Both baselines were executed with **100% GPU
            offload via native AMD ROCm (`gfx1100`)**: `llama-server` compiled with HIPBLAS (`-ngl 999`), and `ollama serve`
            running with CachyOS's hardware-accelerated `ollama-rocm` package (`libggml-hip.so`, all 34 layers pinned to VRAM).

            On the 4B model, `runtime-next` achieved **83.0 tok/s** sustained streaming throughput (**1.16× faster
            than llama.cpp**, **1.15× faster than Ollama**) and **48.5 ms Time-To-First-Token (60–72% lower latency)**.
            Across the wider family, speedups range from **1.28× at 0.8B** to **1.15× at 9B** -- and, as of this
            round's GDN prefill batching work, `runtime-next` now wins Time-To-First-Token at every size from 0.8B
            through 9B, not just decode throughput.

            This report covers the key bottlenecks encountered and resolved: eliminating a 3,300-kernel prefill launch
            storm, fixing an un-flushed 8KB HTTP buffer, isolating streaming loop overhead, and implementing a 4-way
            parallel split-KV attention kernel inspired by `llama.cpp`.
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(json, pathlib):
    _repo_root = pathlib.Path(__file__).resolve().parent.parent
    scorecard = json.loads(
        (_repo_root / "results/benchmarks/4b_engine_comparison_scorecard.json").read_text()
    )
    return (scorecard,)


@app.cell
def _(mo, scorecard):
    mo.md(
        rf"""
        ---
        ## The Test Bench & The Hybrid Architecture

        ### 1. Hardware Environment
        - **GPU**: {scorecard["hardware"]}
        - **VRAM**: 24 GB GDDR6 @ 960 GB/s theoretical bandwidth
        - **Architecture**: AMD RDNA3 (Navi 31, gfx1100, 96 Compute Units, 6,144 Stream Processors)
        - **Software Stack**: Linux x86_64, ROCm 7.2, HIP compiler

        ### 2. Architecture: Qwen 3.5 4B
        Standard Transformers (such as LLaMA or Mistral) use full attention at every layer, meaning Key-Value (KV)
        cache memory scales linearly with sequence length ($O(T)$).

        Qwen 3.5 4B uses a **hybrid architecture** across its 32 layers:
        - **24 Gated DeltaNet (GDN) Linear Attention Layers**: Maintain a **fixed-size recurrent state**
          $S_t \in \mathbb{{R}}^{{32 \times 128 \times 128}}$ (in FP32, 2.0 MB per layer, totaling **48.0 MB** across
          all 24 layers). As tokens arrive, $S_t$ updates in-place via a causal 1D convolution and a gated delta rule.
          Memory footprint is constant $O(1)$ regardless of sequence length.
        - **8 Full Grouped-Query Attention (GQA) Layers**: Positioned at every 4th layer (3, 7, 11, 15, 19, 23, 27, 31).
          These layers store standard KV caches to maintain exact context retrieval over long sequences.

        > **In Plain English (The Detective's Notebook Analogy)**:
        > For 75% of daily work (the 24 GDN layers), a detective writes concise summaries in a small notebook of fixed size,
        > compressing facts as they occur. The notebook never grows heavier. For the remaining 25% of critical evidence
        > (the 8 Full Attention layers), they keep full transcripts of witness interviews, re-reading all past transcripts
        > whenever a new question arises.

        ### 3. Evaluation Methodology
        1. **Weights**: Native BF16 parameters loaded across all three engines.
        2. **Networking**: Standard HTTP/1.1 servers listening on localhost (`runtime-next` on port 8000, `llama-server`
           on port 8001, `ollama serve` on port 11434).
        3. **Streaming**: Requests sent to `POST /v1/chat/completions` with `{{"stream": true}}`. Latency (TTFT) measures
           time from socket opening to the first received token chunk; throughput measures total generated tokens divided
           by total streaming duration.
        4. **Workloads**: Three multi-turn programming tasks (FastAPI asyncpg+pgvector CRUD, PostgreSQL 17 HNSW tuning,
           DuckDB Parquet window analytics) generating ~350 tokens each.
        5. **Hardware Acceleration Parity (AMD ROCm)**: Both baseline engines executed with full ROCm GPU acceleration on
           the RX 7900 XTX (`gfx1100`). `llama-server` ran with `-ngl 999` offloading all 34 layers. `ollama serve` ran
           with the native CachyOS `ollama-rocm` package, dynamically linking `/usr/lib/ollama/rocm_v7_2/libggml-hip.so`
           and offloading all 34 layers (8,023.7 MiB model buffer in VRAM). Neither baseline was run on CPU.
        """
    )
    return


@app.cell
def _(mo):
    mo.Html(r"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px; padding: 20px; margin: 16px 0; font-family: monospace;">
        <div style="font-weight: bold; color: #38bdf8; font-size: 14px; margin-bottom: 12px;">HYBRID MODEL TOPOLOGY: Qwen 3.5 4B (32 Layers Total)</div>
        <div style="display: flex; flex-direction: column; gap: 8px;">
            <div style="background: rgba(56, 189, 248, 0.1); border-left: 4px solid #38bdf8; padding: 10px; border-radius: 4px;">
                <span style="color: #38bdf8; font-weight: bold;">24 Layers: Gated DeltaNet (GDN) Linear Attention</span><br/>
                <span style="color: #94a3b8; font-size: 12px;">• Fixed recurrent state: S_t ∈ ℝ^[32, 128, 128] (FP32, 2.0 MB / layer = 48.0 MB total)</span><br/>
                <span style="color: #94a3b8; font-size: 12px;">• Causal Conv1D state: [8192, 3] BF16 (48 KB / layer = 1.15 MB total)</span><br/>
                <span style="color: #22c55e; font-size: 12px;">• Memory footprint: Constant O(1) — no growth over sequence length</span>
            </div>
            <div style="background: rgba(249, 115, 22, 0.1); border-left: 4px solid #f97316; padding: 10px; border-radius: 4px;">
                <span style="color: #f97316; font-weight: bold;">8 Layers: Full Grouped-Query Attention (Layers 3, 7, 11, 15, 19, 23, 27, 31)</span><br/>
                <span style="color: #94a3b8; font-size: 12px;">• Dynamic KV Cache: [4 heads, seq_len, 256 dim] BF16 (32 KB per token across all 8 layers)</span><br/>
                <span style="color: #ef4444; font-size: 12px;">• Memory footprint: Scales O(T) — creates compute and memory bandwidth demands at long context</span>
            </div>
        </div>
    </div>
    """)
    return


@app.cell
def _(pl, scorecard):
    engine_summary = pl.DataFrame(
        {
            "engine": ["runtime-next (this port)", "llama.cpp", "Ollama"],
            "avg_tok_s": [
                scorecard["avg_runtime_next_tok_s"],
                scorecard["avg_llamacpp_tok_s"],
                scorecard["avg_ollama_tok_s"],
            ],
            "avg_ttft_ms": [
                scorecard["avg_ttft_ms"]["runtime_next"],
                scorecard["avg_ttft_ms"]["llamacpp"],
                scorecard["avg_ttft_ms"]["ollama"],
            ],
        }
    )
    return (engine_summary,)


@app.cell
def _(alt, engine_summary, mo):
    _tput_chart = (
        alt.Chart(engine_summary)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("engine:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("avg_tok_s:Q", title="Streaming tok/s", scale=alt.Scale(domain=[0, 95])),
            color=alt.Color(
                "engine:N",
                legend=None,
                scale=alt.Scale(
                    domain=["runtime-next (this port)", "Ollama", "llama.cpp"],
                    range=["#0ea5e9", "#64748b", "#94a3b8"]
                )
            ),
            tooltip=[alt.Tooltip("engine:N", title="Engine"), alt.Tooltip("avg_tok_s:Q", title="tok/s", format=".2f")],
        )
        .properties(
            width=260,
            height=240,
            title="Sustained Generation Throughput",
            autosize=alt.AutoSizeParams(type="fit", contains="padding")
        )
    )

    _ttft_chart = (
        alt.Chart(engine_summary)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("engine:N", sort="y", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("avg_ttft_ms:Q", title="Time to First Token (ms)", scale=alt.Scale(domain=[0, 130])),
            color=alt.Color(
                "engine:N",
                legend=None,
                scale=alt.Scale(
                    domain=["runtime-next (this port)", "llama.cpp", "Ollama"],
                    range=["#10b981", "#f59e0b", "#f97316"]
                )
            ),
            tooltip=[alt.Tooltip("engine:N", title="Engine"), alt.Tooltip("avg_ttft_ms:Q", title="TTFT (ms)", format=".1f")],
        )
        .properties(
            width=260,
            height=240,
            title="Time-To-First-Token (Lower = Better)",
            autosize=alt.AutoSizeParams(type="fit", contains="padding")
        )
    )

    mo.hstack([_tput_chart, _ttft_chart], justify="center", gap=2)
    return


@app.cell
def _(mo, scorecard):
    mo.md(
        f"""
        On 4B benchmarks, `runtime-next` leads llama.cpp by **{scorecard["runtime_next_speedup_vs_llamacpp"]}**
        and Ollama by **{scorecard["runtime_next_speedup_vs_ollama"]}** in throughput, while reducing Time-To-First-Token
        by **72.5 ms (59.9%) vs. llama.cpp** and **124.7 ms (72.0%) vs. Ollama**.

        Before discussing the HTTP serving pipeline (Acts 1–5), we review how the raw decode loop was tuned
        from 31.7 tok/s up to 82.2 tok/s.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Act 0 — Decode Optimization: Moving from 31.7 to 82 tok/s

    The native decode loop initially ran at **31.7 tok/s** on the RX 7900 XTX—substantially behind mature engines.
    Reaching 82+ tok/s required four targeted optimization passes on the GPU kernels.

    ### Why Rust with HIP?
    HIP kernels compiled for AMD RDNA3 run on the hardware regardless of the host language. The primary advantages
    of Rust in this runtime are architectural:
    - **Enforced Safety Boundaries**: Raw pointers and HIP runtime calls are encapsulated in a minimal FFI module
      (`hip.rs`). The remainder of the 5,000-line model forward pass and server logic is safe Rust.
    - **Zero Foreign Function / Interpreter Overhead**: Moving away from Python removes GIL and runtime dispatch
      overhead from every kernel invocation.
    - **Clean Re-Implementation**: Writing the kernels directly against the model architecture ensured every memory
      layout and compute pattern could be tailored to the RX 7900 XTX.
    """)
    return


@app.cell
def _(pl):
    perf_arc_df = pl.DataFrame(
        {
            "stage": [
                "Baseline (naive port)",
                "+ initial tuning",
                "+ kernel fusion, direct KV reads",
                "+ HIP Graph + custom GEMV + on-device argmax",
                "+ GDN & RMSNorm tuning",
            ],
            "tok_s": [31.7, 44.3, 49.2, 81.2, 82.2],
        }
    )
    return (perf_arc_df,)


@app.cell
def _(alt, mo, perf_arc_df):
    _chart = (
        alt.Chart(perf_arc_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("stage:N", title=None, sort=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y("tok_s:Q", title="Decode tok/s"),
            color=alt.Color(
                "stage:N",
                legend=None,
                scale=alt.Scale(scheme="blues"),
            ),
            tooltip=["stage", alt.Tooltip("tok_s:Q", format=".1f")],
        )
        .properties(
            width=480,
            height=240,
            title="Decode Optimization Progression: 31.7 → 82.2 tok/s",
            autosize=alt.AutoSizeParams(type="fit", contains="padding"),
        )
    )
    mo.vstack([_chart])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What each pass achieved

    - **Pass 1 (31.7 → 44.3 tok/s, +40%)**: Removed redundant host-device synchronization and streamlined buffer allocation.
    - **Pass 2 (44.3 → 49.2 tok/s, +11%)**: Fused projection operations and read directly from combined GEMM output buffers.
      We evaluated `hipBLASLt` as an alternative to `hipblasGemmEx`, but found it slower on these specific matrix shapes,
      so we retained the standard path.
    - **Pass 3 (49.2 → 81.2 tok/s, +65%)**:
      1. *HIP Graph replay*: Captured the per-token decode graph to eliminate CPU dispatch latency.
      2. *Custom vectorized GEMV*: Decode operates on a single token ($M=1$), making general matrix-matrix kernels sub-optimal.
         We implemented a custom matrix-vector kernel using vectorized 4-element reads (`ushort4`), doubling throughput on
         dominant projection dimensions.
      3. *On-device argmax*: Replaced host-side logit transfers (248,320 floats) with a lightweight on-GPU reduction that
         returns a single 4-byte token ID.
    - **Pass 4 (81.2 → 82.2 tok/s)**: Profiled with `rocprofv3` and discovered `gdn_recurrent_decode` utilized only 32
      of the GPU's 96 compute units (one block per head, with 32 heads). Increasing thread block size from 128 to 1,024
      improved occupancy, reducing kernel execution from 60.5 μs to 24.9 μs.
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **Profiling Insight**: Early tuning without GPU profiling relied on indirect timers. Once `rocprofv3`
            was introduced, compute-unit underutilization was identified and resolved in hours. Systematic profiling
            proved far more effective than trial-and-error kernel adjustments.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Act 1 — The 3,300-Kernel Launch Storm

    When first porting prompt prefill, the forward pass looped over tokens sequentially:
    - **Causal Conv1D**: 54 tokens $\times$ 24 GDN layers = **1,296 launches**
    - **GDN Gate/Beta**: 54 tokens $\times$ 24 GDN layers = **1,296 launches**
    - **RoPE Embeddings**: 54 tokens $\times$ 8 Attention layers = **432 launches**
    - **KV-Cache Appends**: 54 tokens $\times$ 8 Attention layers = **432 launches**

    This totaled roughly **3,300 kernel launches for a single 54-token prompt**.

    > **In Plain English (The Chef and Single Grains of Rice)**:
    > A commercial kitchen has 6,144 cooks ready to prepare an order. Instead of bringing in the full bowl of ingredients,
    > an assistant walks into the kitchen 3,300 times to deliver one grain of rice per trip. The cooks spend most of their
    > time waiting for the assistant to open the door.

    At 15–20 μs per launch, dispatch overhead consumed **over 50 ms of CPU driver time** before execution completed.

    ### The Fix: Batched Chunk Kernels
    We redesigned all four operations into batched kernels that process the sequence in a single launch per layer:
    """)
    return


@app.cell
def _(pl):
    batching_df = pl.DataFrame(
        {
            "operation": [
                "Causal Conv1D",
                "GDN Gate & Beta",
                "RoPE Embeddings",
                "KV-Cache Append",
            ],
            "before_launches": [1296, 1296, 432, 432],
            "after_launches": [24, 24, 8, 8],
            "parallel_mechanism": [
                "Bounded lookback (k=4): 1 thread per channel scans all T tokens carrying a 4-element register window",
                "2D parallel grid over (token, head): Zero cross-token recurrence, heads indexed by tid % num_v_heads",
                "Vectorized position indexing: Each token rotates by its precomputed absolute position from position_buf",
                "Direct strided scatter: Each token writes directly into its predetermined cache slot",
            ],
        }
    )
    return (batching_df,)


@app.cell
def _(batching_df, mo):
    mo.ui.table(
        batching_df,
        selection=None,
        label="Four Batched Prefill Kernels Replacing 3,300 Per-Token Launches (-91.5% Host Overhead)",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Batching reduced prefill latency from **78.95 ms to 71.28 ms** (-9.7%) on internal microbenchmarks.
    With host launch overhead addressed, we expected HTTP TTFT to drop proportionally. Instead, we encountered an unexpected delay.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Act 2 — The Phantom Stall: A 600ms Bug Hiding Behind a 70ms Kernel

    Testing the HTTP server with live client requests showed Time-To-First-Token stuck at **613–640 ms**,
    despite internal GPU forward passes completing in ~71 ms.

    ### Investigating the Latency Gap
    We systematically evaluated potential causes:
    1. **Client overhead**: Verified with `curl -N -w "%{time_starttransfer}\n" -s -o /dev/null`, which confirmed the same 600ms+ delay.
    2. **State reset**: Profiled `state.reset()`, which took **0.05 ms**.
    3. **Token decoding**: Measured BPE token decoding across 350 tokens at **3.32 ms total** (~9 μs per token).

    ### The Root Cause: Un-Flushed 8KB HTTP Buffer
    Reviewing the HTTP server implementation (`tiny_http` 0.12.0) revealed that `Response::raw_print` wrapped chunked transfer
    encoding in `chunked_transfer::Encoder`:
    - The encoder defaulted to an **8,192-byte internal buffer**.
    - It performed **no flush until the stream closed**.

    At ~150 bytes per Server-Sent Event (SSE) JSON chunk, 8KB accumulated roughly **55 tokens** before sending any TCP packets.

    > **In Plain English (The Reluctant Mail Carrier)**:
    > A mail carrier establishes a rule: *"I will not walk to the mailbox until my bag weighs 8 kilograms."*
    > Even though your first letter was ready in 70 milliseconds, the recipient waits until you write 54 more letters
    > just to fill the carrier's bag.

    ### The Resolution
    We utilized `Request::into_writer()`, `tiny_http`'s escape hatch for raw socket access, bypassing the buffered
    `Response` abstraction. We wrote HTTP/1.1 chunked framing directly with an explicit `.flush()` after every SSE token frame.
    """)
    return


@app.cell
def _(pl):
    ttft_bug_df = pl.DataFrame(
        {
            "state": ["Before Fix (Cold Start)", "Before Fix (Warm Cache)", "After Fix (Cold Start)", "After Fix (Warm Cache)"],
            "ttft_ms": [700.0, 410.0, 180.0, 75.0],
            "status": ["Buffered (8KB Stall)", "Buffered (8KB Stall)", "Direct SSE Flush", "Direct SSE Flush"],
        }
    )
    return (ttft_bug_df,)


@app.cell
def _(alt, mo, ttft_bug_df):
    _chart = (
        alt.Chart(ttft_bug_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("state:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("ttft_ms:Q", title="Time to First Token (ms)"),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=["Buffered (8KB Stall)", "Direct SSE Flush"], range=["#ef4444", "#10b981"]),
                title="Delivery Mode"
            ),
            tooltip=["state", "ttft_ms", "status"]
        )
        .properties(
            width=480,
            height=240,
            title="Time-To-First-Token: Buffering Fix Impact",
            autosize=alt.AutoSizeParams(type="fit", contains="padding")
        )
    )
    mo.vstack([_chart, mo.md("*Bypassing the 8KB socket buffer dropped warm TTFT from 410 ms to 75 ms.*")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Act 3 — Isolating Streaming Overhead

    Following the buffering fix, full 350-token streaming benchmarks reported **78.7–80.2 tok/s**.
    Earlier isolated tests on short prompts had reached 90+ tok/s. We investigated whether per-token network flushing
    was creating a bottleneck.

    We tested two initial adjustments:
    1. **Flush Coalescing (`FLUSH_INTERVAL`)**: Buffering writes into a 250ms window raised throughput to **85.4 tok/s**,
       but increased TTFT to **336.4 ms**.
    2. **TCP_NODELAY**: Enabling `TCP_NODELAY` on the server socket left throughput unchanged at **79.0 tok/s**.

    ### The In-Process A/B/C Test
    Rather than accepting flush coalescing, we designed an in-process diagnostic to isolate the exact cost of each stage:
    - **Arm A**: `engine.step()` alone (GPU forward pass only).
    - **Arm B**: Arm A + BPE tokenizer decode + JSON string formatting + SSE frame serialization.
    - **Arm C**: Arm B + writing into an in-memory buffer followed by an explicit `.flush()`.
    """)
    return


@app.cell
def _(pl):
    abc_df = pl.DataFrame(
        {
            "configuration": [
                "Arm A: Raw GPU engine.step()",
                "Arm B: Step + Tokenizer + JSON/SSE",
                "Arm C: Step + Tokenizer + SSE + Write + Flush",
            ],
            "tok_s": [79.40, 80.15, 79.51],
        }
    )
    return (abc_df,)


@app.cell
def _(abc_df, alt, mo):
    _chart = (
        alt.Chart(abc_df)
        .mark_bar(clip=True)
        .encode(
            y=alt.Y("configuration:N", sort=None, title=None),
            x=alt.X(
                "tok_s:Q",
                scale=alt.Scale(domain=[75, 82], zero=False, nice=False, clamp=True),
                title="tok/s",
            ),
            color=alt.value("#0ea5e9"),
            tooltip=["configuration", "tok_s"],
        )
        .properties(
            width=460,
            height=180,
            title="In-Process Isolation Test: Arms A, B, and C Within 0.3%",
            autosize=alt.AutoSizeParams(type="fit", contains="padding"),
        )
    )
    mo.vstack([
        _chart,
        mo.md(
            """
            > **In Plain English (The Engine and the Dashboard Light)**:
            > Blaming per-token HTTP flushing for GPU slowdown was like blaming the flashing turn signal on your
            > dashboard for your car losing engine horsepower. The two systems aren't mechanically coupled.
            > 
            > The isolation test showed that formatting and flushing SSE chunks accounts
            > for **less than 0.3% of execution time**.
            """
        )
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    The results confirmed that per-token flushing does not limit decode speed. We removed the flush-coalescing buffer
    and kept immediate per-frame flushes.

    The throughput variation was explained by examining Arm A's segment logs across generation depth.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Act 4 — The KV Cache Bottleneck in Attention Layers

    Timing Arm A (`engine.step()` alone) in 50-token windows revealed a clear trend:
    - Tokens 54 → 104: **80.00 tok/s**
    - Tokens 104 → 154: **80.57 tok/s**
    - Tokens 154 → 204: **80.38 tok/s**
    - Tokens 204 → 254: **79.53 tok/s**
    - Tokens 254 → 304: **77.61 tok/s**
    - Tokens 304 → 354: **78.43 tok/s**

    Throughput gradually decreased as the sequence grew. The earlier 90+ tok/s measurements had been taken on 5-token prompts
    at positions 5–28. Over longer sequences, attention cost grew with cache length.

    ### The Mechanism
    In the 8 full-attention layers, each new token computes attention across all preceding tokens in the KV cache.
    Our initial decode attention kernel assigned **one thread per output dimension** (`head_dim = 256`), with that thread
    running a serial loop over all $K$ past positions ($O(kv\_len)$ scan).

    > **In Plain English (The Single Researcher in an Expanding Archive)**:
    > A researcher reviews past files. With 20 files, one person finishes quickly. When the archive reaches 350 files,
    > that same single researcher must read through all 350 files alone while colleagues remain unassigned.

    ### 4-Way Split-KV Parallel Reduction
    Following the approach in `llama.cpp`'s `fattn-vec.cuh`, we restructured decode attention to split the sequence across parallel threads:
    - **4 cooperative threads** (`kv_split = 4`) work on each output dimension: `tid % head_dim` selects the dimension,
      while `tid / head_dim` assigns the split worker.
    - Each thread scans $\frac{1}{4}$ of the KV cache length concurrently.
    - Partial sums are combined in GPU shared memory (LDS) via a warp reduction tree.
    - **Hardware limit**: `head_dim (256) × 4 = 1,024 threads`, matching the maximum thread count per block on AMD RDNA3.
    """)
    return


@app.cell
def _(pl):
    speedup_df = pl.DataFrame(
        {
            "kv_len": [64, 128, 256, 354],
            "scalar_us": [14.345, 19.553, 32.462, 44.812],
            "split_us": [8.686, 10.374, 14.042, 16.789],
            "speedup": [1.65, 1.88, 2.31, 2.67],
        }
    )
    return (speedup_df,)


@app.cell
def _(alt, mo, speedup_df):
    _chart = (
        alt.Chart(speedup_df)
        .mark_line(point=alt.OverlayMarkDef(size=80, filled=True), strokeWidth=3, color="#0ea5e9")
        .encode(
            x=alt.X("kv_len:Q", title="KV Cache Length (Tokens Generated)"),
            y=alt.Y("speedup:Q", title="Speedup vs. Scalar Kernel", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("kv_len:Q", title="Cache Length"),
                alt.Tooltip("scalar_us:Q", title="Scalar (μs)", format=".2f"),
                alt.Tooltip("split_us:Q", title="Split-4 (μs)", format=".2f"),
                alt.Tooltip("speedup:Q", title="Speedup", format=".2f"),
            ]
        )
        .properties(
            width=480,
            height=240,
            title="Split-KV Kernel Speedup vs. Sequence Length",
            autosize=alt.AutoSizeParams(type="fit", contains="padding")
        )
    )
    mo.vstack([
        _chart,
        mo.md("*At length 64, the split kernel is 1.65× faster. By length 354, it reaches 2.67× (16.8 μs vs. 44.8 μs).*")
    ])
    return


@app.cell
def _(pl):
    segment_df = pl.DataFrame(
        {
            "segment_start_position": [54, 104, 154, 204, 254, 304],
            "Original Scalar Kernel": [80.00, 80.57, 80.38, 79.53, 77.61, 78.43],
            "4-Way Split-KV Kernel": [83.05, 83.86, 83.80, 83.66, 83.41, 83.21],
        }
    ).unpivot(
        index="segment_start_position",
        on=["Original Scalar Kernel", "4-Way Split-KV Kernel"],
        variable_name="kernel",
        value_name="tok_s",
    )
    return (segment_df,)


@app.cell
def _(alt, mo, segment_df):
    _chart = (
        alt.Chart(segment_df)
        .mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=3)
        .encode(
            x=alt.X("segment_start_position:Q", title="Token Position in Generation Sequence"),
            y=alt.Y("tok_s:Q", title="Decode tok/s (50-Token Window)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "kernel:N",
                title="Attention Implementation",
                scale=alt.Scale(domain=["Original Scalar Kernel", "4-Way Split-KV Kernel"], range=["#ef4444", "#10b981"])
            ),
            tooltip=["kernel", "segment_start_position", "tok_s"]
        )
        .properties(
            width=480,
            height=240,
            title="Throughput Across Sequence Depth: Flat Cadence",
            autosize=alt.AutoSizeParams(type="fit", contains="padding")
        )
    )
    mo.vstack([
        _chart,
        mo.md(
            "*With the 4-way split-KV kernel, decode throughput stays at ~83.2–83.8 tok/s "
            "across the entire 350-token trajectory.*"
        )
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Act 5 — Baseline Configuration and Speculative Decoding

    To ensure a fair comparison, we reviewed every compilation and runtime flag used with `llama-server` on this hardware:
    """)
    return


@app.cell
def _(pl):
    flag_audit_df = pl.DataFrame(
        {
            "flag": [
                "-ngl 99",
                "-fa on",
                "-ctk / -ctv q8_0",
                "GGML_HIP_GRAPHS",
                "-b 512 / -ub 512",
                "-t 8",
                "--spec-type ngram-mod",
            ],
            "status_in_baseline": [
                "ACTIVE — 100% of all 32 layers offloaded to GPU VRAM",
                "ACTIVE — Flash Attention enabled for full-attention layers",
                "ACTIVE — 8-bit quantized KV cache (bandwidth advantage for llama.cpp)",
                "ACTIVE — Verified compiled ON in CMakeCache.txt, RDNA3 graph replay enabled",
                "ACTIVE — Prompt processing batch sizes matched to standard server configs",
                "ACTIVE — 8 CPU threads allocated for host-side dispatch",
                "TESTED SEPARATELY — Model-free n-gram speculation evaluated below",
            ],
            "impact_on_comparison": [
                "Parity: Zero CPU fallback for either engine",
                "Parity: Optimal attention kernels active",
                "Advantage llama.cpp: runtime-next uses unquantized BF16 KV buffers",
                "Parity: Both engines replay decode via hardware HIP graphs",
                "Parity: Standard inference parameters",
                "Parity: GPU-bound workload unaffected by host thread pool",
                "Evaluated: Verified whether speculation changes the outcome",
            ],
        }
    )
    return (flag_audit_df,)


@app.cell
def _(flag_audit_df, mo):
    mo.ui.table(flag_audit_df, selection=None, label="Audit of llama.cpp Performance Flags")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Standard acceleration features were enabled for `llama.cpp`: full offload, flash attention, quantized KV cache
    (which reduces memory bandwidth requirements vs. unquantized BF16), and HIP Graph execution.

    ### Why Speculative Decoding Failed on Code Generation
    We evaluated model-free speculative decoding (`--spec-type ngram-mod`) in `llama-server` to see if prompt-lookup
    speculation could improve throughput:
    """)
    return


@app.cell
def _(pl):
    ngram_mod_df = pl.DataFrame(
        {
            "configuration": ["llama.cpp Baseline (No Speculation)", "llama.cpp + ngram-mod (Run 1)", "llama.cpp + ngram-mod (Run 2)"],
            "avg_tok_s": [72.73, 72.75, 71.04],
            "avg_ttft_ms": [108.0, 107.4, 140.7],
            "delta_throughput": ["Baseline", "+0.02 tok/s (0.0%)", "-1.69 tok/s (-2.3%)"],
        }
    )
    return (ngram_mod_df,)


@app.cell
def _(mo, ngram_mod_df):
    mo.ui.table(ngram_mod_df, selection=None, label="llama.cpp Speculative Decoding Benchmark Results")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Telemetry from `llama-server` showed 64 drafted tokens with only 5 accepted—a **7.8% acceptance rate**:
    ```json
    "timings": {
        "prompt_n": 54,
        "predicted_n": 350,
        "draft_n": 64,
        "draft_n_accepted": 5
    }
    ```

    > **In Plain English (The Overconfident Assistant)**:
    > An assistant attempts to predict the remainder of your sentence. If they guess 64 words and 59 are incorrect,
    > you spend more time correcting their guesses than if you had spoken at your normal speed.

    In code generation, token predictions require high contextual accuracy. Because each rejected token incurs validation
    and rollback costs, an acceptance rate under 10% reduces throughput on an already-optimized decode loop.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Act 6 — Multi-Size Scaling: 0.8B, 2B, 4B, and 9B

    To evaluate whether this performance profile generalizes across model sizes, we benchmarked the entire Qwen 3.5 family:
    0.8B, 2B, 4B, and 9B.

    The architecture scales primarily by layer count and layer width:
    """)
    return


@app.cell
def _(pl):
    family_config_df = pl.DataFrame(
        {
            "dimension": ["hidden_size", "intermediate_size", "num_hidden_layers", "attn Q heads", "attn KV heads", "GDN V heads", "head_dim (attn)", "GDN key head_dim", "full_attention_interval", "lm_head tied to embeddings?"],
            "0.8B": ["1,024", "3,584", "24", "8", "2", "16", "256", "128", "4", "Yes"],
            "2B": ["2,048", "6,144", "24", "8", "2", "16", "256", "128", "4", "Yes"],
            "4B": ["2,560", "9,216", "32", "16", "4", "32", "256", "128", "4", "Yes"],
            "9B": ["4,096", "12,288", "32", "16", "4", "32", "256", "128", "4", "No — separate lm_head.weight"],
        }
    )
    return (family_config_df,)


@app.cell
def _(family_config_df, mo):
    mo.ui.table(family_config_df, selection=None, label="Architecture Parameters Across the Qwen 3.5 Family")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Key structural elements—head dimensions (256 for attention, 128 for GDN) and the 4-layer attention cadence—remain
    constant across the family. The main architectural distinction occurs at 9B, which uses an untied output projection
    (`lm_head.weight`) rather than sharing the input embedding table.

    ### Correctness Verification Across Sizes
    To verify numerical correctness across all four sizes, we generated greedy reference completions with Hugging Face `transformers`
    and matched output token IDs exactly:
    """)
    return


@app.cell
def _(pl):
    correctness_df = pl.DataFrame(
        {
            "size": ["0.8B", "2B", "4B", "9B"],
            "generated_ids": [
                "[11751, 13, 198, 760, 6511, 314]",
                "[11751, 13, 198, 32, 13, 2912]",
                "[11751, 13, 198, 32, 13, 2912]",
                "[11751, 13, 198, 760, 6511, 314]",
            ],
            "matches_hf_reference": ["Exact match", "Exact match", "Exact match", "Exact match (untied lm_head)"],
        }
    )
    return (correctness_df,)


@app.cell
def _(correctness_df, mo):
    mo.ui.table(correctness_df, selection=None, label="Token Output Verification vs. Hugging Face Reference")
    return


@app.cell
def _(json, pathlib):
    _repo_root = pathlib.Path(__file__).resolve().parent.parent
    multi_scorecard = json.loads(
        (_repo_root / "results/benchmarks/multi_size_engine_comparison_scorecard.json").read_text()
    )
    return (multi_scorecard,)


@app.cell
def _(multi_scorecard, pl):
    _order = ["0.8B", "2B", "4B", "9B"]
    _rows = []
    for _size in _order:
        _s = multi_scorecard["sizes"][_size]
        for _engine_key, _label in [("llamacpp", "llama.cpp"), ("ollama", "Ollama"), ("runtime_next", "runtime-next (this port)")]:
            _rows.append({
                "size": _size,
                "engine": _label,
                "avg_tok_s": _s[f"avg_{_engine_key}_tok_s"],
                "avg_ttft_ms": _s["avg_ttft_ms"][_engine_key],
            })
    multi_size_df = pl.DataFrame(_rows)
    return (multi_size_df,)


@app.cell
def _(alt, mo, multi_size_df):
    _chart = (
        alt.Chart(multi_size_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("size:N", title="Model Size", sort=["0.8B", "2B", "4B", "9B"], axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("engine:N", sort=["runtime-next (this port)", "Ollama", "llama.cpp"]),
            y=alt.Y("avg_tok_s:Q", title="Streaming tok/s"),
            color=alt.Color(
                "engine:N",
                title="Engine",
                scale=alt.Scale(
                    domain=["runtime-next (this port)", "Ollama", "llama.cpp"],
                    range=["#0ea5e9", "#64748b", "#94a3b8"],
                ),
            ),
            tooltip=["size", "engine", alt.Tooltip("avg_tok_s:Q", format=".1f")],
        )
        .properties(
            width=480,
            height=250,
            title="Throughput Across Qwen 3.5 Model Sizes",
            autosize=alt.AutoSizeParams(type="fit", contains="padding"),
        )
    )
    mo.vstack([_chart])
    return


@app.cell
def _(mo):
    mo.md(r"""
    `runtime-next` maintains higher throughput across all four sizes, though the relative margin narrows at larger scales:
    """)
    return


@app.cell
def _(multi_scorecard, pl):
    _order = ["0.8B", "2B", "4B", "9B"]
    _rows = [
        {
            "size": _size,
            "vs llama.cpp": multi_scorecard["sizes"][_size]["runtime_next_speedup_vs_llamacpp"],
            "vs Ollama": multi_scorecard["sizes"][_size]["runtime_next_speedup_vs_ollama"],
        }
        for _size in _order
    ]
    speedup_by_size_df = pl.DataFrame(_rows).unpivot(
        index="size", on=["vs llama.cpp", "vs Ollama"], variable_name="baseline", value_name="speedup"
    )
    return (speedup_by_size_df,)


@app.cell
def _(alt, mo, speedup_by_size_df):
    _chart = (
        alt.Chart(speedup_by_size_df)
        .mark_line(point=alt.OverlayMarkDef(size=90, filled=True), strokeWidth=3)
        .encode(
            x=alt.X("size:N", title="Model size", sort=["0.8B", "2B", "4B", "9B"]),
            y=alt.Y("speedup:Q", title="Speedup (×)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "baseline:N",
                title=None,
                scale=alt.Scale(domain=["vs llama.cpp", "vs Ollama"], range=["#f59e0b", "#64748b"]),
            ),
            tooltip=["size", "baseline", alt.Tooltip("speedup:Q", format=".3f")],
        )
        .properties(
            width=480,
            height=240,
            title="Speedup Margin Across Model Sizes",
            autosize=alt.AutoSizeParams(type="fit", contains="padding"),
        )
    )
    mo.vstack([
        _chart,
        mo.md(
            "*1.28× at 0.8B, tapering to 1.15× at 9B (vs. llama.cpp) as memory bandwidth dominates.*"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    To understand why the performance margin narrows at 9B, we profiled decode kernel execution using `rocprofv3`:
    """)
    return


@app.cell
def _(pl):
    kernel_share_by_size_df = pl.DataFrame(
        {
            "size": ["0.8B", "4B", "9B"],
            "gemv (weight-read GEMVs)": [72.66, 88.85, 93.64],
            "gdn_recurrent (GDN kernel)": [13.59, 5.40, 3.02],
            "other ops (attn, conv1d, gates, RoPE, norms)": [13.75, 5.75, 3.34],
        }
    ).unpivot(index="size", variable_name="kernel_group", value_name="pct_of_decode_time")
    return (kernel_share_by_size_df,)


@app.cell
def _(alt, kernel_share_by_size_df, mo):
    _chart = (
        alt.Chart(kernel_share_by_size_df)
        .mark_bar()
        .encode(
            x=alt.X("size:N", title="Model Size", sort=["0.8B", "4B", "9B"], axis=alt.Axis(labelAngle=0)),
            y=alt.Y("pct_of_decode_time:Q", title="% of Decode Kernel Time", stack="normalize", axis=alt.Axis(format="%")),
            color=alt.Color(
                "kernel_group:N",
                title=None,
                scale=alt.Scale(
                    domain=["gemv (weight-read GEMVs)", "gdn_recurrent (GDN kernel)", "other ops (attn, conv1d, gates, RoPE, norms)"],
                    range=["#0ea5e9", "#f59e0b", "#94a3b8"],
                ),
            ),
            tooltip=["size", "kernel_group", alt.Tooltip("pct_of_decode_time:Q", format=".1f")],
        )
        .properties(
            width=440,
            height=240,
            title="Kernel Execution Breakdown by Model Size",
            autosize=alt.AutoSizeParams(type="fit", contains="padding"),
        )
    )
    mo.vstack([_chart])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Two architectural factors explain this trend:
    1. **GEMV dominates execution time at larger sizes**: Weight matrix sizes scale with `hidden_size × intermediate_size`
       (1,024 at 0.8B $\to$ 4,096 at 9B). GDN recurrence cost, by contrast, depends on `num_heads × head_dim²` and remains
       comparatively constant within size tiers.
    2. **Memory bandwidth saturation**: At 9B, decode is heavily memory-bandwidth bound (GEMV consumes over 93% of execution time).
       Both runtimes operate near the theoretical bandwidth ceiling of the GPU (960 GB/s), naturally compressing the margin.

    Time-To-First-Token advantages remain substantial across all sizes (e.g. **80.6 ms vs. 160.5 ms / 189.9 ms at 9B**),
    as the prefill batching and HTTP streaming fixes apply uniformly regardless of parameter scale.
    """)
    return


@app.cell
def _(pl, scorecard):
    task_rows = []
    for _engine, _tasks in scorecard["tasks"].items():
        _display_engine = {
            "runtime_next": "runtime-next (this port)",
            "llamacpp": "llama.cpp",
            "ollama": "Ollama",
        }.get(_engine, _engine)
        for _t in _tasks:
            task_rows.append({
                "engine": _display_engine,
                "task": _t["name"],
                "tokens": _t["tokens"],
                "ttft_ms": round(_t["ttft_ms"], 1),
                "tok_per_sec": round(_t["tok_per_sec"], 2),
            })
    scorecard_tasks_df = pl.DataFrame(task_rows)
    return (scorecard_tasks_df,)


@app.cell
def _(mo, scorecard_tasks_df):
    mo.vstack([
        mo.md(
            """
            ---
            ## The Complete Final Scorecard

            Task-by-task results across all three independent HTTP server daemons:
            """
        ),
        mo.ui.table(
            scorecard_tasks_df,
            selection=None,
            label="Head-to-Head Per-Task Benchmark Results (HTTP + SSE Streaming)",
        ),
        mo.md(
            """
            Across each task, `runtime-next` achieved the lowest Time-To-First-Token and the highest sustained throughput.
            """
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Performance Analysis: The Hardware Ceiling

    While **83.0 tok/s** outperforms `llama.cpp` (71.3 tok/s) and Ollama (72.0 tok/s), it is below the initial 100 tok/s aspiration.

    ### Hardware Factors
    1. **Single-Block Thread Limit**: The 4-way split attention kernel uses `head_dim (256) × 4 = 1,024` threads per block,
       which reaches the physical limit per compute block on AMD RDNA3. Moving to `kv_split = 8` requires a multi-block
       launch architecture (full Flash-Decode with separate partial reduction buffers).
    2. **GDN Recurrence Cost**: The 8 full-attention layers now execute in ~16 μs per token. The **24 GDN layers** represent
       the remaining decode execution floor. Further gains will require deeper fusion across the GDN update step and RMSNorm.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Key Engineering Takeaways

    1. **Host-Side Overhead Can Dwarf GPU Execution**:
       The initial prototype's 3,300 kernel launches spent more time in CPU driver queues than in GPU execution.
       Batching operations to minimize host-to-device dispatch is as critical as optimizing GEMM kernels.
    2. **Default Buffering in HTTP Crates Traps Streaming**:
       A standard 8KB buffer in an HTTP library added 540 ms of latency by holding the first 55 tokens.
       Streaming requires direct socket control with per-frame flushing.
    3. **Empirical Isolation Over Intuitive Tradeoffs**:
       When throughput dropped from 90 to 79 tok/s, our initial hypothesis blamed per-token network flushing.
       An in-process isolation test proved network overhead was under 0.3%, preventing the introduction of unnecessary
       flush coalescing buffers.
    4. **Consumer Hardware Performance**:
       With dedicated HIP kernels and low-overhead Rust architecture, a consumer AMD GPU can provide high-throughput,
       low-latency inference on modern hybrid models.
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **Precision Note**: Every benchmark in this article used **unquantized BF16 weights** across all three engines
            for direct comparability. Weight quantization (such as INT4/INT8) involves a distinct set of kernels and tradeoffs,
            which will be covered in Part 2.
            """
        ),
        kind="warn",
    )
    return


if __name__ == "__main__":
    app.run()
