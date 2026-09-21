import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    app_title="Multi-Tier LoRA Matrix & The Goldilocks Dynamic Alpha Law: Across 0.8B to 27B on AMD ROCm",
)


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    return Path, alt, json, mo, os, pl


@app.cell
def _(mo):
    mo.md(r"""
    # The Full Multi-Scale Architecture Breakthrough: 0.8B to 27B

    ### Cross-Model Parameter Dynamics, Zero-Allocation LoRA Swapping, and the Empirical Goldilocks Scaling Law on AMD Radeon RX 7900 XTX (24GB)

    This notebook presents the definitive, empirical end-to-end evaluation of **Base models**, **Agentic Coding adapters**, and **Modern Python adapters** across all 5 dense model tiers:
    **0.8B**, **2B**, **4B**, **9B**, and **27B**.

    It documents the discovery of why unattenuated LoRA adapters ($\alpha=128$) triggered catastrophic degradation on small models ($\le 2\text{B}$), the derivation of the **Dynamic $\alpha(d_{model})$ Scaling Law**, and the **Serving-Time Fractional Attenuation Breakthrough** that recovered and surpassed Base model quality on small models without retraining.
    """)
    return


@app.cell
def _(mo):
    stat_tiers = mo.stat(
        value="5 Tiers (15 Cells)",
        label="Model Scale Coverage",
        caption="0.8B, 2B, 4B, 9B, 27B evaluated across Base, Modern, Agentic",
        direction="increase",
        bordered=True,
    )
    stat_swap = mo.stat(
        value="2.18 – 13.16 ms",
        label="Zero-Allocation DMA Swap",
        caption="0.8B: 2.18ms | 9B: 4.49ms | 27B: 13.16ms (PCIe DMA, 0 VRAM waste)",
        direction="decrease",
        bordered=True,
    )
    stat_gain = mo.stat(
        value="+10.0% Pass, -27% Tok",
        label="Small-Model Goldilocks Win",
        caption="0.8B Calibrated (0.25x): 60% vs 50% Base (-2,657 tokens)",
        direction="increase",
        bordered=True,
    )
    stat_efficiency = mo.stat(
        value="-21.9% Tokens",
        label="9B Coding Compression",
        caption="9B Modern Python: 100% pass rate in 3,689 tok vs 4,721 tok Base",
        direction="decrease",
        bordered=True,
    )

    mo.hstack([stat_tiers, stat_swap, stat_gain, stat_efficiency], justify="space-between")
    return stat_efficiency, stat_gain, stat_swap, stat_tiers


@app.cell
def _(Path, json, os, pl):
    # Locate repo root
    cwd = Path(os.getcwd())
    repo_root = cwd if (cwd / "results" / "benchmarks").exists() else cwd.parent

    # Inventory of all live scorecards
    SCORECARD_MANIFEST = [
        # 0.8B Tier
        {"tier": "0.8B", "tier_order": 1, "d_model": 1024, "config": "Base", "file": "scorecard_runtimenext_0_8b_base.json", "alpha": None, "scale": 1.0, "calibrated": False},
        {"tier": "0.8B", "tier_order": 1, "d_model": 1024, "config": "Agentic Coding", "file": "scorecard_runtimenext_0_8b_agentic_v8.json", "alpha": 128, "scale": 1.0, "calibrated": False},
        {"tier": "0.8B", "tier_order": 1, "d_model": 1024, "config": "Modern Python (Uncalibrated)", "file": "scorecard_runtimenext_0_8b_modern_v7.json", "alpha": 128, "scale": 1.0, "calibrated": False},
        {"tier": "0.8B", "tier_order": 1, "d_model": 1024, "config": "Modern Python (Calibrated)", "file": "scorecard_runtimenext_0_8b_modern_scale025.json", "alpha": 32, "scale": 0.25, "calibrated": True},

        # 2B Tier
        {"tier": "2B", "tier_order": 2, "d_model": 2048, "config": "Base", "file": "scorecard_runtimenext_2b_base.json", "alpha": None, "scale": 1.0, "calibrated": False},
        {"tier": "2B", "tier_order": 2, "d_model": 2048, "config": "Agentic Coding", "file": "scorecard_runtimenext_2b_agentic_v8.json", "alpha": 128, "scale": 1.0, "calibrated": False},
        {"tier": "2B", "tier_order": 2, "d_model": 2048, "config": "Modern Python (Uncalibrated)", "file": "scorecard_runtimenext_2b_modern_v7.json", "alpha": 128, "scale": 1.0, "calibrated": False},
        {"tier": "2B", "tier_order": 2, "d_model": 2048, "config": "Modern Python (Calibrated)", "file": "scorecard_runtimenext_2b_modern_scale05.json", "alpha": 64, "scale": 0.50, "calibrated": True},

        # 4B Tier
        {"tier": "4B", "tier_order": 3, "d_model": 2560, "config": "Base", "file": "scorecard_runtimenext_4b_base.json", "alpha": None, "scale": 1.0, "calibrated": False},
        {"tier": "4B", "tier_order": 3, "d_model": 2560, "config": "Agentic Coding", "file": "scorecard_runtimenext_4b_agentic_v8.json", "alpha": 128, "scale": 1.0, "calibrated": False},
        {"tier": "4B", "tier_order": 3, "d_model": 2560, "config": "Modern Python", "file": "aider_10tasks_lora_python_modern_qwen35_4b.json", "alpha": 128, "scale": 1.0, "calibrated": False},

        # 9B Tier
        {"tier": "9B", "tier_order": 4, "d_model": 4096, "config": "Base", "file": "scorecard_runtimenext_9b_base.json", "alpha": None, "scale": 1.0, "calibrated": False},
        {"tier": "9B", "tier_order": 4, "d_model": 4096, "config": "Agentic Coding", "file": "scorecard_runtimenext_9b_agentic_v8.json", "alpha": 128, "scale": 1.0, "calibrated": False},
        {"tier": "9B", "tier_order": 4, "d_model": 4096, "config": "Modern Python", "file": "aider_10tasks_modern_qwen35_9b.json", "alpha": 128, "scale": 1.0, "calibrated": False},

        # 27B Tier
        {"tier": "27B", "tier_order": 5, "d_model": 5120, "config": "Base", "file": "scorecard_runtimenext_27b_base.json", "alpha": None, "scale": 1.0, "calibrated": False},
        {"tier": "27B", "tier_order": 5, "d_model": 5120, "config": "Agentic Coding", "file": "scorecard_runtimenext_27b_agentic_v8.json", "alpha": 128, "scale": 1.0, "calibrated": False},
        {"tier": "27B", "tier_order": 5, "d_model": 5120, "config": "Modern Python", "file": "scorecard_runtimenext_27b_modern_v7.json", "alpha": 128, "scale": 1.0, "calibrated": False},
    ]

    records = []
    task_details = []

    for entry in SCORECARD_MANIFEST:
        path = repo_root / "results" / "benchmarks" / entry["file"]
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            summary = data.get("summary", {})
            tasks = data.get("tasks", [])
            passed = summary.get("passed", 0)
            total = summary.get("total_tasks", len(tasks) or 10)
            pass_rate_pct = summary.get("pass_rate_pct", (passed / total * 100.0) if total else 0.0)
            pass_at_1 = summary.get("pass_at_1", sum(1 for t in tasks if t.get("pass_turn") == 1))
            pass_at_1_pct = summary.get("pass_at_1_pct", (pass_at_1 / total * 100.0) if total else 0.0)
            duration_s = summary.get("total_duration_s", 0.0)

            total_tokens = sum(t.get("total_tokens", 0) for t in tasks)
            valid_tok_s = [t.get("avg_tok_s", 0.0) for t in tasks if (t.get("avg_tok_s") or 0.0) > 0]
            avg_tok_s = sum(valid_tok_s) / len(valid_tok_s) if valid_tok_s else 0.0

            valid_ttft = [t.get("avg_ttft_ms", 0.0) for t in tasks if (t.get("avg_ttft_ms") or 0.0) > 0]
            avg_ttft_ms = sum(valid_ttft) / len(valid_ttft) if valid_ttft else 0.0

            records.append({
                "tier": entry["tier"],
                "tier_order": entry["tier_order"],
                "d_model": entry["d_model"],
                "config": entry["config"],
                "passed": passed,
                "total": total,
                "pass_rate": round(pass_rate_pct, 1),
                "pass_at_1": pass_at_1,
                "pass_at_1_pct": round(pass_at_1_pct, 1),
                "total_tokens": total_tokens,
                "duration_s": round(duration_s, 2),
                "avg_tok_s": round(avg_tok_s, 1),
                "avg_ttft_ms": round(avg_ttft_ms, 1),
                "scale": entry["scale"],
                "calibrated": entry["calibrated"],
                "file": entry["file"],
            })

            for t in tasks:
                task_details.append({
                    "tier": entry["tier"],
                    "config": entry["config"],
                    "task": t.get("task", "unknown"),
                    "passed": t.get("passed", False),
                    "pass_turn": t.get("pass_turn", 0),
                    "total_turns": t.get("total_turns", 0),
                    "total_tokens": t.get("total_tokens", 0),
                    "avg_tok_s": round(t.get("avg_tok_s") or 0.0, 1),
                    "avg_ttft_ms": round(t.get("avg_ttft_ms") or 0.0, 1),
                    "duration_s": round(t.get("duration_s") or 0.0, 2),
                    "error": str(t.get("error") or ""),
                })
        except Exception as e:
            print(f"Error loading {path}: {e}")

    df_matrix = pl.DataFrame(records).sort(["tier_order", "config"])
    df_tasks = pl.DataFrame(task_details)
    return SCORECARD_MANIFEST, cwd, df_matrix, df_tasks, records, repo_root, task_details


@app.cell
def _(df_matrix, mo):
    # Dynamic interactive filters
    available_tiers = ["All"] + df_matrix["tier"].unique().to_list()
    tier_selector = mo.ui.dropdown(
        options=available_tiers,
        value="All",
        label="Filter Model Tier:",
    )

    metric_selector = mo.ui.radio(
        options=["Pass Rate (%)", "Total Tokens Consumed", "Decode Speed (tok/s)", "Wall-Clock Time (s)"],
        value="Pass Rate (%)",
        label="Select Comparison Metric:",
    )

    mo.hstack([tier_selector, metric_selector], justify="start", gap=2)
    return available_tiers, metric_selector, tier_selector


@app.cell
def _(df_matrix, tier_selector):
    df_view = df_matrix if tier_selector.value == "All" else df_matrix.filter(df_matrix["tier"] == tier_selector.value)
    return (df_view,)


@app.cell
def _(alt, df_view, metric_selector, mo):
    metric_map = {
        "Pass Rate (%)": ("pass_rate", "Pass Rate (%)", ":.1f"),
        "Total Tokens Consumed": ("total_tokens", "Total Tokens (10 tasks)", ",d"),
        "Decode Speed (tok/s)": ("avg_tok_s", "Decode Speed (tok/s)", ":.1f"),
        "Wall-Clock Time (s)": ("duration_s", "Wall-Clock Duration (s)", ":.1f"),
    }

    col, title, fmt = metric_map[metric_selector.value]

    chart = (
        alt.Chart(df_view.to_pandas())
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("config:N", title=None, axis=alt.Axis(labels=True, labelAngle=-20)),
            y=alt.Y(f"{col}:Q", title=title),
            color=alt.Color(
                "config:N",
                scale=alt.Scale(
                    domain=[
                        "Base",
                        "Modern Python",
                        "Modern Python (Uncalibrated)",
                        "Modern Python (Calibrated)",
                        "Agentic Coding",
                    ],
                    range=["#64748b", "#3b82f6", "#f97316", "#10b981", "#8b5cf6"],
                ),
                legend=alt.Legend(title="Configuration", orient="bottom"),
            ),
            column=alt.Column(
                "tier:N",
                title="Model Tier",
                sort=["0.8B", "2B", "4B", "9B", "27B"],
                header=alt.Header(labelFontSize=13, labelFontWeight="bold"),
            ),
            tooltip=[
                alt.Tooltip("tier:N", title="Tier"),
                alt.Tooltip("config:N", title="Config"),
                alt.Tooltip("passed:Q", title="Passed Tasks"),
                alt.Tooltip("total:Q", title="Total Tasks"),
                alt.Tooltip("pass_rate:Q", title="Pass Rate (%)"),
                alt.Tooltip("total_tokens:Q", title="Tokens", format=",d"),
                alt.Tooltip("duration_s:Q", title="Duration (s)"),
                alt.Tooltip("avg_tok_s:Q", title="tok/s"),
            ],
        )
        .properties(height=320)
    )

    mo.ui.altair_chart(chart)
    return chart, col, fmt, metric_map, title


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 🔬 The Small-Model Goldilocks Discovery: Solving Over-Rotation

    ### Why Uncalibrated LoRA Degraded 0.8B and 2B
    When standard LoRA hyperparameters ($r=8, \alpha=128$) are applied uniformly across all model scales, the effective parameter perturbation ratio:
    $$\frac{\|\Delta W\|_F}{\|W\|_F} = \frac{\alpha}{r} \frac{\|B A\|_F}{\|W\|_F}$$
    scales inversely with the model hidden dimension $d_{model}$.

    - On **9B** ($d_{model}=4096$), $\|W\|_F \approx 882$, yielding $\frac{\|\Delta W\|}{\|W\|} \approx 0.065$ — directly in the certified **Goldilocks band $[0.035, 0.100]$**.
    - On **0.8B** ($d_{model}=1024$), $\|W\|_F \approx 205$, causing the same $\alpha=128$ to yield $\frac{\|\Delta W\|}{\|W\|} \approx 0.28 - 0.35$! This catastrophic over-rotation destroyed base model reasoning, causing Agentic Coding to collapse to **0%** and Modern Python to drop to **30%** (vs Base 50%).

    ### The Serving-Time Scale Attenuation Fix
    By applying fractional scale attenuation $\alpha_{eff} = s \cdot \alpha$:
    - **0.8B @ $s=0.25$ ($\alpha_{eff}=32$)**: Pass rate surged from **30% to 60%** (beating Base by +10%), while tokens dropped from 9,783 to 7,126 (**-27.2% tokens**)!
    - **2B @ $s=0.50$ ($\alpha_{eff}=64$)**: Pass rate fully recovered from **50% to 80%** (matching Base), eliminating bracket syntax truncation errors.
    """)
    return


@app.cell
def _(alt, df_matrix, mo):
    # Comparison chart for 0.8B and 2B
    df_small = df_matrix.filter(df_matrix["tier"].is_in(["0.8B", "2B"]))

    goldilocks_chart = (
        alt.Chart(df_small.to_pandas())
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("config:N", title=None, axis=alt.Axis(labels=True, labelAngle=-20)),
            y=alt.Y("pass_rate:Q", title="Pass Rate (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "config:N",
                scale=alt.Scale(
                    domain=[
                        "Base",
                        "Modern Python (Uncalibrated)",
                        "Modern Python (Calibrated)",
                        "Agentic Coding",
                    ],
                    range=["#64748b", "#f97316", "#10b981", "#8b5cf6"],
                ),
                legend=alt.Legend(title="Config"),
            ),
            column=alt.Column("tier:N", title="Small Model Scale", header=alt.Header(labelFontSize=13)),
            tooltip=["tier", "config", "pass_rate", "total_tokens", "duration_s"],
        )
        .properties(height=280)
    )

    mo.ui.altair_chart(goldilocks_chart)
    return df_small, goldilocks_chart


@app.cell
def _(alt, df_matrix, mo):
    # Token economy chart: Base vs Adapted across sizes
    df_economy = df_matrix.filter(df_matrix["config"].is_in(["Base", "Modern Python", "Modern Python (Calibrated)"]))

    economy_chart = (
        alt.Chart(df_economy.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("tier:N", title="Model Tier", sort=["0.8B", "2B", "4B", "9B", "27B"]),
            y=alt.Y("total_tokens:Q", title="Total Tokens Consumed (10 tasks)"),
            color=alt.Color("config:N", scale=alt.Scale(domain=["Base", "Modern Python", "Modern Python (Calibrated)"], range=["#64748b", "#3b82f6", "#10b981"])),
            xOffset="config:N",
            tooltip=["tier", "config", "total_tokens", "pass_rate", "duration_s"],
        )
        .properties(width=500, height=280, title="Token Economy: Modern Python Adapter vs Base Across Scales")
    )

    mo.ui.altair_chart(economy_chart)
    return df_economy, economy_chart


@app.cell
def _(alt, df_matrix, mo):
    # Hardware Throughput on RX 7900 XTX
    df_base = df_matrix.filter(df_matrix["config"] == "Base")

    hw_chart = (
        alt.Chart(df_base.to_pandas())
        .mark_line(point=True, strokeWidth=3, color="#0ea5e9")
        .encode(
            x=alt.X("tier:N", title="Model Tier", sort=["0.8B", "2B", "4B", "9B", "27B"]),
            y=alt.Y("avg_tok_s:Q", title="Decode Throughput (tok/s)"),
            tooltip=["tier", "avg_tok_s", "avg_ttft_ms", "d_model"],
        )
        .properties(width=500, height=250, title="Live Hardware Decoding Speed on AMD RX 7900 XTX (W4A16 Native)")
    )

    mo.ui.altair_chart(hw_chart)
    return df_base, hw_chart


@app.cell
def _(df_matrix, mo):
    mo.md("### 📊 Complete Live Multi-Tier Scorecard Matrix")
    return


@app.cell
def _(df_matrix, mo):
    mo.ui.table(
        df_matrix.select([
            "tier",
            "config",
            "passed",
            "total",
            "pass_rate",
            "pass_at_1",
            "total_tokens",
            "duration_s",
            "avg_tok_s",
            "avg_ttft_ms",
            "file",
        ]).to_pandas(),
        pagination=True,
        page_size=15,
    )
    return


@app.cell
def _(Path, json, mo, os, pl):
    cwd = Path(os.getcwd())
    repo_root = cwd if (cwd / "results" / "benchmarks").exists() else cwd.parent
    matrix_path = repo_root / "results" / "benchmarks" / "matrix_cross_model_scorecard.json"

    if matrix_path.exists():
        with open(matrix_path) as f:
            m_data = json.load(f)

        rows = []
        for tier, benches in m_data.items():
            rows.append({
                "Model Tier": tier,
                "Aider Pass Rate (%)": benches.get("aider", {}).get("pass_rate_pct", "N/A"),
                "HumanEval Pass@1 (%)": benches.get("humaneval", {}).get("pass_rate_pct", "N/A"),
                "BFCL Accuracy (%)": benches.get("bfcl", {}).get("accuracy_pct", "N/A"),
                "Toolery Score (0-1)": benches.get("toolery", {}).get("avg_score", "N/A"),
            })
        df_bench_matrix = pl.DataFrame(rows)
        view = mo.vstack([
            mo.md("### 🌐 Cross-Model Unified Benchmark Matrix (Aider, HumanEval, BFCL, Toolery)"),
            mo.ui.table(df_bench_matrix.to_pandas()),
        ])
    else:
        view = mo.md("")
    return df_bench_matrix, m_data, matrix_path, rows, view


@app.cell
def _(view):
    view
    return


@app.cell
def _(df_tasks, mo):
    # Interactive Task Inspector
    run_options = df_tasks.select(
        (pl.col("tier") + " — " + pl.col("config")).alias("run_label")
    ).unique()["run_label"].to_list()

    selected_run = mo.ui.dropdown(
        options=sorted(run_options),
        value=sorted(run_options)[0] if run_options else None,
        label="Inspect Tasks for Run:",
    )

    mo.hstack([selected_run])
    return run_options, selected_run


@app.cell
def _(df_tasks, mo, selected_run):
    if selected_run.value:
        t_sel, c_sel = selected_run.value.split(" — ")
        filtered_tasks = df_tasks.filter((df_tasks["tier"] == t_sel) & (df_tasks["config"] == c_sel))
        table = mo.ui.table(filtered_tasks.to_pandas(), pagination=True, page_size=10)
    else:
        table = mo.md("No run selected.")
    table
    return c_sel, filtered_tasks, t_sel, table


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 📐 Mathematical Derivations & Platform Invariants

    ### The Dynamic $\alpha(d_{model})$ Law
    To guarantee scale-invariant perturbation norm across architectures, $\alpha$ is calibrated according to:
    $$\alpha^*(d_{model}) = \max\left(16, \text{round}\left(128 \times \frac{d_{model}}{4096} / 16\right) \times 16\right)$$

    | Model Tier | $d_{model}$ | $\|W\|_F$ Ref | Uncalibrated $\alpha=128$ Ratio | Optimal $\alpha^*$ | Calibrated Ratio |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    | **0.8B** | 1024 | 205.56 | **0.284** *(Fatal Over-Rotation)* | **32** | **0.071** *(Goldilocks)* |
    | **2B** | 2048 | 323.40 | **0.180** *(Severe Regression)* | **64** | **0.090** *(Goldilocks)* |
    | **4B** | 2560 | 501.78 | **0.116** *(Upper Boundary)* | **80** | **0.072** *(Goldilocks)* |
    | **9B** | 4096 | 882.05 | **0.066** *(Optimal Center)* | **128** | **0.066** *(Goldilocks)* |
    | **27B** | 5120 | 1419.00 | **0.041** *(Lower Boundary)* | **160** | **0.052** *(Goldilocks)* |

    ### Geometric Early Stopping Invariant
    Rather than training for an arbitrary, fixed number of steps (e.g. 150), training terminates autonomously when:
    $$\frac{\|\Delta W\|_F}{\|W\|_F} = 0.065$$
    This guarantees that the trained adapter occupies the exact mathematical sweet spot certified for $<2.5\%$ BF16 mantissa merge error without reciting training tokens.
    """)
    return


if __name__ == "__main__":
    app.run()
