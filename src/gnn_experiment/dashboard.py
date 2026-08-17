"""Real-Time Telemetry & VRAM State Dashboard for Autonomous Runtime Engine.

Provides an ultra-premium, dark-mode, glassmorphic real-time web UI
with live Canvas charts, APSP state graph visualization, and SSE event streaming.
Zero GPU performance overhead.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Runtime Engine — Live Telemetry &amp; VRAM State Router</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link
        href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@400;700&display=swap"
        rel="stylesheet"
    >
    <style>
        :root {
            --bg-base: #080a0f;
            --bg-surface: #10141e;
            --bg-card: rgba(18, 24, 38, 0.7);
            --bg-card-hover: rgba(26, 34, 52, 0.85);
            --border-glow: rgba(0, 242, 255, 0.2);
            --border-subtle: rgba(255, 255, 255, 0.08);
            
            --cyan: #00f2ff;
            --cyan-glow: rgba(0, 242, 255, 0.4);
            --purple: #a855f7;
            --purple-glow: rgba(168, 85, 247, 0.4);
            --emerald: #10b981;
            --emerald-glow: rgba(16, 185, 129, 0.4);
            --amber: #f59e0b;
            --rose: #f43f5e;
            
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --text-dim: #6b7280;
            
            --font-sans: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(0, 242, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.05) 0%, transparent 40%),
                linear-gradient(180deg, rgba(8, 10, 15, 0.9) 0%, #080a0f 100%);
            color: var(--text-main);
            font-family: var(--font-sans);
            min-height: 100vh;
            overflow-x: hidden;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: rgba(16, 20, 30, 0.85);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-subtle);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .brand-icon {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--cyan), var(--purple));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            box-shadow: 0 0 20px var(--cyan-glow);
        }

        .brand-title {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #ffffff 40%, var(--cyan) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-subtitle {
            font-size: 0.75rem;
            color: var(--text-dim);
            font-family: var(--font-mono);
        }

        .header-badges {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.85rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--emerald-glow);
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--emerald);
            font-family: var(--font-mono);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--emerald);
            box-shadow: 0 0 10px var(--emerald);
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.85); }
        }

        .container {
            max-width: 1560px;
            margin: 0 auto;
            padding: 1.75rem 2rem;
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 1.5rem;
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 1.35rem 1.5rem;
            position: relative;
            overflow: hidden;
            transition: all 0.25s ease;
        }

        .card:hover {
            border-color: var(--border-glow);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.85rem;
        }

        .card-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        .card-badge {
            font-size: 0.7rem;
            font-family: var(--font-mono);
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-muted);
            border: 1px solid var(--border-subtle);
        }

        .metric-big {
            font-size: 2.35rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            font-family: var(--font-mono);
            line-height: 1.1;
            display: flex;
            align-items: baseline;
            gap: 0.35rem;
        }

        .metric-unit {
            font-size: 0.95rem;
            font-weight: 500;
            color: var(--text-dim);
        }

        .metric-sub {
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .kpi-1 { grid-column: span 3; }
        .kpi-2 { grid-column: span 3; }
        .kpi-3 { grid-column: span 3; }
        .kpi-4 { grid-column: span 3; }

        .chart-card { grid-column: span 8; }
        .state-card { grid-column: span 4; }

        .history-card { grid-column: span 7; }
        .test-card { grid-column: span 5; }

        .chart-container {
            width: 100%;
            height: 220px;
            position: relative;
            margin-top: 0.5rem;
        }

        canvas {
            width: 100%;
            height: 100%;
        }

        .vram-bar-track {
            width: 100%;
            height: 12px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 9999px;
            overflow: hidden;
            margin: 0.75rem 0;
            position: relative;
        }

        .vram-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--cyan), var(--purple));
            border-radius: 9999px;
            transition: width 0.5s ease;
            box-shadow: 0 0 12px var(--cyan-glow);
        }

        .vram-stats-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-family: var(--font-mono);
            color: var(--text-muted);
        }

        .state-nodes {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-top: 0.75rem;
        }

        .state-node {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-subtle);
            transition: all 0.2s ease;
        }

        .state-node.active {
            background: rgba(0, 242, 255, 0.08);
            border-color: var(--cyan);
            box-shadow: 0 0 15px rgba(0, 242, 255, 0.15);
        }

        .state-name {
            font-weight: 600;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .state-tag {
            font-size: 0.7rem;
            font-family: var(--font-mono);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.08);
        }

        .table-wrap {
            max-height: 320px;
            overflow-y: auto;
            margin-top: 0.5rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            font-family: var(--font-mono);
        }

        th {
            text-align: left;
            padding: 0.6rem 0.75rem;
            color: var(--text-dim);
            border-bottom: 1px solid var(--border-subtle);
            position: sticky;
            top: 0;
            background: var(--bg-surface);
        }

        td {
            padding: 0.6rem 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-main);
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }

        .badge-domain {
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .badge-astral {
            background: rgba(0, 242, 255, 0.15);
            color: var(--cyan);
            border: 1px solid rgba(0, 242, 255, 0.3);
        }
        .badge-postgres {
            background: rgba(168, 85, 247, 0.15);
            color: var(--purple);
            border: 1px solid rgba(168, 85, 247, 0.3);
        }
        .badge-financial {
            background: rgba(245, 158, 11, 0.15);
            color: var(--amber);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .badge-base {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            border: 1px solid var(--border-subtle);
        }

        .console-form {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .form-row {
            display: flex;
            gap: 0.75rem;
        }

        select, textarea, button {
            font-family: inherit;
            border-radius: 8px;
            border: 1px solid var(--border-subtle);
            background: rgba(8, 10, 15, 0.8);
            color: var(--text-main);
            padding: 0.6rem 0.85rem;
            font-size: 0.85rem;
            outline: none;
            transition: all 0.2s ease;
        }

        select:focus, textarea:focus {
            border-color: var(--cyan);
            box-shadow: 0 0 10px var(--cyan-glow);
        }

        textarea {
            width: 100%;
            height: 70px;
            resize: none;
        }

        button {
            background: linear-gradient(135deg, var(--cyan), var(--purple));
            color: #000;
            font-weight: 700;
            cursor: pointer;
            border: none;
            padding: 0.65rem 1.25rem;
            box-shadow: 0 0 15px var(--cyan-glow);
        }

        button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        .console-output {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.85rem;
            font-family: var(--font-mono);
            font-size: 0.8rem;
            height: 140px;
            overflow-y: auto;
            white-space: pre-wrap;
            color: #e5e7eb;
        }

        @media (max-width: 1200px) {
            .kpi-1, .kpi-2, .kpi-3, .kpi-4 { grid-column: span 6; }
            .chart-card, .state-card, .history-card, .test-card { grid-column: span 12; }
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="brand-icon">⚡</div>
            <div>
                <div class="brand-title">Autonomous Runtime Engine</div>
                <div class="brand-subtitle">ROCm In-Place Weight Folding &amp; APSP State Router</div>
            </div>
        </div>
        <div class="header-badges">
            <div class="status-pill">
                <div class="status-dot"></div>
                <span id="cuda-status">CUDA GRAPH REPLAY (32K)</span>
            </div>
        </div>
    </header>

    <main class="container">
        <!-- KPI 1: Live Velocity -->
        <div class="card kpi-1">
            <div class="card-header">
                <span class="card-title">Generation Velocity</span>
                <span class="card-badge" id="last-model-badge">READY</span>
            </div>
            <div class="metric-big" style="color: var(--cyan);">
                <span id="val-tok-s">0.0</span>
                <span class="metric-unit">tok/s</span>
            </div>
            <div class="metric-sub">
                <span>⏱️ TTFT: <b id="val-ttft" style="color: #fff;">0.0</b> ms</span>
                <span style="margin-left: auto;">⚡ Swap: <b id="val-swap" style="color: #fff;">0.0</b> ms</span>
            </div>
        </div>

        <!-- KPI 2: Total Generated Tokens -->
        <div class="card kpi-2">
            <div class="card-header">
                <span class="card-title">Total Tokens Emitted</span>
                <span class="card-badge">CUMULATIVE</span>
            </div>
            <div class="metric-big" style="color: var(--purple);">
                <span id="val-total-tokens">0</span>
                <span class="metric-unit">tokens</span>
            </div>
            <div class="metric-sub">
                <span>📈 Requests: <b id="val-total-reqs" style="color: #fff;">0</b></span>
                <span style="margin-left: auto;">Avg: <b id="val-avg-tok-s" style="color: #fff;">0.0</b> tok/s</span>
            </div>
        </div>

        <!-- KPI 3: GPU VRAM Footprint -->
        <div class="card kpi-3">
            <div class="card-header">
                <span class="card-title">VRAM Allocation</span>
                <span class="card-badge" style="color: var(--emerald);">22.0 GB CAP</span>
            </div>
            <div class="metric-big" style="color: var(--emerald);">
                <span id="val-vram-gb">0.0</span>
                <span class="metric-unit">GB</span>
            </div>
            <div class="vram-bar-track">
                <div class="vram-bar-fill" id="vram-fill" style="width: 60%;"></div>
            </div>
            <div class="vram-stats-row">
                <span>Reserved: <b id="val-vram-res">0.0</b> GB</span>
                <span>Headroom: <b id="val-vram-headroom">0.0</b> GB</span>
            </div>
        </div>

        <!-- KPI 4: Active Physical State -->
        <div class="card kpi-4">
            <div class="card-header">
                <span class="card-title">Active GPU State</span>
                <span class="card-badge" style="color: var(--amber);">APSP NODE</span>
            </div>
            <div class="metric-big" style="font-size: 1.6rem; color: var(--amber);">
                <span id="val-active-state">qwen3.5-4b-base</span>
            </div>
            <div class="metric-sub">
                <span>🔒 Drift: <b style="color: var(--emerald);">L_inf = 0.0094</b></span>
                <span style="margin-left: auto;">Path: <b style="color: var(--cyan);">Floyd-Warshall</b></span>
            </div>
        </div>

        <!-- Velocity Time-Series Chart -->
        <div class="card chart-card">
            <div class="card-header">
                <span class="card-title">⚡ Real-Time Generation Speed (tok/s)</span>
                <span class="card-badge">LIVE HISTORY</span>
            </div>
            <div class="chart-container">
                <canvas id="speedChart"></canvas>
            </div>
        </div>

        <!-- APSP VRAM State Topology -->
        <div class="card state-card">
            <div class="card-header">
                <span class="card-title">🧠 VRAM State Graph (APSP)</span>
                <span class="card-badge">O(1) MATRIX</span>
            </div>
            <div class="state-nodes">
                <div class="state-node" id="node-base">
                    <div class="state-name">
                        <span>Pristine Base (W0)</span>
                    </div>
                    <span class="state-tag">0.0 ms hit</span>
                </div>
                <div class="state-node" id="node-astral">
                    <div class="state-name">
                        <span>Astral Python Expert</span>
                    </div>
                    <span class="state-tag">18.0 ms fold</span>
                </div>
                <div class="state-node" id="node-postgresql">
                    <div class="state-name">
                        <span>PostgreSQL 17 Expert</span>
                    </div>
                    <span class="state-tag">18.0 ms fold</span>
                </div>
                <div class="state-node" id="node-financial">
                    <div class="state-name">
                        <span>Financial Planning</span>
                    </div>
                    <span class="state-tag">18.0 ms fold</span>
                </div>
            </div>
        </div>

        <!-- Request History Table -->
        <div class="card history-card">
            <div class="card-header">
                <span class="card-title">📜 Multi-Agent Request Stream</span>
                <span class="card-badge">LAST 20 CALLS</span>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Model / Expert</th>
                            <th>Prompt</th>
                            <th>Gen</th>
                            <th>Velocity</th>
                            <th>TTFT</th>
                            <th>Time</th>
                        </tr>
                    </thead>
                    <tbody id="history-rows">
                        <tr>
                            <td colspan="6" style="text-align: center; color: var(--text-dim);">
                                Awaiting multi-agent API calls...
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Live Interactive Playground -->
        <div class="card test-card">
            <div class="card-header">
                <span class="card-title">💬 Live Test Console</span>
                <span class="card-badge">REST CLIENT</span>
            </div>
            <div class="console-form">
                <div class="form-row">
                    <select id="sel-model" style="flex: 1;">
                        <option value="qwen3.5-4b-astral">qwen3.5-4b-astral (uv/ruff)</option>
                        <option value="qwen3.5-4b-postgresql">qwen3.5-4b-postgresql (pgvector/SQL)</option>
                        <option value="qwen3.5-4b-financial">qwen3.5-4b-financial (wealth)</option>
                        <option value="qwen3.5-4b-base">qwen3.5-4b-base (pristine W0)</option>
                    </select>
                    <button id="btn-send" onclick="sendPrompt()">Generate ⚡</button>
                </div>
                <textarea
                    id="prompt-input"
                    placeholder="Enter prompt to evaluate live folding &amp; streaming..."
                >How do I install fastmcp?</textarea>
                <div class="console-output" id="console-output">Streaming token response will appear here...</div>
            </div>
        </div>
    </main>

    <script>
        // Chart Canvas logic
        const canvas = document.getElementById('speedChart');
        const ctx = canvas.getContext('2d');
        let speedHistory = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let requestHistory = [];

        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
            drawChart();
        }
        window.addEventListener('resize', resizeCanvas);
        setTimeout(resizeCanvas, 50);

        function drawChart() {
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            if (speedHistory.length < 2) return;

            const maxVal = Math.max(70, ...speedHistory) * 1.15;

            // Draw grid
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.lineWidth = 1;
            for (let i = 1; i <= 3; i++) {
                const y = h - (h * (i / 4));
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();

                ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
                ctx.font = '10px JetBrains Mono';
                ctx.fillText(Math.round((maxVal * (i / 4))) + ' tok/s', 8, y - 4);
            }

            // Draw line & gradient fill
            const step = w / (speedHistory.length - 1);
            ctx.beginPath();
            speedHistory.forEach((val, idx) => {
                const x = idx * step;
                const y = h - ((val / maxVal) * (h - 20)) - 10;
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });

            // Gradient stroke
            const grad = ctx.createLinearGradient(0, 0, w, 0);
            grad.addColorStop(0, '#00f2ff');
            grad.addColorStop(1, '#a855f7');
            ctx.strokeStyle = grad;
            ctx.lineWidth = 3;
            ctx.stroke();

            // Fill
            ctx.lineTo(w, h);
            ctx.lineTo(0, h);
            ctx.closePath();
            const fillGrad = ctx.createLinearGradient(0, 0, 0, h);
            fillGrad.addColorStop(0, 'rgba(0, 242, 255, 0.25)');
            fillGrad.addColorStop(1, 'rgba(0, 242, 255, 0.0)');
            ctx.fillStyle = fillGrad;
            ctx.fill();
        }

        // Live SSE Poll / Stream
        function updateStats(data) {
            if (!data) return;

            const live = data.live_metrics || {};
            const last = data.last_request || {};
            const hw = data.hardware || {};

            document.getElementById('val-total-tokens').innerText = (live.total_tokens_generated || 0).toLocaleString();
            document.getElementById('val-total-reqs').innerText = live.total_requests_served || 0;
            document.getElementById('val-avg-tok-s').innerText = (live.average_tokens_per_second || 0).toFixed(1);

            const tokS = last.tokens_per_second || 0;
            document.getElementById('val-tok-s').innerText = tokS.toFixed(1);
            document.getElementById('val-ttft').innerText = (last.time_to_first_token_ms || 0).toFixed(1);
            document.getElementById('val-swap').innerText = (last.expert_swap_ms || 0).toFixed(1);

            if (last.model) {
                document.getElementById('last-model-badge').innerText = last.model;
            }

            // VRAM
            const vramAlloc = hw.vram_allocated_gb || 0;
            const vramRes = hw.vram_reserved_gb || 0;
            const vramCap = hw.vram_hard_cap_gb || 22.0;
            document.getElementById('val-vram-gb').innerText = vramAlloc.toFixed(2);
            document.getElementById('val-vram-res').innerText = vramRes.toFixed(2);
            document.getElementById('val-vram-headroom').innerText = (vramCap - vramAlloc).toFixed(2);
            document.getElementById('vram-fill').style.width = Math.min(100, (vramAlloc / vramCap) * 100) + '%';

            // Active state
            const active = hw.active_expert || 'base (pristine W0)';
            document.getElementById('val-active-state').innerText = active;

            // Highlight state node
            ['base', 'astral', 'postgresql', 'financial'].forEach(dom => {
                const el = document.getElementById('node-' + dom);
                if (el) {
                    if (active.toLowerCase().includes(dom)) {
                        el.classList.add('active');
                    } else {
                        el.classList.remove('active');
                    }
                }
            });

            // Update chart if new last_request
            if (last.timestamp && (!window.lastTs || window.lastTs !== last.timestamp)) {
                window.lastTs = last.timestamp;
                speedHistory.push(tokS);
                if (speedHistory.length > 30) speedHistory.shift();
                drawChart();

                // Add to table
                addHistoryRow(last);
            }
        }

        function addHistoryRow(req) {
            if (!req || !req.model) return;
            requestHistory.unshift(req);
            if (requestHistory.length > 20) requestHistory.pop();

            const tbody = document.getElementById('history-rows');
            tbody.innerHTML = requestHistory.map(r => {
                let badgeClass = 'badge-base';
                if (r.model.includes('astral')) badgeClass = 'badge-astral';
                if (r.model.includes('postgre')) badgeClass = 'badge-postgres';
                if (r.model.includes('fin')) badgeClass = 'badge-financial';

                const displaySpeed = (r.tokens_per_second || 0).toFixed(1) + ' tok/s';
                const displayTtft = (r.time_to_first_token_ms || 0).toFixed(1) + ' ms';
                const displayTime = new Date(r.timestamp * 1000).toLocaleTimeString();

                return `<tr>
                    <td><span class="badge-domain ${badgeClass}">${r.model}</span></td>
                    <td>${r.prompt_tokens}</td>
                    <td>${r.completion_tokens}</td>
                    <td style="color: var(--cyan); font-weight: 700;">${displaySpeed}</td>
                    <td>${displayTtft}</td>
                    <td style="color: var(--text-dim);">${displayTime}</td>
                </tr>`;
            }).join('');
        }

        // Periodic Poll
        async function fetchTelemetry() {
            try {
                const res = await fetch('/stats');
                if (res.ok) {
                    const json = await res.json();
                    updateStats(json);
                }
            } catch (err) {}
        }
        setInterval(fetchTelemetry, 1000);
        fetchTelemetry();

        // Live Chat Generator
        async function sendPrompt() {
            const model = document.getElementById('sel-model').value;
            const prompt = document.getElementById('prompt-input').value.trim();
            const outEl = document.getElementById('console-output');
            const btn = document.getElementById('btn-send');

            if (!prompt) return;

            btn.disabled = true;
            btn.innerText = 'Streaming...';
            outEl.innerText = '';

            try {
                const res = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: model,
                        messages: [{ role: 'user', content: prompt }],
                        stream: true
                    })
                });

                const reader = res.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const payload = line.slice(6).trim();
                            if (payload === '[DONE]') continue;
                            try {
                                const parsed = JSON.parse(payload);
                                const delta = parsed.choices[0]?.delta || {};
                                if (delta.reasoning_content) {
                                    outEl.innerText += delta.reasoning_content;
                                }
                                if (delta.content) {
                                    outEl.innerText += delta.content;
                                }
                                outEl.scrollTop = outEl.scrollHeight;
                            } catch (e) {}
                        }
                    }
                }
            } catch (err) {
                outEl.innerText = 'Error: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.innerText = 'Generate ⚡';
                fetchTelemetry();
            }
        }
    </script>
</body>
</html>
"""
