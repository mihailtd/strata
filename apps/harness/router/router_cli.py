"""CLI bridge for the DSH moa-router-tool plugin.

Invoked by the JS plugin as a subprocess or fallback:
    uv run python -m apps.harness.router.router_cli "<prompt>"

Outputs a single JSON object to stdout. Exits 0 on success, 1 on error.
No torch / GPU deps — pure Python registry matching and HTTP sidecar probe.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "apps") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "apps"))

from apps.factory.decision_engine.registry import AdapterRegistry


def route_via_service(
    prompt: str,
    url: str = "http://127.0.0.1:8100/route",
    timeout: float = 0.10,
) -> dict | None:
    """Attempt fast neural decision routing via local Decision Service daemon."""
    try:
        data = json.dumps({"prompt": prompt}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                res = json.loads(resp.read().decode("utf-8"))
                res["engine"] = "neural_system_one_sidecar"
                return res
    except Exception:
        pass
    return None


def route(prompt: str) -> dict:
    t0 = time.perf_counter()

    # Fast path: Check if local System One Decision Service daemon is running
    neural_result = route_via_service(prompt)
    if neural_result is not None:
        return neural_result

    # Offline fallback path: Pure-Python dynamic matching using AdapterRegistry (zero torch deps)
    registry = AdapterRegistry()
    active_weights = registry.match_offline_keywords(prompt)

    if not active_weights:
        return {
            "is_multi_expert": False,
            "experts": {"general": 1.0},
            "recommended_model_id": registry.fallback_general_model,
            "active_domains": [],
            "system_prompt_prefix": registry.format_system_prompt({"general": 1.0}),
            "rationale": "Offline router fallback: No domain specialist match — using base general engine.",
            "routing_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "engine": "offline_registry_fallback",
        }

    is_multi = len(active_weights) >= 2
    sorted_domains = sorted(active_weights.keys())
    model_id = registry.resolve_model(sorted_domains)
    sys_prefix = registry.format_system_prompt(active_weights)

    if is_multi:
        detected_str = ", ".join(
            f"{d} (γ={w:.2f})"
            for d, w in sorted(active_weights.items(), key=lambda x: -x[1])
        )
        rationale = f"Offline dynamic multi-domain detected: {detected_str}."
    else:
        domain = next(iter(active_weights))
        rationale = f"Offline dynamic single domain detected: {domain} (γ=1.00)."

    return {
        "is_multi_expert": is_multi,
        "experts": active_weights,
        "recommended_model_id": model_id,
        "active_domains": sorted_domains,
        "system_prompt_prefix": sys_prefix,
        "rationale": rationale,
        "routing_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "engine": "offline_registry_fallback",
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: router_cli.py <prompt>"}))
        sys.exit(1)
    prompt = " ".join(sys.argv[1:])
    try:
        result = route(prompt)
        print(json.dumps(result))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
