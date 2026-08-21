"""Automated test validating JavaScript syntax and tab structure in DASHBOARD_HTML."""

import re
import subprocess
import pytest
from runtime.dashboard import DASHBOARD_HTML


def test_dashboard_tabs_and_elements_exist():
    """Verify all tabs and view panel IDs exist in the HTML."""
    expected_ids = [
        "tab-telemetry",
        "tab-chat",
        "tab-training",
        "view-telemetry",
        "view-chat",
        "view-training",
        "btn-engine-toggle",
        "engine-status-pill",
        "dag-hit-rate",
        "dag-saved-ms",
        "dag-edges-tbody",
    ]
    for element_id in expected_ids:
        assert element_id in DASHBOARD_HTML, f"Missing {element_id} in DASHBOARD_HTML"


def test_dashboard_javascript_syntax():
    """Extracts all <script> blocks from DASHBOARD_HTML and verifies syntax using node --check."""
    script_matches = re.findall(r"<script>(.*?)</script>", DASHBOARD_HTML, re.DOTALL)
    assert len(script_matches) > 0, "No <script> block found in DASHBOARD_HTML"

    full_js = "\n".join(script_matches)
    
    # Run node --check via stdin
    res = subprocess.run(
        ["node", "--check"],
        input=full_js,
        text=True,
        capture_output=True,
    )
    assert res.returncode == 0, f"JavaScript SyntaxError detected in DASHBOARD_HTML:\n{res.stderr}"
