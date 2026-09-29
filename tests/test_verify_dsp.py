"""
The narrative verification script must keep passing.

tools/verify_dsp.py is a deliverable in its own right — it prints a readable
proof that the filters do what the drawn curve claims. Running it here means a
regression fails CI rather than waiting for someone to run it by hand.
"""

from __future__ import annotations

import subprocess
import sys


def test_verify_dsp_reports_all_checks_passing(root):
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "verify_dsp.py")],
        capture_output=True, text=True, cwd=root, check=False,
    )
    assert result.returncode == 0, (
        f"verify_dsp.py reported failures:\n{result.stdout}\n{result.stderr}"
    )
    assert "ALL DSP CHECKS PASSED" in result.stdout
    assert "[FAIL]" not in result.stdout
