"""One-command refresh: fetch -> compact -> rebuild snapshots."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str):
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    run("fetch_data.py")
    run("compact_data.py")
    run("build_snapshots.py")
    run("self_check.py")
    print("\nAll data refreshed. Start the app with: streamlit run app.py")
