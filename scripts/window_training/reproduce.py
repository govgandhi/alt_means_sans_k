"""Rebuild both exploration lanes and their local PR36 references from scratch.

Install the pinned requirements and patched Gensim extension first (see results
README). Run: python3 scripts/window_training/reproduce.py --output /fresh/path
All commands are checked, deterministic, and resumable under the same source and
runtime. A fresh platform rebuilds its references instead of trusting archived
binary hashes. This includes the complete earlier baseline/trajectory sweeps.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script: Path, *arguments: str) -> None:
    subprocess.run([sys.executable, str(script), *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    scripts = Path(__file__).resolve().parents[1]
    references = destination / "references"
    run_script(
        scripts / "training_trajectory/reproduce.py", "--output", str(references)
    )
    run_script(
        scripts / "window_training/schedule_fork.py",
        "--output",
        str(destination / "schedule"),
        "--reference",
        str(references / "training_trajectory"),
    )
    run_script(
        scripts / "window_training/expected_window.py",
        "--output",
        str(destination / "windows"),
    )
    for name in ("report.py", "verify.py"):
        run_script(scripts / "window_training" / name, "--output", str(destination))


if __name__ == "__main__":
    main()
