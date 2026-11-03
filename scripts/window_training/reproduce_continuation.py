"""Regenerate original window fits, their continuation, reports and verification.

Inputs: graph seeds/configuration and pinned numerical dependencies in the repo.
Outputs: fresh windows/ and continuation/ trees under the requested destination.
No archived factors or Gensim trajectory references are needed.
Run: python3 scripts/window_training/reproduce_continuation.py --output /fresh/path
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from window_training import continue_windows as continuation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("At least one worker is required.")
    destination = args.output.resolve()
    scripts = Path(__file__).resolve().parent
    reference, output = destination / "windows", destination / "continuation"
    subprocess.run(
        [
            sys.executable,
            str(scripts / "expected_window.py"),
            "--output",
            str(reference),
        ],
        check=True,
    )
    continuation.run(
        continuation.Config(reference=reference, output=output),
        list(continuation.Config.seeds),
        args.workers,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "report_continuation.py"),
            "--output",
            str(output),
        ],
        check=True,
    )
    verified = subprocess.run(
        [
            sys.executable,
            str(scripts / "verify_continuation.py"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (output / "report" / "verification.json").write_text(verified.stdout)


if __name__ == "__main__":
    main()
