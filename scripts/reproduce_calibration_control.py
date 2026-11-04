"""Reproduce the historical calibration control with its exact original sources.

Inputs: archived source snapshots and the original control's plan/input artifacts.
Outputs: a fresh diagnostic directory supplied with --output. No current code or
data is replaced. Run in the pinned corrected environment:
python3 scripts/reproduce_calibration_control.py --output /tmp/fresh-control
Use --seeds 0 for the one-graph reproducibility check. All copied inputs are checked
against the recorded hashes before invoking the historical script in a temporary root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def prepare(root: Path, destination: Path) -> None:
    snapshot = root / "archive/negative_sampling_control/calibration_source"
    plan = json.loads(
        (root / "results/negative_sampling_control/plan.json").read_text()
    )
    for name, expected in plan["source_sha256"].items():
        source = (snapshot if name.startswith("scripts/") else root) / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Historical input hash mismatch: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory.")
    root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="calibration-control-") as directory:
        temporary = Path(directory)
        prepare(root, temporary)
        command = [
            sys.executable,
            str(temporary / "scripts/negative_sampling_control.py"),
            "--output",
            str(output),
        ]
        if args.seeds is not None:
            command += ["--seeds", *map(str, args.seeds)]
        subprocess.run(command, cwd=temporary, check=True)


if __name__ == "__main__":
    main()
