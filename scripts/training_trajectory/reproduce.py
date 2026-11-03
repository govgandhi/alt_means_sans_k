"""Reproduce the baseline and trajectories in a fresh environment/output directory.

Install the pinned runtime and build the checked Gensim extension first. Then:
python3 scripts/training_trajectory/reproduce.py --output /fresh/path

This regenerates the local reference before the trajectories, so a platform's
newly compiled binary need not match the archived macOS binary byte for byte.
Within the fresh run, exact-prefix, runtime, source and output checks still apply.
Completed invocations verify the existing files instead of retraining.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import run as experiment
from threadpoolctl import threadpool_limits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    reference = destination / "subsampling_origin"
    trajectories = destination / "training_trajectory"
    with threadpool_limits(limits=1):
        experiment.baseline.run(experiment.baseline.Config(output=reference))
        experiment.run(experiment.Config(output=trajectories, reference=reference))
    for script in ("report.py", "verify.py"):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name(script)),
                "--output",
                str(trajectories),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
