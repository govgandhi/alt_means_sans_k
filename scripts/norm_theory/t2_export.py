"""Export PR #36 objective arrays and Gensim initial factors for the modified-loss flow.

For seeds 0-4: positive pair counts per epoch P (input a, output b), expected negative
counts N (as in pr36_common.setup), negative proposal q, degree, and the Gensim initial
center/context vectors used by every clock run (clock_replica.initial).
Output: results/norm_theory/t2_setup_s<seed>.npz
Run: .venv/bin/python -I scripts/norm_theory/t2_export.py
"""

import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/norm_tests"))
import pr36_common  # noqa: E402, F401
from clock_replica import initial  # noqa: E402
from pr36_common import T, setup  # noqa: E402


def main() -> None:
    out = ROOT / "results/norm_theory"
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        for seed in range(5):
            s = setup(seed)
            W, C = initial(s, seed)
            np.savez_compressed(
                out / f"t2_setup_s{seed}.npz",
                P=s["positive"],
                N=s["negative"],
                q=s["proposal"],
                degree=s["degree"],
                W0=W,
                C0=C,
                negatives=T.negatives,
            )
            print(f"seed {seed}", flush=True)


if __name__ == "__main__":
    main()
