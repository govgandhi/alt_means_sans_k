# /// script
# dependencies = ["numpy", "numba", "networkx"]
# ///
"""Print binned balanced norms per epoch for saved runs, next to the step0 Gensim runs."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from sgns import binned, OUT

GENSIM = Path(__file__).resolve().parents[2] / "results" / "step0" / "runs"
for setting in ["B", "A", "SW"]:
    g = np.load(GENSIM / f"ba_{setting}_s0.npz")
    x, y = binned(g["x"], g["balanced_norm"])
    print(f"GENSIM {setting:2s} deg", np.round(x, 1))
    print(f"GENSIM {setting:2s} bal", np.round(y, 3))
for name in sys.argv[1:]:
    z = np.load(OUT / f"{name}.npz")
    eps = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("bal_"))
    for e in eps:
        x, y = binned(z["deg"], z[f"bal_{e}"])
        print(f"{name:10s} ep{e:<2d} bal", np.round(y, 3), f" deg {np.round(x[[0, -1]], 0)}")
