"""R6 check: is the controlled trainer under-trained relative to the Gensim runs it explains?

Uses the worktree's own allocate_pairs/train_sgns (simulate_turnover_surface.py) with
the paper's Config, changing only the number of epochs (the trainer's linear
0.025 -> 0.0025 schedule is stretched over the run, exactly as in train_sgns).
Per-node training progress sum(eta * updates) at 4 epochs is ~1/11 of the Gensim
five-epoch runs; 40-48 epochs roughly matches it.
Prints binned raw/balanced curves (paper binning) for beta in {0, 1/3, 1}, lambda = 0.

Run: PYTHONDONTWRITEBYTECODE=1 r6/.venv/bin/python r6/check_controlled_budget.py [seeds]
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time

sys.dont_write_bytecode = True
WT = "<scratch>/round/wt"
sys.path.insert(0, WT + "/scripts")

import numpy as np  # noqa: E402
from simulate_exact_center_exposure import (  # noqa: E402
    balanced_left_norm,
    binned_curve,
    configuration_graph,
    graph_arrays,
    train_sgns,
)
from simulate_turnover_surface import (  # noqa: E402
    Condition,
    Config,
    allocate_pairs,
    transition_matrix,
)


def main(seeds: list[int]) -> None:
    base = Config()
    for seed in seeds:
        graph = configuration_graph(base.nodes, base.attachment, seed)
        adjacency, _, degree = graph_arrays(graph)
        stationary = degree / degree.sum()
        transition = transition_matrix(adjacency, degree)
        for beta in (0.0, 1.0 / 3.0, 1.0):
            cond = Condition("x", beta, 0.0, None, False)
            centers, contexts, _, noise, *_ = allocate_pairs(
                transition, stationary, base, cond
            )
            for epochs in (4, 16, 48):
                t0 = time.time()
                config = dataclasses.replace(base, epochs=epochs)
                u, v = train_sgns(centers, contexts, noise, config, seed=100_000 + seed)
                out = {"seed": seed, "beta": round(beta, 3), "epochs": epochs}
                for name, values in (
                    ("raw", np.linalg.norm(u, axis=1)),
                    ("bal", balanced_left_norm(u, v)),
                ):
                    x, y, _ = binned_curve(degree, values)
                    peak = int(np.argmax(y))
                    out[name] = {
                        "bins": np.round(y, 3).tolist(),
                        "peak_bin": peak,
                        "interior": bool(0 < peak < len(y) - 1),
                        "peak_minus_low": round(float(y[peak] - y[0]), 3),
                        "peak_minus_high": round(float(y[peak] - y[-1]), 3),
                    }
                out["bin_degree"] = np.round(x, 2).tolist()
                out["seconds"] = round(time.time() - t0, 1)
                print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1])
