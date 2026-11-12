"""R6 check: what the integer allocation does to the context-mixing intervention.

Calls the worktree's own allocate_pairs (simulate_turnover_surface.py) for the five
calibration graphs at beta in {0, 1/3, 1} and lambda in {0, 1/3, 2/3, 1}, then reports
  * TV distance between the realized context marginal (= negative distribution used
    in training) and the intended q = c T,
  * contexts with zero realized mass (never a positive, never a negative),
  * whether every center received an identical context list (degenerate design),
  * Spearman of intended vs empirical KL across nodes, and the degree slope of each,
  * the share of each row's lambda*q mass that lands on the same node set in every row.
No embeddings are trained.

Run: PYTHONDONTWRITEBYTECODE=1 r6/.venv/bin/python r6/check_context_mixing_allocation.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
WT = "<scratch>/round/wt"
sys.path.insert(0, WT + "/scripts")

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from simulate_exact_center_exposure import configuration_graph, graph_arrays  # noqa: E402
from simulate_turnover_surface import (  # noqa: E402
    Condition,
    Config,
    allocate_pairs,
    transition_matrix,
)


def main() -> None:
    config = Config()
    for seed in range(5):
        graph = configuration_graph(config.nodes, config.attachment, seed)
        adjacency, _, degree = graph_arrays(graph)
        stationary = degree / degree.sum()
        transition = transition_matrix(adjacency, degree)
        for beta in (0.0, 1.0 / 3.0, 1.0):
            for lam in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
                cond = Condition("x", beta, lam, None, False)
                centers, contexts, counts, realized_q, kl_int, kl_emp, _, _ = (
                    allocate_pairs(transition, stationary, config, cond)
                )
                share = counts / counts.sum()
                intended_q = share @ transition
                n = len(degree)
                matrix = np.zeros((n, n), dtype=np.int64)
                np.add.at(matrix, (centers, contexts), 1)
                identical_rows = (
                    bool(np.all(matrix == matrix[0])) if beta == 0 else None
                )
                support = (matrix > 0).sum(axis=1)
                log_d = np.log(degree)
                slope_int = np.polyfit(log_d, kl_int, 1)[0]
                slope_emp = np.polyfit(log_d, kl_emp, 1)[0]
                rho = (
                    spearmanr(kl_int, kl_emp).statistic
                    if np.ptp(kl_int) > 1e-12
                    else float("nan")
                )
                # mass allocated to non-neighbours (the lambda*q component) per row
                nonnbr = matrix * (adjacency == 0)
                nonnbr_nodes_every_row = int(np.sum((nonnbr > 0).all(axis=0)))
                print(
                    f"seed={seed} beta={beta:.2f} lambda={lam:.2f} "
                    f"TV(realized q, intended q)={0.5 * np.abs(realized_q - intended_q).sum():.4f} "
                    f"zero-mass contexts={int((realized_q == 0).sum())} "
                    f"identical rows={identical_rows} "
                    f"row support median={int(np.median(support))} "
                    f"KL intended/empirical mean={kl_int.mean():.3f}/{kl_emp.mean():.3f} "
                    f"slope on log d intended/empirical={slope_int:.3f}/{slope_emp:.3f} "
                    f"spearman(int,emp)={rho:.3f} "
                    f"non-neighbour contexts present in every row={nonnbr_nodes_every_row}"
                )


if __name__ == "__main__":
    main()
