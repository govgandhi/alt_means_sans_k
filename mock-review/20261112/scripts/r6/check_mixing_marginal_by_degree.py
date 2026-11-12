"""R6 check: degree profile of realized/intended context marginal under context mixing (beta=0).

Uses the worktree's allocate_pairs; prints, per log-degree bin, the ratio of realized to
intended context mass (the negative distribution used in training) and the share of
nodes in the bin with zero realized mass. No training.
Run: PYTHONDONTWRITEBYTECODE=1 r6/.venv/bin/python r6/check_mixing_marginal_by_degree.py
"""
import sys
sys.dont_write_bytecode = True
WT = "<scratch>/round/wt"
sys.path.insert(0, WT + "/scripts")
import numpy as np
from simulate_exact_center_exposure import configuration_graph, graph_arrays
from simulate_turnover_surface import Condition, Config, allocate_pairs, transition_matrix

config = Config()
for seed in (0, 1):
    graph = configuration_graph(config.nodes, config.attachment, seed)
    adjacency, _, degree = graph_arrays(graph)
    stationary = degree / degree.sum()
    transition = transition_matrix(adjacency, degree)
    log_d = np.log(degree)
    edges = np.unique(np.quantile(log_d, np.linspace(0, 1, 9)))
    labels = np.clip(np.digitize(log_d, edges[1:-1], right=True), 0, len(edges) - 2)
    for lam in (1/3, 2/3, 1.0):
        _, _, counts, realized, _, _, _, _ = allocate_pairs(transition, stationary, config, Condition("x", 0.0, lam, None, False))
        intended = (counts / counts.sum()) @ transition
        out = []
        for b in range(len(edges) - 1):
            s = labels == b
            if s.any():
                out.append(f"d~{np.exp(log_d[s].mean()):.1f}: ratio={realized[s].sum()/intended[s].sum():.2f}, zero={np.mean(realized[s]==0):.2f}")
        print(f"seed={seed} lambda={lam:.2f} | " + " | ".join(out))
