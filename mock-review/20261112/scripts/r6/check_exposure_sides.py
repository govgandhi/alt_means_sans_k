"""R6 check: what the exposure intervention equalises (input side) and what it does not.

For the paper's beta in {0, 1/3, 1} (lambda=0), per log-degree bin: positive-center
(input-vector) updates, positive-context (output-vector) updates, and expected negative
(output-vector) updates per epoch, from the worktree's allocate_pairs (k=5).
Run: PYTHONDONTWRITEBYTECODE=1 r6/.venv/bin/python r6/check_exposure_sides.py
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "<scratch>/round/wt/scripts")
import numpy as np
from simulate_exact_center_exposure import configuration_graph, graph_arrays
from simulate_turnover_surface import Condition, Config, allocate_pairs, transition_matrix

config = Config()
graph = configuration_graph(config.nodes, config.attachment, 0)
adjacency, _, degree = graph_arrays(graph)
stationary = degree / degree.sum()
transition = transition_matrix(adjacency, degree)
log_d = np.log(degree)
edges = np.unique(np.quantile(log_d, np.linspace(0, 1, 9)))
labels = np.clip(np.digitize(log_d, edges[1:-1], right=True), 0, len(edges) - 2)
for beta in (0.0, 1 / 3, 1.0):
    centers, contexts, counts, q, *_ = allocate_pairs(transition, stationary, config, Condition("x", beta, 0.0, None, False))
    ctx = np.bincount(contexts, minlength=len(degree))
    neg = config.negatives * len(centers) * q
    cells = []
    for b in range(len(edges) - 1):
        s = labels == b
        cells.append(f"d~{np.exp(log_d[s].mean()):.1f}: center {counts[s].mean():.0f}, context {ctx[s].mean():.0f}, negative {neg[s].mean():.0f}")
    print(f"beta={beta:.2f} | " + " | ".join(cells))
