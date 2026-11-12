"""R6 check: context mixing (beta=0) under longer budgets of the authors' controlled trainer.
Same as check_controlled_budget.py but lambda in {1/3, 2/3} at beta=0.
Run: PYTHONDONTWRITEBYTECODE=1 r6/.venv/bin/python r6/check_controlled_mixing_budget.py [seeds]
"""
import dataclasses, json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, "<scratch>/round/wt/scripts")
import numpy as np
from simulate_exact_center_exposure import balanced_left_norm, binned_curve, configuration_graph, graph_arrays, train_sgns
from simulate_turnover_surface import Condition, Config, allocate_pairs, transition_matrix
base = Config()
for seed in [int(s) for s in sys.argv[1:]] or [0, 1]:
    graph = configuration_graph(base.nodes, base.attachment, seed)
    adjacency, _, degree = graph_arrays(graph)
    transition = transition_matrix(adjacency, degree)
    for lam in (1 / 3, 2 / 3):
        centers, contexts, _, noise, *_ = allocate_pairs(transition, degree / degree.sum(), base, Condition("x", 0.0, lam, None, False))
        for epochs in (4, 16, 48):
            u, v = train_sgns(centers, contexts, noise, dataclasses.replace(base, epochs=epochs), seed=100_000 + seed)
            _, y, _ = binned_curve(degree, np.linalg.norm(u, axis=1))
            print(json.dumps({"seed": seed, "lambda": round(lam, 3), "epochs": epochs, "raw_bins": np.round(y, 3).tolist(), "peak_bin": int(np.argmax(y))}), flush=True)
