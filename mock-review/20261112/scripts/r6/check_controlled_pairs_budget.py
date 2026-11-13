"""R6 check: same as check_controlled_budget.py, but scale the pair budget instead of epochs.

Keeps the paper's four-epoch 0.025 -> 0.0025 schedule and multiplies events_per_node
(240 -> 960, 2880), so per-node progress matches the 16- and 48-epoch runs without
stretching the learning-rate decay. Uses the worktree's allocate_pairs/train_sgns.
Run: PYTHONDONTWRITEBYTECODE=1 r6/.venv/bin/python r6/check_controlled_pairs_budget.py [seeds]
"""
import dataclasses, json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, "<scratch>/round/wt/scripts")
import numpy as np
from simulate_exact_center_exposure import balanced_left_norm, binned_curve, configuration_graph, graph_arrays, train_sgns
from simulate_turnover_surface import Condition, Config, allocate_pairs, transition_matrix
for seed in [int(s) for s in sys.argv[1:]] or [0, 1]:
    graph = configuration_graph(600, 3, seed)
    adjacency, _, degree = graph_arrays(graph)
    transition = transition_matrix(adjacency, degree)
    for events in (960, 2880):
        config = dataclasses.replace(Config(), events_per_node=events)
        for beta in (0.0, 1 / 3, 1.0):
            centers, contexts, _, noise, *_ = allocate_pairs(transition, degree / degree.sum(), config, Condition("x", beta, 0.0, None, False))
            u, v = train_sgns(centers, contexts, noise, config, seed=100_000 + seed)
            out = {"seed": seed, "events_per_node": events, "beta": round(beta, 3)}
            for name, vals in (("raw", np.linalg.norm(u, axis=1)), ("bal", balanced_left_norm(u, v))):
                _, y, _ = binned_curve(degree, vals)
                p = int(np.argmax(y))
                out[name] = {"bins": np.round(y, 3).tolist(), "peak_bin": p, "interior": bool(0 < p < len(y) - 1), "peak_minus_high": round(float(y[p] - y[-1]), 3)}
            print(json.dumps(out), flush=True)
