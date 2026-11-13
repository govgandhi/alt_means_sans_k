"""R6 check: how far does the peak move with training budget, relative to its movement with
the subsampling threshold that the paper's predictor is credited with tracking?

Literal Gensim (internal subsampling), the paper's calibration graphs/walks/seeds,
orig5 (0.025 -> 0.0001 over 5 epochs) vs Gensim's own default when epochs=20
(0.025 -> 0.0001 over 20 epochs). Prints peak bin (paper binning) and drop.
Run: PYTHONDONTWRITEBYTECODE=1 r6/.venv/bin/python r6/check_budget_vs_threshold.py [seeds]
"""
import json
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "<scratch>/round/r6")
from check_lr_schedule import (Word2Vec, configuration_graph, curve, fix_table, graph_arrays,
                               linear, np, run_epochs, sample_walks)

for seed in [int(s) for s in sys.argv[1:]] or [0, 1, 2]:
    graph = configuration_graph(600, 3, seed)
    _, neighbors, degree = graph_arrays(graph)
    walks, _ = sample_walks(neighbors, degree / degree.sum(), 600 * 300, 5, np.random.default_rng(seed + 10_000))
    for sample in (3e-4, 1e-3, 3e-3):
        for label, epochs in (("orig5", 5), ("gensim20", 20)):
            m = Word2Vec(vector_size=64, window=1, min_count=50, sg=1, negative=5, ns_exponent=1.0,
                         workers=1, epochs=epochs, seed=seed + 100_000, sample=sample, alpha=0.025,
                         min_alpha=0.0001, sorted_vocab=1, shrink_windows=False)
            m.build_vocab(walks)
            fix_table(m)
            run_epochs(m, walks, linear(0.025, 0.0001, epochs))
            c = curve(m, degree)
            print(json.dumps({"seed": seed, "t": sample, "schedule": label,
                              "raw_peak_bin": c["raw"]["peak_bin"], "raw_interior": c["raw"]["interior"],
                              "raw_drop": c["raw"]["peak_minus_high"], "raw_bins": c["raw"]["bins"],
                              "bal_peak_bin": c["bal"]["peak_bin"]}), flush=True)
