"""R6 check: are the unfiltered Gensim curves the same at word2vec's default ns_exponent=0.75?

The manuscript never states that every Gensim run sets ns_exponent=1.0 (the
default is 0.75 in word2vec.c, Gensim and node2vec). Same graphs/walks/seeds/schedule
as the corrected no-subsampling control (orig5), official wheel, slot-ordered table.
Run: PYTHONDONTWRITEBYTECODE=1 r6/.venv/bin/python r6/check_ns_exponent.py [seeds]
"""
import copy
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
    for a in (1.0, 0.75, 0.5):
        for sample in (0.0, 1e-3):
            m = Word2Vec(vector_size=64, window=1, min_count=50, sg=1, negative=5, ns_exponent=a,
                         workers=1, epochs=5, seed=seed + 100_000, sample=sample, alpha=0.025,
                         min_alpha=0.0001, sorted_vocab=1, shrink_windows=False)
            m.build_vocab(walks)
            fix_table(m)
            run_epochs(m, walks, linear(0.025, 0.0001, 5))
            print(json.dumps({"seed": seed, "ns_exponent": a, "sample": sample, **curve(m, degree)}), flush=True)
