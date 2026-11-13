"""R6 check: does the Gensim turnover depend on Gensim's skipped updates outside (-6, 6)?

Run with r6/.venv-noskip (official 4.4.0 source, rebuilt in scratch with (i) the
sdot/dsdot noexcept fix and (ii) w2v_fast_sentence_sg_neg computing the exact sigmoid
instead of skipping the update when |score| >= 6; build steps in r6/build_noskip/).
Same graphs/walks/seeds/settings as check_lr_schedule.py; prints the binned curves
for orig5 (0.025 -> 0.0001 over 5 epochs) and Gensim-default 20-epoch decay.
Run: PYTHONDONTWRITEBYTECODE=1 r6/.venv-noskip/bin/python r6/check_noskip.py [seeds]
"""
import copy
import json
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "<scratch>/round/r6")
from check_lr_schedule import (Word2Vec, configuration_graph, curve, fix_table, graph_arrays,
                               linear, np, run_epochs, sample_walks)
from gensim.models import word2vec_inner

for seed in [int(s) for s in sys.argv[1:]] or [0, 1, 2]:
    graph = configuration_graph(600, 3, seed)
    _, neighbors, degree = graph_arrays(graph)
    walks, _ = sample_walks(neighbors, degree / degree.sum(), 600 * 300, 5, np.random.default_rng(seed + 10_000))
    base = Word2Vec(vector_size=64, window=1, min_count=50, sg=1, negative=5, ns_exponent=1.0,
                    workers=1, epochs=5, seed=seed + 100_000, sample=0.0, alpha=0.025,
                    min_alpha=0.0001, sorted_vocab=1, shrink_windows=False)
    base.build_vocab(walks)
    fix_table(base)
    for label, sched in (("orig5", linear(0.025, 0.0001, 5)), ("gensim20", linear(0.025, 0.0001, 20)),
                         ("const20", [(0.0025, 0.0025)] * 20)):
        m = copy.deepcopy(base)
        run_epochs(m, walks, sched)
        scores_out = None
        print(json.dumps({"build": word2vec_inner.__file__.split("/")[-4], "seed": seed, "schedule": label, **curve(m, degree)}), flush=True)
