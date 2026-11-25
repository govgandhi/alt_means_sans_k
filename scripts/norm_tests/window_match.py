"""Test 4: does the window act through update count (speed) or through the target?

Actual BA graph: networkx.barabasi_albert_graph(10000, 5, seed=k), 10 walks per node
of 80 steps started in proportion to degree, uniform neighbour steps (Fig. 1(c) law).
Gensim defaults (A: d=100, exponent 0.75, sample 1e-3) with:
  w5e5   window 5 (shrinking, mean 3 per side -> ~6 contexts/token), 5 epochs
  w1e5   window 1 (2 contexts/token), 5 epochs
  w1e15  window 1, 15 epochs: same updates and rate range as w5e5
  w5lr3  window 5, 5 epochs, rates / 3: same updates x rate as w1e5
Speed-only predicts w1e15 ~ w5e5 and w5lr3 ~ w1e5.
Output: results/norm_tests/window/<config>_s<seed>.npz (norms at every epoch).
Run: .venv/bin/python -I scripts/norm_tests/window_match.py w1e15 0
"""

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/step0"))
from corpora import csr, walks  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from gensim_training import require_correct_blas  # noqa: E402
from simulate_norm_frequency import balanced_left_norm  # noqa: E402
from train import COLUMNS, Walks  # noqa: E402

CONFIGS = {
    "w5e5": dict(window=5, epochs=5, scale=1.0),
    "w1e5": dict(window=1, epochs=5, scale=1.0),
    "w1e15": dict(window=1, epochs=15, scale=1.0),
    "w5lr3": dict(window=5, epochs=5, scale=1 / 3),
}


def ba_walks(seed: int):
    g = nx.barabasi_albert_graph(10_000, 5, seed=seed)
    e = np.array(g.edges(), dtype=np.int64)
    src, dst = np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]]
    graph = csr(10_000, src, dst, np.ones(len(src)))
    return walks(graph, np.random.default_rng(9000 + seed)), graph[3]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", choices=list(CONFIGS))
    p.add_argument("seed", type=int)
    p.add_argument("--out", type=Path, default=ROOT / "results/norm_tests/window")
    a = p.parse_args()
    c = CONFIGS[a.config]
    corpus, degree = ba_walks(a.seed)
    names = np.array([f"n{i}" for i in range(len(degree))], dtype=object)
    settings = COLUMNS["A"] | dict(
        window=c["window"],
        epochs=c["epochs"],
        sg=1,
        hs=0,
        alpha=0.025 * c["scale"],
        min_alpha=1e-4 * c["scale"],
        workers=1,
        seed=a.seed,
    )
    a0, a1, E = settings["alpha"], settings["min_alpha"], settings["epochs"]
    require_correct_blas()
    raw, bal = [], []
    with threadpool_limits(1):
        model = Word2Vec(**settings)
        model.build_vocab(Walks(corpus, names))
        for e in range(E):
            model.train(
                Walks(corpus, names),
                total_examples=model.corpus_count,
                epochs=1,
                start_alpha=a0 - (a0 - a1) * e / E,
                end_alpha=a0 - (a0 - a1) * (e + 1) / E,
            )
            W = model.wv.vectors.astype(np.float64)
            raw.append(np.linalg.norm(W, axis=1))
            bal.append(balanced_left_norm(W, model.syn1neg.astype(np.float64)))
    tokens = list(model.wv.index_to_key)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out / f"{a.config}_s{a.seed}.npz",
        tokens=np.array(tokens),
        x=degree[[int(t[1:]) for t in tokens]],
        raw_norm=np.array(raw),
        balanced_norm=np.array(bal),
        settings=json.dumps(settings),
    )
    print(f"window {a.config} s{a.seed} done", flush=True)


if __name__ == "__main__":
    main()
