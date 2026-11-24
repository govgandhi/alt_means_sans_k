"""Test 1: does a fresh corpus every epoch remove the Fig. 1(c)/(d) norm-degree shape?

Real (corrected) Gensim, Fig. 1 settings A/B/SW, seeds 0-4. The epoch loop calls
train() once per epoch with the global linear schedule split at epoch boundaries, so
the learning rate matches a single train() call. Arms:
  reused: the committed Fig. 1 corpus every epoch (checks the loop against the npz)
  fresh:  epoch 1 uses the committed corpus; later epochs draw a new corpus by the same
          law (ba: iid draws in proportion to degree; blog: new walks on BlogCatalog)
Vocabulary, counts, subsampling and the negative table come from the committed corpus.
Prediction if the shape is overfit corpus noise: (d) flattens, (c) does not change.
Output: results/norm_tests/fresh/<corpus>_<column>_<arm>_s<seed>.npz with x, raw and
balanced norms at every epoch.
Run: .venv/bin/python -I scripts/norm_tests/fresh_corpus.py ba B fresh 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/step0"))
from corpora import csr, walks  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from gensim_training import require_correct_blas  # noqa: E402
from simulate_norm_frequency import balanced_left_norm  # noqa: E402
from train import COLUMNS, Lines, Walks  # noqa: E402

DATA = ROOT / "data/derived/step0"
MAT = ROOT / "data/external/blogcatalog/downloads/blogcatalog.mat"


def blog_graph():
    adjacency = sp.coo_matrix(scipy.io.loadmat(MAT)["network"])
    return csr(
        adjacency.shape[0],
        adjacency.row.astype(np.int64),
        adjacency.col.astype(np.int64),
        adjacency.data,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("corpus", choices=["ba", "blog", "text"])
    p.add_argument("column", choices=["A", "B", "SW", "A50", "B50"])
    p.add_argument("arm", choices=["reused", "fresh"])
    p.add_argument("seed", type=int)
    p.add_argument("--out", type=Path, default=ROOT / "results/norm_tests/fresh")
    a = p.parse_args()
    if a.corpus == "text":
        if a.arm != "reused":
            raise ValueError("text has no generative law; reused arm only")
        base, strength, names = None, None, None
    else:
        z = np.load(DATA / f"{a.corpus}_walks_s{a.seed}.npz")
        strength = z["strength"]
        names = np.array([f"n{i}" for i in range(len(strength))], dtype=object)
        base = z["walks"]
    settings = COLUMNS[a.column] | dict(
        sg=1, hs=0, alpha=0.025, min_alpha=1e-4, workers=1, seed=a.seed
    )
    epochs = settings["epochs"]
    rng = np.random.default_rng(70_000 + 100 * a.seed + ord(a.corpus[0]))
    graph = blog_graph() if (a.corpus == "blog" and a.arm == "fresh") else None
    p_degree = None if strength is None else strength / strength.sum()

    def corpus_for(epoch: int):
        if a.corpus == "text":
            return Lines(DATA / "hepth_tokens.txt")
        if a.arm == "reused" or epoch == 0:
            return Walks(base, names)
        if a.corpus == "ba":
            return Walks(
                rng.choice(len(strength), size=base.shape, p=p_degree).astype(np.int32),
                names,
            )
        return Walks(walks(graph, rng), names)

    require_correct_blas()
    a0, a1 = settings["alpha"], settings["min_alpha"]
    snaps = {"raw": [], "balanced": []}
    with threadpool_limits(1):
        model = Word2Vec(**settings)
        model.build_vocab(corpus_for(0))
        for e in range(epochs):
            start = a0 - (a0 - a1) * e / epochs
            end = a0 - (a0 - a1) * (e + 1) / epochs
            model.train(
                corpus_for(e),
                total_examples=model.corpus_count,
                epochs=1,
                start_alpha=start,
                end_alpha=end,
            )
            center = model.wv.vectors.astype(np.float64)
            context = model.syn1neg.astype(np.float64)
            snaps["raw"].append(np.linalg.norm(center, axis=1))
            snaps["balanced"].append(balanced_left_norm(center, context))
            print(f"{a.corpus} {a.column} {a.arm} s{a.seed} epoch {e + 1}", flush=True)
    tokens = list(model.wv.index_to_key)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out / f"{a.corpus}_{a.column}_{a.arm}_s{a.seed}.npz",
        tokens=np.array(tokens),
        x=(
            np.array([model.wv.get_vecattr(t, "count") for t in tokens], float)
            if strength is None
            else strength[[int(t[1:]) for t in tokens]]
        ),
        raw_norm=np.array(snaps["raw"]),
        balanced_norm=np.array(snaps["balanced"]),
        settings=json.dumps(settings),
    )


if __name__ == "__main__":
    main()
