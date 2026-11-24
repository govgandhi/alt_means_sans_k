"""Train B50 / A50 on BlogCatalog walks or hep-th text and save factors at checkpoints.

Same per-epoch loop and settings as fresh_corpus.py (reused arm), which reproduces the
committed runs exactly. Saves center W and context C at epochs 1, 2, 5, 10, 20, 50 so
the norm-KL relation of Oyama et al. (EMNLP 2023) can be checked on the factors.
--window overrides the column's window (tests which setting matters).
Output: results/norm_tests/oyama/<corpus>_<column>[w<window>]_s<seed>.npz
Run: .venv/bin/python -I scripts/norm_tests/oyama_train.py blog B50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/step0"))
from gensim.models import Word2Vec  # noqa: E402
from gensim_training import require_correct_blas  # noqa: E402
from train import COLUMNS, Lines, Walks  # noqa: E402

DATA = ROOT / "data/derived/step0"
CHECKPOINTS = (1, 2, 5, 10, 20, 50)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("corpus", choices=["blog", "text"])
    p.add_argument("column", choices=["A50", "B50"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int)
    a = p.parse_args()
    if a.corpus == "blog":
        z = np.load(DATA / f"blog_walks_s{a.seed}.npz")
        base, strength = z["walks"], z["strength"]
        names = np.array([f"n{i}" for i in range(len(strength))], dtype=object)

    def corpus():
        if a.corpus == "text":
            return Lines(DATA / "hepth_tokens.txt")
        return Walks(base, names)

    settings = COLUMNS[a.column] | dict(
        sg=1, hs=0, alpha=0.025, min_alpha=1e-4, workers=1, seed=a.seed
    )
    if a.window:
        settings["window"] = a.window
    E, a0, a1 = settings["epochs"], settings["alpha"], settings["min_alpha"]
    require_correct_blas()
    W, C, done = [], [], []
    with threadpool_limits(1):
        model = Word2Vec(**settings)
        model.build_vocab(corpus())
        for e in range(E):
            model.train(
                corpus(),
                total_examples=model.corpus_count,
                epochs=1,
                start_alpha=a0 - (a0 - a1) * e / E,
                end_alpha=a0 - (a0 - a1) * (e + 1) / E,
            )
            if e + 1 in CHECKPOINTS:
                W.append(model.wv.vectors.astype(np.float32).copy())
                C.append(model.syn1neg.astype(np.float32).copy())
                done.append(e + 1)
                print(f"{a.column} epoch {e + 1}", flush=True)
    tokens = list(model.wv.index_to_key)
    out = ROOT / "results/norm_tests/oyama"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out
        / f"{a.corpus}_{a.column}{f'w{a.window}' if a.window else ''}_s{a.seed}.npz",
        tokens=np.array(tokens),
        counts=np.array([model.wv.get_vecattr(t, "count") for t in tokens], float),
        epochs=np.array(done),
        W=np.array(W),
        C=np.array(C),
        settings=json.dumps(settings),
    )


if __name__ == "__main__":
    main()
