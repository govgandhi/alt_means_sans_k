"""Does step size at matched summed learning rate change the curve on real corpora?

One-hop setting by default (column B: d=300, window 1, negative exponent 1, no
subsampling); --column A gives the default setting. BlogCatalog walks (seed 0) or
hep-th. Constant rate TAU/E for E epochs, with TAU = 0.06275 the summed rate of
Gensim's default 5-epoch schedule (0.025 -> 0.0001), so every run ends at the same
tau as the committed Fig. 1 run. Same per-epoch loop as fresh_corpus.py.
Output: results/norm_tests/stepsize_real/<corpus>[_<column>]_E<epochs>.npz with x,
balanced and
raw norms after every epoch, tau after every epoch, and the final factors.
Run: .venv/bin/python -I scripts/norm_tests/stepsize_real.py blog 25
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
from simulate_norm_frequency import balanced_left_norm  # noqa: E402
from train import COLUMNS, Lines, Walks  # noqa: E402

DATA = ROOT / "data/derived/step0"
TAU = 5 * (0.025 + 1e-4) / 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("corpus", choices=["blog", "text"])
    p.add_argument("epochs", type=int)
    p.add_argument("--column", default="B", choices=["A", "B", "SW"])
    a = p.parse_args()
    if a.corpus == "blog":
        z = np.load(DATA / "blog_walks_s0.npz")
        base, strength = z["walks"], z["strength"]
        names = np.array([f"n{i}" for i in range(len(strength))], dtype=object)

    def corpus():
        if a.corpus == "text":
            return Lines(DATA / "hepth_tokens.txt")
        return Walks(base, names)

    rate = TAU / a.epochs
    settings = COLUMNS[a.column] | dict(
        sg=1,
        hs=0,
        alpha=rate,
        min_alpha=rate,
        epochs=a.epochs,
        workers=1,
        seed=0,
    )
    require_correct_blas()
    raw, bal = [], []
    with threadpool_limits(1):
        model = Word2Vec(**settings)
        model.build_vocab(corpus())
        for e in range(a.epochs):
            model.train(
                corpus(),
                total_examples=model.corpus_count,
                epochs=1,
                start_alpha=rate,
                end_alpha=rate,
            )
            W = model.wv.vectors.astype(np.float64)
            C = model.syn1neg.astype(np.float64)
            raw.append(np.linalg.norm(W, axis=1).astype(np.float32))
            bal.append(balanced_left_norm(W, C).astype(np.float32))
            print(f"{a.corpus} E{a.epochs} epoch {e + 1}", flush=True)
    tokens = list(model.wv.index_to_key)
    counts = np.array([model.wv.get_vecattr(t, "count") for t in tokens], float)
    out = ROOT / "results/norm_tests/stepsize_real"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{a.corpus}{'' if a.column == 'B' else '_' + a.column}_E{a.epochs}.npz",
        tokens=np.array(tokens),
        x=counts if a.corpus == "text" else strength[[int(t[1:]) for t in tokens]],
        counts=counts,
        tau=rate * np.arange(1, a.epochs + 1),
        raw_norm=np.array(raw),
        balanced_norm=np.array(bal),
        W=model.wv.vectors.astype(np.float32),
        C=model.syn1neg.astype(np.float32),
        settings=json.dumps(settings),
    )


if __name__ == "__main__":
    main()
