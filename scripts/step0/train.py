"""Train one step-0 embedding and save per-token counts, raw norms and balanced norms.

Usage: python scripts/step0/train.py CORPUS COLUMN SEED --data data/derived/step0 --out
  results/step0/runs
  CORPUS: real | null | pop | ba | blog | text
  COLUMN: A | B | SW | A50 | B50 (A or B at 50 epochs)
  (SW = Schakel & Wilson 2015 published settings; text, blog, ba).
  Corrected Gensim build, one worker.
Network tokens are strings "n<index>", which avoids Gensim's integer-key negative-table
bug (#35).
Output: <out>/<corpus>_<column>_s<seed>.npz with tokens, counts, raw_norm,
  balanced_norm, x
(x = node strength for networks, corpus count for text) and the settings used.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gensim.models import Word2Vec  # noqa: E402
from gensim_training import require_correct_blas  # noqa: E402
from simulate_norm_frequency import balanced_left_norm  # noqa: E402

COLUMNS = {
    "A": dict(
        vector_size=100,
        window=5,
        shrink_windows=True,
        negative=5,
        ns_exponent=0.75,
        sample=1e-3,
        epochs=5,
        min_count=5,
    ),
    "B": dict(
        vector_size=300,
        window=1,
        shrink_windows=False,
        negative=5,
        ns_exponent=1.0,
        sample=0.0,
        epochs=5,
        min_count=50,
    ),
    "SW": dict(
        vector_size=100,
        window=10,
        shrink_windows=True,
        negative=5,
        ns_exponent=0.75,
        sample=1e-4,
        epochs=20,
        min_count=1,
    ),
}
COLUMNS["A50"] = COLUMNS["A"] | {"epochs": 50}
COLUMNS["B50"] = COLUMNS["B"] | {"epochs": 50}


class Walks:
    def __init__(self, walks: np.ndarray, names: np.ndarray):
        self.walks, self.names = walks, names

    def __iter__(self):
        for row in self.walks:
            yield self.names[row].tolist()


class Lines:
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        with self.path.open() as f:
            for line in f:
                yield line.split()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("corpus", choices=["real", "null", "pop", "ba", "blog", "text"])
    p.add_argument("column", choices=list(COLUMNS))
    p.add_argument("seed", type=int)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.column == "SW" and a.corpus not in ("text", "blog", "ba"):
        raise ValueError("SW settings are run on the text, blog and ba corpora only")
    if a.corpus == "text":
        corpus, strength = Lines(a.data / "hepth_tokens.txt"), None
    else:
        z = np.load(a.data / f"{a.corpus}_walks_s{a.seed}.npz")
        names = np.array([f"n{i}" for i in range(len(z["strength"]))], dtype=object)
        corpus, strength = Walks(z["walks"], names), z["strength"]
    settings = COLUMNS[a.column] | dict(
        sg=1, hs=0, alpha=0.025, min_alpha=1e-4, workers=1, seed=a.seed
    )
    require_correct_blas()
    with threadpool_limits(1):
        model = Word2Vec(**settings)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        tokens = list(model.wv.index_to_key)
        center = model.wv.vectors.astype(np.float64)
        context = model.syn1neg.astype(np.float64)
        balanced = balanced_left_norm(center, context)
    counts = np.array([model.wv.get_vecattr(t, "count") for t in tokens])
    x = (
        counts.astype(float)
        if strength is None
        else strength[[int(t[1:]) for t in tokens]]
    )
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out / f"{a.corpus}_{a.column}_s{a.seed}.npz",
        tokens=np.array(tokens),
        counts=counts,
        raw_norm=np.linalg.norm(center, axis=1),
        balanced_norm=balanced,
        x=x,
        settings=json.dumps(settings),
        vocabulary_before_cutoff=len(model.raw_vocab)
        if hasattr(model, "raw_vocab") and model.raw_vocab
        else -1,
    )
    print(f"{a.corpus} {a.column} s{a.seed}: {len(tokens)} tokens kept", flush=True)


if __name__ == "__main__":
    main()
