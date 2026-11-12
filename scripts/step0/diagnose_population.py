"""Diagnose why Gensim departs from the flat population optimum on the independent-draw
corpus.

Trains PNAS settings (column B) on pop_walks_s0 for 5 and for 50 epochs, keeps the
factors, and
reports, for the score matrix X = W C^T over the kept vocabulary:
  - expected skip-gram loss per positive pair, against the optimum (every score = -log 5
  adjusted
    for Gensim's skipped self-collisions: a(v) = -log 5 - log(1 - pi_v));
  - share of positive and of negative training weight on pairs with |score| >= 6, where
  Gensim's
    skip-gram code skips the update;
  - score quantiles.
Positive weight of (w, v) is proportional to pi_w pi_v (independent tokens); negative
weight to
5 pi_w pi_v (1 - pi_v) (exponent 1, collisions skipped).
Run: python scripts/step0/diagnose_population.py --data data/derived/step0 --out
  results/step0/diagnose_population.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gensim.models import Word2Vec  # noqa: E402
from gensim_training import require_correct_blas  # noqa: E402
from train import COLUMNS, Walks  # noqa: E402


def logsig(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0, -x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    z = np.load(a.data / "pop_walks_s0.npz")
    names = np.array([f"n{i}" for i in range(len(z["strength"]))], dtype=object)
    corpus = Walks(z["walks"], names)
    require_correct_blas()
    report = {}
    for column in ("B", "B50"):
        settings = COLUMNS[column] | dict(
            sg=1, hs=0, alpha=0.025, min_alpha=1e-4, workers=1, seed=0
        )
        with threadpool_limits(4):
            model = Word2Vec(**settings)
            model.build_vocab(corpus)
            model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
            w, c = model.wv.vectors.astype(np.float64), model.syn1neg.astype(np.float64)
            counts = np.array(
                [model.wv.get_vecattr(t, "count") for t in model.wv.index_to_key],
                dtype=float,
            )
            pi = counts / counts.sum()
            x = w @ c.T
        positive = np.outer(pi, pi)
        negative = 5 * np.outer(pi, pi * (1 - pi))
        optimum = -np.log(5) - np.log(1 - pi)[None, :]

        def loss(s: np.ndarray) -> float:
            return float(-(positive * logsig(s)).sum() - (negative * logsig(-s)).sum())

        outside = np.abs(x) >= 6
        report[column] = {
            "vocabulary": int(len(pi)),
            "loss_trained": loss(x),
            "loss_optimum": loss(np.broadcast_to(optimum, x.shape)),
            "positive_weight_outside_6": float(
                positive[outside].sum() / positive.sum()
            ),
            "negative_weight_outside_6": float(
                negative[outside].sum() / negative.sum()
            ),
            "share_of_pairs_below_minus6": float((x <= -6).mean()),
            "score_quantiles_1_25_50_75_99": np.quantile(
                x, [0.01, 0.25, 0.5, 0.75, 0.99]
            )
            .round(3)
            .tolist(),
            "weighted_mean_score": float((positive * x).sum() / positive.sum()),
        }
        print(column, json.dumps(report[column]), flush=True)
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
