"""Test whether long training drives never-observed pairs away from the population
optimum.

Retrains PNAS settings (column B) on pop_walks_s0 for 5 and 50 epochs (same seed as
diagnose_population.py), counts window-1 co-occurrences in that corpus, and reports
scores of
pairs observed at least once versus never observed, plus how each node's balanced norm
relates to
its share of never-observed pairs (weighted by pi).
Run: python scripts/step0/diagnose_unseen_pairs.py --data data/derived/step0 --out
  results/step0/diagnose_unseen_pairs.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gensim.models import Word2Vec  # noqa: E402
from gensim_training import require_correct_blas  # noqa: E402
from simulate_norm_frequency import balanced_left_norm  # noqa: E402
from train import COLUMNS, Walks  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    z = np.load(a.data / "pop_walks_s0.npz")
    walks = z["walks"]
    names = np.array([f"n{i}" for i in range(len(z["strength"]))], dtype=object)
    corpus = Walks(walks, names)
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
            keys = model.wv.index_to_key
            slot = np.full(len(names), -1)
            slot[[int(k[1:]) for k in keys]] = np.arange(len(keys))
            w, c = model.wv.vectors.astype(np.float64), model.syn1neg.astype(np.float64)
            x = w @ c.T
            balanced = balanced_left_norm(w, c)
        left, right = slot[walks[:, :-1].ravel()], slot[walks[:, 1:].ravel()]
        keep = (left >= 0) & (right >= 0)
        n = len(keys)
        counts = np.zeros((n, n), dtype=np.int32)
        np.add.at(counts, (left[keep], right[keep]), 1)
        np.add.at(counts, (right[keep], left[keep]), 1)
        observed = counts > 0
        tokens = np.array([model.wv.get_vecattr(k, "count") for k in keys], dtype=float)
        pi = tokens / tokens.sum()
        unseen_share = ((~observed) * pi[None, :]).sum(1)
        report[column] = {
            "share_pairs_observed": float(observed.mean()),
            "score_observed_q05_50_95": np.quantile(x[observed], [0.05, 0.5, 0.95])
            .round(3)
            .tolist(),
            "score_unseen_q05_50_95": np.quantile(x[~observed], [0.05, 0.5, 0.95])
            .round(3)
            .tolist(),
            "share_below_minus6_that_are_unseen": float(
                (~observed & (x <= -6)).sum() / max((x <= -6).sum(), 1)
            ),
            "spearman_balanced_norm_vs_unseen_share": float(
                spearmanr(balanced, unseen_share)[0]
            ),
            "spearman_balanced_norm_vs_frequency": float(
                spearmanr(balanced, tokens)[0]
            ),
        }
        print(column, json.dumps(report[column]), flush=True)
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
