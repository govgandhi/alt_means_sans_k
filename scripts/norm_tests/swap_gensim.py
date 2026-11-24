"""Does large-step hub shrinkage persist after a switch to small steps (and back)?

PR #36 graphs, real Gensim, one-hop window. Schedules (rate per epoch):
  big2small: 0.0314 for 2 epochs (tau 0.0628), then 0.000625 for 100 epochs
  small2big: 0.000625 for 100 epochs (tau 0.0625), then 0.0314 for 2 epochs
If the first-order penalty acts only while steps are large, big2small relaxes toward
the small-step curve at a rate set by the loss curvature along hub-norm directions;
a slow relaxation would explain why learning-rate history persists (PR #36).
Output: results/norm_tests/swap/gensim_<schedule>_s<seed>.npz (clock_gensim format).
Run: .venv/bin/python -I scripts/norm_tests/swap_gensim.py big2small 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clock_gensim  # noqa: E402
from clock_gensim import Recorder, Scheduled  # noqa: E402
from gensim_training import require_correct_blas  # noqa: E402
from pr36_common import ROOT, gensim_model, setup  # noqa: E402

BIG, SMALL = 0.0314, 0.000625
PLAN = {"big2small": [(BIG, 2), (SMALL, 100)], "small2big": [(SMALL, 100), (BIG, 2)]}


def rate(schedule: str, t: float) -> float:
    edge = 0.0
    for r, n in PLAN[schedule]:
        edge += n
        if t < edge:
            return r
    return PLAN[schedule][-1][0]


def tau(schedule: str, epoch: int) -> float:
    total, left = 0.0, epoch
    for r, n in PLAN[schedule]:
        take = min(left, n)
        total += r * take
        left -= take
    return total


clock_gensim.rate, clock_gensim.tau = rate, tau


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("schedule", choices=list(PLAN))
    p.add_argument("seed", type=int)
    a = p.parse_args()
    epochs = sum(n for _, n in PLAN[a.schedule])
    require_correct_blas()
    s = setup(a.seed)
    Scheduled.schedule = a.schedule
    model = gensim_model(
        s["walks"],
        a.seed,
        rate(a.schedule, 0),
        PLAN[a.schedule][-1][0],
        epochs,
        Scheduled,
    )
    rec = Recorder(s, a.schedule)
    with threadpool_limits(1):
        model.train(
            s["walks"],
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words,
            epochs=epochs,
            start_alpha=model.alpha,
            end_alpha=model.min_alpha,
            callbacks=[rec],
        )
    out = ROOT / "results/norm_tests/swap"
    out.mkdir(parents=True, exist_ok=True)
    keys = rec.rows[0].keys()
    np.savez_compressed(
        out / f"gensim_{a.schedule}_s{a.seed}.npz",
        degree=s["degree"],
        **{k: np.array([r[k] for r in rec.rows]) for k in keys},
    )
    print(f"swap {a.schedule} s{a.seed}: {epochs} epochs", flush=True)


if __name__ == "__main__":
    main()
