"""Tests 2-3 in real Gensim: is the norm profile a function of cumulative learning rate?

PR #36 graphs (pr36_common.py). Schedules: 'orig' (0.025 -> 0.0001 over 5 epochs, then
0.0001; PR #36 baseline) and constant rates. Every node shares the schedule, so at equal
summed learning rate (tau) every node has had the same updates x rate. If tau is the
clock, profiles at equal tau coincide across schedules. Snapshots every epoch: balanced
and raw norms, per-node loss, per-node fraction of negative mass with score <= -6 (where
Gensim skips updates).
Output: results/norm_tests/clock/gensim_<schedule>_s<seed>.npz
Run: .venv/bin/python -I scripts/norm_tests/clock_gensim.py orig 0
     .venv/bin/python -I scripts/norm_tests/clock_gensim.py 0.0025 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr36_common  # noqa: E402, F401  (sets sys.path)
from gensim.models import Word2Vec  # noqa: E402
from gensim.models.callbacks import CallbackAny2Vec  # noqa: E402
from gensim_training import require_correct_blas  # noqa: E402
from pr36_common import ROOT, T, gensim_model, metrics, setup  # noqa: E402

TAU_END = 0.1


def rate(schedule: str, t: float) -> float:
    if schedule == "orig":
        return max(1e-4, 0.025 - (0.025 - 1e-4) * t / 5)
    return float(schedule)


def tau(schedule: str, epoch: int) -> float:
    grid = np.linspace(0, epoch, 20_001)
    return float(np.trapezoid([rate(schedule, t) for t in grid], grid))


class Scheduled(Word2Vec):
    schedule = "orig"

    def _get_next_alpha(self, epoch_progress: float, cur_epoch: int) -> float:
        alpha = rate(self.schedule, cur_epoch + epoch_progress)
        self.min_alpha_yet_reached = alpha
        return alpha


class Recorder(CallbackAny2Vec):
    def __init__(self, s, schedule):
        self.s, self.schedule, self.epoch, self.rows = s, schedule, 0, []

    def capture(self, model):
        slots = np.array([model.wv.key_to_index[n] for n in range(T.nodes)])
        m = metrics(
            model.wv.vectors[slots].astype(np.float64),
            model.syn1neg[slots].astype(np.float64),
            self.s,
        )
        self.rows.append(
            dict(epoch=self.epoch, tau=tau(self.schedule, self.epoch), **m)
        )

    def on_train_begin(self, model):
        self.capture(model)

    def on_epoch_end(self, model):
        self.epoch += 1
        self.capture(model)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("schedule")
    p.add_argument("seed", type=int)
    p.add_argument("--out", type=Path, default=ROOT / "results/norm_tests/clock")
    a = p.parse_args()
    epochs = 20 if a.schedule == "orig" else int(round(TAU_END / float(a.schedule)))
    require_correct_blas()
    s = setup(a.seed)
    Scheduled.schedule = a.schedule
    start = rate(a.schedule, 0)
    model = gensim_model(
        s["walks"], a.seed, start, rate(a.schedule, epochs), epochs, Scheduled
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
    a.out.mkdir(parents=True, exist_ok=True)
    keys = rec.rows[0].keys()
    np.savez_compressed(
        a.out / f"gensim_{a.schedule}_s{a.seed}.npz",
        degree=s["degree"],
        **{k: np.array([r[k] for r in rec.rows]) for k in keys},
    )
    print(f"gensim {a.schedule} s{a.seed}: {epochs} epochs", flush=True)


if __name__ == "__main__":
    main()
