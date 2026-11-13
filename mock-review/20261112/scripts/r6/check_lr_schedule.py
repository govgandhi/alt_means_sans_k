"""R6 check: is "no turnover at twenty epochs under constant 0.0025" under-training,
or does the curve shape track the *current* learning rate (annealing)?

Unfiltered literal Gensim (sample=0), the paper's calibration graphs/walks/seeds,
window 1, d=64, k=5, ns_exponent=1, one worker, vocabulary-slot negative table
(same fix as scripts/gensim_training.set_negative_table). Official 4.4.0 wheel
(unpatched BLAS: at most ~1 affected dot product per run, see check_gensim_upstream).

Schedules (piecewise-linear per epoch, Gensim per-job decay inside each epoch):
  orig5        0.025 -> 0.0001 over 5 epochs                    (paper's baseline)
  gensim20     0.025 -> 0.0001 over 20 epochs                   (Gensim default at epochs=20)
  const        0.0025 constant, checkpoints 5/20/40/80          (paper's alternative, extended)
  const+anneal const to epoch 20, then 0.0025 -> 0.0001 over 5  (does annealing reveal turnover?)
  high5        0.025 constant for 5 epochs                      (high rate, no annealing)
  high5+anneal high5 then 0.025 -> 0.0001 over 5
  orig5+const  orig5 then constant 0.0025 for 20 epochs         (re-heating after annealing)
Readout: raw input norm and balanced norm, binned on 8 requested log-degree quantiles.

Run: PYTHONDONTWRITEBYTECODE=1 r6/.venv/bin/python r6/check_lr_schedule.py [seeds...]
"""

from __future__ import annotations

import copy
import json
import sys
import time

sys.dont_write_bytecode = True
WT = "<scratch>/round/wt"
sys.path.insert(0, WT + "/scripts")

import numpy as np  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from simulate_norm_frequency import (  # noqa: E402
    balanced_left_norm,
    configuration_graph,
    graph_arrays,
    sample_walks,
)

TABLE_DOMAIN = 2**31 - 1


def fix_table(model: Word2Vec) -> None:
    counts = model.wv.expandos["count"].astype(np.float64)
    cumulative = np.cumsum(counts ** float(model.ns_exponent))
    model.cum_table = np.rint(cumulative / cumulative[-1] * TABLE_DOMAIN).astype(
        np.uint32
    )


def curve(model: Word2Vec, degree: np.ndarray) -> dict[str, object]:
    n = len(degree)
    slots = np.array([model.wv.key_to_index[i] for i in range(n)])
    u = model.wv.vectors[slots].astype(np.float64)
    v = model.syn1neg[slots].astype(np.float64)
    out = {}
    log_d = np.log(degree)
    edges = np.unique(np.quantile(log_d, np.linspace(0, 1, 9)))
    labels = np.clip(np.digitize(log_d, edges[1:-1], right=True), 0, len(edges) - 2)
    for name, values in (
        ("raw", np.linalg.norm(u, axis=1)),
        ("bal", balanced_left_norm(u, v)),
    ):
        means = np.array(
            [
                values[labels == b].mean()
                for b in range(len(edges) - 1)
                if (labels == b).any()
            ]
        )
        peak = int(np.argmax(means))
        out[name] = {
            "bins": np.round(means, 3).tolist(),
            "peak_bin": peak,
            "interior": bool(0 < peak < len(means) - 1),
            "peak_minus_high": round(float(means[peak] - means[-1]), 3),
        }
    return out


def run_epochs(model, walks, schedule):
    """schedule: list of (start_alpha, end_alpha) per epoch."""
    for a0, a1 in schedule:
        model.train(
            walks,
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words,
            epochs=1,
            start_alpha=a0,
            end_alpha=a1,
        )


def linear(a0: float, a1: float, epochs: int):
    grid = np.linspace(a0, a1, epochs + 1)
    return list(zip(grid[:-1], grid[1:], strict=True))


def main(seeds: list[int]) -> None:
    results = []
    for seed in seeds:
        graph = configuration_graph(600, 3, seed)
        _, neighbors, degree = graph_arrays(graph)
        walks, _ = sample_walks(
            neighbors,
            degree / degree.sum(),
            600 * 300,
            5,
            np.random.default_rng(seed + 10_000),
        )
        base = Word2Vec(
            vector_size=64,
            window=1,
            min_count=50,
            sg=1,
            negative=5,
            ns_exponent=1.0,
            workers=1,
            epochs=5,
            seed=seed + 100_000,
            sample=0.0,
            alpha=0.025,
            min_alpha=0.0001,
            sorted_vocab=1,
            shrink_windows=False,
        )
        base.build_vocab(walks)
        fix_table(base)

        def record(label, model, epoch, rate_integral):
            row = {
                "seed": seed,
                "schedule": label,
                "epoch": epoch,
                "sum_alpha": round(rate_integral, 4),
                **curve(model, degree),
            }
            results.append(row)
            print(json.dumps(row), flush=True)

        t0 = time.time()
        # orig5 (+ re-heat at constant 0.0025)
        m = copy.deepcopy(base)
        run_epochs(m, walks, linear(0.025, 0.0001, 5))
        record("orig5", m, 5, 5 * 0.01255)
        run_epochs(m, walks, [(0.0025, 0.0025)] * 20)
        record("orig5+const", m, 25, 5 * 0.01255 + 20 * 0.0025)
        # gensim20
        m = copy.deepcopy(base)
        run_epochs(m, walks, linear(0.025, 0.0001, 20))
        record("gensim20", m, 20, 20 * 0.01255)
        # const with checkpoints, anneal fork at 20
        m = copy.deepcopy(base)
        done = 0
        for stop in (5, 20, 40, 80):
            run_epochs(m, walks, [(0.0025, 0.0025)] * (stop - done))
            done = stop
            record("const", m, stop, stop * 0.0025)
            if stop == 20:
                fork = copy.deepcopy(m)
                run_epochs(fork, walks, linear(0.0025, 0.0001, 5))
                record("const+anneal", fork, 25, 20 * 0.0025 + 5 * 0.0013)
        # high5 (+ anneal)
        m = copy.deepcopy(base)
        run_epochs(m, walks, [(0.025, 0.025)] * 5)
        record("high5", m, 5, 5 * 0.025)
        run_epochs(m, walks, linear(0.025, 0.0001, 5))
        record("high5+anneal", m, 10, 5 * 0.025 + 5 * 0.01255)
        print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    path = "<scratch>/round/r6/lr_schedule_results.json"
    with open(path, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
