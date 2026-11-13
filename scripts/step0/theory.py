"""Population-theory line for the step-0 null (expected graph of the weighted
configuration model).

With walks started in proportion to strength, every token of a walk on the expected
graph is an
independent draw from pi (strength share). For input w and output v, Gensim's skip-gram
expected
positive and negative weights give the optimal score
    a(v) = log( pi'(v) / (b * q(v) * (1 - pi'(v))) ),
pi' = retained share after subsampling, q = negative-sampling distribution
(count^exponent),
and (1 - pi') from Gensim skipping a negative equal to the true target. The score matrix
is
1 a^T (rank one): every input row is identical, so center norms are flat in any gauge
with full-
rank context factors; in the balanced gauge the common norm is (mean_v a(v)^2)^(1/4).
The same holds for the Barabasi-Albert expected graph of Fig. 1(d) (--corpus ba), with
pi = degree share.
Inputs: results/step0/runs/<corpus>_{A,B}_s*.npz (vocabulary and counts Gensim kept).
Output: --out (default results/step0/theory.json). Run: python scripts/step0/theory.py
  --runs results/step0/runs [--corpus ba --out results/step0/theory_ba.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulate_norm_frequency import balanced_left_norm  # noqa: E402

NEGATIVES = 5
SETTINGS = {
    "A": {"sample": 1e-3, "exponent": 0.75},
    "B": {"sample": 0.0, "exponent": 1.0},
    "SW": {"sample": 1e-4, "exponent": 0.75},
}


def keep_probability(counts: np.ndarray, sample: float) -> np.ndarray:
    if sample == 0:
        return np.ones_like(counts, dtype=float)
    threshold = (
        sample * counts.sum()
    )  # Gensim 4: threshold_count = sample * retain_total
    return np.minimum(1.0, (np.sqrt(counts / threshold) + 1) * (threshold / counts))


def scores(counts: np.ndarray, sample: float, exponent: float) -> np.ndarray:
    retained = counts * keep_probability(counts, sample)
    pi = retained / retained.sum()
    q = counts**exponent / (counts**exponent).sum()
    return np.log(pi / (NEGATIVES * q * (1 - pi)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--corpus", default="null")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = {}
    for column, s in SETTINGS.items():
        levels, checks = [], []
        paths = sorted(a.runs.glob(f"{a.corpus}_{column}_s*.npz"))
        if not paths:
            continue
        for path in paths:
            counts = np.load(path)["counts"].astype(float)
            score = scores(counts, s["sample"], s["exponent"])
            level = float(np.mean(score**2) ** 0.25)
            numeric = balanced_left_norm(np.ones((len(score), 1)), score[:, None])
            checks.append(float(np.max(np.abs(numeric - level))))
            levels.append(
                {
                    "run": path.name,
                    "level": level,
                    "nodes": len(score),
                    "score_min": float(score.min()),
                    "score_max": float(score.max()),
                    "outside_gensim_range": int(np.sum(np.abs(score) >= 6)),
                }
            )
        out[column] = {
            "level_mean": float(np.mean([r["level"] for r in levels])),
            "level_range": [
                min(r["level"] for r in levels),
                max(r["level"] for r in levels),
            ],
            "max_numeric_vs_formula_error": max(checks),
            "runs": levels,
        }
    out["note"] = (
        "Flat in any gauge (context factors full rank); "
        "level shown is the balanced gauge."
    )
    (a.out or a.runs.parent / "theory.json").write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {
                c: {k: v for k, v in out[c].items() if k != "runs"}
                for c in SETTINGS
                if c in out
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
