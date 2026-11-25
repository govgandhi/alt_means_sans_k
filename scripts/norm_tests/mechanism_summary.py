"""Summaries for the step-size mechanism: absolute lengths, swap trajectories, theory.

PR #36 graphs, 5 seeds, degree bins 1-2 ... 14+ (analyze.py).
- abs: absolute balanced length by bin at tau = 0.0628 for gradient flow and three rates
- history: top-bin gap and loss against summed learning rate for gradient flow,
  constant 0.000625, constant 0.025, and the two swap schedules (swap_gensim.py)
- theory: top-bin gap at tau = 0.0628 for Gensim, the exact SGD copy, and the
  first-order modified-loss flow (results/norm_theory/t2.json, when present)
Output: results/norm_tests/mechanism.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/norm_tests"))
from analyze import PR36_EDGES, PR36_LABELS, interp_snap, pr36_profile  # noqa: E402

R = ROOT / "results/norm_tests"


def load(name, d):
    return [np.load(R / d / f"{name}_s{s}.npz") for s in range(5)]


def abs_bins(zs, t):
    out = []
    for z in zs:
        b = interp_snap(z, "balanced_norm", t)
        lab = np.digitize(z["degree"], PR36_EDGES) - 1
        out.append([b[lab == k].mean() for k in range(len(PR36_LABELS))])
    return np.array(out)


def trajectory(zs):
    taus = zs[0]["tau"]
    gaps, loss = [], []
    for j in range(1, len(taus)):
        g = [
            (lambda p: p[-1] - p[-2])(pr36_profile(z["degree"], z["balanced_norm"][j]))
            for z in zs
        ]
        gaps.append(g)
        loss.append(np.mean([z["expected_loss"][j] for z in zs]))
    g = np.array(gaps)
    return dict(
        tau=[round(float(t), 5) for t in taus[1:]],
        gap=[round(float(v), 4) for v in g.mean(1)],
        se=[round(float(v), 4) for v in g.std(1, ddof=1) / np.sqrt(g.shape[1])],
        loss=[round(float(v), 4) for v in loss],
    )


def main() -> None:
    t = 0.0628
    runs = {
        "Gradient flow": ("replica_gf-exact_gf", "clock"),
        "0.000625": ("gensim_0.000625", "clock"),
        "0.0025": ("gensim_0.0025", "clock"),
        "0.025": ("gensim_0.025", "clock"),
    }
    out = {"labels": PR36_LABELS, "abs": {}, "history": {}}
    for k, (name, d) in runs.items():
        A = abs_bins(load(name, d), t)
        out["abs"][k] = dict(
            mean=[round(float(v), 3) for v in A.mean(0)],
            se=[round(float(v), 3) for v in A.std(0, ddof=1) / np.sqrt(len(A))],
        )
    hist = {
        "Gradient flow": ("replica_gf-exact_gf", "clock"),
        "Constant 0.000625": ("gensim_0.000625", "clock"),
        "Constant 0.025": ("gensim_0.025", "clock"),
        "0.0314, then 0.000625": ("gensim_big2small", "swap"),
        "0.000625, then 0.0314": ("gensim_small2big", "swap"),
    }
    for k, (name, d) in hist.items():
        out["history"][k] = trajectory(load(name, d))
    out["stochasticity"] = {}
    for r in ("0.0314", "0.00628"):
        for v in ("sampled", "expected", "shuffled"):
            zs = [np.load(R / "stochasticity" / f"{v}_{r}_s{s}.npz") for s in range(5)]
            g = np.array(
                [
                    (lambda p: p[-1] - p[-2])(
                        pr36_profile(z["degree"], z["balanced_norm"][-1])
                    )
                    for z in zs
                ]
            )
            A = np.array(
                [
                    [
                        z["balanced_norm"][-1][
                            np.digitize(z["degree"], PR36_EDGES) - 1 == k
                        ].mean()
                        for k in range(len(PR36_LABELS))
                    ]
                    for z in zs
                ]
            )
            out["stochasticity"][f"{v}:{r}"] = dict(
                gap=round(float(g.mean()), 4),
                se=round(float(g.std(ddof=1) / np.sqrt(len(g))), 4),
                abs=[round(float(x), 3) for x in A.mean(0)],
                loss=round(float(np.mean([z["expected_loss"][-1] for z in zs])), 4),
            )
    t2 = ROOT / "results/norm_theory/t2.json"
    if t2.exists():
        out["theory"] = json.loads(t2.read_text()).get(str(t))
    (R / "mechanism.json").write_text(json.dumps(out, indent=1))
    for k, v in out["abs"].items():
        print(k, v["mean"])


if __name__ == "__main__":
    main()
