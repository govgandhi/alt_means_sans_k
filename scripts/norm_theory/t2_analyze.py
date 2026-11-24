"""Compare SGD at matched tau with the modified-loss flow and the diffusion model.

PR #36 graphs, tau = 0.0628 (and 0.1), 5 seeds, degree bins 1-2, 3-4, 5-6, 7-9,
10-13, 14+. For every source: absolute bin means of balanced norms, mean-normalized
profile, top-bin gap (top minus preceding bin of the normalized profile), and the
per-node shift log(norm / norm_GF) after mean normalization. Model shifts use our
own eps = 0 flow as reference (same integrator); SGD shifts use the replica's
gradient flow (clock_replica.py gf-exact), as in stepsize_specificity.py.
Output: results/norm_theory/t2.json
Run: .venv/bin/python -I scripts/norm_theory/t2_analyze.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/norm_tests"))
from analyze import PR36_EDGES, interp_snap  # noqa: E402

R = ROOT / "results/norm_tests/clock"
M = ROOT / "results/norm_theory"
EPS = ["0.000625", "0.00125", "0.0025", "0.00625", "0.0125", "0.025"]


def load(path, t):
    if not path.exists():
        return None
    z = np.load(path)
    if (
        "tau" in z.files
        and len(z["tau"]) == 5
        and np.allclose(z["tau"], [0.0225, 0.04, 0.0526, 0.0628, 0.1])
        and path.parent == M
    ):
        i = int(np.argmin(np.abs(z["tau"] - t)))
        return z["balanced_norm"][i], z["degree"]
    v = interp_snap(z, "balanced_norm", t)
    return (None if v is None else v), z["degree"]


def bins(deg):
    return np.digitize(deg, PR36_EDGES) - 1


def summarize(vals, degs):
    absm, prof, gaps = [], [], []
    for v, d in zip(vals, degs):
        lab = bins(d)
        a = np.array([v[lab == b].mean() for b in range(6)])
        p = np.array([(v / v.mean())[lab == b].mean() for b in range(6)])
        absm.append(a)
        prof.append(p)
        gaps.append(p[-1] - p[-2])
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))  # noqa: E731
    return dict(
        abs_bins=np.round(np.mean(absm, 0), 3).tolist(),
        profile=np.round(np.mean(prof, 0), 4).tolist(),
        gap=round(float(np.mean(gaps)), 4),
        gap_se=round(se(gaps), 4),
    )


def shift(v, ref):
    return np.log((v / v.mean()) / (ref / ref.mean()))


def main() -> None:
    out = {}
    for t in (0.0628, 0.1):
        res = {}
        ref_sgd = [load(R / f"replica_gf-exact_gf_s{s}.npz", t) for s in range(5)]
        ref_mod = [load(M / f"t2_flow_eps0.0_s{s}.npz", t) for s in range(5)]
        res["gf_replica"] = summarize([r[0] for r in ref_sgd], [r[1] for r in ref_sgd])
        if all(r is not None for r in ref_mod):
            res["flow_eps0"] = summarize(
                [r[0] for r in ref_mod], [r[1] for r in ref_mod]
            )
            res["flow_eps0_vs_replica_max_rel"] = float(
                max(np.abs(a[0] / b[0] - 1).max() for a, b in zip(ref_mod, ref_sgd))
            )
        for e in EPS:
            entry = {}
            sources = {
                "gensim": [R / f"gensim_{e}_s{s}.npz" for s in range(5)],
                "replica_exact": [R / f"replica_exact_{e}_s{s}.npz" for s in range(5)],
                "flow": [M / f"t2_flow_eps{float(e)}_s{s}.npz" for s in range(5)],
                "diff_neg": [
                    M / f"t2_diff_neg_eps{float(e)}_s{s}.npz" for s in range(5)
                ],
                "diff_full": [
                    M / f"t2_diff_full_eps{float(e)}_s{s}.npz" for s in range(5)
                ],
            }
            loaded = {}
            for k, paths in sources.items():
                vs = [load(p, t) for p in paths]
                if k in ("flow", "diff_neg", "diff_full"):
                    vs = [v for v in vs if v is not None and v[0] is not None]
                    vs = vs if len(vs) >= 3 else [None]
                if all(v is not None and v[0] is not None for v in vs):
                    loaded[k] = vs
                    entry[k] = summarize([v[0] for v in vs], [v[1] for v in vs])
            for target in ("gensim", "replica_exact"):
                if target not in loaded:
                    continue
                sgd = [shift(v[0], r[0]) for v, r in zip(loaded[target], ref_sgd)]
                for k in ("flow", "diff_neg", "diff_full"):
                    if k not in loaded:
                        continue
                    seeds = min(len(loaded[k]), len(sgd))
                    mod = [shift(v[0], r[0]) for v, r in zip(loaded[k], ref_mod)]
                    rho, slope, within = [], [], []
                    for a, b, d in zip(
                        sgd[:seeds], mod[:seeds], [v[1] for v in loaded[target]]
                    ):
                        rho.append(spearmanr(a, b).statistic)
                        slope.append(float(np.polyfit(b, a, 1)[0]))
                        lab = bins(d)
                        rr = [
                            spearmanr(a[lab == j], b[lab == j]).statistic
                            for j in range(6)
                            if (lab == j).sum() >= 20
                        ]
                        within.append(np.mean(rr))
                    entry[f"{k}_vs_{target}_shift"] = dict(
                        spearman=round(float(np.mean(rho)), 3),
                        within_bin_spearman=round(float(np.mean(within)), 3),
                        slope_sgd_on_model=round(float(np.mean(slope)), 3),
                    )
            res[e] = entry
        out[str(t)] = res
    (M / "t2.json").write_text(json.dumps(out, indent=1))
    for t, res in out.items():
        print("tau", t)
        print(" gf_replica", res["gf_replica"]["gap"], res["gf_replica"]["abs_bins"])
        if "flow_eps0" in res:
            print(
                " flow_eps0 ",
                res["flow_eps0"]["gap"],
                res["flow_eps0"]["abs_bins"],
                "max rel diff",
                round(res["flow_eps0_vs_replica_max_rel"], 4),
            )
        for e in EPS:
            en = res[e]
            line = [f"  {e}:"]
            for k in ("gensim", "replica_exact", "flow", "diff_neg", "diff_full"):
                if k in en:
                    line.append(
                        f"{k} gap {en[k]['gap']:+.3f} top {en[k]['abs_bins'][-1]:.2f}"
                    )
            print(" | ".join(line))
            for key, val in en.items():
                if key.endswith("_shift"):
                    print("     ", key, val)


if __name__ == "__main__":
    main()
