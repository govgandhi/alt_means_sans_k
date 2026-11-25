"""Summarize the norm tests: fresh corpus, clock/saturation, window matching, frames.

Fig. 1 binning (quarter decades, bins with >= 10 nodes in every seed, position = mean
log x). PR #36 graphs use fixed degree bins 1-2, 3-4, 5-6, 7-9, 10-13, 14+; per graph
bin means of norm / graph mean, then mean +/- SE over the 5 graphs.
'upper gap' = top bin minus the preceding bin (negative = interior peak / decline).
Output: results/norm_tests/summary.json, results/norm_tests/frames.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/norm_tests"
COMMITTED = ROOT / "results/step0/runs"
PR36_EDGES = [1, 3, 5, 7, 10, 14, 10**9]
PR36_LABELS = ["1–2", "3–4", "5–6", "7–9", "10–13", "14+"]
TAUS = [0.0225, 0.04, 0.0526, 0.0628, 0.1]


def fig1_curve(xs, ys, min_per_bin=10):
    """xs, ys: lists over seeds. Returns position, mean, se (se=0 for one seed)."""
    allx = np.concatenate(xs)
    lo = np.floor(np.log10(allx.min()) * 4) / 4
    hi = np.ceil(np.log10(allx.max()) * 4) / 4
    edges = 10 ** np.arange(lo, hi + 0.25, 0.25)
    per = []
    for x, y in zip(xs, ys):
        y = y / y.mean()
        idx = np.digitize(x, edges)
        per.append(
            {
                b: (y[idx == b].mean(), (idx == b).sum(), np.log(x[idx == b]).mean())
                for b in np.unique(idx)
            }
        )
    keep = sorted(
        b for b in per[0] if all(b in s and s[b][1] >= min_per_bin for s in per)
    )
    m = np.array([[s[b][0] for b in keep] for s in per])
    pos = np.exp(np.array([[s[b][2] for b in keep] for s in per]).mean(0))
    se = m.std(0, ddof=1) / np.sqrt(len(m)) if len(m) > 1 else np.zeros(m.shape[1])
    return pos, m.mean(0), se


def rnd(a, k=4):
    return [round(float(v), k) for v in np.atleast_1d(a)]


def pack(pos, m, se):
    return dict(x=rnd(pos, 2), mean=rnd(m), se=rnd(se))


def peak_x(pos, m):
    i = int(np.argmax(m))
    return dict(x=round(float(pos[i]), 1), interior=bool(0 < i < len(m) - 1))


# ---------- test 1: fresh corpus ----------
def fresh():
    out = {}
    for corpus in ("ba", "blog"):
        for col in ("A", "B", "SW"):
            fr = [R / f"fresh/{corpus}_{col}_fresh_s{s}.npz" for s in range(5)]
            if not all(p.exists() for p in fr):
                continue
            cm = [np.load(COMMITTED / f"{corpus}_{col}_s{s}.npz") for s in range(5)]
            fz = [np.load(p) for p in fr]
            entry = {}
            for key in ("balanced_norm", "raw_norm"):
                reused = fig1_curve([z["x"] for z in cm], [z[key] for z in cm])
                fresh_ = fig1_curve([z["x"] for z in fz], [z[key][-1] for z in fz])
                entry[key] = dict(
                    reused=pack(*reused),
                    fresh=pack(*fresh_),
                    range_reused=round(float(np.ptp(reused[1])), 4),
                    range_fresh=round(float(np.ptp(fresh_[1])), 4),
                    peak_reused=peak_x(reused[0], reused[1]),
                    peak_fresh=peak_x(fresh_[0], fresh_[1]),
                )
            loop = R / f"fresh/{corpus}_{col}_reused_s0.npz"
            if loop.exists():
                lz = np.load(loop)
                entry["loop_check_max_abs_diff_balanced"] = float(
                    np.abs(lz["balanced_norm"][-1] - cm[0]["balanced_norm"]).max()
                )
                entry["loop_check_rel"] = float(
                    np.abs(lz["balanced_norm"][-1] - cm[0]["balanced_norm"]).max()
                    / cm[0]["balanced_norm"].mean()
                )
                # per-epoch range, reused vs fresh (seed 0)
                f0 = fz[0]
                entry["range_by_epoch"] = dict(
                    reused=[
                        round(float(np.ptp(fig1_curve([lz["x"]], [b])[1])), 4)
                        for b in lz["balanced_norm"]
                    ],
                    fresh=[
                        round(float(np.ptp(fig1_curve([f0["x"]], [b])[1])), 4)
                        for b in f0["balanced_norm"]
                    ],
                )
            out[f"{corpus}_{col}"] = entry
    return out


# ---------- tests 2-3: clock and saturation ----------
def pr36_profile(degree, values):
    v = values / values.mean()
    lab = np.digitize(degree, PR36_EDGES) - 1
    return np.array([v[lab == b].mean() for b in range(len(PR36_LABELS))])


def pr36_binmean(degree, values):
    lab = np.digitize(degree, PR36_EDGES) - 1
    return np.array([values[lab == b].mean() for b in range(len(PR36_LABELS))])


def interp_snap(z, key, t):
    taus = z["tau"]
    if t > taus[-1] + 1e-9 or t < taus[0]:
        return None
    j = int(np.searchsorted(taus, t))
    if j == 0:
        return z[key][0]
    w = (t - taus[j - 1]) / (taus[j] - taus[j - 1])
    return (1 - w) * z[key][j - 1] + w * z[key][j]


def clock():
    runs = {}
    for name in (
        ["gensim_orig"]
        + [
            f"gensim_{r}"
            for r in ("0.025", "0.0125", "0.00625", "0.0025", "0.00125", "0.000625")
        ]
        + [
            f"replica_{v}_{r}"
            for v in ("skip", "exact")
            for r in ("orig", "0.025", "0.0025", "0.000625")
        ]
        + ["replica_gf-exact_gf", "replica_gf-skip_gf"]
    ):
        paths = [R / f"clock/{name}_s{s}.npz" for s in range(5)]
        if all(p.exists() for p in paths):
            runs[name] = [np.load(p) for p in paths]
    out = {"labels": PR36_LABELS, "taus": TAUS, "runs": {}}
    for name, zs in runs.items():
        entry = {}
        for t in TAUS:
            profs, frozen, losses, nl = [], [], [], []
            for z in zs:
                b = interp_snap(z, "balanced_norm", t)
                if b is None:
                    break
                profs.append(pr36_profile(z["degree"], b))
                frozen.append(
                    pr36_binmean(z["degree"], interp_snap(z, "neg_frozen", t))
                )
                nl.append(pr36_binmean(z["degree"], interp_snap(z, "node_loss", t)))
                losses.append(float(interp_snap(z, "expected_loss", t)))
            if len(profs) < len(zs):
                continue
            P = np.array(profs)
            gaps = P[:, -1] - P[:, -2]
            entry[str(t)] = dict(
                profile=rnd(P.mean(0)),
                profile_se=rnd(P.std(0, ddof=1) / np.sqrt(len(P))),
                upper_gap=round(float(gaps.mean()), 4),
                upper_gap_se=round(float(gaps.std(ddof=1) / np.sqrt(len(gaps))), 4),
                top_minus_bottom=round(float((P[:, -1] - P[:, 0]).mean()), 4),
                interior_peaks=int(sum(0 < np.argmax(p) < len(p) - 1 for p in P)),
                neg_frozen_by_bin=rnd(np.mean(frozen, 0)),
                node_loss_by_bin=rnd(np.mean(nl, 0)),
                expected_loss=round(float(np.mean(losses)), 5),
            )
        out["runs"][name] = entry
    return out


# ---------- test 4: window matching ----------
def window():
    out = {}
    for c in ("w5e5", "w1e5", "w1e15", "w5lr3"):
        paths = [R / f"window/{c}_s{s}.npz" for s in range(5)]
        if not all(p.exists() for p in paths):
            continue
        zs = [np.load(p) for p in paths]
        e = {}
        for key in ("balanced_norm", "raw_norm"):
            cur = fig1_curve([z["x"] for z in zs], [z[key][-1] for z in zs])
            e[key] = pack(*cur) | dict(peak=peak_x(cur[0], cur[1]))
        out[c] = e
    return out


# ---------- frames for the animation ----------
def frames():
    out = {}
    for name in (
        "ba_B50_reused",
        "ba_B50_fresh",
        "blog_B50_reused",
        "blog_A50_reused",
        "text_B50_reused",
        "text_A50_reused",
    ):
        p = R / f"anim/{name}_s0.npz"
        if not p.exists():
            continue
        z = np.load(p)
        fr = []
        for b in z["balanced_norm"]:
            pos, m, _ = fig1_curve([z["x"]], [b])
            fr.append(dict(x=rnd(pos, 2), y=rnd(m)))
        out[name] = fr
    for c in ("w5e5", "w1e15"):
        p = R / f"window/{c}_s0.npz"
        if p.exists():
            z = np.load(p)
            out[f"window_{c}"] = [
                dict(
                    x=rnd(fig1_curve([z["x"]], [b])[0], 2),
                    y=rnd(fig1_curve([z["x"]], [b])[1]),
                )
                for b in z["balanced_norm"]
            ]
    return out


def main():
    summary = dict(fresh=fresh(), clock=clock(), window=window())
    (R / "summary.json").write_text(json.dumps(summary, indent=1))
    (R / "frames.json").write_text(json.dumps(frames()))
    print("wrote summary.json, frames.json")


if __name__ == "__main__":
    main()
