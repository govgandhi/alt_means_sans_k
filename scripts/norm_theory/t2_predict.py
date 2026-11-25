"""Closed-form per-node prediction of the step-size shrinkage from a trained model.

Inputs: gradient-flow factors at tau = 0.0628 (t2_ablation.py 'all' at eps = 0).
Near calibration the per-example penalty is about (eps/4) sum_pairs (P + N) s'(1 - s')
(||w_i||^2 + ||c_j||^2) with s' = sigma(s_ij), i.e. (eps/4) tr(Hessian). Treating the
partner vectors as fixed and moving node i only along its own radius gives
  radial:    dlog||w_i|| = -(eps/2) ||w_i||^2 / <s_ij^2>_i,
<.>_i the (P + N) s'(1 - s')-weighted mean over node i's pairs (as input; the column
version for context vectors c_j). The 'projected' predictor replaces the calibrated
trace by the exact first-order penalty R: dlog||w_i|| = -(eps/4) (grad_{w_i} R . w_i)
/ (||w_i||^2 kappa_i), kappa_i = sum_j (P + N) s'(1 - s') (w_i/||w_i|| . c_j)^2.
Targets at tau = 0.0628: modified-loss flows (t2_ablation.py 'all', raw and balanced
norms) and the exact-sigmoid SGD replica (results/norm_tests/clock/replica_exact_*),
each as the log ratio to the matching gradient flow.
Output: results/norm_theory/t2_predict.json
Run: uv run --no-project --python 3.13 --with torch --with numpy --with scipy \
         python -I scripts/norm_theory/t2_predict.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2_ablation import USE, objective  # noqa: E402
from t2_modified_flow import ROOT, balanced_left_norm  # noqa: E402

M = ROOT / "results/norm_theory"
CLOCK = ROOT / "results/norm_tests/clock"
EDGES = [1, 3, 5, 7, 10, 14, 10**9]
TAU = 0.0628


def interp(z, key, t):
    taus = z["tau"]
    j = int(np.searchsorted(taus, t))
    w = (t - taus[j - 1]) / (taus[j] - taus[j - 1])
    return (1 - w) * z[key][j - 1] + w * z[key][j]


def predictors(W, C, P, N, q, K, eps):
    S = W @ C.T
    sp = 1 / (1 + np.exp(-S))
    A = (P + N) * sp * (1 - sp)
    out = {}
    out["radial_w"] = -(eps / 2) * (W**2).sum(1) / ((A * S**2).sum(1) / A.sum(1))
    out["radial_c"] = -(eps / 2) * (C**2).sum(1) / ((A * S**2).sum(0) / A.sum(0))
    Wt = torch.tensor(W, requires_grad=True)
    Ct = torch.tensor(C, requires_grad=True)
    tot = objective(
        Wt, Ct, torch.tensor(P), torch.tensor(N), torch.tensor(q), K, 1.0, USE["all"]
    )
    loss = objective(
        Wt, Ct, torch.tensor(P), torch.tensor(N), torch.tensor(q), K, 0.0, USE["all"]
    )
    gW, gC = torch.autograd.grad(tot - loss, (Wt, Ct))  # = grad R / 4
    gW, gC = 4 * gW.numpy(), 4 * gC.numpy()
    wn, cn = np.linalg.norm(W, axis=1), np.linalg.norm(C, axis=1)
    kw = (A * (S / wn[:, None]) ** 2).sum(1)
    kc = (A * (S / cn[None, :]) ** 2).sum(0)
    out["projected_w"] = -(eps / 4) * (gW * W).sum(1) / (wn**2 * kw)
    out["projected_c"] = -(eps / 4) * (gC * C).sum(1) / (cn**2 * kc)
    out["radial_mean"] = 0.5 * (out["radial_w"] + out["radial_c"])
    out["projected_mean"] = 0.5 * (out["projected_w"] + out["projected_c"])
    # coupled radial response: x_i = dlog||w_i||, y_j = dlog||c_j||, scores change by
    # s_ij (x_i + y_j); quadratic loss in (x, y) with weights A s^2; solve H z = -rhs
    # (min-norm solution; H is singular along the gauge x = d, y = -d)
    n = len(W)
    B = A * S**2
    H = np.block([[np.diag(B.sum(1)), B], [B.T, np.diag(B.sum(0))]])
    rhs_trace = (eps / 2) * np.r_[A.sum(1) * wn**2, A.sum(0) * cn**2]
    rhs_exact = (eps / 4) * np.r_[(gW * W).sum(1), (gC * C).sum(1)]
    for name, rhs in (("coupled_trace", rhs_trace), ("coupled_exact", rhs_exact)):
        z = -np.linalg.lstsq(H, rhs, rcond=1e-10)[0]
        x, y = z[:n], z[n:]
        out[f"{name}_w"] = x
        out[f"{name}_c"] = y
        out[f"{name}_balanced"] = np.log(
            balanced_left_norm(W * np.exp(x)[:, None], C * np.exp(y)[:, None])
            / balanced_left_norm(W, C)
        )
    # free-drift limit: no restoring force over the trajectory, penalty gradient
    # integrated over tau at its endpoint value (dlog rho = -(eps/2) tau a_i)
    out["drift_trace_w"] = -(eps / 2) * TAU * A.sum(1)
    out["drift_trace_c"] = -(eps / 2) * TAU * A.sum(0)
    out["drift_exact_w"] = -(eps / 4) * TAU * (gW * W).sum(1) / wn**2
    out["drift_exact_c"] = -(eps / 4) * TAU * (gC * C).sum(1) / cn**2
    for name in ("drift_trace", "drift_exact"):
        x, y = out[f"{name}_w"], out[f"{name}_c"]
        out[f"{name}_balanced"] = np.log(
            balanced_left_norm(W * np.exp(x)[:, None], C * np.exp(y)[:, None])
            / balanced_left_norm(W, C)
        )
    out["fisher_mass_w"] = -A.sum(1)
    out["pair_loss_w"] = -(
        (P * np.logaddexp(0, -S) + N * np.logaddexp(0, S)).sum(1) / P.sum(1)
    )
    return out


def score(pred, actual, deg):
    lab = np.digitize(deg, EDGES) - 1
    within = [
        spearmanr(pred[lab == b], actual[lab == b]).statistic
        for b in range(6)
        if (lab == b).sum() >= 20
    ]
    slope = float(np.polyfit(pred, actual, 1)[0])
    return dict(
        spearman=float(spearmanr(pred, actual).statistic),
        within_bin=float(np.mean(within)),
        slope_actual_on_pred=slope,
        bin_pred=[float(pred[lab == b].mean()) for b in range(6)],
        bin_actual=[float(actual[lab == b].mean()) for b in range(6)],
    )


def main() -> None:
    torch.set_num_threads(1)
    res = {}
    for eps in [float(x) for x in (sys.argv[1:] or ["0.000625", "0.0025"])]:
        per = []
        for s in range(5):
            z = np.load(M / f"t2_setup_s{s}.npz")
            P, N, q, deg = z["P"], z["N"], z["q"], z["degree"]
            K = int(z["negatives"])
            g = np.load(M / f"t2_abl_all_eps0.0_s{s}.npz")
            f = np.load(M / f"t2_abl_all_eps{eps}_s{s}.npz")
            W, C = g["W"], g["C"]
            pr = predictors(W, C, P, N, q, K, eps)
            targets = {
                "flow_raw_w": np.log(
                    np.linalg.norm(f["W"], axis=1) / np.linalg.norm(W, axis=1)
                ),
                "flow_raw_c": np.log(
                    np.linalg.norm(f["C"], axis=1) / np.linalg.norm(C, axis=1)
                ),
                "flow_balanced": np.log(f["balanced_norm"] / g["balanced_norm"]),
            }
            rp = CLOCK / f"replica_exact_{eps}_s{s}.npz"
            if rp.exists():
                rz, rg = np.load(rp), np.load(CLOCK / f"replica_gf-exact_gf_s{s}.npz")
                targets["sgd_raw_w"] = np.log(
                    interp(rz, "raw_norm", TAU) / interp(rg, "raw_norm", TAU)
                )
                targets["sgd_balanced"] = np.log(
                    interp(rz, "balanced_norm", TAU) / interp(rg, "balanced_norm", TAU)
                )
            row = {}
            for tn, tv in targets.items():
                for pn, pv in pr.items():
                    if tn.endswith("_w") and not pn.endswith(("_w", "_mean")):
                        continue
                    if tn.endswith("_c") and not pn.endswith(("_c", "_mean")):
                        continue
                    if (
                        tn.endswith("_balanced")
                        and pn.startswith(("coupled", "drift"))
                        and not pn.endswith("_balanced")
                    ):
                        continue
                    row[f"{tn}|{pn}"] = score(pv, tv, deg)
            per.append(row)
        keys = per[0].keys()
        res[str(eps)] = {
            k: {
                m: (
                    float(np.mean([p[k][m] for p in per if k in p]))
                    if m in ("spearman", "within_bin", "slope_actual_on_pred")
                    else np.round(
                        np.mean([p[k][m] for p in per if k in p], 0), 4
                    ).tolist()
                )
                for m in (
                    "spearman",
                    "within_bin",
                    "slope_actual_on_pred",
                    "bin_pred",
                    "bin_actual",
                )
            }
            for k in keys
        }
        for k, v in res[str(eps)].items():
            print(
                eps,
                k,
                round(v["spearman"], 3),
                round(v["within_bin"], 3),
                round(v["slope_actual_on_pred"], 3),
                flush=True,
            )
    (M / "t2_predict.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
