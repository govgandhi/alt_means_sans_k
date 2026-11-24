"""Which part of the first-order SGD penalty shrinks hubs? Ablated modified-loss flows.

R = E sum_k ||grad l_k||^2 (t2_modified_flow.py) splits, per example (a, b) with K
negatives, into
  center, positive  r^2 ||v_b||^2
  center, cross     -2 K r v_b . m_a
  center, negative  K sum_c q_c p_ac^2 ||v_c||^2 + K(K-1) ||m_a||^2
  context, positive r^2 ||u_a||^2
  context, negative ||u_a||^2 (K sum_c q_c p_ac^2 + K(K-1) sum_c q_c^2 p_ac^2)
Variants: all; center (center terms only); context; positive (positive terms only);
negative (negative terms only); nocross (all but the cross term).
Flow to tau = 0.0628; saves balanced norms and the final factors.
Output: results/norm_theory/t2_abl_<variant>_eps<eps>_s<seed>.npz
Run: uv run --no-project --python 3.13 --with torch --with numpy \
         python -I scripts/norm_theory/t2_ablation.py center 0.0025 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2_modified_flow import ROOT, balanced_left_norm  # noqa: E402

TAU = 0.0628
USE = {
    "all": dict(cp=1, cx=1, cn=1, xp=1, xn=1),
    "center": dict(cp=1, cx=1, cn=1, xp=0, xn=0),
    "context": dict(cp=0, cx=0, cn=0, xp=1, xn=1),
    "positive": dict(cp=1, cx=0, cn=0, xp=1, xn=0),
    "negative": dict(cp=0, cx=0, cn=1, xp=0, xn=1),
    "nocross": dict(cp=1, cx=0, cn=1, xp=1, xn=1),
}


def objective(W, C, P, N, q, K, eps, u):
    S = W @ C.T
    loss = (P * torch.nn.functional.softplus(-S)).sum() + (
        N * torch.nn.functional.softplus(S)
    ).sum()
    if eps == 0:
        return loss
    r, p = torch.sigmoid(-S), torch.sigmoid(S)
    v2, u2 = (C * C).sum(1), (W * W).sum(1)
    m = (p * q[None, :]) @ C
    n = P.sum(1)
    p2 = p * p
    terms = (
        u["cp"] * (P * r * r * v2[None, :]).sum()
        + u["cx"] * (P * (-2 * K * r * (m @ C.T))).sum()
        + u["cn"]
        * (n * (K * ((p2 * q[None, :]) @ v2) + K * (K - 1) * (m * m).sum(1))).sum()
        + u["xp"] * (P * r * r * u2[:, None]).sum()
        + u["xn"]
        * (
            n
            * u2
            * (
                K * (p2 * q[None, :]).sum(1)
                + K * (K - 1) * (p2 * (q * q)[None, :]).sum(1)
            )
        ).sum()
    )
    return loss + 0.25 * eps * terms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=list(USE))
    ap.add_argument("eps", type=float)
    ap.add_argument("seed", type=int)
    ap.add_argument("--step", type=float, default=1e-5)
    a = ap.parse_args()
    torch.set_num_threads(1)
    z = np.load(ROOT / f"results/norm_theory/t2_setup_s{a.seed}.npz")
    dt = torch.float64
    P, N, q = (torch.tensor(z[k], dtype=dt) for k in ("P", "N", "q"))
    K = int(z["negatives"])
    W = torch.tensor(z["W0"], dtype=dt, requires_grad=True)
    C = torch.tensor(z["C0"], dtype=dt, requires_grad=True)
    steps = int(round(TAU / a.step))
    for _ in range(steps):
        gW, gC = torch.autograd.grad(
            objective(W, C, P, N, q, K, a.eps, USE[a.variant]), (W, C)
        )
        with torch.no_grad():
            W -= a.step * gW
            C -= a.step * gC
    Wn, Cn = W.detach().numpy(), C.detach().numpy()
    np.savez_compressed(
        ROOT / f"results/norm_theory/t2_abl_{a.variant}_eps{a.eps}_s{a.seed}.npz",
        tau=TAU,
        balanced_norm=balanced_left_norm(Wn, Cn),
        W=Wn,
        C=Cn,
        degree=z["degree"],
        step=a.step,
    )
    print(f"{a.variant} eps {a.eps} s{a.seed} done", flush=True)


if __name__ == "__main__":
    main()
