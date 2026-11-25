"""Diffusion alternative for the step-size effect: gradient flow plus matched SGD noise.

Euler-Maruyama in tau for dtheta = -grad L dtau + sqrt(eps) Sigma^(1/2) dB, where
Sigma is the per-epoch covariance of the sum of Gensim's per-example gradients at
fixed parameters (so one epoch of SGD, tau-length eps, carries noise eps^2 Sigma).
'neg': only the negative-sampling noise. Gensim visits every positive pair once per
  epoch in a fixed order, so positives add no noise to the epoch sum; each of the
  n_a K negative draws for input a adds x = p_ac v_c to grad u_a and p_ac u_a to
  grad v_c, so Cov(grad u_a) = n_a K (sum_c q_c x x^T - m_a m_a^T). Sampled as
  sqrt(n_a K q_c) Z_ac (p_ac v_c - m_a) and sqrt(n_a K q_c) Z_ac p_ac u_a with one
  Gaussian Z for both (same draws move both vectors).
'full': adds positive-pair noise as if pairs were drawn with replacement,
  sqrt(P_ab) Z'_ab r_ab (v_b, u_a), with r = sigma(-s).
Output: results/norm_theory/t2_diff_<kind>_eps<eps>_s<seed>.npz
Run: uv run --no-project --python 3.13 --with torch --with numpy \
         python -I scripts/norm_theory/t2_diffusion.py neg 0.0025 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2_modified_flow import ROOT, TAUS, balanced_left_norm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["neg", "full"])
    ap.add_argument("eps", type=float)
    ap.add_argument("seed", type=int)
    ap.add_argument("--step", type=float, default=5e-6)
    a = ap.parse_args()
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(12_345 + 100 * a.seed + int(a.kind == "full"))
    z = np.load(ROOT / f"results/norm_theory/t2_setup_s{a.seed}.npz")
    dt = torch.float64
    P, N, q = (torch.tensor(z[k], dtype=dt) for k in ("P", "N", "q"))
    K = int(z["negatives"])
    W = torch.tensor(z["W0"], dtype=dt)
    C = torch.tensor(z["C0"], dtype=dt)
    n = P.sum(1)
    sq = torch.sqrt(q)
    sP = torch.sqrt(P)
    amp = np.sqrt(a.eps * a.step)
    marks = list(TAUS)
    snaps, t = [], 0.0
    nodes = W.shape[0]
    while marks:
        S = W @ C.T
        p, r = torch.sigmoid(S), torch.sigmoid(-S)
        D = -P * r + N * p
        gW, gC = D @ C, D.T @ W
        Z = torch.randn(nodes, nodes, generator=gen, dtype=dt)
        A = torch.sqrt(n * K)[:, None] * sq[None, :] * Z  # (a, c)
        m = (p * q[None, :]) @ C
        xiW = (A * p) @ C - (Z @ sq)[:, None] * torch.sqrt(n * K)[:, None] * m
        xiC = (A * p).T @ W
        if a.kind == "full":
            Zp = torch.randn(nodes, nodes, generator=gen, dtype=dt)
            B = sP * Zp * r
            xiW = xiW - B @ C
            xiC = xiC - B.T @ W
        W = W - a.step * gW - amp * xiW
        C = C - a.step * gC - amp * xiC
        t += a.step
        if t >= marks[0] - 1e-12:
            snaps.append(balanced_left_norm(W.numpy(), C.numpy()))
            S = W @ C.T
            loss = (P * torch.nn.functional.softplus(-S)).sum() + (
                N * torch.nn.functional.softplus(S)
            ).sum()
            print(
                f"{a.kind} eps {a.eps} s{a.seed} tau {marks[0]} "
                f"loss {float(loss / P.sum()):.5f}",
                flush=True,
            )
            marks.pop(0)
    np.savez_compressed(
        ROOT / f"results/norm_theory/t2_diff_{a.kind}_eps{a.eps}_s{a.seed}.npz",
        tau=np.array(TAUS),
        balanced_norm=np.array(snaps),
        degree=z["degree"],
        step=a.step,
    )


if __name__ == "__main__":
    main()
