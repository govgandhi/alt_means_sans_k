"""Gradient flow on the first-order SGD modified loss for Gensim's SGNS updates.

Smith, Dherin, Barrett & De (ICLR 2021, Eq. 1): random-reshuffling SGD with step eps
follows, to first order in eps, the flow of
    L_eps = L + (eps / 4) * sum_k ||grad l_k||^2,
L = sum_k l_k the per-epoch loss and l_k the loss of example k. In Gensim an example is
one positive pair (a, b) plus K negatives drawn from q, all sharing the input vector
u_a and applied in one step. With r = sigma(-s_ab), p_ac = sigma(s_ac),
m_a = sum_c q_c p_ac v_c, and expectations over the negatives,
    E||grad_u l_k||^2 = ||r v_b - K m_a||^2 + K (sum_c q_c p_ac^2 ||v_c||^2 - ||m_a||^2)
    E||grad_v l_k||^2 = ||u_a||^2 (r^2 + K sum_c q_c p_ac^2 + K(K-1) sum_c q_c^2 p_ac^2)
(the second line counts repeated negatives). Summing over the epoch's examples gives
R(W, C); the flow integrates dtheta/dtau = -grad(L + eps R / 4) by explicit Euler in
tau (summed learning rate) from Gensim's initial vectors, as clock_replica.py 'gf'.
Runs in an environment with torch (no Gensim needed); inputs from t2_export.py.
Output: results/norm_theory/t2_flow_eps<eps>_s<seed>.npz (balanced norms at TAUS)
Run: uv run --no-project --python 3.13 --with torch --with numpy \
         python -I scripts/norm_theory/t2_modified_flow.py 0.0025 0
"""

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TAUS = (0.0225, 0.04, 0.0526, 0.0628, 0.1)


def balanced_left_norm(W: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Same computation as simulate_norm_frequency.balanced_left_norm."""
    Ql, Rl = np.linalg.qr(W, mode="reduced")
    _, Rr = np.linalg.qr(C, mode="reduced")
    U, s, _ = np.linalg.svd(Rl @ Rr.T, full_matrices=False)
    return np.linalg.norm(Ql @ (U * np.sqrt(np.maximum(s, 0.0))[None, :]), axis=1)


def objective(W, C, P, N, q, K, eps):
    S = W @ C.T
    loss = (P * torch.nn.functional.softplus(-S)).sum() + (
        N * torch.nn.functional.softplus(S)
    ).sum()
    if eps == 0:
        return loss, loss
    r, p = torch.sigmoid(-S), torch.sigmoid(S)
    v2, u2 = (C * C).sum(1), (W * W).sum(1)
    m = (p * q[None, :]) @ C
    n = P.sum(1)
    pos = (P * (r * r * (v2[None, :] + u2[:, None]) - 2 * K * r * (m @ C.T))).sum()
    p2 = p * p
    A = (p2 * q[None, :]) @ v2
    B = (p2 * q[None, :]).sum(1)
    D = (p2 * (q * q)[None, :]).sum(1)
    m2 = (m * m).sum(1)
    neg = (n * (K * A + K * (K - 1) * m2 + u2 * (K * B + K * (K - 1) * D))).sum()
    return loss + 0.25 * eps * (pos + neg), loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eps", type=float)
    ap.add_argument("seed", type=int)
    ap.add_argument("--step", type=float, default=5e-6, help="Euler step in tau")
    a = ap.parse_args()
    torch.set_num_threads(1)
    z = np.load(ROOT / f"results/norm_theory/t2_setup_s{a.seed}.npz")
    dt = torch.float64
    P, N, q = (torch.tensor(z[k], dtype=dt) for k in ("P", "N", "q"))
    K = int(z["negatives"])
    W = torch.tensor(z["W0"], dtype=dt, requires_grad=True)
    C = torch.tensor(z["C0"], dtype=dt, requires_grad=True)
    marks = list(TAUS)
    snaps, losses, t = [], [], 0.0
    while marks:
        total, loss = objective(W, C, P, N, q, K, a.eps)
        gW, gC = torch.autograd.grad(total, (W, C))
        with torch.no_grad():
            W -= a.step * gW
            C -= a.step * gC
        t += a.step
        if t >= marks[0] - 1e-12:
            Wn, Cn = W.detach().numpy(), C.detach().numpy()
            snaps.append(balanced_left_norm(Wn, Cn))
            losses.append(float(loss) / float(P.sum()))
            print(
                f"eps {a.eps} s{a.seed} tau {marks[0]} loss {losses[-1]:.5f}",
                flush=True,
            )
            marks.pop(0)
    np.savez_compressed(
        ROOT / f"results/norm_theory/t2_flow_eps{a.eps}_s{a.seed}.npz",
        tau=np.array(TAUS),
        balanced_norm=np.array(snaps),
        expected_loss=np.array(losses),
        degree=z["degree"],
        step=a.step,
    )


if __name__ == "__main__":
    main()
