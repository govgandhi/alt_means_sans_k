# /// script
# dependencies = ["numpy", "numba", "networkx", "matplotlib"]
# ///
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from sgns import binned, OUT

COL = {"A": "#2f6fdb", "B": "#e5622d", "SW": "#1f9e72"}
LAB = {"A": "Gensim defaults", "B": "PNAS settings", "SW": "Schakel & Wilson"}
THEORY = {"A": 1.313, "B": 1.269, "SW": 1.32}
LAST = {"A": 5, "B": 5, "SW": 20}

def last(z):
    e = max(int(k.split("_")[1]) for k in z.files if k.startswith("bal_"))
    return z[f"bal_{e}"]

def panel(ax, prefix, title, theory=True):
    for s in ["A", "B", "SW"]:
        for kind, ls in [("rep", "-"), ("fresh", "--")]:
            f = OUT / f"{prefix}{s}_{kind}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            x, y = binned(z["deg"], last(z))
            ax.plot(x, y, ls, color=COL[s], lw=2, marker="o" if kind == "rep" else None, ms=4)
        if theory:
            ax.axhline(THEORY[s], color=COL[s], lw=1, ls=":")
    ax.set_xscale("log"); ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("node degree"); ax.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

has_graph = (OUT / "G_A_rep.npz").exists()
fig, axes = plt.subplots(1, 3 if has_graph else 2, figsize=(13 if has_graph else 9, 3.8))
panel(axes[0], "", "(a) BA expected graph: theory says flat")
axes[0].set_ylabel("balanced norm")
z = np.load(OUT / "B50_rep.npz")
for e, c in zip([5, 10, 20, 50], plt.cm.Oranges(np.linspace(0.35, 1.0, 4))):
    x, y = binned(z["deg"], z[f"bal_{e}"])
    axes[1].plot(x, y, "-o", color=c, lw=2, ms=3.5)
    axes[1].annotate(f"ep {e}", (x[0], y[0]), xytext=(-4, 0), textcoords="offset points",
                     ha="right", va="center", fontsize=8, color="0.3")
axes[1].axhline(THEORY["B"], color=COL["B"], lw=1, ls=":")
axes[1].set_xscale("log"); axes[1].set_xlabel("node degree")
axes[1].set_title("(b) BA expected, PNAS settings, one 50-epoch run", fontsize=10, loc="left")
axes[1].grid(alpha=0.25, lw=0.5); axes[1].set_xlim(2.2, 160)
for sp in ("top", "right"):
    axes[1].spines[sp].set_visible(False)
if has_graph:
    ax = axes[2]
    for s_ in ["A", "B", "SW"]:
        for kind, ls in [("rep", "-"), ("fresh", "--")]:
            z = np.load(OUT / f"G_{s_}_{kind}.npz")
            x, y = binned(z["deg"], last(z))
            ax.plot(x, y / y.mean(), ls, color=COL[s_], lw=2, marker="o" if kind == "rep" else None, ms=4)
    z = np.load(OUT / "G_B_ep20.npz")
    x, y = binned(z["deg"], z["bal_20"])
    ax.plot(x, y / y.mean(), "-o", color="#8c2d04", lw=2, ms=4)
    ax.annotate("PNAS settings,\n20 epochs", (x[3], (y / y.mean())[3]), xytext=(-8, 6), textcoords="offset points",
                ha="right", va="bottom", fontsize=8, color="#8c2d04")
    ax.set_xscale("log"); ax.set_xlabel("node degree"); ax.set_ylabel("balanced norm / mean")
    ax.set_title("(c) actual BA graph: fresh walks change nothing", fontsize=10, loc="left")
    ax.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=COL[s], lw=2, label=LAB[s]) for s in ["A", "B", "SW"]]
h += [Line2D([], [], color="0.3", lw=2, marker="o", ms=4, label="same corpus every epoch"),
      Line2D([], [], color="0.3", lw=2, ls="--", label="fresh corpus every epoch"),
      Line2D([], [], color="0.3", lw=1, ls=":", label="theory (exact optimum)")]
fig.legend(handles=h, ncol=3, loc="upper center", frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(OUT.parent / "norm_degree_diagnosis.png", dpi=180)
print("saved")
