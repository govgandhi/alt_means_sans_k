"""Simulate deuteranopia/protanopia (Machado et al. 2009, severity 1.0) on PNGs.

Usage: python cvd.py out_dir in1.png [in2.png ...]
Also prints the CIE76 Delta-E between named matplotlib colors before/after simulation.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

DEUTER = np.array(
    [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]
)
PROTAN = np.array(
    [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]
)


def srgb_to_lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055) * 255.0


def simulate(rgb, m):
    lin = srgb_to_lin(rgb.astype(float))
    out = lin @ m.T
    return lin_to_srgb(out)


def to_lab(rgb):
    lin = srgb_to_lin(np.asarray(rgb, float))
    M = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )
    xyz = lin @ M.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def de(c1, c2):
    return float(np.linalg.norm(to_lab(c1) - to_lab(c2)))


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], float)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(exist_ok=True)
    for p in sys.argv[2:]:
        im = np.asarray(Image.open(p).convert("RGB"))
        for tag, m in (("deut", DEUTER), ("prot", PROTAN)):
            sim = simulate(im, m).astype(np.uint8)
            Image.fromarray(sim).save(out / f"{Path(p).stem}_{tag}.png")
    palettes = {
        "tab10": {
            "blue": "#1f77b4",
            "orange": "#ff7f0e",
            "green": "#2ca02c",
            "red": "#d62728",
            "purple": "#9467bd",
        },
        "okabe": {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73"},
    }
    for pal, cols in palettes.items():
        names = list(cols)
        print(f"palette {pal}: pairwise Delta-E76 normal -> deuteranopia / protanopia")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = hexrgb(cols[names[i]]), hexrgb(cols[names[j]])
                ad, bd = simulate(a[None], DEUTER)[0], simulate(b[None], DEUTER)[0]
                ap, bp = simulate(a[None], PROTAN)[0], simulate(b[None], PROTAN)[0]
                print(
                    f"  {names[i]:>6}-{names[j]:<6} {de(a, b):6.1f} -> {de(ad, bd):6.1f} / {de(ap, bp):6.1f}"
                )
