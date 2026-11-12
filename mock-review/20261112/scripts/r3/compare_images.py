"""R3 check: pixel-compare a committed figure against a scratch regeneration.

Usage: python compare_images.py <committed.pdf|png> <regenerated.pdf|png> [dpi]
PDFs are rasterized with pdftoppm into a temporary directory. Reports image sizes,
fraction of pixels whose RGB differs by more than 8/255, and the bounding box of the
differing region, so that label-only and data-level differences can be told apart.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def load(path: Path, dpi: int, tmp: Path) -> np.ndarray:
    if path.suffix.lower() == ".pdf":
        stem = tmp / (path.stem + str(abs(hash(str(path)))))
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(path), str(stem)],
            check=True,
        )
        path = stem.with_suffix(".png")
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.int16)


def main() -> None:
    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    with tempfile.TemporaryDirectory() as tmp:
        a = load(a_path, dpi, Path(tmp))
        b = load(b_path, dpi, Path(tmp))
    print(f"committed   {a_path.name}: {a.shape[1]}x{a.shape[0]}")
    print(f"regenerated {b_path.name}: {b.shape[1]}x{b.shape[0]}")
    if a.shape != b.shape:
        print("RESULT: size differs; cannot pixel-compare directly")
        return
    diff = np.abs(a - b).max(axis=2)
    mask = diff > 8
    frac = mask.mean()
    print(
        f"pixels differing (>8/255): {mask.sum()} ({100 * frac:.4f}%)  max diff {diff.max()}"
    )
    if mask.any():
        ys, xs = np.nonzero(mask)
        print(
            f"diff bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}] of {a.shape[1]}x{a.shape[0]}"
        )
    print(
        "RESULT:",
        "IDENTICAL"
        if not mask.any()
        else ("NEAR-IDENTICAL" if frac < 1e-4 else "DIFFERENT"),
    )


if __name__ == "__main__":
    main()
