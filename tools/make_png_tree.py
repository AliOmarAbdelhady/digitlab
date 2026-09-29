#!/usr/bin/env python3
"""Create local/data/pngtree — a small local mirror of the Kaggle dataset layout
(mnist_png/training/<0..9>/*.png) — so local smoke tests exercise the exact
Kaggle PNG loading code path."""
import pathlib
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle" / "src"))
from _common import _load_idx_images, _load_idx_labels  # noqa: E402

IDX = ROOT / "local" / "data" / "mnist"
TREE = ROOT / "local" / "data" / "pngtree" / "mnist_png"
N_TRAIN, N_TEST = 3000, 600


def build(X, y, out: pathlib.Path, n: int, rng):
    idx = np.concatenate([rng.choice(np.where(y == c)[0], n // 10, replace=False)
                          for c in range(10)])
    for k, i in enumerate(idx):
        d = out / str(int(y[i]))
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(X[i].reshape(28, 28)).save(d / f"{k:05d}.png")
    return len(idx)


def main():
    rng = np.random.default_rng(42)
    Xtr = _load_idx_images(f"{IDX}/train-images-idx3-ubyte")
    ytr = _load_idx_labels(f"{IDX}/train-labels-idx1-ubyte")
    Xte = _load_idx_images(f"{IDX}/t10k-images-idx3-ubyte")
    yte = _load_idx_labels(f"{IDX}/t10k-labels-idx1-ubyte")
    n1 = build(Xtr, ytr, TREE / "training", N_TRAIN, rng)
    n2 = build(Xte, yte, TREE / "testing", N_TEST, rng)
    print(f"pngtree ready: training={n1} testing={n2} at {TREE}")


if __name__ == "__main__":
    main()
