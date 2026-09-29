#!/usr/bin/env python3
"""Generate core/test/fixtures/fixtures.json — a fixed set of MNIST test images
plus expected predictions computed with the same math the JS engine uses.
Tests the engines against the currently installed models/*.json (demo models
right after smoke training; regenerate after swapping in real Kaggle models)."""
import base64
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle" / "src"))

# exec both notebook-common sources into one namespace (same as flattening)
_ns: dict = {}
exec((ROOT / "kaggle" / "src" / "_common.py").read_text(), _ns)
exec((ROOT / "kaggle" / "src" / "_torch_common.py").read_text(), _ns)
_load_idx_images = _ns["_load_idx_images"]
_load_idx_labels = _ns["_load_idx_labels"]
b64_u8 = _ns["b64_u8"]
np_forward = _ns["np_forward"]
mlp_json_forward = _ns["mlp_json_forward"]
softmax_np = _ns["softmax_np"]

N = 256
IDX = ROOT / "local" / "data" / "mnist"
OUT = ROOT / "core" / "test" / "fixtures" / "fixtures.json"


def svc_json_forward(blob, X01):
    m = blob["model"]
    sv = np.frombuffer(base64.b64decode(m["sv_u8"]), np.uint8).astype(np.float32).reshape(-1, 784) / 255.0
    dual = np.frombuffer(base64.b64decode(m["dual_f32"]), "<f4").reshape(-1, 10)
    b = np.frombuffer(base64.b64decode(m["intercept_f32"]), "<f4")
    g = m["gamma"]
    scores = np.empty((len(X01), 10), np.float32)
    sv2 = (sv ** 2).sum(1)[None, :]
    for i0 in range(0, len(X01), 256):
        xb = X01[i0:i0 + 256]
        d2 = np.maximum((xb ** 2).sum(1, keepdims=True) + sv2 - 2.0 * (xb @ sv.T), 0.0)
        scores[i0:i0 + 256] = np.exp(-g * d2) @ dual + b[None, :]
    return softmax_np(scores)


def main():
    X = _load_idx_images(f"{IDX}/t10k-images-idx3-ubyte")
    y = _load_idx_labels(f"{IDX}/t10k-labels-idx1-ubyte")
    rng = np.random.default_rng(123)
    idx = np.concatenate([
        rng.choice(np.where(y == c)[0], N // 10 + (1 if c < N % 10 else 0), replace=False)
        for c in range(10)
    ])
    imgs = X[idx]
    X01 = imgs.astype(np.float32) / 255.0

    fixtures = {
        "n": int(N),
        "labels": y[idx].tolist(),
        "images_u8": b64_u8(imgs),
        "models": {},
    }
    for kind, path, fwd in [
        ("svc", ROOT / "models" / "svc_mnist.json", lambda mj: svc_json_forward(mj, X01)),
        ("mlp", ROOT / "models" / "mlp_mnist.json", lambda mj: mlp_json_forward(mj["model"]["layers"], X01)),
        ("cnn", ROOT / "models" / "cnn_mnist.json", lambda mj: np_forward(mj["model"]["ops"], X01)),
    ]:
        mj = json.loads(path.read_text())
        probs = fwd(mj).astype(np.float64)
        fixtures["models"][kind] = {
            "preds": probs.argmax(1).tolist(),
            "probs": [[round(float(p), 6) for p in row] for row in probs],
        }
        acc = float((probs.argmax(1) == y[idx]).mean())
        print(f"{kind}: fixture accuracy on sampled set = {acc*100:.2f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixtures))
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
