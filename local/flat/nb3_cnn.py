# %% [markdown]
# ## Common — data loading, preprocessing, evaluation, export helpers
#
# **DigitLab · MNIST 3-model project** (shared cells inlined into every notebook)
#
# ### Dataset
# - **Kaggle dataset:** [`hojjatk/mnist-dataset`](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
#   ("The MNIST Database") — the canonical full MNIST: **60,000 training + 10,000 test**
#   28×28 grayscale PNGs, digit *white on black*, one folder per class:
#   `/kaggle/input/mnist-dataset/mnist_png/{training,testing}/<0..9>/*.png`
# - Attach it via **Add Input → Datasets → "MNIST Dataset" (hojjatk)**.
#
# ### Preprocessing (identical for training AND serving in the apps)
# 1. Load PNG → 28×28 uint8 array `[0..255]`, flatten to 784.
# 2. Scale to `float32` in `[0,1]` (**÷255**). No mean/std standardization — this keeps
#    the exact same numeric pipeline trivially reproducible in the JavaScript inference
#    core used by the web/mobile apps (train/serve parity).
# 3. Model selection splits (never touching the official test set until the end):
#    - hyperparameter search / early stopping: stratified **54k train / 6k val** from the 60k;
#    - final reported metrics: the official **10k test** set.
# 4. NN notebooks add *training-only* geometric augmentation (rotation / translation /
#    scaling with zero padding) to reduce overfitting; validation/test are never augmented.
#
# ### Reproducibility
# Global `SEED = 42` for NumPy / random / torch; deterministic config grids; all
# artifacts (metrics, curves, hashes) written to `models/` and `figures/`.

# %%
import os, sys, glob, json, time, math, base64, random, hashlib, warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Local smoke-test mode (NOT used on Kaggle): shrinks datasets/search so the whole
# notebook (incl. export + parity checks) runs in a couple of minutes on a laptop CPU.
SMOKE = os.environ.get("DIGITLAB_SMOKE", "0") == "1"

def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed)

def log(*a):
    print(*a, flush=True)

def sz(n: int) -> int:
    """Cap a dataset size in SMOKE mode."""
    return min(n, 4000) if SMOKE else n

# %%
# ---------------------------------------------------------------- data loading
from PIL import Image

def _find_split_dirs(root: str):
    """Auto-discover `training` and `testing` digit-folder trees under `root`.
    Robust to the exact intermediate folder name (mnist_png/...)."""
    found = {}
    if not os.path.isdir(root):
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath).lower()
        digit_subs = [d for d in dirnames if d in [str(i) for i in range(10)]]
        if len(digit_subs) == 10:
            kind = "train" if "train" in base else ("test" if "test" in base else None)
            if kind and kind not in found:
                found[kind] = dirpath
                dirnames[:] = []  # do not descend into the digit folders
        else:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")
                           and d not in ("__output__",)]
    return found

def _load_png_split(dirpath: str):
    """Load one split (folder-per-class layout) into X uint8 (N,784), y (N,)."""
    files, labels = [], []
    for d in range(10):
        p = os.path.join(dirpath, str(d))
        fs = sorted(glob.glob(os.path.join(p, "*.png")))
        files += fs; labels += [d] * len(fs)
    assert files, f"no PNGs under {dirpath}"
    X = np.empty((len(files), 784), dtype=np.uint8)
    def _read(i_f):
        i, f = i_f
        with Image.open(f) as im:
            a = np.asarray(im.convert("L"), dtype=np.uint8)
        assert a.shape == (28, 28), f"unexpected image size {a.shape} in {f}"
        X[i] = a.reshape(-1)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_read, enumerate(files)))
    return X, np.array(labels, dtype=np.int64)

def _load_idx_images(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        head = f.read(16)
        magic, n, h, w = np.frombuffer(head, dtype=">u4")
        assert magic == 2051, f"bad magic in {path}"
        return np.frombuffer(f.read(n * h * w), dtype=np.uint8).reshape(n, 784)

def _load_idx_labels(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        head = f.read(8)
        magic, n = np.frombuffer(head, dtype=">u4")
        assert magic == 2049, f"bad magic in {path}"
        return np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int64)

def load_mnist():
    """Return X_train, y_train, X_test, y_test as uint8 arrays (N,784)/(N,).
    Resolution order: env override → /kaggle/input (attached Kaggle dataset)
    → ./local/data/pngtree (local mirror of the Kaggle layout) → raw IDX files."""
    candidates = []
    env_tr, env_te = os.environ.get("DIGITLAB_TRAIN_DIR"), os.environ.get("DIGITLAB_TEST_DIR")
    if env_tr and env_te:
        candidates.append({"train": env_tr, "test": env_te})
    if os.path.isdir("/kaggle/input"):
        candidates.append(_find_split_dirs("/kaggle/input"))
    if os.path.isdir("local/data/pngtree"):
        candidates.append(_find_split_dirs("local/data/pngtree"))

    for dirs in candidates:
        if "train" in dirs and "test" in dirs:
            log(f"[data] loading PNG split tree: {dirs}")
            Xtr, ytr = _load_png_split(dirs["train"])
            Xte, yte = _load_png_split(dirs["test"])
            break
    else:
        idx_dir = os.environ.get("DIGITLAB_IDX_DIR", "local/data/mnist")
        log(f"[data] PNG tree not found — falling back to IDX files in {idx_dir}")
        Xtr = _load_idx_images(f"{idx_dir}/train-images-idx3-ubyte")
        ytr = _load_idx_labels(f"{idx_dir}/train-labels-idx1-ubyte")
        Xte = _load_idx_images(f"{idx_dir}/t10k-images-idx3-ubyte")
        yte = _load_idx_labels(f"{idx_dir}/t10k-labels-idx1-ubyte")

    if SMOKE:  # stratified subsample for fast local end-to-end testing
        rng = np.random.default_rng(SEED)
        def _sub(X, y, n):
            keep = np.concatenate([rng.choice(np.where(y == c)[0], n // 10, replace=False)
                                   for c in range(10)])
            keep.sort()
            return X[keep], y[keep]
        Xtr, ytr = _sub(Xtr, ytr, 3000)
        Xte, yte = _sub(Xte, yte, 600)

    log(f"[data] train={Xtr.shape} labels={sorted(set(ytr.tolist()))} | test={Xte.shape}")
    assert Xtr.shape[1] == 784 and Xtr.dtype == np.uint8
    return Xtr, ytr, Xte, yte

# %%
# ------------------------------------------------------------------- EDA
import matplotlib
if SMOKE: matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

def eda_overview(X, y, tag="train"):
    """Sample grid + class distribution — dataset sanity check."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    idx = np.random.default_rng(SEED).choice(len(X), 40, replace=False)
    grid = np.concatenate([np.concatenate(
        [X[i].reshape(28, 28) for i in idx[r * 8:(r + 1) * 8]], axis=1)
        for r in range(5)], axis=0)
    axes[0].imshow(grid, cmap="gray"); axes[0].set_title(f"{tag}: 40 samples (white digit on black)")
    axes[0].axis("off")
    counts = np.bincount(y, minlength=10)
    axes[1].bar(range(10), counts, color="#4f8cff")
    for c, v in enumerate(counts):
        axes[1].text(c, v, str(v), ha="center", va="bottom", fontsize=8)
    axes[1].set_title(f"{tag}: class distribution"); axes[1].set_xticks(range(10))
    axes[1].set_ylim(0, counts.max() * 1.15)
    fig.tight_layout(); fig.savefig(f"figures/eda_{tag}.png", dpi=110); plt.show()
    log(f"[eda] {tag} pixel range [{X.min()},{X.max()}], class counts={counts.tolist()}")

# %%
# --------------------------------------------------------------- evaluation
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def eval_classification(y_true, y_pred):
    """Standard metric bundle: accuracy, macro/weighted F1, per-class table, confusion."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }

def plot_confusion(cm, title, fname):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(np.array(cm), cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    thr = np.array(cm).max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i][j] > thr else "black")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(f"figures/{fname}", dpi=110); plt.show()

def measure_latency(predict_fn, X, n=1000, per_call=1):
    """Median end-to-end latency (ms/sample) of a predict-like callable."""
    Xs = X[:n]
    t0 = time.perf_counter(); predict_fn(Xs); _ = time.perf_counter() - t0  # warmup
    t0 = time.perf_counter()
    for _ in range(per_call):
        predict_fn(Xs)
    dt = time.perf_counter() - t0
    return 1000.0 * dt / (per_call * len(Xs))

# %%
# ------------------------------------------------------- artifact export utils
def b64_f32(arr) -> str:
    """float32 little-endian → base64 (portable model weights for the JS core)."""
    a = np.ascontiguousarray(arr, dtype="<f4")
    return base64.b64encode(a.tobytes()).decode("ascii")

def b64_u8(arr) -> str:
    a = np.ascontiguousarray(arr, dtype=np.uint8)
    return base64.b64encode(a.tobytes()).decode("ascii")

def write_model_json(payload: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    log(f"[export] wrote {path} ({os.path.getsize(path)/1e6:.2f} MB)")

def write_metrics(obj: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    log(f"[export] wrote {path}")

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def report_row(name, m, extra=""):
    """One markdown row for the final comparison table in REPORT.md."""
    log(f"| {name} | {m['accuracy']*100:.3f}% | {m['macro_f1']*100:.3f}% | {extra} |")

try:
    from IPython.display import display as _display
except Exception:  # flattened .py runs outside a notebook
    def _display(*a):
        for x in a:
            print(x)
display = _display


# %% [markdown]
# ## Common (PyTorch) — training loop, early stopping, augmentation, export
#
# Shared by the MLP and CNN notebooks.
#
# - **Early stopping** monitors validation accuracy (mode=max) with a patience and a
#   minimum-delta threshold, keeps the best-weight snapshot in memory and restores it at
#   the end — the delivered model is always the best-validation epoch, never an
#   overfit one.
# - **Augmentation** is applied *on-device per batch* (vectorized `affine_grid` /
#   `grid_sample`, per-sample random parameters) so it costs almost nothing and keeps
#   validation/test data untouched.
# - **Export**: trained weights are serialized to the portable DigitLab JSON format.
#   BatchNorm layers are *folded* into the preceding conv/linear layer at export time
#   (standard inference optimization), so the JS engine only needs conv/relu/pool/
#   linear/softmax primitives.
# - **Parity**: `np_forward` re-implements the exported ops list in pure NumPy exactly
#   the way the JavaScript engine will run it — used to prove export correctness.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd

torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(max(1, os.cpu_count() or 2))
log(f"[torch] {torch.__version__} · device={DEVICE}"
    + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

def set_torch_seed(seed: int = SEED):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

# %%
# ---------------------------------------------------------- augmentation
def batch_affine(x: torch.Tensor, max_deg: float, translate: float,
                 scale_range=None) -> torch.Tensor:
    """Per-sample random rotation / translation / scaling (zero padding), batched."""
    B = x.size(0)
    ang = np.random.uniform(-max_deg, max_deg, B)
    sc = (np.random.uniform(*scale_range, B) if scale_range else np.ones(B))
    tx = np.random.uniform(-translate, translate, B)
    ty = np.random.uniform(-translate, translate, B)
    ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    rows = np.stack([sc * ca, -sc * sa, tx,
                     sc * sa,  sc * ca, ty], axis=1).astype(np.float32)  # (B, 6)
    theta = torch.from_numpy(rows.reshape(B, 2, 3)).to(x.device)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

# %%
# ---------------------------------------------------------- early stopping
class EarlyStopping:
    """Stop when val accuracy hasn't improved by `min_delta` for `patience` epochs."""

    def __init__(self, patience: int = 6, min_delta: float = 2e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.wait = -1.0, -1, 0
        self.best_state = None

    def step(self, epoch: int, score: float, model: nn.Module) -> bool:
        if score > self.best + self.min_delta:
            self.best, self.best_epoch, self.wait = score, epoch, 0
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            return False
        self.wait += 1
        return self.wait >= self.patience

# %%
# ---------------------------------------------------------- training loop
def to_img_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(X).view(-1, 1, 28, 28)  # uint8 (N,1,28,28)

@torch.no_grad()
def _eval_split(model, X: torch.Tensor, y: torch.Tensor, device, batch=1000):
    model.eval()
    loss_sum, correct, n = 0.0, 0, len(y)
    for i in range(0, n, batch):
        xb = X[i:i + batch].to(device).float().div_(255.0)
        yb = y[i:i + batch].to(device)
        out = model(xb)
        loss_sum += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return loss_sum / n, correct / n

def train_model(model: nn.Module, tr_X, tr_y, va_X, va_y, *, epochs, lr, wd,
                batch_size, aug=None, patience=6, min_delta=2e-4,
                scheduler=True, seed=SEED, tag="model", log_every=1):
    """Full training with early stopping on val accuracy; restores best weights."""
    set_torch_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2) if scheduler else None)
    es = EarlyStopping(patience, min_delta)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    n = len(tr_y)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        run_loss, run_correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = tr_X[idx].to(DEVICE).float().div_(255.0)
            if aug is not None:
                xb = aug(xb)
            yb = tr_y[idx].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
            run_correct += (out.argmax(1) == yb).sum().item()
        tr_loss, tr_acc = run_loss / n, run_correct / n
        va_loss, va_acc = _eval_split(model, va_X, va_y, DEVICE)
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss);   hist["val_acc"].append(va_acc)
        hist["lr"].append(opt.param_groups[0]["lr"])
        if sched is not None:
            sched.step(va_acc)
        if epoch % log_every == 0 or epoch == 1:
            log(f"[{tag}] epoch {epoch:3d}/{epochs} · train loss {tr_loss:.4f} acc {tr_acc*100:.2f}%"
                f" · val loss {va_loss:.4f} acc {va_acc*100:.2f}% · lr {hist['lr'][-1]:.2e}")
        if es.step(epoch, va_acc, model):
            log(f"[{tag}] early stopping at epoch {epoch} "
                f"(best val acc {es.best*100:.2f}% @ epoch {es.best_epoch})")
            break
    if es.best_state is not None:
        model.load_state_dict(es.best_state)
    model.eval()
    return hist, es.best_epoch, float(es.best)

@torch.no_grad()
def predict_torch(model, X_u8: np.ndarray, batch=1000) -> np.ndarray:
    model.eval()
    Xt = to_img_tensor(X_u8)
    outs = []
    for i in range(0, len(Xt), batch):
        xb = Xt[i:i + batch].to(DEVICE).float().div_(255.0)
        outs.append(model(xb).argmax(1).cpu())
    return torch.cat(outs).numpy()

@torch.no_grad()
def torch_latency_ms(model, n_iter=100) -> float:
    model.eval()
    x = torch.rand(1, 1, 28, 28, device=DEVICE)
    model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_iter

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

def plot_curves(hist, tag, fname):
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    best_i = int(np.argmax(hist["val_acc"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ep, hist["train_loss"], label="train"); axes[0].plot(ep, hist["val_loss"], label="val")
    axes[0].axvline(ep[best_i], ls="--", c="gray", lw=1); axes[0].set_title(f"{tag}: loss")
    axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(ep, np.array(hist["train_acc"]) * 100, label="train")
    axes[1].plot(ep, np.array(hist["val_acc"]) * 100, label="val")
    axes[1].axvline(ep[best_i], ls="--", c="gray", lw=1,
                    label=f"best val (ep {ep[best_i]})")
    axes[1].set_title(f"{tag}: accuracy (%)"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(f"figures/{fname}", dpi=110); plt.show()

# %%
# ---------------------------------------------------------- portable export
def _sd(module) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}

def _fold_bn(w, b, bn):
    """Fold y = BN(Wx+b) into a single affine (W', b'). Works for 2-D and 4-D W."""
    sd = _sd(bn)
    s = sd["weight"] / np.sqrt(sd["running_var"] + bn.eps)
    w2 = w * s[:, None, None, None] if w.ndim == 4 else w * s[:, None]
    return w2, (b - sd["running_mean"]) * s + sd["bias"]

def export_mlp_layers(model: nn.Module):
    """Sequential Flatten(Linear→[BN]→ReLU→Dropout)*…→Linear ➜ [{'w','b','in','out'}].
    Assumes ReLU between consecutive layers (never after the last) — true for our MLP factory."""
    raw, pend = [], None
    for m in model:
        if isinstance(m, (nn.Flatten, nn.Dropout, nn.ReLU, nn.GELU)):
            continue
        if isinstance(m, nn.Linear):
            if pend is not None:
                raw.append(pend)
            sd = _sd(m)
            pend = [sd["weight"].astype(np.float32), sd["bias"].astype(np.float32), None]
        elif isinstance(m, nn.BatchNorm1d) and pend is not None:
            pend[2] = m
        else:
            raise ValueError(f"unsupported MLP module {m}")
    if pend is not None:
        raw.append(pend)
    layers = []
    for w, b, bn in raw:
        if bn is not None:
            w, b = _fold_bn(w, b, bn)
        layers.append({"in": int(w.shape[1]), "out": int(w.shape[0]),
                       "w": b64_f32(w), "b": b64_f32(b)})
    return layers

def export_cnn_ops(model: nn.Module):
    """Sequential conv/BN/pool/dense network ➜ ops list for the JS engine (BN folded)."""
    ops, pend_w, pend_b, pend_kind, pend_bn = [], None, None, None, None

    def _flush():
        nonlocal pend_w, pend_b, pend_kind, pend_bn
        if pend_kind is None:
            return
        w, b = _fold_bn(pend_w, pend_b, pend_bn) if pend_bn is not None else (pend_w, pend_b)
        if pend_kind == "conv":
            oc, ic, kh, kw = w.shape
            ops.append({"op": "conv", "in_c": int(ic), "out_c": int(oc),
                        "k": int(kh), "pad": 1, "w": b64_f32(w), "b": b64_f32(b)})
        else:
            ops.append({"op": "linear", "in": int(w.shape[1]), "out": int(w.shape[0]),
                        "w": b64_f32(w), "b": b64_f32(b)})
        pend_w = pend_b = pend_kind = pend_bn = None

    for m in model:
        if isinstance(m, nn.Conv2d):
            _flush()
            sd = _sd(m)
            assert m.kernel_size == (3, 3) and m.stride == (1, 1) and m.padding == (1, 1), \
                "export supports 3×3 stride-1 pad-1 convs"
            pend_w, pend_b, pend_kind = sd["weight"].copy(), sd["bias"].copy(), "conv"
        elif isinstance(m, nn.Linear):
            _flush()
            sd = _sd(m)
            pend_w, pend_b, pend_kind = sd["weight"].copy(), sd["bias"].copy(), "linear"
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            pend_bn = m
        elif isinstance(m, (nn.ReLU, nn.GELU)):
            _flush(); ops.append({"op": "relu"})
        elif isinstance(m, nn.MaxPool2d):
            _flush()
            assert m.kernel_size == 2 and m.stride == 2, "export supports 2×2 maxpool"
            ops.append({"op": "maxpool2"})
        elif isinstance(m, (nn.Flatten, nn.Dropout)):
            _flush()
            if isinstance(m, nn.Flatten):
                ops.append({"op": "flatten"})
        else:
            raise ValueError(f"unsupported CNN module {m}")
    _flush()
    ops.append({"op": "softmax"})
    return ops

# %%
# ------------------------------------------------ NumPy reference engine (parity)
def np_forward(ops, X01: np.ndarray) -> np.ndarray:
    """Run the exported ops on (N,784) float32 inputs — mirrors the JS engine exactly."""
    from numpy.lib.stride_tricks import sliding_window_view
    x = X01.reshape(-1, 1, 28, 28).astype(np.float32)
    for op in ops:
        kind = op["op"]
        if kind == "conv":
            w = np.frombuffer(base64.b64decode(op["w"]), "<f4").reshape(op["out_c"], op["in_c"], op["k"], op["k"])
            b = np.frombuffer(base64.b64decode(op["b"]), "<f4")
            xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
            win = sliding_window_view(xp, (3, 3), axis=(2, 3))       # (N,C,H,W,3,3)
            x = np.einsum("nchwij,ocij->nohw", win, w, optimize=True) + b[None, :, None, None]
        elif kind == "linear":
            w = np.frombuffer(base64.b64decode(op["w"]), "<f4").reshape(op["out"], op["in"])
            b = np.frombuffer(base64.b64decode(op["b"]), "<f4")
            x = x.reshape(x.shape[0], -1) @ w.T + b
        elif kind == "relu":
            x = np.maximum(x, 0.0)
        elif kind == "maxpool2":
            n, c, h, w_ = x.shape
            x = x.reshape(n, c, h // 2, 2, w_ // 2, 2).max(axis=(3, 5))
        elif kind == "flatten":
            x = x.reshape(x.shape[0], -1)
        elif kind == "softmax":
            z = x - x.max(axis=1, keepdims=True)
            e = np.exp(z); x = e / e.sum(axis=1, keepdims=True)
        else:
            raise ValueError(kind)
    return x

def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def mlp_json_forward(layers, X01):
    """Reference forward for exported MLP layers (mirrors JS engine)."""
    x = X01.astype(np.float32)
    for i, L in enumerate(layers):
        w = np.frombuffer(base64.b64decode(L["w"]), "<f4").reshape(L["out"], L["in"])
        b = np.frombuffer(base64.b64decode(L["b"]), "<f4")
        x = x @ w.T + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0.0)
    return softmax_np(x)

def model_json_payload(kind, model_blob, metrics, extra=None):
    return {
        "format": "digitlab-model", "version": 1, "kind": kind,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "framework": f"torch-{torch.__version__}",
        "classes": list(range(10)),
        "preprocess": {"scale": 255, "shape": [1, 28, 28], "layout": "CHW",
                       "note": "white digit on black; divide by 255"},
        "metrics": metrics, "model": model_blob, **(extra or {}),
    }


# %% [markdown]
# # DigitLab · Notebook 3/3 — Deep Learning CV: Convolutional Neural Network
#
# **How to run on Kaggle:** import this .ipynb → **Add Input → "MNIST Dataset"
# (hojjatk)** → Settings → Accelerator **GPU T4 x2 or P100** (CPU works, ~4× slower) →
# *Run All*. Runtime ≈ 15 min on GPU / ~1 h on CPU.
#
# ## Architecture (introductory CNN — deliberately kept basic)
# A standard lecture-style two-block CNN (VGG-flavoured, *no* residual/pretrained parts):
#
# ```
# (1×28×28 input)
# Conv3×3(w) → BN → ReLU → Conv3×3(w)   → BN → ReLU → MaxPool2×2 → Dropout(p)
# Conv3×3(2w)→ BN → ReLU → Conv3×3(2w) → BN → ReLU → MaxPool2×2 → Dropout(p)
# Flatten → Linear(7·7·2w → 128) → BN → ReLU → Dropout(0.5) → Linear(128 → 10)
# w = 32 (≈467k params) or 48 (≈1.0M params)
# ```
# Every conv is 3×3, stride 1, padding 1 (so MaxPool 2×2 is what shrinks
# 28→14→7). BatchNorm accelerates training; dropout + augmentation fight overfitting.
#
# ## Hyperparameter search (9 configs × 4 epochs, fixed budget)
# Knobs: **learning rate, dropout, width, batch size** — ranked by best val accuracy
# across the 4 epochs. The winner retrains from scratch with **early stopping**
# (patience 5 on val accuracy, best-weights restore, max 40 epochs).
#
# | # | width w | dropout conv | lr | batch |
# |---|--------|--------------|----|-------|
# | 1 | 32 | 0.25 | 1e-3 | 128 | ← reference |
# | 2 | 32 | 0.15 | 1e-3 | 128 |
# | 3 | 32 | 0.35 | 1e-3 | 128 |
# | 4 | 32 | 0.25 | 5e-4 | 128 |
# | 5 | 32 | 0.25 | 2e-3 | 128 |
# | 6 | 32 | 0.25 | 1e-3 | 64 |
# | 7 | 48 | 0.25 | 1e-3 | 128 |
# | 8 | 48 | 0.30 | 2e-3 | 128 |
# | 9 | 32 | 0.25 | 1e-3 | 256 |
#
# ## Augmentation (train only)
# Rotation ±10°, translation ±10%, scale 92–108% (per-sample, zero padding) —
# matches how real drawings vary: slanted, shifted, bigger/smaller digits.

# %%
from sklearn.model_selection import train_test_split

X_train_u8, y_train, X_test_u8, y_test = load_mnist()
eda_overview(X_train_u8, y_train, "train")

idx_tr, idx_va = train_test_split(np.arange(len(y_train)), test_size=10_000 if not SMOKE else 600,
                                  stratify=y_train, random_state=SEED)
tr_X = to_img_tensor(X_train_u8[idx_tr]); tr_y = torch.from_numpy(y_train[idx_tr])
va_X = to_img_tensor(X_train_u8[idx_va]); va_y = torch.from_numpy(y_train[idx_va])
log(f"[split] train={len(tr_y)} val={len(va_y)} test={len(y_test)}")

def make_cnn(width=32, dropout=0.25):
    w = width
    return nn.Sequential(
        nn.Conv2d(1,  w, 3, padding=1),  nn.BatchNorm2d(w),  nn.ReLU(),
        nn.Conv2d(w, w, 3, padding=1),   nn.BatchNorm2d(w),  nn.ReLU(),
        nn.MaxPool2d(2),                                       # 28 → 14
        nn.Dropout(dropout),
        nn.Conv2d(w,  2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(),
        nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(),
        nn.MaxPool2d(2),                                       # 14 → 7
        nn.Dropout(dropout),
        nn.Flatten(),
        nn.Linear(7 * 7 * 2 * w, 128), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10),
    )

def cnn_aug(x):
    return batch_affine(x, max_deg=10, translate=0.10, scale_range=(0.92, 1.08))

# %%
# ------------------------------------------------------------- HP search
SEARCH_CONFIGS = [
    dict(name="cnn-01", width=32, dropout=0.25, lr=1e-3, bs=128),
    dict(name="cnn-02", width=32, dropout=0.15, lr=1e-3, bs=128),
    dict(name="cnn-03", width=32, dropout=0.35, lr=1e-3, bs=128),
    dict(name="cnn-04", width=32, dropout=0.25, lr=5e-4, bs=128),
    dict(name="cnn-05", width=32, dropout=0.25, lr=2e-3, bs=128),
    dict(name="cnn-06", width=32, dropout=0.25, lr=1e-3, bs=64),
    dict(name="cnn-07", width=48, dropout=0.25, lr=1e-3, bs=128),
    dict(name="cnn-08", width=48, dropout=0.30, lr=2e-3, bs=128),
    dict(name="cnn-09", width=32, dropout=0.25, lr=1e-3, bs=256),
]

search_epochs = 1 if SMOKE else 4
search_rows = []
for i, cfg in enumerate(SEARCH_CONFIGS):
    if SMOKE and i % 2:  # shrink grid in smoke mode
        continue
    set_torch_seed(SEED + 200 + i)
    model = make_cnn(cfg["width"], cfg["dropout"]).to(DEVICE)
    hist, best_ep, best_acc = train_model(
        model, tr_X, tr_y, va_X, va_y, epochs=search_epochs, lr=cfg["lr"], wd=1e-4,
        batch_size=cfg["bs"], aug=cnn_aug, scheduler=True, patience=99,
        seed=SEED + 200 + i, tag=cfg["name"], log_every=10)
    search_rows.append({**cfg, "best_val_acc": best_acc, "best_epoch": best_ep,
                        "params": count_params(model)})
    del model

search_df = pd.DataFrame(search_rows).sort_values("best_val_acc", ascending=False)
display(search_df)
best_cfg = next(r for r in search_rows if r["name"] == search_df.iloc[0]["name"])
log(f"[search] winner: {best_cfg['name']} → width={best_cfg['width']} "
    f"dropout={best_cfg['dropout']} lr={best_cfg['lr']} bs={best_cfg['bs']} "
    f"(val acc {best_cfg['best_val_acc']*100:.3f}%)")

# %%
# ------------------------------------------------- final training (early stop)
FINAL_EPOCHS = 4 if SMOKE else 40
PATIENCE = 2 if SMOKE else 5
set_torch_seed(SEED)
final_cnn = make_cnn(best_cfg["width"], best_cfg["dropout"])
log(f"[final] params: {count_params(final_cnn):,}")
t0 = time.perf_counter()
hist, best_ep, best_va = train_model(
    final_cnn, tr_X, tr_y, va_X, va_y, epochs=FINAL_EPOCHS, lr=best_cfg["lr"], wd=1e-4,
    batch_size=best_cfg["bs"], aug=cnn_aug, patience=PATIENCE, seed=SEED, tag="cnn-final")
cnn_fit_s = time.perf_counter() - t0
plot_curves(hist, "CNN", "cnn_curves.png")
gap = hist["train_acc"][best_ep - 1] - hist["val_acc"][best_ep - 1]
log(f"[final] best val acc {best_va*100:.3f}% @ epoch {best_ep}/{len(hist['val_acc'])} "
    f"(train−val gap at best epoch: {gap*100:.2f} pts — small gap ⇒ no overfitting)")

# %%
# --------------------------------------------------------------- evaluation
cnn_pred = predict_torch(final_cnn, X_test_u8)
cnn_metrics = eval_classification(y_test, cnn_pred)
cnn_metrics.update({"best_val_acc": best_va, "best_epoch": best_ep,
                    "epochs_run": len(hist["val_acc"]),
                    "train_val_gap_at_best": float(gap),
                    "fit_seconds": cnn_fit_s,
                    "latency_ms_per_sample": torch_latency_ms(final_cnn),
                    "params": count_params(final_cnn)})
log(f"[eval] CNN test accuracy = {cnn_metrics['accuracy']*100:.3f}%")
plot_confusion(cnn_metrics["confusion"], "CNN — test confusion", "cnn_confusion.png")
print(classification_report(y_test, cnn_pred, digits=4))

# %%
# ------------------------------------------------------------ save artifacts
torch.save(final_cnn.state_dict(), "models/cnn_mnist.pt")
cnn_payload = model_json_payload("cnn", {"ops": export_cnn_ops(final_cnn)},
                                 {"test_accuracy": cnn_metrics["accuracy"],
                                  "best_val_accuracy": best_va})
write_model_json(cnn_payload, "models/cnn_mnist.json")

# parity: NumPy ops-engine recompute from the JSON (exactly what the JS engine does)
N_CHECK = 512
xs = X_test_u8[:N_CHECK].astype(np.float32) / 255.0
ops = cnn_payload["model"]["ops"]
probs_json = np_forward(ops, xs)
json_pred = probs_json.argmax(1)
with torch.no_grad():
    probs_torch = torch.softmax(final_cnn(to_img_tensor(X_test_u8[:N_CHECK]).to(DEVICE)
                                          .float().div_(255.0)), dim=1).cpu().numpy()
agree = float((json_pred == probs_torch.argmax(1)).mean())
max_pd = float(np.abs(probs_json - probs_torch).max())
log(f"[parity] JSON-vs-torch prediction agreement: {agree*100:.2f}% · max prob diff {max_pd:.2e}")
assert agree >= 0.998 and max_pd < 5e-3, "CNN JSON export parity failed!"

cnn_summary = {
    "model": f"CNN w={best_cfg['width']} (2 conv blocks + BN + dropout {best_cfg['dropout']})",
    "best_config": best_cfg, "search": {"epochs_per_config": search_epochs,
                                        "configs": search_rows},
    "final": cnn_metrics, "curves": hist,
    "artifacts": {"pt": "models/cnn_mnist.pt", "json": "models/cnn_mnist.json",
                  "pt_sha256_16": sha256_file("models/cnn_mnist.pt"),
                  "json_sha256_16": sha256_file("models/cnn_mnist.json")},
}
write_metrics(cnn_summary, "models/metrics_cnn.json")
log("\n=== REPORT ROW ===")
report_row("CNN", cnn_metrics,
           f"{count_params(final_cnn):,} params · {cnn_metrics['epochs_run']} epochs")
log("\n[notebook 3 done] artifacts:")
for f in ["models/cnn_mnist.pt", "models/cnn_mnist.json", "models/metrics_cnn.json"]:
    log(f"  · {f}  ({os.path.getsize(f)/1e6:.2f} MB, sha256:{sha256_file(f)})")
