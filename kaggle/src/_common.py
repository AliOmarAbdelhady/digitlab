# %% [markdown]
# ## Common — data loading, preprocessing, evaluation, export helpers
#
# **DigitLab · MNIST 3-model project** (shared cells inlined into every notebook)
#
# ### Dataset
# - **Kaggle dataset:** [`hojjatk/mnist-dataset`](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
#   ("The MNIST Database") — the canonical full MNIST as the **original IDX
#   binary files**: 60,000 training + 10,000 test 28×28 grayscale images
#   (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`,
#   `t10k-labels.idx1-ubyte`), digit white on black.
# - Attach it via **Add Input → Datasets → "MNIST Dataset" (hojjatk)**; it mounts
#   at `/kaggle/input/mnist-dataset/`.
# - The loader auto-discovers the data in either form: a PNG folder-per-class
#   tree (`training/<0..9>/*.png`) or the canonical IDX files (one or two
#   directory levels deep, either filename spelling) — it works unchanged with
#   any faithful MNIST mirror.
#
# ### Preprocessing (identical for training AND serving in the apps)
# 1. Parse IDX (or PNGs) → 28×28 uint8 arrays `[0..255]`, flattened to 784.
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
    """Auto-discover `training`/`testing` (or train/test) digit-folder PNG trees
    under `root` — used when a dataset ships PNGs (folder-per-class layout)."""
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
        magic, n, h, w = np.frombuffer(f.read(16), dtype=">u4")
        assert magic == 2051, f"bad magic in {path}"
        return np.frombuffer(f.read(n * h * w), dtype=np.uint8).reshape(n, 784)

def _load_idx_labels(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n = np.frombuffer(f.read(8), dtype=">u4")
        assert magic == 2049, f"bad magic in {path}"
        return np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int64)

# canonical MNIST IDX filenames — datasets use either spelling, sometimes
# nested one directory deep (e.g. hojjatk/mnist-dataset ships both variants)
_IDX_NAMES = {
    "trX": ["train-images-idx3-ubyte", "train-images.idx3-ubyte"],
    "trY": ["train-labels-idx1-ubyte", "train-labels.idx1-ubyte"],
    "teX": ["t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"],
    "teY": ["t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"],
}

def _find_idx_files():
    """Locate the 4 canonical IDX files. Search order: $DIGITLAB_IDX_DIR,
    /kaggle/input/** (attached Kaggle datasets — one or two levels deep),
    ./local/data/mnist (local mirror)."""
    candidates = [os.environ.get("DIGITLAB_IDX_DIR"), "local/data/mnist"]
    if os.path.isdir("/kaggle/input"):
        for d in sorted(os.listdir("/kaggle/input")):
            candidates.append(f"/kaggle/input/{d}")
        candidates.append("/kaggle/input")
    for c in candidates:
        if not c or not os.path.isdir(c):
            continue
        bases = [c] + [os.path.join(c, d) for d in sorted(os.listdir(c))
                       if os.path.isdir(os.path.join(c, d))]
        paths = {}
        for key, alts in _IDX_NAMES.items():
            for base in bases:
                hit = next((os.path.join(base, n) for n in alts
                            if os.path.isfile(os.path.join(base, n))), None)
                if hit:
                    paths[key] = hit
                    break
        if len(paths) == 4:
            return paths
    return None

def load_mnist():
    """Return X_train, y_train, X_test, y_test as uint8 arrays (N,784)/(N,).
    Resolution order: explicit env override → attached Kaggle dataset (PNG
    folder-per-class tree, else canonical IDX files) → local PNG mirror →
    local IDX mirror."""
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
        idx = _find_idx_files()
        if idx is None:
            raise RuntimeError(
                "MNIST not found — attach the Kaggle dataset 'hojjatk/mnist-dataset' "
                "(canonical IDX files) or point DIGITLAB_IDX_DIR at an IDX folder.")
        log(f"[data] loading canonical IDX files: { {k: os.path.basename(os.path.dirname(p)) or p for k, p in idx.items()} }")
        Xtr = _load_idx_images(idx["trX"])
        ytr = _load_idx_labels(idx["trY"])
        Xte = _load_idx_images(idx["teX"])
        yte = _load_idx_labels(idx["teY"])

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
os.makedirs("models", exist_ok=True)

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
