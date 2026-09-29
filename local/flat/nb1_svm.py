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
# # DigitLab · Notebook 1/3 — Classical ML: RBF Support Vector Machine
#
# **How to run on Kaggle:** *Create Notebook → File → Import Notebook* (this .ipynb) →
# **Add Input → Datasets → "MNIST Dataset" (hojjatk)** → Accelerator **CPU** is fine →
# *Run All*. Runtime ≈ 1.5–2.5 h (CPU). All artifacts land in `/kaggle/working/models`.
#
# ## Plan
# 1. Load + preprocess (see shared preprocessing notes above).
# 2. Baseline classical models for context: multinomial **Logistic Regression** and
#    **Random Forest** (trained once, default-ish settings — they set the reference bar).
# 3. **Hyperparameter search** for the main model `sklearn.svm.SVC` (RBF kernel) with
#    `RandomizedSearchCV` (3-fold CV, 12 sampled configs over the two decisive
#    hyperparameters) on a stratified 15k subsample:
#    - `C` ∈ log-uniform [1, 30] (regularization strength inverse — margin tightness),
#    - `gamma` ∈ log-uniform [5e-4, 5e-3] (RBF locality; note `1/784 ≈ 1.3e-3` brackets the sweet spot).
#    *Subsample search keeps each CV fit minutes-fast; the optimum transfers to the full
#    60k fit (standard practice for kernel machines).*
# 4. Final `SVC` fit on all 60k with the best `(C, γ)` → official 10k test metrics.
# 5. Equivalent **One-vs-Rest SVC ensemble** with the same hyperparameters — trained
#    because its per-class binary coefficients export *losslessly* to the portable JSON
#    format used by the apps (libsvm's native multiclass packing does not). Prediction
#    agreement with the canonical SVC is measured and reported.
# 6. Save `models/svc_mnist.joblib` (+ `models/svc_mnist.json` for the apps) with a
#    parity check that re-computes predictions exactly the way the JS engine will
#    (including uint8 quantization of support vectors) and asserts agreement.

# %%
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import loguniform
import joblib
import pandas as pd

X_train_u8, y_train, X_test_u8, y_test = load_mnist()
eda_overview(X_train_u8, y_train, "train")

X_train = (X_train_u8.astype(np.float32) / 255.0)
X_test = (X_test_u8.astype(np.float32) / 255.0)
log(f"[prep] X_train {X_train.shape} float32 ∈ [{X_train.min():.1f}, {X_train.max():.1f}]")

# %%
# ---------------------------------------------------------------- baselines
baseline_rows = []

def run_baseline(name, est, Xtr, ytr):
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    pred = est.predict(X_test)
    m = eval_classification(y_test, pred)
    lat_ms = measure_latency(est.predict, X_test)
    baseline_rows.append({"model": name, "test_acc": m["accuracy"],
                          "macro_f1": m["macro_f1"], "fit_s": fit_s,
                          "lat_ms_per_sample": lat_ms})
    log(f"[baseline] {name:<22} test acc {m['accuracy']*100:.3f}%  (fit {fit_s:.0f}s)")

run_baseline("LogisticRegression", LogisticRegression(max_iter=1000, n_jobs=-1),
             X_train[:sz(20000)], y_train[:sz(20000)])
run_baseline("RandomForest(300)", RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                         random_state=SEED),
             X_train[:sz(20000)], y_train[:sz(20000)])
display(pd.DataFrame(baseline_rows))

# %%
# ----------------------------------------------- hyperparameter search (SVC)
# Search on a stratified 15k subsample (60k full fits × 12 configs × 3 folds would
# take ~10× longer for near-identical (C, gamma) selection).
search_n = min(15000, int(len(X_train) * 0.9))
Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=search_n,
                                stratify=y_train, random_state=SEED)

param_dist = {"C": loguniform(1, 30), "gamma": loguniform(5e-4, 5e-3)}
n_iter = 3 if SMOKE else 12

search = RandomizedSearchCV(
    SVC(cache_size=800), param_distributions=param_dist, n_iter=n_iter, cv=3,
    scoring="accuracy", n_jobs=-1, random_state=SEED, return_train_score=True,
    verbose=1, refit=False, error_score="raise")
t0 = time.perf_counter()
search.fit(Xs, ys)
log(f"[search] done in {time.perf_counter()-t0:.0f}s")

res = pd.DataFrame(search.cv_results_)
res["C"] = res["params"].apply(lambda p: round(float(p["C"]), 3))
res["gamma"] = res["params"].apply(lambda p: float(f"{p['gamma']:.2e}"))
cols = ["C", "gamma", "mean_test_score", "std_test_score", "mean_fit_time", "rank_test_score"]
display(res[cols].sort_values("rank_test_score"))

best_params = search.best_params_
log(f"[search] best params: C={best_params['C']:.3f}, gamma={best_params['gamma']:.2e} "
    f"(CV acc {search.best_score_*100:.3f}%)")

# %%
# ------------------------------------------------- final canonical SVC (60k)
final_svc = SVC(C=best_params["C"], gamma=best_params["gamma"], cache_size=800)
t0 = time.perf_counter()
final_svc.fit(X_train, y_train)
svc_fit_s = time.perf_counter() - t0
log(f"[final] SVC fit on {len(X_train)} samples in {svc_fit_s:.0f}s "
    f"({len(final_svc.support_vectors_)} support vectors)")

svc_pred = final_svc.predict(X_test)
svc_metrics = eval_classification(y_test, svc_pred)
svc_metrics["fit_seconds"] = svc_fit_s
svc_metrics["n_support_vectors"] = int(len(final_svc.support_vectors_))
svc_metrics["latency_ms_per_sample"] = measure_latency(final_svc.predict, X_test)
log(f"[final] canonical SVC test accuracy = {svc_metrics['accuracy']*100:.3f}%")
plot_confusion(svc_metrics["confusion"], "SVC (RBF) — test confusion", "svc_confusion.png")
print(classification_report(y_test, svc_pred, digits=4))

# %%
# ------------------------------- One-vs-Rest SVC (portable-export equivalent)
ovr = OneVsRestClassifier(
    SVC(C=best_params["C"], gamma=best_params["gamma"], cache_size=800), n_jobs=-1)
t0 = time.perf_counter()
ovr.fit(X_train, y_train)
log(f"[ovr] fit in {time.perf_counter()-t0:.0f}s")

ovr_pred = ovr.predict(X_test)
ovr_metrics = eval_classification(y_test, ovr_pred)
agree_canonical = float((ovr_pred == svc_pred).mean())
log(f"[ovr] test accuracy = {ovr_metrics['accuracy']*100:.3f}% · "
    f"agreement with canonical SVC = {agree_canonical*100:.2f}%")

# %%
# ------------------------------------------------------------ save artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(final_svc, "models/svc_mnist.joblib")

# %%
# ------------------------------------------- portable JSON export (for apps)
# Per class c: decision_c(x) = Σ_i α_i·exp(−γ·‖x − sv_i‖²) + b_c
# Support vectors are stored as uint8 in *pixel units* (0..255); the JS engine divides
# by 255 — quantization error ≤ 1/510 per pixel is negligible in the kernel argument.
sv_blocks, dual_cols, intercepts, offsets, total_sv = [], [], [], [0], 0
for est in ovr.estimators_:
    sv = est.support_vectors_                       # (n_c, 784) float32 in [0,1]
    sv_blocks.append(np.clip(np.rint(sv * 255.0), 0, 255).astype(np.uint8))
    dual_cols.append(est.dual_coef_[0].astype(np.float32))   # α_i·y_i
    intercepts.append(np.float32(est.intercept_[0]))
    total_sv += len(sv)
    offsets.append(total_sv)

dual = np.zeros((total_sv, 10), dtype="<f4")
for c in range(10):
    dual[offsets[c]:offsets[c + 1], c] = dual_cols[c]

svc_payload = {
    "format": "digitlab-model", "version": 1, "kind": "svc",
    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "framework": f"scikit-learn-{__import__('sklearn').__version__}",
    "classes": list(range(10)),
    "preprocess": {"scale": 255, "shape": [1, 28, 28], "layout": "CHW",
                   "note": "white digit on black; divide by 255"},
    "metrics": {"test_accuracy": ovr_metrics["accuracy"],
                "canonical_svc_test_accuracy": svc_metrics["accuracy"],
                "agreement_with_canonical": agree_canonical},
    "model": {
        "gamma": float(best_params["gamma"]), "C": float(best_params["C"]),
        "n_sv": int(total_sv), "n_features": 784,
        "class_offsets": offsets,
        "sv_u8": b64_u8(np.concatenate(sv_blocks, axis=0)),
        "dual_f32": b64_f32(dual),
        "intercept_f32": b64_f32(np.array(intercepts, dtype="<f4")),
    },
}
write_model_json(svc_payload, "models/svc_mnist.json")

# %%
# -------------------------------------------------- export parity check (JS view)
# Recompute predictions *from the JSON* exactly like the JS engine and compare.
N_CHECK = 512
xs = X_test[:N_CHECK]

sv_q = np.frombuffer(base64.b64decode(svc_payload["model"]["sv_u8"]),
                     np.uint8).astype(np.float32).reshape(-1, 784) / 255.0
dual_q = np.frombuffer(base64.b64decode(svc_payload["model"]["dual_f32"]),
                       "<f4").reshape(-1, 10)
b_q = np.frombuffer(base64.b64decode(svc_payload["model"]["intercept_f32"]), "<f4")
g = svc_payload["model"]["gamma"]

scores = np.empty((N_CHECK, 10), dtype=np.float32)
sv2 = (sv_q ** 2).sum(1)[None, :]
for i0 in range(0, N_CHECK, 256):
    xb = xs[i0:i0 + 256]
    # squared distances via the expanded form — identical to the JS engine
    xb2 = (xb ** 2).sum(1, keepdims=True)
    d2 = np.maximum(xb2 + sv2 - 2.0 * (xb @ sv_q.T), 0.0)
    K = np.exp(-g * d2)
    scores[i0:i0 + 256] = K @ dual_q + b_q[None, :]

json_pred = scores.argmax(1)
agreement_ovr = float((json_pred == ovr_pred[:N_CHECK]).mean())
agreement_canon = float((json_pred == svc_pred[:N_CHECK]).mean())
max_score_diff = float(np.abs(scores - ovr.decision_function(xs)).max())
log(f"[parity] JSON-vs-OvR prediction agreement: {agreement_ovr*100:.2f}%")
log(f"[parity] JSON-vs-canonical-SVC agreement:  {agreement_canon*100:.2f}%")
log(f"[parity] max |score − sklearn decision_function| = {max_score_diff:.4f}")
assert agreement_ovr >= 0.999, "JSON export diverges from sklearn OvR!"

# %%
# ---------------------------------------------------------------- wrap-up
svc_summary = {
    "model": "SVC (RBF kernel)",
    "best_params": {k: float(v) for k, v in best_params.items()},
    "search": {"method": "RandomizedSearchCV", "cv": 3, "n_iter": n_iter,
               "subsample": int(search_n),
               "space": {"C": "loguniform[1,30]", "gamma": "loguniform[5e-4,5e-3]"},
               "cv_best_accuracy": float(search.best_score_)},
    "canonical_svc": svc_metrics, "ovr_svc": ovr_metrics,
    "ovr_agreement_with_canonical": agree_canonical,
    "json_export_parity": {"vs_ovr": agreement_ovr, "vs_canonical": agreement_canon},
    "baselines": baseline_rows,
    "artifacts": {"joblib": "models/svc_mnist.joblib", "json": "models/svc_mnist.json",
                  "joblib_sha256_16": sha256_file("models/svc_mnist.joblib"),
                  "json_sha256_16": sha256_file("models/svc_mnist.json")},
}
write_metrics(svc_summary, "models/metrics_svc.json")
log("\n=== REPORT ROW (canonical SVC) ===")
report_row("SVC (RBF)", svc_metrics,
           f"{len(final_svc.support_vectors_)} SVs · {svc_fit_s:.0f}s fit")
log("\n[notebook 1 done] artifacts:")
for f in ["models/svc_mnist.joblib", "models/svc_mnist.json", "models/metrics_svc.json"]:
    log(f"  · {f}  ({os.path.getsize(f)/1e6:.2f} MB, sha256:{sha256_file(f)})")
