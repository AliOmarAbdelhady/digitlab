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
