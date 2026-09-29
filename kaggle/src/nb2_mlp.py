# %% [markdown]
# # DigitLab · Notebook 2/3 — Neural Network: Multi-Layer Perceptron
#
# **How to run on Kaggle:** import this .ipynb → **Add Input → "MNIST Dataset"
# (hojjatk)** → Settings → Accelerator **GPU T4 x2 or P100** (CPU also works, ~4×
# slower) → *Run All*. Runtime ≈ 20 min on GPU / ~1.5 h on CPU.
#
# ## Architecture family (fully-connected only)
# `784 → Flatten → [Linear → BatchNorm1d → ReLU → Dropout]×L → Linear(→10)`
# BatchNorm + Dropout + weight decay + light geometric augmentation are the
# regularization team that keeps a high-capacity MLP from overfitting 54k images.
#
# ## Hyperparameter search (systematic grid, 12 configs × 8 epochs)
# The four knobs that matter most for an MLP: **hidden layout, dropout, learning
# rate, batch size** (plus weight decay). Each config trains for a *fixed* 8 epochs
# on the 54k train split and is ranked by its **best validation accuracy across
# epochs** (fixed budget = fair comparison; per-epoch best = no config is punished
# by a slow start). The winner is then retrained from scratch with **early stopping**
# (patience 8 on val accuracy, best-weights restore, max 60 epochs).
#
# | # | hidden layers | dropout | lr | batch | wd |
# |---|--------------|---------|----|-------|-----|
# | 1 | (512, 256) | 0.20 | 1e-3 | 128 | 1e-4 | ← reference |
# | 2 | (512, 256) | 0.30 | 1e-3 | 128 | 1e-4 |
# | 3 | (512, 256) | 0.10 | 1e-3 | 128 | 1e-4 |
# | 4 | (512, 256, 128) | 0.20 | 1e-3 | 128 | 1e-4 |
# | 5 | (256,) | 0.10 | 1e-3 | 128 | 1e-4 |
# | 6 | (256, 128, 64) | 0.20 | 1e-3 | 128 | 1e-4 |
# | 7 | (1024, 512, 256) | 0.30 | 1e-3 | 128 | 1e-4 |
# | 8 | (512, 256) | 0.20 | **3e-3** | 128 | 1e-4 |
# | 9 | (512, 256) | 0.20 | **5e-4** | 128 | 1e-4 |
# | 10 | (512, 256) | 0.20 | 1e-3 | **64** | 1e-4 |
# | 11 | (512, 256) | 0.20 | 1e-3 | **256** | 1e-4 |
# | 12 | (512, 256) | 0.20 | 1e-3 | 128 | **0** (no wd probe) |
# | 13 | (512, 256) | 0.20 | 1e-3 | 128 | **1e-3** (strong wd probe) |

# %%
from sklearn.model_selection import train_test_split

X_train_u8, y_train, X_test_u8, y_test = load_mnist()
eda_overview(X_train_u8, y_train, "train")

# stratified 54k/6k split for search + early stopping (official 10k test untouched)
idx_tr, idx_va = train_test_split(np.arange(len(y_train)), test_size=10_000 if not SMOKE else 600,
                                  stratify=y_train, random_state=SEED)
tr_X = to_img_tensor(X_train_u8[idx_tr]); tr_y = torch.from_numpy(y_train[idx_tr])
va_X = to_img_tensor(X_train_u8[idx_va]); va_y = torch.from_numpy(y_train[idx_va])
log(f"[split] train={len(tr_y)} val={len(va_y)} test={len(y_test)}")

def make_mlp(hidden, dropout):
    layers, prev = [nn.Flatten()], 784
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, 10)]
    return nn.Sequential(*layers)

# %%
# ------------------------------------------------------------- HP search
SEARCH_CONFIGS = [
    dict(name="mlp-01", hidden=(512, 256),      dropout=0.20, lr=1e-3, bs=128, wd=1e-4),
    dict(name="mlp-02", hidden=(512, 256),      dropout=0.30, lr=1e-3, bs=128, wd=1e-4),
    dict(name="mlp-03", hidden=(512, 256),      dropout=0.10, lr=1e-3, bs=128, wd=1e-4),
    dict(name="mlp-04", hidden=(512, 256, 128), dropout=0.20, lr=1e-3, bs=128, wd=1e-4),
    dict(name="mlp-05", hidden=(256,),          dropout=0.10, lr=1e-3, bs=128, wd=1e-4),
    dict(name="mlp-06", hidden=(256, 128, 64),  dropout=0.20, lr=1e-3, bs=128, wd=1e-4),
    dict(name="mlp-07", hidden=(1024, 512, 256), dropout=0.30, lr=1e-3, bs=128, wd=1e-4),
    dict(name="mlp-08", hidden=(512, 256),      dropout=0.20, lr=3e-3, bs=128, wd=1e-4),
    dict(name="mlp-09", hidden=(512, 256),      dropout=0.20, lr=5e-4, bs=128, wd=1e-4),
    dict(name="mlp-10", hidden=(512, 256),      dropout=0.20, lr=1e-3, bs=64,  wd=1e-4),
    dict(name="mlp-11", hidden=(512, 256),      dropout=0.20, lr=1e-3, bs=256, wd=1e-4),
    dict(name="mlp-12", hidden=(512, 256),      dropout=0.20, lr=1e-3, bs=128, wd=0.0),
    dict(name="mlp-13", hidden=(512, 256),      dropout=0.20, lr=1e-3, bs=128, wd=1e-3),
]

def mlp_aug(x):  # light augmentation for an MLP (heavy warps hurt dense-only models)
    return batch_affine(x, max_deg=7, translate=0.06)

search_epochs = 2 if SMOKE else 8
search_rows = []
for i, cfg in enumerate(SEARCH_CONFIGS):
    if SMOKE and i % 3:          # shrink the grid in smoke mode
        continue
    set_torch_seed(SEED + 100 + i)
    model = make_mlp(cfg["hidden"], cfg["dropout"]).to(DEVICE)
    hist, best_ep, best_acc = train_model(
        model, tr_X, tr_y, va_X, va_y, epochs=search_epochs, lr=cfg["lr"], wd=cfg["wd"],
        batch_size=cfg["bs"], aug=mlp_aug, scheduler=True, patience=99,
        seed=SEED + 100 + i, tag=cfg["name"], log_every=10)
    search_rows.append({**cfg, "best_val_acc": best_acc, "best_epoch": best_ep,
                        "params": count_params(model)})
    del model

search_df = pd.DataFrame(search_rows).sort_values("best_val_acc", ascending=False)
display(search_df)
best_cfg = next(r for r in search_rows if r["name"] == search_df.iloc[0]["name"])
log(f"[search] winner: {best_cfg['name']} → hidden={best_cfg['hidden']} "
    f"dropout={best_cfg['dropout']} lr={best_cfg['lr']} bs={best_cfg['bs']} wd={best_cfg['wd']} "
    f"(val acc {best_cfg['best_val_acc']*100:.3f}%)")

# %%
# ------------------------------------------------- final training (early stop)
FINAL_EPOCHS = 6 if SMOKE else 60
PATIENCE = 3 if SMOKE else 8
set_torch_seed(SEED)
final_mlp = make_mlp(best_cfg["hidden"], best_cfg["dropout"])
log(f"[final] params: {count_params(final_mlp):,}")
t0 = time.perf_counter()
hist, best_ep, best_va = train_model(
    final_mlp, tr_X, tr_y, va_X, va_y, epochs=FINAL_EPOCHS, lr=best_cfg["lr"],
    wd=best_cfg["wd"], batch_size=best_cfg["bs"], aug=mlp_aug,
    patience=PATIENCE, seed=SEED, tag="mlp-final")
mlp_fit_s = time.perf_counter() - t0
plot_curves(hist, "MLP", "mlp_curves.png")
gap = hist["train_acc"][best_ep - 1] - hist["val_acc"][best_ep - 1]
log(f"[final] best val acc {best_va*100:.3f}% @ epoch {best_ep}/{len(hist['val_acc'])} "
    f"(train−val gap at best epoch: {gap*100:.2f} pts — small gap ⇒ no overfitting)")

# %%
# --------------------------------------------------------------- evaluation
mlp_pred = predict_torch(final_mlp, X_test_u8)
mlp_metrics = eval_classification(y_test, mlp_pred)
mlp_metrics.update({"best_val_acc": best_va, "best_epoch": best_ep,
                    "epochs_run": len(hist["val_acc"]),
                    "train_val_gap_at_best": float(gap),
                    "fit_seconds": mlp_fit_s,
                    "latency_ms_per_sample": torch_latency_ms(final_mlp),
                    "params": count_params(final_mlp)})
log(f"[eval] MLP test accuracy = {mlp_metrics['accuracy']*100:.3f}%")
plot_confusion(mlp_metrics["confusion"], "MLP — test confusion", "mlp_confusion.png")
print(classification_report(y_test, mlp_pred, digits=4))

# %%
# ------------------------------------------------------------ save artifacts
torch.save(final_mlp.state_dict(), "models/mlp_mnist.pt")
mlp_payload = model_json_payload("mlp", {"activation": "relu",
                                         "layers": export_mlp_layers(final_mlp)},
                                 {"test_accuracy": mlp_metrics["accuracy"],
                                  "best_val_accuracy": best_va})
write_model_json(mlp_payload, "models/mlp_mnist.json")

# parity: NumPy recompute from the JSON (exactly what the JS engine does) vs torch
N_CHECK = 512
xs = X_test_u8[:N_CHECK].astype(np.float32) / 255.0
layers = mlp_payload["model"]["layers"]
probs_json = mlp_json_forward(layers, xs)
json_pred = probs_json.argmax(1)
with torch.no_grad():
    probs_torch = torch.softmax(final_mlp(to_img_tensor(X_test_u8[:N_CHECK]).to(DEVICE)
                                          .float().div_(255.0)), dim=1).cpu().numpy()
agree = float((json_pred == probs_torch.argmax(1)).mean())
max_pd = float(np.abs(probs_json - probs_torch).max())
log(f"[parity] JSON-vs-torch prediction agreement: {agree*100:.2f}% · max prob diff {max_pd:.2e}")
assert agree >= 0.998 and max_pd < 5e-3, "MLP JSON export parity failed!"

mlp_summary = {
    "model": f"MLP {best_cfg['hidden']} (BN+ReLU+Dropout {best_cfg['dropout']})",
    "best_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_cfg.items()},
    "search": {"epochs_per_config": search_epochs, "configs": search_rows},
    "final": mlp_metrics, "curves": hist,
    "artifacts": {"pt": "models/mlp_mnist.pt", "json": "models/mlp_mnist.json",
                  "pt_sha256_16": sha256_file("models/mlp_mnist.pt"),
                  "json_sha256_16": sha256_file("models/mlp_mnist.json")},
}
write_metrics(mlp_summary, "models/metrics_mlp.json")
log("\n=== REPORT ROW ===")
report_row("MLP", mlp_metrics,
           f"{count_params(final_mlp):,} params · {mlp_metrics['epochs_run']} epochs")
log("\n[notebook 2 done] artifacts:")
for f in ["models/mlp_mnist.pt", "models/mlp_mnist.json", "models/metrics_mlp.json"]:
    log(f"  · {f}  ({os.path.getsize(f)/1e6:.2f} MB, sha256:{sha256_file(f)})")
