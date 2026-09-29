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
