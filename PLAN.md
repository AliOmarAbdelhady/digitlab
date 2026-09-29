# DigitLab — MNIST 3-Model Project Plan (master state document)

This file is the durable plan + state log for the whole project. Update the
"Status log" as stages complete. If context is ever summarized, this file is
the source of truth.

## Goal

1. Three Kaggle-ready notebooks (classical ML / MLP / CNN) trained on the
   **Kaggle dataset `hojjatk/mnist-dataset`** (canonical full MNIST:
   60,000 train + 10,000 test PNGs at `/kaggle/input/mnist-dataset/mnist_png/{training,testing}/<digit>/*.png`).
2. Hyperparameter search + early stopping + no under/overfitting, max achievable accuracy.
3. Saved models in standard formats (joblib / torch state_dict) + a portable
   JSON export used by the apps.
4. Web app (canvas drawing, model switcher, predict) — hosted publicly
   (GitHub Pages via `gh`, account `AliOmarAbdelhady`, repo `digitlab`).
5. Expo React Native mobile app, same features, premium UI; release APK built
   locally (Android SDK at `~/Android/Sdk`, Java 17) and installed via adb;
   APK also hosted on the website for download.
6. REPORT.md + READMEs + run instructions.

## Key environment facts (verified)

- python3.14 default; **python3.12 + uv** at `~/.local/bin/uv` → local venv `local/.venv` (numpy, scikit-learn, pillow, matplotlib, torch CPU).
- Node v26.8.1 via nvm → `export PATH="$HOME/.nvm/versions/node/v26.8.1/bin:$PATH"`.
- Java 17, Android SDK `~/Android/Sdk` (set `ANDROID_HOME=/home/ali/Android/Sdk`), adb present, **no device attached yet** at recon time.
- `gh` authenticated (AliOmarAbdelhady, repo scope) → GitHub Pages hosting.
- Disk tight (~5.6 GB free): use `--no-cache`/clean npm+pip caches, clean gradle after APK build.
- Torch local CPU wheel: use `--index-url https://download.pytorch.org/whl/cpu`.

## Architecture

```
Kaggle notebooks (user runs)  ──outputs──▶  models/*.joblib|*.pt (standard formats)
                                           models/*.json       (portable app format)
                                                    │
                     ┌──────────────────────────────┴───────────────────┐
              web/ (Vite+React, static)                          mobile/ (Expo RN, APK)
                     └──────────── both import core/ (TypeScript) ──────┘
        core = pure-JS inference: strokes→28×28 MNIST preprocess + SVC/MLP/CNN engines
```

- All 3 models use identical preprocessing at train and serve time: **X/255.0,
  CHW 1×28×28, white digit on black** (no extra standardization → train/serve parity).
- Apps preprocess drawings exactly like MNIST was built: crop ink bounding box →
  scale longer side to 20px (area-average) → center by center-of-mass in 28×28.

## Model specs

| # | Model | Framework | Search | Final fit | Expected test acc |
|---|-------|-----------|--------|-----------|-------------------|
| 1 | RBF-SVM (`sklearn.svm.SVC`) + LogReg/RF baselines | sklearn | RandomizedSearchCV (12 iter ×3 fold, 15k subsample) over C∈logU[1,30], γ∈logU[5e-4,5e-3] | best (C,γ) on full 60k, cache 800MB | ~98.4–98.7% |
| 2 | MLP (dense only, BN+dropout) | PyTorch | 12 configs × 8 epochs (hidden sizes, dropout, lr, batch, wd) | best config, ≤60 ep, early stop patience 8 (val acc), restore best | ~98.3–98.6% |
| 3 | Intro CNN (2×[conv32,conv32]→pool, 2×[conv64,conv64]→pool → FC128; BN, dropout, aug) | PyTorch | ~9 configs × 4 epochs (lr, dropout, batch, width) | best config, ≤40 ep, early stop patience 5 | ~99.4–99.6% |

Early stopping: custom class on val accuracy, restore best weights; curves plotted
to prove no under/overfit. SVM: capacity control via C/γ (documented analogue).

## Portable JSON model format (`models/*.json`)

```jsonc
{ "format":"digitlab-model", "version":1, "kind":"svc|mlp|cnn",
  "classes":[0..9], "created":iso, "metrics":{...},
  "preprocess":{"scale":255,"shape":[1,28,28],"layout":"CHW"},
  "model": { /* kind-specific */ } }
```
- svc: `{gamma, n_sv, sv_u8:base64(n_sv×784, pixel units 0-255), dual_f32:base64(n_sv×10), intercept_f32:base64(10)}` — decision = Σ dual·exp(-γ‖x-sv‖²)+b, softmax over classes.
- mlp: `{layers:[{w_f32:base64([out,in] row-major), b_f32:base64([out]), in, out}], activation:"relu"}` → logits.
- cnn: `{ops:[{op:"conv",w,b,in_c,out_c,k,pad}|{op:"relu"}|{op:"maxpool2"}|{op:"flatten"}|{op:"linear",w,b,in,out}|{op:"softmax"}]}` — **BatchNorm folded into preceding conv/linear at export**.
- Base64 decoded by hand-written decoder in core (Hermes/old-browser safe). Float32 little-endian.

## Verification strategy

1. Notebook code paths smoke-tested locally (subset, SMOKE env flag) before .ipynb conversion.
2. Python export-parity cell in each notebook: recompute predictions from the JSON exactly as the JS engine will (incl. uint8 SV quantization) vs framework predictions — require ≥99.9% agreement.
3. core/ node tests: fixtures (small locally-trained models + 256 MNIST test images + expected predictions from Python) → JS engine must match ≥99.6% argmax, prob diff <1e-3.
4. Web app tested in real browser (drawing injection hook `window.__digitlabTest`).
5. Mobile app: adb install + console check when device available.

## Repo layout

```
digitlab/
  PLAN.md README.md REPORT.md .gitignore
  kaggle/            3 .ipynb + README (how to run on Kaggle) + src/*.py (percent-format sources)
  models/            *.json (app models, committed), *.joblib/*.pt (NOT committed; from Kaggle)
  core/              TS package (engines + preprocess + base64), node-tested
  web/               Vite React app; dist → gh-pages branch → Pages URL
  mobile/            Expo app; android release APK via gradle (local SDK)
  tools/             py→ipynb converter, fixture generators
  local/             venv + fixtures (gitignored)
```

## Hosting plan

- `gh repo create digitlab --public --source . --push`
- Build web (vite, `base:'./'`), copy models/*.json into dist, orphan `gh-pages`
  branch, push, enable Pages via `gh api`.
- APK copied to dist as `digitlab.apk` (download link + QR on site).

## Status log

- [x] Recon: env + dataset verified (hojjatk/mnist-dataset, mnist_png/training|testing/<d>/*.png)
- [x] Scaffold + PLAN.md
- [ ] Local python env (uv, bg install)
- [ ] Notebook sources (3) + smoke test + .ipynb conversion
- [ ] core/ TS engines + fixtures + tests
- [ ] web/ app + browser test
- [ ] Hosting (gh-pages)
- [ ] mobile/ Expo app + APK + install
- [ ] REPORT.md + READMEs
