# DigitLab · Kaggle Training Notebooks

Three self-contained notebooks that train the three MNIST models on the
**[`hojjatk/mnist-dataset`](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)**
dataset (the canonical full MNIST — 60,000 training + 10,000 test 28×28 images,
shipped as the **original IDX binary files**):

| notebook | model | framework | accelerator | approx. runtime |
|---|---|---|---|---|
| `01_classical_svm.ipynb` | RBF-SVM (`sklearn.svm.SVC`) + LogReg/RF baselines | scikit-learn | CPU | 1.5–2.5 h |
| `02_neural_network_mlp.ipynb` | Multi-layer perceptron (dense only) | PyTorch | GPU T4/P100 (CPU ok) | ~25 min GPU |
| `03_cnn.ipynb` | Intro 2-block CNN | PyTorch | GPU T4/P100 (CPU ok) | ~20 min GPU |

## How to run (each notebook)

1. On Kaggle: **Create Notebook** → **File → Import Notebook** → upload the `.ipynb`.
2. **Add Input → Datasets** → search **"MNIST Dataset"** by **hojjatk** → attach it.
   It mounts at `/kaggle/input/mnist-dataset/`; the loader parses the canonical
   IDX files directly (it also accepts PNG folder-per-class mirrors unchanged).
3. For notebooks 2 and 3: **Settings → Accelerator → GPU T4 x2** (or P100).
   Notebook 1 is fine on CPU.
4. **Run All**. Every notebook ends having written its artifacts to
   `/kaggle/working/models/`:

| file | what it is |
|---|---|
| `svc_mnist.joblib` / `mlp_mnist.pt` / `cnn_mnist.pt` | the trained models in each framework's **standard format** |
| `svc_mnist.json` / `mlp_mnist.json` / `cnn_mnist.json` | **portable export** consumed by the web + mobile apps |
| `metrics_*.json` | hyperparameter search tables, final metrics, training curves, artifact hashes |

5. Open the **Output** panel of the notebook → download the `models/` folder →
   copy the three `*_mnist.json` files into the repo's `models/` directory
   (replacing the demo models trained on a small subset), then re-run
   `web npm run build` + `mobile npm run sync` to update the apps.

## What each notebook does (summary)

- **Preprocessing** (identical in all three, and identical to the serving pipeline
  in the apps): PNG → uint8 `[0..255]` → `float32 ÷ 255`. No further
  standardization, so training math equals serving math. Hyperparameter search and
  early stopping use a stratified 54k/6k split of the training data; the official
  10k test set is touched exactly once, at the very end.
- **Hyperparameter search**: a documented grid (MLP/CNN) or `RandomizedSearchCV`
  over `C`/`gamma` (SVM), never defaults-only. Full search tables are printed and
  saved to `metrics_*.json`.
- **Early stopping** (notebooks 2–3): validation-accuracy monitor with patience,
  minimum-delta and best-weights restore; training curves prove the absence of
  under/overfitting. The SVM's analogue to early stopping is capacity control via
  `C` and `gamma` (kernel machines are not iterative over epochs).
- **Export parity check** (all three): predictions are recomputed *from the
  portable JSON* with exactly the arithmetic the JavaScript engine performs
  (including uint8 quantization of support vectors) and compared against the
  framework's own predictions — the assert requires ≥99.9% agreement.
- **Reproducibility**: global seed 42, fixed config grids, artifact SHA-256 hashes.

## Notes

- The notebooks print a `REPORT ROW` line at the end — paste those rows into
  `REPORT.md` to finalize the comparison table with your run's numbers.
- GPU runs introduce tiny nondeterminism (cuDNN); expect ±0.05% accuracy jitter
  between identical runs.
