# Model files

The apps (web + mobile) load models from the **portable DigitLab JSON format**
produced by the Kaggle notebooks (see `kaggle/README.md`).

| file | used by | produced by |
|---|---|---|
| `svc_mnist.json` | apps — RBF-SVM engine (uint8 support vectors + dual coefficients) | `01_classical_svm.ipynb` |
| `mlp_mnist.json` | apps — dense MLP weights (BatchNorm folded) | `02_neural_network_mlp.ipynb` |
| `cnn_mnist.json` | apps — CNN ops list (conv/relu/pool/linear, BatchNorm folded) | `03_cnn.ipynb` |
| `*.joblib`, `*.pt` | framework-standard deliverables (not used by the apps) | same notebooks — download from Kaggle Output, not committed to git |

## ⚠️ Current contents: DEMO models

The JSON files currently in this folder are **demo models trained on a 3,000-image
subset** during automated local testing (test accuracy ≈ 93–97%). They exist so
the web/mobile apps work end-to-end before the real training runs.

**After running the notebooks on Kaggle:** download the `models/` output folder
from each notebook and replace these three files with the real ones
(≈60k-image training → SVM ~98.5%, MLP ~98.4%, CNN ~99.5% expected). Then:

```bash
cd web    && npm run sync && npm run build     # refreshes web/public/models
cd mobile && npm run sync                       # refreshes mobile/assets/models
```

and redeploy (`gh-pages` branch / rebuild the APK).
