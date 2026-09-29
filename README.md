# DigitLab — MNIST 3-Model Lab & Drawing Apps

Train **three MNIST models on Kaggle** (classical SVM · MLP · CNN), then recognize
**your own hand-drawn digits** with any of them — in the browser and on Android —
using a dependency-free TypeScript inference core.

```
Kaggle notebooks (you run) ──▶ models/*.joblib|*.pt   (standard formats)
                             ▶ models/*.json          (portable app format)
                                        │
        ┌───────────────────────────────┴───────────────────────┐
   web/ (Vite + React)                                     mobile/ (Expo RN)
   canvas · model picker · predict                          same, as an APK
        └────────────── both import core/ (TS engines) ────────┘
```

- **Live web app:** <https://aliomarabdelhady.github.io/digitlab/>
- **APK:** downloadable from the live site (`digitlab.apk` link in the footer area
  of the repo's gh-pages) — see *Mobile* below.

## Repository layout

| path | contents |
|---|---|
| `kaggle/` | the three Kaggle-ready notebooks + `README.md` with exact run steps |
| `models/` | portable `*_mnist.json` (apps) — **currently demo models**; replace with Kaggle outputs. `README.md` explains |
| `core/` | shared TS inference core: strokes→784 MNIST preprocessing + SVC/MLP/CNN engines (no runtime deps) + parity tests |
| `web/` | Vite + React web app (premium dark UI, canvas drawing, model switcher, confidence ring) |
| `mobile/` | Expo (React Native) app with the same design; builds a release APK |
| `tools/` | notebook builder, local smoke tests, fixture generator |
| `REPORT.md` | the project report (preprocessing, HP search, comparison, best-model justification) |
| `PLAN.md` | full architecture decisions + status log |

## 1 · Train the models (Kaggle)

See **`kaggle/README.md`** — import each `.ipynb`, attach dataset
`hojjatk/mnist-dataset`, run. Download each notebook's `models/` output and drop
the three `*_mnist.json` files into `models/` (they currently hold demo models
trained on a 3k subset purely so the apps run before real training).

## 2 · Web app

```bash
cd web
npm install
npm run dev        # http://localhost:5173  (models auto-synced into public/)
npm run build      # production build in dist/ (relative paths — host anywhere)
```

The production build is deployed to GitHub Pages (`gh-pages` branch):

```bash
cd web && npm run build
cd .. && git checkout --orphan gh-pages   # or reuse the existing one
# (the live deployment was produced from web/dist — see PLAN.md §hosting)
```

Everything (all three models) runs **client-side** — no server, no API keys.

## 3 · Mobile app (Expo → APK)

```bash
cd mobile
npm install
npm run sync         # copy models/*.json → assets/models/*.dlmodel

# development on a connected device / emulator:
npx expo run:android

# release APK (debug-signed — fine for personal sideloading):
npx expo prebuild -p android
cd android && ./gradlew assembleRelease
# → android/app/build/outputs/apk/release/app-release.apk
adb install -r android/app/build/outputs/apk/release/app-release.apk
```

Requires Android SDK (`ANDROID_HOME`), JDK 17, and Node ≥ 18. The release APK is
signed with the debug keystore (Expo default) — perfect for installing on your
own phone, not for store distribution.

## 4 · Verification (what has been tested)

- `tools/smoke.sh` — all three notebook pipelines end-to-end on a small subset,
  including the export-parity asserts (JSON recompute vs framework predictions).
- `core/` — `npm test`: 12/12 passing; engines match Python-computed predictions
  on 256 fixed MNIST test images (≥99.6% argmax agreement, prob diff < 1e-3).
- Web app — browser-automated e2e: synthetic strokes for 3/1/7/0 recognized by
  all three models; real pointer drawing + real button click verified; canvas
  ink verified at pixel level; deployed site re-verified in production.
- Mobile — TypeScript-clean, Metro bundle builds (Hermes), APK via Gradle.

## 5 · Swapping in the real Kaggle models

```bash
# after downloading Kaggle outputs into models/:
cd web    && npm run sync && npm run build     # web app picks them up
cd mobile && npm run sync                       # APK assets pick them up
# rebuild/redeploy: push web/dist to gh-pages; rebuild the APK
```

## Notes

- The three models share one serving preprocessing pipeline (crop → 20px box →
  center-of-mass → ÷255) that mirrors how MNIST itself was constructed — this is
  why canvas drawings classify correctly, not just test-set images.
- The SVM serving engine stores support vectors as uint8 (quantization verified
  lossless: 100% prediction agreement vs sklearn in the notebook parity cell).
