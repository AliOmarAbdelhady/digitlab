#!/usr/bin/env bash
# Local end-to-end smoke test of all three notebook code paths on a small subset.
# Uses the PNG tree (exact Kaggle loading path), shrinks search/epochs, and runs
# every cell including export + parity assertions.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=local/.venv/bin/python

export DIGITLAB_SMOKE=1
[ -d local/data/pngtree ] || $PY tools/make_png_tree.py
$PY tools/build_notebooks.py

for nb in nb1_svm nb2_mlp nb3_cnn; do
  echo ""
  echo "================ SMOKE: $nb ================"
  $PY "local/flat/$nb.py"
done
echo ""
echo "ALL SMOKE TESTS PASSED"
