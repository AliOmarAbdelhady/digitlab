#!/usr/bin/env bash
# Deploy web build + APK to GitHub Pages via a git worktree (never disturbs the
# working tree — safe even while gradle is running).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.nvm/versions/node/v26.8.1/bin:$PATH"

APK=mobile/android/app/build/outputs/apk/release/app-release.apk
test -f "$APK" || { echo "APK missing at $APK — build it first"; exit 1; }

echo "==> building web"
( cd web && npm run sync >/dev/null && npm run build >/dev/null )
cp "$APK" web/dist/digitlab.apk

echo "==> deploying to gh-pages (worktree)"
WT=$(mktemp -d /tmp/digitlab-pages.XXXXXX)
if git show-ref --verify --quiet refs/heads/gh-pages; then
  git worktree add "$WT" gh-pages >/dev/null
  ( cd "$WT" && git rm -rq . 2>/dev/null || true )
else
  git worktree add --orphan -b gh-pages "$WT" >/dev/null
fi
cp -r web/dist/. "$WT"/
( cd "$WT" && git add -A \
  && git -c user.name="Ali Omar Abdelhady" -c user.email="AliOmarAbdelhady@users.noreply.github.com" \
      commit -qm "deploy: DigitLab web + APK" \
  && git push -q origin gh-pages )
git worktree remove --force "$WT"
echo "==> deployed: https://aliomarabdelhady.github.io/digitlab/"
