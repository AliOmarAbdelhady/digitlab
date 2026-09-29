#!/usr/bin/env bash
# Finalize everything after real Kaggle models land in models/:
#   fixtures + core parity tests → web rebuild + Pages deploy → mobile sync +
#   APK rebuild → adb install on the connected phone.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.nvm/versions/node/v26.8.1/bin:$PATH"
export ANDROID_HOME=/home/ali/Android/Sdk
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

echo "=== 1/6 regenerating core fixtures + parity tests (real models) ==="
local/.venv/bin/python tools/make_fixtures.py
( cd core && npx vitest run 2>&1 | tail -6 )

echo "=== 2/6 web: sync models + build ==="
( cd web && npm run sync >/dev/null && npm run build 2>&1 | tail -3 )

echo "=== 3/6 deploy Pages (web + new APK after rebuild below is separate) ==="
bash tools/deploy-pages.sh

echo "=== 4/6 mobile: sync models ==="
( cd mobile && npm run sync )

echo "=== 5/6 mobile: rebuild release APK (arm64) ==="
( cd mobile/android && ./gradlew assembleRelease --no-daemon -q \
    -PreactNativeArchitectures=arm64-v8a 2>&1 | tail -3 )
ls -la mobile/android/app/build/outputs/apk/release/app-release.apk

echo "=== 6/6 install on phone (if connected) ==="
if adb devices | grep -q "device$"; then
  adb install -r mobile/android/app/build/outputs/apk/release/app-release.apk
else
  echo "no device connected — install manually: adb install -r mobile/android/app/build/outputs/apk/release/app-release.apk"
fi
echo "=== done ==="
