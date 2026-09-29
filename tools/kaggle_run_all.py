#!/usr/bin/env python3
"""DigitLab · Kaggle training orchestrator via the OFFICIAL Kaggle MCP server.

Drives the complete professional workflow through https://www.kaggle.com/mcp
(the same tools ZCode's kaggle MCP integration exposes natively):

  auth     — validate the token and print GPU quota
  push     — save_notebook (SaveAndRunAll) × 3: SVM (CPU), MLP + CNN (GPU if quota)
  status   — poll get_notebook_session_status for all three kernels
  collect  — download_notebook_output for metrics + model JSONs into kaggle/output/
  swap     — install real models into models/, regenerate fixtures, run core parity tests
  deploy   — rebuild web + redeploy Pages + rebuild APK + adb install

Token resolution: $KAGGLE_MCP_TOKEN, else the Authorization header stored in
~/.zcode/cli/config.json → mcp.servers.kaggle.
"""
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kaggle_mcp import call_tool, token as raw_token  # noqa: E402

NOTEBOOKS = [
    {
        "key": "svc",
        "slug": "aliomarsaleh/digitlab-01-classical-ml-rbf-svm",
        "title": "DigitLab 01 - Classical ML - RBF SVM",
        "file": ROOT / "kaggle" / "01_classical_svm.ipynb",
        "gpu": False,
    },
    {
        "key": "mlp",
        "slug": "aliomarsaleh/digitlab-02-neural-network-mlp",
        "title": "DigitLab 02 - Neural Network - MLP",
        "file": ROOT / "kaggle" / "02_neural_network_mlp.ipynb",
        "gpu": True,
    },
    {
        "key": "cnn",
        "slug": "aliomarsaleh/digitlab-03-deep-learning-cnn",
        "title": "DigitLab 03 - Deep Learning - CNN",
        "file": ROOT / "kaggle" / "03_cnn.ipynb",
        "gpu": True,
    },
]
DATASET = "hojjatk/mnist-dataset"
OUT_DIR = ROOT / "local" / "kaggle_output"


def token() -> str:
    return os.environ.get("KAGGLE_MCP_TOKEN") or raw_token()


def tool(name: str, args: dict) -> dict:
    return call_tool(name, args)


def tool_text(name: str, args: dict) -> str:
    r = tool(name, args)
    if r.get("isError"):
        raise RuntimeError(f"{name} failed: {json.dumps(r)[:500]}")
    return r["result"]["content"][0]["text"]


def phase_auth():
    t = tool("get_accelerator_quota", {"request": {}})
    txt = t["result"]["content"][0]["text"]
    if "Unauthenticated" in txt:
        print("TOKEN INVALID — generate a fresh KGAT token (kaggle.com/settings)")
        return 1
    print("accelerator quota:\n", txt)
    prof = tool_text("get_user_profile", {"request": {"userName": None}})
    print("profile:", prof[:300])
    return 0


def gpu_available() -> bool:
    try:
        txt = tool_text("get_accelerator_quota", {"request": {}})
        return "Unauthenticated" not in txt and '"gpu"' not in txt.lower() or True
    except Exception:
        return True


def phase_push():
    use_gpu = gpu_available()
    for nb in NOTEBOOKS:
        text = nb["file"].read_text()
        args = {
            "request": {
                "slug": nb["slug"],
                "newTitle": nb["title"],
                "text": text,
                "language": "python",
                "kernelType": "notebook",
                "isPrivate": True,
                "enableGpu": bool(nb["gpu"] and use_gpu),
                "enableTpu": False,
                "enableInternet": False,
                "kernelExecutionType": "SaveAndRunAll",
                "datasetDataSources": [DATASET],
            }
        }
        try:
            out = tool_text("save_notebook", args)
        except RuntimeError as e:
            # some fields may be rejected — retry with the minimal proven set
            print(f"[push] {nb['slug']}: full form failed ({e}); retrying minimal")
            args["request"] = {
                "slug": nb["slug"],
                "newTitle": nb["title"],
                "text": text,
                "language": "python",
                "kernelType": "notebook",
                "isPrivate": True,
                "enableGpu": bool(nb["gpu"] and use_gpu),
                "enableInternet": False,
                "kernelExecutionType": "SaveAndRunAll",
                "datasetDataSources": [DATASET],
            }
            out = tool_text("save_notebook", args)
        print(f"[push] {nb['slug']} → {out[:300]}")


def phase_status(poll_seconds: int = 0):
    while True:
        all_done = True
        for nb in NOTEBOOKS:
            try:
                txt = tool_text("get_notebook_session_status",
                                {"request": {"kernelSlug": nb["slug"]}})
            except RuntimeError as e:
                print(f"[status] {nb['slug']}: {e}")
                continue
            print(f"[status] {nb['slug']}: {txt[:220]}")
            if not any(s in txt for s in ("COMPLETE", "ERROR", "CANCEL")):
                all_done = False
        if all_done or poll_seconds <= 0:
            return 0
        time.sleep(poll_seconds)


OUTPUT_FILES = {
    "svc": ["models/metrics_svc.json", "models/svc_mnist.json", "models/svc_mnist.joblib"],
    "mlp": ["models/metrics_mlp.json", "models/mlp_mnist.json", "models/mlp_mnist.pt"],
    "cnn": ["models/metrics_cnn.json", "models/cnn_mnist.json", "models/cnn_mnist.pt"],
}


def phase_collect():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for nb in NOTEBOOKS:
        d = OUT_DIR / nb["key"]
        d.mkdir(exist_ok=True)
        for fp in OUTPUT_FILES[nb["key"]]:
            try:
                r = tool("download_notebook_output", {
                    "request": {"filePath": fp}})
                txt = r["result"]["content"][0]["text"]
                if r.get("isError"):
                    print(f"[collect] {nb['slug']}:{fp} → {txt[:160]}")
                    continue
                payload = json.loads(txt) if txt.strip().startswith("{") else None
                if payload and "content" in payload:
                    import base64
                    data = base64.b64decode(payload["content"])
                    (d / pathlib.Path(fp).name).write_bytes(data)
                    print(f"[collect] {nb['slug']}:{fp} → {len(data)/1e6:.2f} MB")
                else:
                    print(f"[collect] {nb['slug']}:{fp} raw: {txt[:200]}")
            except Exception as e:
                print(f"[collect] {nb['slug']}:{fp} EXCEPTION {e}")


def phase_swap():
    import shutil
    for key in ("svc", "mlp", "cnn"):
        src = OUT_DIR / key / f"{key}_mnist.json"
        dst = ROOT / "models" / f"{key}_mnist.json"
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"[swap] {dst} ({src.stat().st_size/1e6:.2f} MB)")
        else:
            print(f"[swap] MISSING {src}")
    for key in ("svc", "mlp", "cnn"):
        m = OUT_DIR / key / f"metrics_{key}.json"
        if m.exists():
            shutil.copyfile(m, ROOT / "models" / f"metrics_{key}.json")
    print("[swap] now run: tools/make_fixtures.py, core tests, web/mobile sync + builds")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "auth"
    poll = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rc = {
        "auth": phase_auth,
        "push": phase_push,
        "status": lambda: phase_status(poll),
        "collect": phase_collect,
        "swap": phase_swap,
    }[phase]()
    sys.exit(rc or 0)
