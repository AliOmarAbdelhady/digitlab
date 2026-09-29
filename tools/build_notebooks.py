#!/usr/bin/env python3
"""Build Kaggle .ipynb notebooks and flattened smoke-test scripts from kaggle/src.

- kaggle/<NN>_<name>.ipynb  : importable into Kaggle (markdown + code cells)
- local/flat/<nb>.py        : exact concatenation, runnable as a plain script
                              (markdown cells are already '#' comments)
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kaggle" / "src"

NOTEBOOKS = {
    "01_classical_svm.ipynb": (["_common.py", "nb1_svm.py"], "nb1_svm"),
    "02_neural_network_mlp.ipynb": (["_common.py", "_torch_common.py", "nb2_mlp.py"], "nb2_mlp"),
    "03_cnn.ipynb": (["_common.py", "_torch_common.py", "nb3_cnn.py"], "nb3_cnn"),
}


def parse_cells(text: str):
    """Split a percent-format file into (type, lines) cells."""
    cells, cur_type, cur_lines = [], None, []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if cur_type is not None:
                cells.append((cur_type, cur_lines))
            cur_type = "markdown" if "[markdown]" in line else "code"
            cur_lines = []
        else:
            if cur_type is None:
                cur_type = "code"
            cur_lines.append(line)
    if cur_type is not None:
        cells.append((cur_type, cur_lines))
    return cells


def md_source(lines):
    """Strip the leading '# ' from markdown-cell lines."""
    out = []
    for l in lines:
        out.append(l[2:] if l.startswith("# ") else ("") if l == "#" else l)
    return "\n".join(out).strip("\n")


def main():
    flat_dir = ROOT / "local" / "flat"
    flat_dir.mkdir(parents=True, exist_ok=True)
    for out_name, (parts, flat_name) in NOTEBOOKS.items():
        texts = [(SRC / p).read_text() for p in parts]
        (flat_dir / f"{flat_name}.py").write_text("\n\n".join(texts))

        cells = []
        for text in texts:
            for ctype, lines in parse_cells(text):
                src = md_source(lines) if ctype == "markdown" else "\n".join(lines).strip("\n")
                if not src:
                    continue
                if ctype == "markdown":
                    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
                else:
                    cells.append({"cell_type": "code", "metadata": {},
                                  "execution_count": None, "outputs": [], "source": src})
        nb = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python",
                               "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        out = ROOT / "kaggle" / out_name
        out.write_text(json.dumps(nb, indent=1))
        print(f"built {out.relative_to(ROOT)} ({len(cells)} cells) "
              f"+ local/flat/{flat_name}.py")


if __name__ == "__main__":
    sys.exit(main())
