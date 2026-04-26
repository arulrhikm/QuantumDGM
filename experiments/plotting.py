from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "linear": "#1f77b4",
    "clique": "#2ca02c",
    "full": "#d62728",
    "vqc": "#6a3d9a",
    "mps": "#ff7f0e",
}


def ensure_figures_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "figures")
    os.makedirs(out, exist_ok=True)
    return out


def savefig(name: str):
    out_dir = ensure_figures_dir()
    path = os.path.join(out_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def style_ax(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")


def mean_std(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    uniq = np.unique(xs)
    m = []
    s = []
    for u in uniq:
        vals = ys[xs == u]
        m.append(float(np.mean(vals)))
        s.append(float(np.std(vals)))
    return uniq, np.array(m), np.array(s)

