"""Figures for Project 5.

PROVIDED: you do not need to change this file (but you may restyle the figures).
Each function takes the dictionaries your tasks return and writes one PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_c_sweep(linear: dict, path: Path) -> None:
    """Task 1: training and test F1 against C (log scale)."""
    rows = sorted(linear["sweep"], key=lambda r: r["C"])
    cs = [r["C"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(cs, [r["train"]["f1"] for r in rows], "o-", label="train F1")
    ax.plot(cs, [r["test"]["f1"] for r in rows], "s-", label="test F1")
    ax.set_xscale("log"); ax.set_xlabel("C"); ax.set_ylabel("F1"); ax.set_title("Linear SVM: F1 vs. C")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_rbf_heatmap(rbf: dict, path: Path) -> None:
    """Task 2: cross-validation F1 over the C x gamma grid, with the train-CV gap in each cell."""
    cs = sorted({r["C"] for r in rbf["grid"]})
    gs = sorted({r["gamma"] for r in rbf["grid"]})
    cv = np.full((len(cs), len(gs)), np.nan)
    gap = np.full_like(cv, np.nan)
    for r in rbf["grid"]:
        i, j = cs.index(r["C"]), gs.index(r["gamma"])
        cv[i, j], gap[i, j] = r["cv_f1"], r["train_f1"] - r["cv_f1"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(cv, cmap="viridis", origin="lower")
    ax.set_xticks(range(len(gs)), [str(g) for g in gs]); ax.set_yticks(range(len(cs)), [str(c) for c in cs])
    ax.set_xlabel("gamma"); ax.set_ylabel("C"); ax.set_title("RBF SVM: CV F1 (train - CV gap)")
    for i in range(len(cs)):
        for j in range(len(gs)):
            if not np.isnan(cv[i, j]):
                ax.text(j, i, f"{cv[i, j]:.2f}\n({gap[i, j]:+.2f})", ha="center", va="center",
                        fontsize=7, color="white" if cv[i, j] < np.nanmean(cv) else "black")
    fig.colorbar(im, ax=ax, label="CV F1"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_per_category(one_class: dict, path: Path) -> None:
    """Task 3: detection rate for each attack category, one bar group per nu."""
    runs = sorted(one_class["runs"], key=lambda r: r["nu"])
    cats = sorted({c for r in runs for c in r["per_category"]})
    x = np.arange(len(cats)); w = 0.8 / max(len(runs), 1)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, r in enumerate(runs):
        ax.bar(x + k * w, [r["per_category"].get(c, np.nan) for c in cats], w, label=f"nu = {r['nu']}")
    ax.set_xticks(x + w * (len(runs) - 1) / 2, cats, rotation=35, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("detection rate"); ax.set_title("One-class SVM: detection by attack category")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
