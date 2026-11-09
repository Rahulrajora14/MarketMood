"""House style for every figure in the project (reports, slides, dashboard)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = "#FAF7F2"
INK = "#1E2430"
MUTED = "#6B6B6B"
GRID = "#E4DED3"

COLORS = {
    "price": "#7A8B99",      # steel
    "attention": "#588B8B",  # sage
    "lm": "#E0A458",         # ochre
    "finbert": "#C8553D",    # terracotta
    "all_text": "#2F4858",   # slate
    "baseline": "#B9B2A6",   # sand
}
REGIME_COLORS = {"low": "#9DB4C0", "mid": "#5C6B73", "high": "#253237"}

LABELS = {
    "price": "Price only",
    "attention": "Price + news volume",
    "lm": "Price + LM lexicon",
    "finbert": "Price + FinBERT",
    "all_text": "Price + FinBERT + Reddit",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def subtitle(ax, text: str) -> None:
    """Grey line under the title. Call after ax.set_title."""
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=22)
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
