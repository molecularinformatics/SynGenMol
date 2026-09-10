"""Render the PPO score curves published for each demonstration.

Two figures, one per demonstration, each showing the per-update batch mean and
batch maximum GeoDiff score across the 500 PPO updates.  Faint lines are the raw
per-update values; heavy lines are a centred 15-update rolling mean, drawn in the
same hue so each series keeps one identity.

Only matplotlib is required.  Usage, from the repository root:
    python docs/figures/plot_ppo_curves.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
MEAN_HUE = "#2a78d6"  # categorical slot 1
MAX_HUE = "#eb6834"   # categorical slot 2
WINDOW = 15

FIGURES = [
    ("outputs/syngenmol_demo_unconstraint/results/ppo_score_curve.csv",
     "docs/figures/ppo_curve_unconstrained.png",
     "Unconstrained generation"),
    ("outputs/syngenmol_demo_warhead_constraint/results/ppo_score_curve.csv",
     "docs/figures/ppo_curve_warhead_constrained.png",
     "Warhead-protected generation (first-token constraint)"),
]


def rolling(values: list[float], window: int) -> list[float]:
    """Centred rolling mean that shortens the window at both edges."""
    half = window // 2
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def render(csv_path: Path, png_path: Path, title: str) -> None:
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    step = [int(r["step"]) for r in rows]
    series = [
        ("Batch mean", [float(r["score_mean"]) for r in rows], MEAN_HUE),
        ("Batch maximum", [float(r["score_max"]) for r in rows], MAX_HUE),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for label, values, hue in series:
        ax.plot(step, values, color=hue, lw=0.8, alpha=0.22, zorder=2)
        smooth = rolling(values, WINDOW)
        ax.plot(step, smooth, color=hue, lw=2.0, label=label, zorder=3,
                solid_capstyle="round")
        # Direct label at the right end, so identity never rests on colour alone.
        ax.annotate(f"{label}  {smooth[-1]:.2f}",
                    xy=(step[-1], smooth[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=INK_MUTED, zorder=4)

    ax.set_xlabel("PPO update", fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("PED-GeoDiff score", fontsize=9.5, color=INK_MUTED)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=10)
    ax.set_xlim(0, step[-1])
    # Both figures share these limits so they can be compared side by side.
    ax.set_ylim(0.40, 0.91)

    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3, width=0.8)

    legend = ax.legend(loc="lower right", frameon=False, fontsize=9,
                       labelcolor=INK_MUTED, handlelength=1.6)
    legend.set_zorder(5)

    # Leave room for the right-hand direct labels.
    fig.subplots_adjust(left=0.085, right=0.78, top=0.89, bottom=0.13)
    fig.savefig(png_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {png_path}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    for csv_rel, png_rel, title in FIGURES:
        render(root / csv_rel, root / png_rel, title)


if __name__ == "__main__":
    main()
