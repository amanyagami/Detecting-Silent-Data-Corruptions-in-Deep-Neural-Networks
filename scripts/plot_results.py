"""Render results/detection.png from the campaign JSON files (matplotlib; `plots` extra)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
ALPHA = 0.01
ROWS = [
    ("tau1", "τ1"),
    ("tau2", "τ2"),
    ("tau3", "τ3"),
    ("tau1+2", "τ1 + τ2"),
    ("tau1/k+2", "τ1/k + τ2"),
    ("all", "τ1 + τ2 + τ3"),
]
PANELS = [
    ("campaign_seed0.json", "Random bit (0-31)"),
    ("campaign_seed1_exponent.json", "Exponent bits only (23-30)"),
]


def main(results: str = "results") -> None:
    """Draw one panel per campaign: SDC detection rate with 95% intervals."""
    root = Path(results)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True, facecolor=SURFACE)
    for ax, (fname, title) in zip(axes, PANELS, strict=True):
        res = json.loads((root / fname).read_text())
        n_sdc = res["detection"][f"all@{ALPHA}/sdc"]["n"]
        ax.set_facecolor(SURFACE)
        for i, (key, _label) in enumerate(ROWS):
            d = res["detection"][f"{key}@{ALPHA}/sdc"]
            fpr = res["false_positive_rate"][f"{key}@{ALPHA}"]["rate"]
            y = len(ROWS) - 1 - i
            ax.barh(y, d["rate"], height=0.42, color=SERIES, zorder=2)
            ax.plot(d["ci95"], [y, y], color=INK, lw=1.1, zorder=3, solid_capstyle="butt")
            ax.text(1.02, y, f"{d['rate']:.0%}", va="center", ha="left", color=INK, fontsize=9.5)
            ax.text(1.2, y, f"{fpr:.1%}", va="center", ha="left", color=MUTED, fontsize=9.5)
        ax.text(1.02, len(ROWS) - 0.35, "detected", color=MUTED, fontsize=8.5, ha="left")
        ax.text(1.2, len(ROWS) - 0.35, "false alarms", color=MUTED, fontsize=8.5, ha="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, len(ROWS) - 0.2)
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([lab for _, lab in ROWS][::-1], color=INK, fontsize=10)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=9)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_title(f"{title}\n{n_sdc} SDC trials", loc="left", color=INK, fontsize=10.5)
    fig.suptitle(
        "Share of silent data corruptions detected, at a 1% false-alarm budget",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.015,
        "Bars: detection rate; line: 95% Wilson interval; "
        "false alarms measured on held-out clean images.",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.1, right=0.86, top=0.78, bottom=0.14, wspace=0.62)
    fig.savefig(root / "detection.png", dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    main()
