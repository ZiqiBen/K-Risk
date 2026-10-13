"""Standalone panel: unsafe-decision rate of GPT-4o / GPT-4.1, zero-shot vs 20 in-context failure cases,
on the 352 held-out events (Wilson 95% CI; exact McNemar p). Same style as panel d of fig8_closedloop.py,
sized (4:3) to replace the closed-loop panel of the framework figure.

Reads results/icl_eval_n20.csv; writes figures/fig_icl.{png,pdf,svg}.
"""
import os
import sys
from math import comb, sqrt

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import AV_GREY, INK, MM, NEUTRAL, ORIG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W_MM, H_MM = 100.0, 75.0
FS_TICK, FS_LABEL, FS_VAL = 10, 11, 9.5


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    E = pd.read_csv(os.path.join(HERE, "results", "icl_eval_n20.csv"))
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    ax = fig.add_axes([0.14, 0.2, 0.84, 0.77])
    tr = blended_transform_factory(ax.transData, fig.transFigure)
    x = 0.0
    for m, mname in [("gpt4o", "GPT-4o"), ("gpt41", "GPT-4.1")]:
        W = E[E.model == m].pivot(index="event", columns="cond", values="unsafe").dropna()
        n = len(W)
        for c, col in [("zero-shot", AV_GREY), ("ICL-20", ORIG)]:
            k = int(W[c].sum())
            lo, hi = wilson(k, n)
            ax.bar(x, 100 * k / n, color=col, width=0.6, edgecolor="none")
            ax.plot([x, x], [100 * lo, 100 * hi], color=INK, lw=1.0, solid_capstyle="butt")
            ax.text(x, 100 * hi + 0.4, f"{100 * k / n:.2f}%\n({k}/{n})", ha="center", va="bottom",
                    fontsize=FS_VAL, color=INK, linespacing=1.25)
            x += 1
        b10 = int(((W["zero-shot"] == 1) & (W["ICL-20"] == 0)).sum())
        b01 = int(((W["zero-shot"] == 0) & (W["ICL-20"] == 1)).sum())
        p = min(1.0, 2 * sum(comb(b10 + b01, j) for j in range(min(b10, b01) + 1)) / 2 ** (b10 + b01))
        ax.plot([x - 2.3, x - 0.7], [0.115, 0.115], transform=tr, color=NEUTRAL["mid"], lw=0.9, clip_on=False)
        ax.text(x - 1.5, 0.1, f"{mname} (p = {p:.3f})", transform=tr, ha="center", va="top",
                fontsize=FS_TICK, color=INK)
        x += 0.5
    ax.set_xticks([0, 1, 2.5, 3.5])
    ax.set_xticklabels(["Zero-shot", "ICL", "Zero-shot", "ICL"], fontsize=FS_TICK)
    ax.set_xlim(-0.6, 4.1)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylim(0, 20)
    ax.set_yticks(range(0, 21, 4))
    ax.set_ylabel("Unsafe decision rate (%)", fontsize=FS_LABEL)
    out = os.path.join(HERE, "figures", "fig_icl")
    fig.savefig(out + ".png", dpi=600, facecolor="white")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".svg")
    print("written", out + ".{png,pdf,svg}")


if __name__ == "__main__":
    main()
