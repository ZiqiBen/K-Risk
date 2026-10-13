"""Figure: closed-loop validation of the risk levels with the collision-free simulator,
and the LLM-agent trials on the extreme subset.

a  Distribution of the number of unsafe actions of the five-action schema per event
   (CitySim events, n = 9,820) at the peak-risk frame.
b  Unsafe rate of each action by risk level (Wilson 95% CI).
c  Mean number of unsafe actions by risk level under nine simulator settings.
d  Collision rate of the LLM agent over three reflection trials on 262 extreme events,
   with the recorded human-driver reference (Wilson 95% CI).
   With --icl: unsafe-decision rate of GPT-4o / GPT-4.1, zero-shot vs 20 in-context failure
   cases, on 352 held-out events (results/icl_eval_n20.csv; exact McNemar p).
Source data: results/checker_*.csv, results/sensitivity_checker.csv
"""
import math
import os
import sys
from math import comb

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fig_style import (AV_GREY, CAR_GREY, FS, INK, NEUTRAL, ORIG, RISK, RISK_DARK, WARM,
                       axes_mm, new_figure, panel_letter, save)

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
B = pd.read_csv(os.path.join(R, "checker_base.csv"))
B = B[B.n_unsafe.notna()]
K = pd.read_csv(os.path.join(R, "sensitivity_checker.csv"))
LV = ["moderate", "high", "extreme"]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


H_MM = 95.0
fig = new_figure(H_MM)
X1, X2 = 0.3, 84.5                      # panel-letter columns
ROW = [(7.5, 36.0), (55.0, 81.5)]       # axes top/bottom of the two rows (mm from the top)
LET = [4.2, 51.7]                       # panel-letter baselines
axa = axes_mm(fig, 18.5, ROW[0][0], 77.0, ROW[0][1])
axb = axes_mm(fig, 96.0, ROW[0][0], 175.5, ROW[0][1])
axc = axes_mm(fig, 18.5, ROW[1][0], 77.0, ROW[1][1])
axd = axes_mm(fig, 96.0, ROW[1][0], 175.5, ROW[1][1])
for s, x, y in [("a", X1, LET[0]), ("b", X2, LET[0]), ("c", X1, LET[1]), ("d", X2, LET[1])]:
    panel_letter(fig, s, x, y)
LEG_DY = 4.0                            # legend centre, mm above the axes top


def inline_legend(ax, title, handles, labels, top_mm, bottom_mm, cs=0.9, **kw):
    """Legend in one row above the axes, its title on the left of the entries."""
    yc = 1 + LEG_DY / (bottom_mm - top_mm)
    t = ax.text(0.0, yc, title, transform=ax.transAxes, ha="left", va="center",
                fontsize=FS["legend"], color=INK)
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transAxes.inverted())
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(bb.x1 + 0.035, yc), ncol=len(handles),
              fontsize=FS["legend"], borderaxespad=0.0, borderpad=0.0, handletextpad=0.35,
              columnspacing=cs, **kw)


# ---------------------------------------------------------------- a
cols = ["#E4E4E4", WARM[1], "#E7B5A8", "#D58E82", "#BD645E", WARM[4]]   # 0-5 unsafe actions
for i, l in enumerate(LV):
    g = B[B.level == l]
    left = 0
    for k in range(6):
        f = (g.n_unsafe == k).mean() * 100
        axa.barh(i, f, left=left, color=cols[k], height=0.66, edgecolor="white", lw=0.5)
        if f >= 7:
            axa.text(left + f / 2, i, f"{f:.0f}", ha="center", va="center", fontsize=FS["text"],
                     color="white" if k >= 3 else INK)
        left += f
    axa.text(102, i, f"n = {len(g):,}\nmean {g.n_unsafe.mean():.2f}", va="center", ha="left",
             fontsize=FS["note"], color=NEUTRAL["dark"], linespacing=1.15)
axa.set_yticks(range(3))
axa.set_yticklabels([l.capitalize() for l in LV])
axa.set_ylabel("Risk level")
axa.set_ylim(2.5, -0.5)
axa.set_xlim(0, 119)
axa.spines["bottom"].set_bounds(0, 100)
axa.set_xticks(range(0, 101, 20))
axa.set_xlabel("Share of events (%)", x=100 / 119 / 2)
axa.spines["left"].set_visible(False)
axa.tick_params(axis="y", length=0)
inline_legend(axa, "Unsafe actions:",
              [Patch(facecolor=c, edgecolor="none") for c in cols], [str(k) for k in range(6)],
              *ROW[0], handlelength=0.9, handleheight=0.9, cs=0.7)

# ---------------------------------------------------------------- b
acts = ["IDLE", "LEFT", "RIGHT", "ACC", "DEC"]
names = ["IDLE", "Turn Left", "Turn Right", "Acceleration", "Deceleration"]
off = {"moderate": -0.27, "high": 0.0, "extreme": 0.27}
for l in LV:
    g = B[B.level == l]
    for j, a in enumerate(acts):
        k, n = int(g[f"unsafe_{a}"].sum()), int(g[f"unsafe_{a}"].notna().sum())
        lo, hi = wilson(k, n)
        x = j + off[l]
        axb.bar(x, 100 * k / n, width=0.26, color=RISK[l], edgecolor="none")
        axb.plot([x, x], [100 * lo, 100 * hi], color=INK, lw=0.7, solid_capstyle="butt")
axb.set_xticks(range(5))
axb.set_xticklabels(names)
axb.set_xlim(-0.5, 4.5)
axb.tick_params(axis="x", length=0)
axb.set_ylabel("Unsafe within 2 s (%)")
axb.set_ylim(0, 100)
inline_legend(axb, "Risk level:", [Patch(facecolor=RISK[l], edgecolor="none") for l in LV],
              [l.capitalize() for l in LV], *ROW[0], handlelength=0.9, handleheight=0.9)

# ---------------------------------------------------------------- c
order = ["base", "h15", "h30", "d3", "d6", "v075", "v15", "a1", "a3"]
lab = ["Base", "H\n1.5 s", "H\n3 s", "Dec\n3", "Dec\n6", "Lat\n0.75", "Lat\n1.5", "Acc\n1", "Acc\n3"]
K = K.set_index("tag").loc[order]
x = np.arange(len(order))
axc.axvspan(-0.4, 0.4, color=CAR_GREY, alpha=0.45, lw=0)
for l in LV:
    axc.plot(x, K[f"{l}_n_unsafe"], "-o", ms=3.6, lw=1.2, color=RISK_DARK[l], mfc=RISK[l],
             mec=RISK_DARK[l], mew=0.7)
axc.set_xticks(x)
axc.set_xticklabels(lab, linespacing=1.25)
axc.set_xlim(-0.5, len(order) - 0.5)
axc.set_xlabel("Collision-free simulator setting")
axc.set_ylabel("Mean unsafe actions\nper event")
axc.set_ylim(1.0, 4.0)
inline_legend(axc, "Risk level:",
              [Line2D([], [], color=RISK_DARK[l], lw=1.2, marker="o", ms=3.6, mfc=RISK[l],
                      mec=RISK_DARK[l], mew=0.7) for l in LV],
              [l.capitalize() for l in LV], *ROW[1], handlelength=1.6)

# ---------------------------------------------------------------- d
from matplotlib.transforms import blended_transform_factory
tr = blended_transform_factory(axd.transData, fig.transFigure)
yb = 1 - (ROW[1][1] + 5.6) / H_MM
ICL = "--icl" in sys.argv
if ICL:
    E = pd.read_csv(os.path.join(R, "icl_eval_n20.csv"))
    x = 0
    for m, mname in [("gpt4o", "GPT-4o"), ("gpt41", "GPT-4.1")]:
        W = E[E.model == m].pivot(index="event", columns="cond", values="unsafe").dropna()
        n = len(W)
        for c, col in [("zero-shot", AV_GREY), ("ICL-20", ORIG)]:
            k = int(W[c].sum())
            lo, hi = wilson(k, n)
            axd.bar(x, 100 * k / n, color=col, width=0.58, edgecolor="none")
            axd.plot([x, x], [100 * lo, 100 * hi], color=INK, lw=0.7, solid_capstyle="butt")
            axd.text(x, 100 * hi + 0.3, f"{100 * k / n:.2f}%\n({k}/{n})", ha="center", va="bottom",
                     fontsize=FS["text"], linespacing=1.3)
            x += 1
        b10 = int(((W["zero-shot"] == 1) & (W["ICL-20"] == 0)).sum())
        b01 = int(((W["zero-shot"] == 0) & (W["ICL-20"] == 1)).sum())
        p = min(1.0, 2 * sum(comb(b10 + b01, j) for j in range(min(b10, b01) + 1)) / 2 ** (b10 + b01))
        axd.plot([x - 2.25, x - 0.75], [yb, yb], transform=tr, color=NEUTRAL["mid"], lw=0.7, clip_on=False)
        axd.text(x - 1.5, yb - 0.6 / H_MM, f"{mname} (p = {p:.3f})", transform=tr, ha="center", va="top",
                 fontsize=FS["tick"], color=INK)
        x += 0.4
    axd.set_xticks([0, 1, 2.4, 3.4])
    axd.set_xticklabels(["Zero-shot", "ICL", "Zero-shot", "ICL"])
    axd.set_xlim(-0.55, 3.95)
    axd.tick_params(axis="x", length=0)
    axd.set_ylabel("Unsafe decision rate (%)")
    axd.set_ylim(0, 20)
    axd.set_yticks(range(0, 21, 4))
    out = os.path.join(HERE, "figures", "fig_closedloop_icl")
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(f"{out}.{ext}", **kw)
    print("saved", out)
    sys.exit()

trials = [("Recorded\nhuman driver", 14), ("Trial 1", 12), ("Trial 2", 7), ("Trial 3", 5)]
n = 262
for i, (t, k) in enumerate(trials):
    lo, hi = wilson(k, n)
    axd.bar(i, 100 * k / n, color=AV_GREY if i == 0 else ORIG, width=0.58, edgecolor="none")
    axd.plot([i, i], [100 * lo, 100 * hi], color=INK, lw=0.7, solid_capstyle="butt")
    axd.text(i, 100 * hi + 0.3, f"{100 * k / n:.2f}%\n({k}/{n})", ha="center", va="bottom",
             fontsize=FS["text"], linespacing=1.3)
axd.set_xticks(range(4))
axd.set_xticklabels([t for t, _ in trials], linespacing=1.25)
axd.set_xlim(-0.55, 3.55)
axd.tick_params(axis="x", length=0)
axd.set_ylabel("Collision rate (%)")
axd.set_ylim(0, 12.5)
axd.set_yticks(range(0, 13, 2))
# trials 1-3 belong to the LLM agent
axd.plot([0.75, 3.25], [yb, yb], transform=tr, color=NEUTRAL["mid"], lw=0.7, clip_on=False)
axd.text(2.0, yb - 0.6 / H_MM, "LLM agent", transform=tr, ha="center", va="top",
         fontsize=FS["tick"], color=INK)

out = os.path.join(HERE, "figures", "fig_closedloop")
save(fig, out, align=True, panel_ids={axa: "a", axb: "b", axc: "c", axd: "d"},
     row_groups=[["a", "b"], ["c", "d"]], column_groups=[["a", "c"], ["b", "d"]])
print("saved", out)
