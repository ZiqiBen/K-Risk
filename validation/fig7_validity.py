"""Figure: complementarity of the screening criteria and validity of the risk levels.

a  Three-set Venn diagram (not area-proportional) of the 21.2 million vehicle-frames of
   the complete highD recordings: DRF in the recording's top 10%, TTC < 5 s, and a
   hard maneuver of the ego vehicle or one of its eight neighbors.
b  Composition of the DRF-top-10% frames by TTC state and of the TTC < 5 s frames by
   DRF state.
c  Recording-stratified Cliff's delta (95% bootstrap CI) between risk levels on indicators
   that the risk-level rules do not use; > 0 means the higher risk level is riskier.
Source data: results/highd_frame_level.json, results/validity_effects.csv
"""
import json
import os

import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from fig_style import (BLUE, CAR_GREY, FS, IND, IND_LABEL, INK, NEUTRAL, RED, TEAL, WARM,
                       axes_mm, new_figure, panel_letter, save)

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
J = json.load(open(os.path.join(R, "highd_frame_level.json")))
E = pd.read_csv(os.path.join(R, "validity_effects.csv"))


def num(x, fmt):
    return format(x, fmt).replace("-", "−")


fig = new_figure(88)
# left column: a (top), b (bottom); right column: c
axa = axes_mm(fig, 3.0, 1.0, 72.0, 42.0)
axb = axes_mm(fig, 20.5, 61.5, 72.0, 78.5)
axc = axes_mm(fig, 107.0, 12.0, 175.0, 78.5)
COL2 = 77.0
panel_letter(fig, "a", 0.3, 4.2)
panel_letter(fig, "b", 0.3, 50.5)
panel_letter(fig, "c", COL2, 4.2)

# ---------------------------------------------------------------- a: Venn
N = J["n_frames"]
v = J["venn"]
w_mm, h_mm = 69.0, 41.0
ylo, yhi = -1.98, 1.60
half = (yhi - ylo) * w_mm / h_mm / 2
axa.set_xlim(-half, half)
axa.set_ylim(ylo, yhi)
axa.set_aspect("equal", adjustable="box")
axa.axis("off")
C = {"A": ((-0.62, 0.30), BLUE), "B": ((0.62, 0.30), RED), "C": ((0.0, -0.68), TEAL)}
for (x, y), col in C.values():
    axa.add_patch(Circle((x, y), 1.0, facecolor=col, alpha=0.22, edgecolor="none", lw=0))


def pct(n):
    p = 100 * n / N
    return f"{p:.2g}%" if p >= 0.01 else f"{p:.1g}%"


lab = {"100": (-1.06, 0.56), "010": (1.06, 0.56), "001": (0.0, -1.26), "110": (0.0, 0.60),
       "101": (-0.68, -0.42), "011": (0.68, -0.42), "111": (0.0, -0.04)}
for k, (x, y) in lab.items():
    axa.text(x, y, pct(v["venn_" + k]), ha="center", va="center", fontsize=FS["text"],
             fontweight="bold" if k == "111" else "normal")
axa.text(-1.50, 1.05, "DRF top 10%", color=BLUE, ha="right", va="center", fontsize=FS["legend"])
axa.text(1.50, 1.05, "TTC < 5 s", color=RED, ha="left", va="center", fontsize=FS["legend"])
axa.text(0.0, -1.86, "Hard maneuver (ego vehicle or neighbor)", color=TEAL, ha="center",
         va="center", fontsize=FS["legend"])
axa.text(half, -1.30, f"Outside: {pct(v['venn_000'])}", ha="right", va="center",
         fontsize=FS["note"], color=NEUTRAL["mid"])
axa.text(0.0, yhi, f"highD, {N / 1e6:.1f} million vehicle-frames", ha="center", va="top",
         fontsize=FS["text"], color=INK)

# ---------------------------------------------------------------- b: composition
rows = [("DRF top 10%\nframes", [("TTC undefined", J["drf10_ttc_undef_frac"], CAR_GREY),
                                  ("TTC ≥ 5 s", J["drf10_ttc_gt5_frac"], WARM[2]),
                                  ("TTC < 5 s", 1 - J["drf10_ttc_undef_frac"] - J["drf10_ttc_gt5_frac"], RED)]),
        ("TTC < 5 s\nframes", [("DRF below top 10%", J["ttc5_not_drf10_frac"], CAR_GREY),
                               ("DRF top 10%", 1 - J["ttc5_not_drf10_frac"], BLUE)])]
for i, (name, parts) in enumerate(rows):
    left = 0
    for lbl, f, col in parts:
        axb.barh(i, f * 100, left=left, color=col, height=0.62, edgecolor="white", lw=0.5)
        if f > 0.2:
            axb.text(left + f * 50, i, f"{lbl}\n{100 * f:.1f}%", ha="center", va="center",
                     fontsize=FS["text"], linespacing=1.1,
                     color="white" if col in (RED, BLUE) else INK)
        left += f * 100
    small = [(lbl, f) for lbl, f, _ in parts if f <= 0.2]
    if small:
        axb.text(100, i - 0.36, "  ".join(f"{lbl}: {100 * f:.1f}%" for lbl, f in small), va="bottom",
                 ha="right", fontsize=FS["note"], color=NEUTRAL["dark"])
axb.set_yticks([0, 1])
axb.set_yticklabels([r[0] for r in rows])
axb.set_ylim(1.42, -0.62)
axb.set_xlim(0, 100)
axb.set_xlabel("Share of frames (%)")
axb.spines["left"].set_visible(False)
axb.tick_params(axis="y", length=0)
lo, hi = J["rho_drfpos_iqr"]
fig.text(20.5 / 175.9, 1 - 54.0 / 88, "Spearman ρ(DRF, 1/TTC) per recording:\nmedian "
         f"{num(J['rho_drfpos_median'], '.3f')} (IQR {num(lo, '.3f')} to {num(hi, '.3f')})",
         fontsize=FS["note"], color=NEUTRAL["dark"], ha="left", va="top", linespacing=1.2)

# ---------------------------------------------------------------- c: forest plot
groups = [("High vs\nmoderate", [("highd", "highD"), ("ind", "inD"), ("round", "rounD"), ("ngsim", "NGSIM"),
                                 ("expresswayA", "ExpresswayA"), ("freewayB", "FreewayB")], "high>moderate"),
          ("Extreme\nvs high", [("expresswayA", "ExpresswayA"), ("freewayB", "FreewayB")], "extreme>high"),
          ("Extreme vs\nmoderate", [("expresswayA", "ExpresswayA"), ("freewayB", "FreewayB")], "extreme>moderate")]
GAP = 0.55
inds = list(IND)
off = np.linspace(-0.28, 0.28, len(inds))
yt, yl, y = [], [], 0.0
spans = []
for g, (gname, srcs, contrast) in enumerate(groups):
    y0 = y
    for s, name in srcs:
        for k, ind in enumerate(inds):
            row = E[(E.src == s) & (E.contrast == contrast) & (E.indicator == ind)]
            if row.empty:
                continue
            row = row.iloc[0]
            yy = y + off[k]
            axc.plot([row.sci_lo, row.sci_hi], [yy, yy], color=IND[ind], lw=1.0, solid_capstyle="butt")
            sig = row.sci_lo > 0 or row.sci_hi < 0
            axc.plot(row.sdelta, yy, "o", ms=3.4, mfc=IND[ind] if sig else "white", mec=IND[ind], mew=0.8)
        yt.append(y)
        yl.append(name)
        y += 1
    spans.append((gname, y0, y - 1))
    if g < len(groups) - 1:
        axc.axhline(y - 0.5 + GAP / 2, color=CAR_GREY, lw=0.6)
    y += GAP
axc.axvline(0, color=NEUTRAL["dark"], lw=0.6, ls=(0, (2, 2)))
axc.set_yticks(yt)
axc.set_yticklabels(yl)
ymax = y - GAP - 1
axc.set_ylim(ymax + 0.5, -0.5)
axc.set_xlim(-0.33, 0.67)
axc.set_xlabel("Stratified Cliff's δ (> 0: higher risk level riskier)")
axc.tick_params(axis="y", length=0)
axc.spines["left"].set_visible(False)
# risk-level contrast of each block, drawn as in Figure 9a (bar + rotated label)
from matplotlib.transforms import blended_transform_factory
tr = blended_transform_factory(fig.transFigure, axc.transData)
xbar = (COL2 + 7.2) / 175.9
for gname, a0, a1 in spans:
    axc.plot([xbar, xbar], [a0 - 0.32, a1 + 0.32], transform=tr, color=NEUTRAL["mid"], lw=1.4,
             solid_capstyle="butt", clip_on=False)
    axc.text(xbar - 1.0 / 175.9, (a0 + a1) / 2, gname, transform=tr, rotation=90, ha="right",
             va="center", fontsize=FS["note"], fontweight="bold", color=NEUTRAL["dark"],
             linespacing=1.3, multialignment="center")
handles = [axc.plot([], [], "o", ms=3.4, color=IND[ind], label=IND_LABEL[ind])[0] for ind in inds]
axc.legend(handles=handles, loc="center left", bbox_to_anchor=(-0.30, 1 + (12.0 - 4.6) / (78.5 - 12.0)), ncol=2, fontsize=FS["legend"],
           handletextpad=0.15, columnspacing=1.2, borderaxespad=0.0, handlelength=1.0, labelspacing=0.35)

out = os.path.join(HERE, "figures", "fig_validity")
# free layout (axis-less Venn beside a forest plot spanning both rows): no comparable panel
# groups for the alignment gate; b and c share their bottom edge and the letters share
# their x (a, b) and baseline (a, c) by construction
save(fig, out, align=False)
print("saved", out)
