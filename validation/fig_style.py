"""Shared settings for the validation figures (fig_validity, fig_closedloop, fig_icl; matplotlib).

Palette, text sizes and panel letters follow the dataset overview figure (fig_statistic,
drawn by ../plot_overview.py) as it appears in the manuscript.
The figures are drawn at the text width of the manuscript (\\linewidth of
wlscirep: letter paper with 2-cm margins), so all sizes below are printed sizes.
"""
import os
import sys

import matplotlib as mpl

# Optional panel-alignment audit (audit_panel_alignment.py, not distributed); set
# KR_FIG_AUDIT to its folder to enable it. Figures are written without it otherwise.
if os.environ.get("KR_FIG_AUDIT"):
    sys.path.insert(0, os.environ["KR_FIG_AUDIT"])

MM = 1 / 25.4
TEXTWIDTH_MM = 175.9

INK = "#2B2B2B"
# printed text sizes (pt), matched to the main panels (a, b) of Figure 7 (fig_statistic):
# panel letters 10 pt bold, axis titles about 8.5 pt, tick and category labels about 7-7.5 pt
# (enlarged by 1 pt on 2026-10-09; previous values in figures/backup_1008/fig_style.py)
FS = {"panel": 10, "label": 8.5, "tick": 7.5, "legend": 7.5, "text": 7.0, "note": 6.5}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": FS["text"],
    "axes.titlesize": FS["label"],
    "axes.labelsize": FS["label"],
    "xtick.labelsize": FS["tick"],
    "ytick.labelsize": FS["tick"],
    "legend.fontsize": FS["legend"],
    "legend.title_fontsize": FS["legend"],
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "axes.labelpad": 2.5,
    "legend.frameon": False,
    "lines.linewidth": 1.0,
})

# ---- palette of Figure 9 -------------------------------------------------------
RISK = {"moderate": "#C6D3E3", "high": "#E7B98F", "extreme": "#B5524E"}
# darker members of the same palette, for thin marks (lines, marker edges)
RISK_DARK = {"moderate": "#86A8C4", "high": "#C0763F", "extreme": "#9E3A37"}
BLUE = "#4C72A8"      # high-speed highway
TEAL = "#4E9D9B"      # urban freeway
PURPLE = "#9A6FB0"    # intersection
GREEN = "#76A86B"     # roundabout
RED = "#B5524E"       # extreme / K-Risk
ORIG = "#5B7FB4"      # reference series
AV_GREY = "#A9ACB5"
CAR_GREY = "#D2D2D2"
HEAVY = "#8A8D99"
WARM = ["#FFFFFF", "#F4DAD2", "#E2A89A", "#C8736A", "#9E3A37"]   # density ramp of Figure 9b
NEUTRAL = {"light": CAR_GREY, "mid": HEAVY, "dark": "#5A5A5A", "black": INK}

# independent indicators (keys as in results/validity_effects.csv) and their labels
IND = {"Minimum clearance": BLUE, "PET": TEAL, "DRAC": PURPLE, "Predicted 2-s conflict": RED}
IND_LABEL = {"Minimum clearance": "Minimum clearance", "PET": "PET", "DRAC": "DRAC",
             "Predicted 2-s conflict": "Two-second trajectory conflict"}


def new_figure(height_mm, width_mm=TEXTWIDTH_MM):
    import matplotlib.pyplot as plt
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


def axes_mm(fig, x0, y0, x1, y1, **kw):
    """Axes spanning (x0, y0)-(x1, y1) in mm, measured from the top-left corner of the figure."""
    w, h = fig.get_size_inches() / MM
    return fig.add_axes([x0 / w, 1 - y1 / h, (x1 - x0) / w, (y1 - y0) / h], **kw)


def panel_letter(fig, s, x, y):
    """Bold panel letter with its left edge at x mm and its baseline y mm below the top edge."""
    w, h = fig.get_size_inches() / MM
    fig.text(x / w, 1 - y / h, s, fontsize=FS["panel"], fontweight="bold", ha="left",
             va="baseline", color=INK)


def save(fig, path, align=True, **kw):
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        align = False
    fig.canvas.draw()
    if align:
        require_matplotlib_panel_alignment(fig, json_out=path + ".alignment.json",
                                           overlay_svg=path + ".alignment.svg",
                                           tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, **kw)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".svg")
    fig.savefig(path + ".png", dpi=300)
