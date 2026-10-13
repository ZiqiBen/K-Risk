"""Redraw the 14 NGSIM events of the 100-event spot-check sample from the current NGSIM
release (7 moderate, 7 high; separate seed so the other strata are unchanged), render
them with the same layout as 09_spotcheck_render.py and update spotcheck_sample.csv."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(__file__), "09_spotcheck_render.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
from kr_common import FIG, OUT, event_index

idx = event_index(); sc.IDX = idx.set_index("event")
rng = np.random.default_rng(20261006)
picks = []
for l in ("moderate", "high"):
    g = idx[(idx.src == "ngsim") & (idx.level == l)]
    picks += list(g.event.to_numpy()[rng.choice(len(g), 7, replace=False)])
rows = []
for sh in range(0, 14, 4):
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.6), gridspec_kw=dict(height_ratios=[2.2, 1]))
    for j in range(4):
        if sh + j >= 14:
            axs[0, j].axis("off"); axs[1, j].axis("off"); continue
        ev = picks[sh + j]; r = sc.IDX.loc[ev]
        rows.append(sc.draw(axs[0, j], axs[1, j], r.src, r.level, ev, 53 + sh + j))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "spotcheck", f"sheet_ngsim_{sh // 4 + 1}.png"), dpi=130); plt.close(fig)
S = pd.read_csv(os.path.join(OUT, "spotcheck_sample.csv"))
S = pd.concat([S[S.src != "ngsim"], pd.DataFrame(rows)]).sort_values("n")
S.to_csv(os.path.join(OUT, "spotcheck_sample.csv"), index=False)
print(pd.DataFrame(rows)[["n", "event", "level", "trigger_agent"]].to_string())
