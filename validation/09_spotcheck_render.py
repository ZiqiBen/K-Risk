"""Render a stratified random sample of 100 released HV events for visual inspection
(R2-5 extraction spot-check). Each panel pair shows the bird's-eye trajectories of the
ego (blue), the agent carrying the trigger (orange) and the other agents (grey), and the
time series of the triggering quantity with its threshold.

Output: figures/spotcheck/sheet_XX.png and results/spotcheck_sample.csv
"""
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from kr_common import FIG, FPS, OUT, SLOTS, ego_of, event_index, load_event, smooth_acc  # noqa: E402

PLAN = {("highd", "moderate"): 10, ("highd", "high"): 10, ("ind", "moderate"): 8, ("ind", "high"): 8,
        ("round", "moderate"): 8, ("round", "high"): 8, ("ngsim", "moderate"): 7, ("ngsim", "high"): 7,
        ("expresswayA", "moderate"): 6, ("expresswayA", "high"): 6, ("expresswayA", "extreme"): 5,
        ("freewayB", "moderate"): 6, ("freewayB", "high"): 6, ("freewayB", "extreme"): 5}
REL = {"precedingId": "preceding", "followingId": "following", "leftPrecedingId": "left_preceding",
       "leftAlongsideId": "left_alongside", "leftFollowingId": "left_following",
       "rightPrecedingId": "right_preceding", "rightAlongsideId": "right_alongside",
       "rightFollowingId": "right_following"}


def trigger(src, level, ev, a, ego):
    """Return (agent id, time series label, series(t, y), threshold, frame) of the trigger."""
    fps = FPS[src]
    e = a[a.id == ego].sort_values("frame")
    if src == "highd":
        m = re.match(r"highd_\d+_(\d+)_(\w+?Id)_(\w+?)_frame", ev)
        rel, beh = REL[m.group(2)], m.group(3)
        fl = a.set_index(["frame", "id"])[beh]
        tid, tfr = None, None
        for fr, sid in zip(e.frame, e["slot_" + rel]):
            if pd.notna(sid) and bool(fl.get((fr, int(sid)), False)):
                tid, tfr = int(sid), fr
                break
        g = a[a.id == tid].sort_values("frame") if tid is not None else e
        if beh in ("yaw_left", "yaw_right"):
            yv = np.gradient(g.y.to_numpy(), 1 / fps)
            return tid, f"{rel} {beh}: lateral speed (m/s)", (g.t.to_numpy(), yv), None, tfr
        return tid, f"{rel} {beh}: long. acc. (m/s$^2$)", (g.t.to_numpy(), g.acc.to_numpy()), None, tfr
    if src == "ngsim":
        t = pd.to_numeric(e.ttc, errors="coerce").where(lambda s: s > 0)
        k = int(np.nanargmin(t.to_numpy())) if t.notna().any() else 0
        return int(e.slot_preceding.iloc[k]) if pd.notna(e.slot_preceding.iloc[k]) else None, "ego TTC (s)", \
            (e.t.to_numpy(), t.clip(upper=10).to_numpy()), (1.5 if level == "high" else 3.0), e.frame.iloc[k]
    if src in ("ind", "round"):
        k = int(np.nanargmax(e.drf.to_numpy())) if e.drf.notna().any() else 0
        return None, "ego DRF", (e.t.to_numpy(), e.drf.to_numpy()), None, e.frame.iloc[k]
    # CitySim
    if level == "extreme":
        if "coll2s" in a and a.coll2s.any():
            c = a[a.coll2s]
            k = int(np.nanargmin(c.coll_t.fillna(9).to_numpy()))
            r = c.iloc[k]
            pid = int(r.coll_id) if pd.notna(r.coll_id) else None
            if int(r.id) != ego and pid != ego:
                pid = int(r.id)
            return pid, "recorded time to overlap (s)", (c.t.to_numpy(), c.coll_t.to_numpy()), 2.0, r.frame
        return None, "no conflict record (ego speed m/s)", (e.t.to_numpy(), e.speed.to_numpy()), None, e.frame.iloc[len(e) // 2]
    if level == "high":
        k = int(np.nanargmax(e.acc.abs().fillna(0).to_numpy()))
        return ego, "ego acc. raw (thin) / smoothed (thick)", (e.t.to_numpy(), e.acc.to_numpy()), 5.0, e.frame.iloc[k]
    slots = set()
    for r_ in SLOTS:
        slots |= set(pd.to_numeric(e["slot_" + r_], errors="coerce").dropna().astype(int))
    o = a[a.id.isin(slots)]
    k = int(np.nanargmax(o.acc.abs().fillna(0).to_numpy()))
    tid = int(o.iloc[k].id)
    g = a[a.id == tid].sort_values("frame")
    return tid, "neighbour acc. raw (thin) / smoothed (thick)", (g.t.to_numpy(), g.acc.to_numpy()), 5.0, o.iloc[k].frame


def draw(ax_b, ax_t, src, level, ev, n):
    path = IDX.loc[ev, "path"]
    a = load_event(src, path)
    ego = ego_of(src, ev)
    e = a[a.id == ego].sort_values("frame")
    tid, lab, (tt, yy), thr, tfr = trigger(src, level, ev, a, ego)
    for oid, g in a[a.id != ego].groupby("id"):
        g = g.sort_values("frame")
        col, lw, z = ("#E28E2C", 1.4, 3) if oid == tid else ("#BBBBBB", 0.6, 1)
        ax_b.plot(g.x, g.y, color=col, lw=lw, zorder=z)
        ax_b.plot(g.x.iloc[-1], g.y.iloc[-1], marker=">", ms=2.5, color=col, zorder=z)
    ax_b.plot(e.x, e.y, color="#0F4D92", lw=1.6, zorder=4)
    ax_b.plot(e.x.iloc[0], e.y.iloc[0], "o", ms=3, color="#0F4D92", zorder=5)
    ax_b.plot(e.x.iloc[-1], e.y.iloc[-1], ">", ms=3.5, color="#0F4D92", zorder=5)
    at = a[a.frame == tfr]
    for _, r in at.iterrows():
        b = np.asarray(r.box)
        c = "#0F4D92" if r.id == ego else ("#E28E2C" if r.id == tid else "#999999")
        ax_b.fill(b[:, 0], b[:, 1], facecolor="none", edgecolor=c, lw=0.8, zorder=6)
    xs = np.r_[e.x, a[a.id == tid].x] if tid is not None else e.x
    ys = np.r_[e.y, a[a.id == tid].y] if tid is not None else e.y
    cx, cy = np.nanmean(xs), np.nanmean(ys)
    span = max(np.nanmax(xs) - np.nanmin(xs), np.nanmax(ys) - np.nanmin(ys), 30) * 0.65
    ax_b.set_xlim(cx - span, cx + span)
    ax_b.set_ylim(cy - span, cy + span)
    ax_b.set_aspect("equal")
    ax_b.tick_params(labelsize=5)
    ax_b.set_title(f"#{n:03d} {src} | {level}\n{ev[:58]}", fontsize=5.5)
    ax_t.plot(tt, yy, color="#E28E2C" if tid != ego else "#0F4D92", lw=0.6)
    if "acc" in lab and src in ("expresswayA", "freewayB"):
        g = a[a.id == tid].sort_values("frame")
        ax_t.plot(g.t, smooth_acc(g.speed, FPS[src]), color="k", lw=1.2)
        ax_t.axhline(-thr, color="r", lw=0.5, ls="--")
        ax_t.axhline(-3, color="g", lw=0.5, ls=":")
        ax_t.axhline(3, color="g", lw=0.5, ls=":")
    if thr is not None:
        ax_t.axhline(thr, color="r", lw=0.5, ls="--")
    tf = (tfr - a.frame.min()) / FPS[src]
    ax_t.axvline(tf, color="k", lw=0.5, ls=":")
    ax_t.set_title(lab, fontsize=5.5)
    ax_t.tick_params(labelsize=5)
    ax_t.set_xlabel("t (s)", fontsize=5)
    return dict(n=n, event=ev, src=src, level=level, trigger_agent=tid, trigger_frame=int(tfr))


def main():
    global IDX
    idx = event_index()
    IDX = idx.set_index("event")
    rng = np.random.default_rng(20261005)
    picks = []
    for (s, l), k in PLAN.items():
        g = idx[(idx.src == s) & (idx.level == l)]
        picks += list(g.event.to_numpy()[rng.choice(len(g), k, replace=False)])
    os.makedirs(os.path.join(FIG, "spotcheck"), exist_ok=True)
    rows = []
    per = 4
    for sh in range(0, len(picks), per):
        fig, axs = plt.subplots(2, 4, figsize=(11, 5.6), gridspec_kw=dict(height_ratios=[2.2, 1]))
        for j in range(per):
            if sh + j >= len(picks):
                axs[0, j].axis("off")
                axs[1, j].axis("off")
                continue
            ev = picks[sh + j]
            r = IDX.loc[ev]
            try:
                rows.append(draw(axs[0, j], axs[1, j], r.src, r.level, ev, sh + j + 1))
            except Exception as exc:  # noqa: BLE001
                axs[0, j].set_title(f"#{sh + j + 1} {ev[:40]}\nERROR {exc!r}"[:120], fontsize=5)
                rows.append(dict(n=sh + j + 1, event=ev, src=r.src, level=r.level, error=repr(exc)))
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "spotcheck", f"sheet_{sh // per + 1:02d}.png"), dpi=130)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "spotcheck_sample.csv"), index=False)
    print(len(rows))


if __name__ == "__main__":
    main()
