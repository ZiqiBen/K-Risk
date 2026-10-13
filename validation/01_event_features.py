"""Per-event feature table for all 28,828 released HV events.

Rule-side features (what the extraction protocol used): ego DRF, ego TTC, raw and
smoothed accelerations of the ego and of its slot agents, behaviour flags, and the
extreme-subset conflict fields.

Independent indicators (NOT used by the grading rules), computed from the released
trajectories:
  min_clear  minimum bounding-box clearance between the ego and any agent (m)
  overlap    number of frames in which the ego box overlaps another box
  pet        post-encroachment time: minimum time separation with which the ego and
             another agent occupy the same location (centres within 1 m), in s
  drac       maximum deceleration rate to avoid a crash w.r.t. the preceding agent
  jerk       maximum absolute longitudinal jerk of the ego (smoothed), m/s^3
Slot coverage (R1-2): agents present but not in any of the 8 slots, their distance,
and whether the closest agent / conflict partner is a slot agent.

Run: python 01_event_features.py  ->  results/event_features.csv
"""
import os
import sys
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, os.path.dirname(__file__))
from kr_common import FPS, OUT, SLOTS, ego_of, event_index, load_event, smooth_acc  # noqa: E402

PET_R = 1.0


def features(row):
    src, path, ev = row["src"], row["path"], row["event"]
    fps = FPS[src]
    try:
        a = load_event(src, path)
    except Exception as exc:  # noqa: BLE001
        return dict(event=ev, error=repr(exc))
    ego = ego_of(src, ev)
    e = a[a.id == ego].sort_values("frame").reset_index(drop=True)
    o = a[a.id != ego]
    f = dict(event=ev, n_frames=len(e), n_agents=a.id.nunique() - 1,
             dur_s=(e.frame.max() - e.frame.min()) / fps if len(e) else np.nan)
    if len(e) < 3:
        f["error"] = "ego missing"
        return f
    # ---------------- rule-side features
    f["drf_max"] = pd.to_numeric(e.drf, errors="coerce").max()
    ttc = pd.to_numeric(e.ttc, errors="coerce")
    f["ttc_min_pos"] = ttc[ttc > 0].min() if (ttc > 0).any() else np.nan
    f["ego_amax_raw"] = e.acc.abs().max()
    f["ego_amin_raw"] = e.acc.min()
    f["ego_apos_raw"] = e.acc.max()
    sa = smooth_acc(e.speed, fps)
    f["ego_amax_s"] = np.nanmax(np.abs(sa))
    f["ego_amin_s"] = np.nanmin(sa)
    jerk = np.gradient(sa, 1.0 / fps)
    f["jerk"] = np.nanmax(np.abs(jerk))
    for c in ("acc_high", "brake_high", "yaw_left", "yaw_right"):
        if c in e:
            f["ego_" + c] = int(e[c].sum())
    # slot agents over the window
    slot_ids = {}
    for r in SLOTS:
        ids = pd.to_numeric(e["slot_" + r], errors="coerce").dropna().astype(int)
        slot_ids[r] = set(ids)
    all_slot = set().union(*slot_ids.values())
    so = o[o.id.isin(all_slot)]
    f["slot_amax_raw"] = so.acc.abs().max() if len(so) else np.nan
    if len(so):
        sm = []
        for _, g in so.groupby("id"):
            g = g.sort_values("frame")
            if len(g) >= 3:
                sm.append(np.nanmax(np.abs(smooth_acc(g.speed, fps))))
        f["slot_amax_s"] = max(sm) if sm else np.nan
    for c in ("acc_high", "brake_high", "yaw_left", "yaw_right"):
        if c in so:
            f["slot_" + c] = int(so[c].sum())
    # flagged slot agents at the same frame as the slot relation (attribution)
    flagcols = [c for c in ("acc_high", "brake_high", "yaw_left", "yaw_right") if c in a]
    if flagcols:
        fl = a.set_index(["frame", "id"])[flagcols]
        trig = set()
        for r in SLOTS:
            s = e[["frame", "slot_" + r]].dropna()
            for fr, sid in zip(s.frame, s["slot_" + r].astype(int)):
                k = (fr, sid)
                if k in fl.index and fl.loc[k].to_numpy().any():
                    trig.add((r, sid))
        f["n_flagged_slot_agents"] = len({sid for _, sid in trig})
        f["flagged_roles"] = "|".join(sorted({r for r, _ in trig}))
    if "coll2s" in a:
        f["coll_rows_any"] = int(a.coll2s.sum())
        f["coll_agents_any"] = int(a[a.coll2s].id.nunique())
        f["ego_is_partner"] = int((a.coll_id == ego).any())
        f["coll_slot_agents"] = int(a[a.coll2s & a.id.isin(all_slot)].id.nunique())
        f["coll_t_min_any"] = a[a.coll2s].coll_t.min() if a.coll2s.any() else np.nan
    if "coll2s" in e:
        cf = e[e.coll2s]
        f["coll2s_frames"] = len(cf)
        f["coll_t_min"] = cf.coll_t.min() if len(cf) else np.nan
        cids = set(cf.coll_id.dropna().astype(int))
        f["n_coll_partners"] = len(cids)
        # is every conflict partner one of the 8 slot agents at that frame?
        ins = 0
        for _, rr in cf.dropna(subset=["coll_id"]).iterrows():
            ids_here = {int(rr["slot_" + r]) for r in SLOTS if pd.notna(rr["slot_" + r])}
            ins += int(int(rr.coll_id) in ids_here)
        f["coll_partner_in_slot_frac"] = ins / max(1, cf.coll_id.notna().sum())
        f["peak_frame"] = int(cf.frame.iloc[int(np.nanargmin(cf.coll_t.to_numpy()))]) if cf.coll_t.notna().any() else np.nan
    # ---------------- independent indicators
    egob = {fr: b for fr, b in zip(e.frame, e.box)}
    eg_poly = {fr: shapely.polygons(np.asarray(b)) for fr, b in egob.items()}
    oo = o[o.frame.isin(egob)]
    okb = np.array([np.isfinite(np.asarray(b, float)).all() for b in oo.box]) if len(oo) else np.array([], bool)
    oo = oo[okb]
    eg_poly = {fr: p for fr, p in eg_poly.items() if np.isfinite(np.asarray(egob[fr], float)).all()}
    oo = oo[oo.frame.isin(eg_poly)]
    if len(oo):
        ob = shapely.polygons(np.stack(oo.box.to_numpy()))
        ep = np.array([eg_poly[fr] for fr in oo.frame])
        dist = shapely.distance(ep, ob)
        oo = oo.assign(dist=dist)
        f["min_clear"] = float(np.nanmin(dist))
        f["overlap_frames"] = int((dist <= 0).sum())
        f["overlap_partners"] = int(oo[oo.dist <= 0].id.nunique())
        j = int(np.nanargmin(dist))
        cl = oo.iloc[j]
        er = e[e.frame == cl.frame].iloc[0]
        ids_here = {int(er["slot_" + r]) for r in SLOTS if pd.notna(er["slot_" + r])}
        f["closest_in_slot"] = int(int(cl.id) in ids_here)
        f["closest_cls"] = cl.cls
        # slot coverage per frame
        n_oth = oo.groupby("frame").id.nunique()
        ins, nd = [], []
        for fr, g in oo.groupby("frame"):
            er = e[e.frame == fr].iloc[0]
            ids_here = {int(er["slot_" + r]) for r in SLOTS if pd.notna(er["slot_" + r])}
            m = g.id.isin(ids_here)
            ins.append(int(m.sum()))
            nd += list(np.hypot(g.x[~m] - er.x, g.y[~m] - er.y))
        f["others_per_frame"] = float(n_oth.mean())
        f["slot_per_frame"] = float(np.mean(ins))
        f["nonslot_med_dist"] = float(np.median(nd)) if nd else np.nan
        f["nonslot_within20m_per_frame"] = float(np.sum(np.asarray(nd) < 20) / len(n_oth)) if nd else 0.0
        f["vru_per_frame"] = float(oo.cls.isin(["pedestrian", "bicycle", "motorcycle"]).sum() / len(n_oth))
        # PET (centres within PET_R)
        exy = e[["x", "y"]].to_numpy()
        et = e.t.to_numpy()
        pet = np.inf
        for _, g in oo.groupby("id"):
            gxy = g[["x", "y"]].to_numpy()
            if np.min(np.hypot(*(exy.mean(0) - gxy.mean(0)))) > 200:
                continue
            dmat = np.hypot(exy[:, None, 0] - gxy[None, :, 0], exy[:, None, 1] - gxy[None, :, 1])
            ii, jj = np.nonzero(dmat < PET_R)
            if len(ii):
                pet = min(pet, float(np.min(np.abs(et[ii] - g.t.to_numpy()[jj]))))
        f["pet"] = pet if np.isfinite(pet) else np.nan
        # DRAC w.r.t. the preceding agent
        dr = 0.0
        pre = e[["frame", "slot_preceding", "vx", "vy", "yaw"]].dropna(subset=["slot_preceding"])
        if len(pre):
            idx = oo.set_index(["frame", "id"])
            for fr, pid, vx, vy, yaw in pre.itertuples(index=False):
                k = (fr, int(pid))
                if k not in idx.index:
                    continue
                pr = idx.loc[k]
                if isinstance(pr, pd.DataFrame):
                    pr = pr.iloc[0]
                h = np.array([np.cos(yaw), np.sin(yaw)])
                vrel = float(np.dot([vx - pr.vx, vy - pr.vy], h))
                gap = float(pr.dist)
                if vrel > 0.5 and gap > 0.5 and pr.cls not in ("pedestrian", "bicycle"):
                    dr = max(dr, vrel * vrel / (2 * gap))
        f["drac"] = min(dr, 30.0)
    return f


def main():
    idx = event_index()
    print(idx.groupby(["src", "level"]).size(), len(idx))
    rows = idx.to_dict("records")
    with Pool(22) as p:
        res = []
        for i, r in enumerate(p.imap_unordered(features, rows, chunksize=8)):
            res.append(r)
            if i % 2000 == 0:
                print(i, flush=True)
    F = pd.DataFrame(res)
    out = idx.merge(F, on="event", how="left")
    out.to_csv(os.path.join(OUT, "event_features.csv"), index=False)
    print(out.groupby(["src", "level"]).size())
    print("errors", out.error.notna().sum() if "error" in out else 0)


if __name__ == "__main__":
    main()
