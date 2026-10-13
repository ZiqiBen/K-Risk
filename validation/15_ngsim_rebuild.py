"""Rebuild the surrounding-agent records of the released NGSIM I-80 events from the
complete original trajectories (data.transportation.gov, DOI 10.21949/1504477).

For every released NGSIM event file the ego vehicle and the frame window are kept.
At each frame the eight roles are recomputed from the complete snapshot of the same
recording period: preceding and following from the original Preceding/Following
fields; left/right roles from the adjacent lanes (lane - 1 / lane + 1, lane 7 being
the on-ramp to the right of lane 6) by longitudinal overlap of the vehicle extents
[Local_Y - v_Length, Local_Y] (alongside = overlap, preceding/following = nearest
non-overlapping vehicle ahead/behind). Every vehicle that occupies a role of the ego
in the window is written for all frames of the window, with its own roles and TTC, in
the released schema and native units (ft, ft/s).

Output: analysis/ngsim_v2/<folder>/<same file name>.json
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from kr_common import HV  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngsim_source")
DST = os.path.join(os.path.dirname(__file__), "ngsim_v2")
TTC_MODE = os.environ.get("NGSIM_TTC_MODE", "gap")  # set from 14_ngsim_diagnose.py
ORIG = None


def load():
    global ORIG
    d = pd.read_pickle(os.path.join(SRC, "i80.parquet.pkl"))
    d = d.sort_values(["global_time", "lane_id", "local_y"]).reset_index(drop=True)
    ORIG = d
    return d


def left_of(lane):
    return lane - 1 if lane >= 2 else None


def right_of(lane):
    return lane + 1 if lane <= 6 else None


def roles(snap, vid):
    """Eight role identifiers (-1 if none) of vehicle vid in one frame snapshot."""
    me = snap.loc[vid]
    out = {"preceding_id": float(me.preceding) if me.preceding > 0 else -1.0,
           "Following": float(me.following) if me.following > 0 else -1.0}
    ye, le = me.local_y, me.v_length
    for side, lane in (("left", left_of(int(me.lane_id))), ("right", right_of(int(me.lane_id)))):
        pre, alo, fol = -1.0, -1.0, -1.0
        if lane is not None:
            nb = snap[(snap.lane_id == lane)]
            nb = nb[nb.index != vid]
            if len(nb):
                front, rear = nb.local_y.to_numpy(), (nb.local_y - nb.v_length).to_numpy()
                ids = nb.index.to_numpy()
                ov = (rear < ye) & (front > ye - le)
                if ov.any():
                    c = np.abs((front + rear) / 2 - (ye - le / 2))
                    alo = float(ids[ov][np.argmin(c[ov])])
                ah = rear >= ye
                if ah.any():
                    pre = float(ids[ah][np.argmin(rear[ah])])
                be = front <= ye - le
                if be.any():
                    fol = float(ids[be][np.argmax(front[be])])
        out[f"{side}PrecedingId"], out[f"{side}AlongsideId"], out[f"{side}FollowingId"] = pre, alo, fol
    return out


def ttc(snap, vid):
    me = snap.loc[vid]
    p = me.preceding
    if p <= 0 or p not in snap.index:
        return np.nan
    q = snap.loc[p]
    dv = me.v_vel - q.v_vel
    if dv <= 0:
        return np.nan
    if TTC_MODE == "sh":
        return float(me.space_headway / dv)
    gap = (q.local_y - q.v_length) - me.local_y
    return float(gap / dv) if gap > 0 else np.nan


def rebuild(path):
    name = os.path.basename(path)
    folder = os.path.basename(os.path.dirname(path))
    with open(path) as f:
        old = json.load(f)
    ego = int(name.split("car")[1].split("_")[0])
    f0, f1 = map(int, name[:-5].split("frame_")[1].split("_to_"))
    gt = [r["Global_Time"] for r in old if int(r["Vehicle_ID"]) == ego]
    g0 = min(gt) - (min(r["Frame_ID"] for r in old if int(r["Vehicle_ID"]) == ego) - f0) * 100
    times = [g0 + (f - f0) * 100 for f in range(f0, f1 + 1)]
    gtv = ORIG.global_time.to_numpy()
    i0, i1 = np.searchsorted(gtv, times[0], "left"), np.searchsorted(gtv, times[-1], "right")
    sub = ORIG.iloc[i0:i1]
    snaps = {t: g.set_index("vehicle_id") for t, g in sub.groupby("global_time")}
    keep = {ego}
    ego_roles = {}
    for t in times:
        s = snaps.get(t)
        if s is None or ego not in s.index:
            continue
        r = roles(s, ego)
        ego_roles[t] = r
        keep |= {int(v) for v in r.values() if v > 0}
    recs = []
    for t in times:
        s = snaps.get(t)
        if s is None:
            continue
        for vid in sorted(keep):
            if vid not in s.index:
                continue
            o = s.loc[vid]
            r = ego_roles[t] if vid == ego and t in ego_roles else roles(s, vid)
            rec = {"Vehicle_ID": int(vid), "Frame_ID": int(o.frame_id), "Lane_ID": int(o.lane_id),
                   "Local_X": float(o.local_x), "Local_Y": float(o.local_y), "v_Length": float(o.v_length),
                   "preceding_id": r["preceding_id"], "Following": r["Following"],
                   "leftPrecedingId": r["leftPrecedingId"], "leftAlongsideId": r["leftAlongsideId"],
                   "leftFollowingId": r["leftFollowingId"], "rightPrecedingId": r["rightPrecedingId"],
                   "rightAlongsideId": r["rightAlongsideId"], "rightFollowingId": r["rightFollowingId"],
                   "Total_Frames": int(o.total_frames), "Global_Time": int(t), "Global_X": float(o.global_x),
                   "Global_Y": float(o.global_y), "v_Width": float(o.v_width), "v_Class": int(o.v_class),
                   "v_Vel": float(o.v_vel), "v_Acc": float(o.v_acc), "Space_Headway": float(o.space_headway),
                   "Time_Headway": float(o.time_headway), "TTC": ttc(s, vid)}
            recs.append(rec)
    os.makedirs(os.path.join(DST, folder), exist_ok=True)
    with open(os.path.join(DST, folder, name), "w") as f:
        json.dump(recs, f, allow_nan=True)
    e = [r for r in recs if r["Vehicle_ID"] == ego]
    t_old = [r["TTC"] for r in old if int(r["Vehicle_ID"]) == ego and r["TTC"] is not None and r["TTC"] > 0]
    t_new = [r["TTC"] for r in e if r["TTC"] == r["TTC"] and r["TTC"] > 0]
    return dict(file=name, folder=folder, ego_frames=len(e), window=f1 - f0 + 1, vehicles=len(keep),
                ttc_min_old=min(t_old) if t_old else np.nan, ttc_min_new=min(t_new) if t_new else np.nan)


def main():
    load()
    files = sorted(glob.glob(os.path.join(HV, "ngsim", "*", "*.json")))
    if len(sys.argv) > 1:
        files = files[: int(sys.argv[1])]
    with Pool(16) as p:  # workers inherit ORIG by fork
        res = p.map(rebuild, files, chunksize=8)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(DST, "rebuild_log.csv"), index=False)
    print(R.describe().round(3).to_string())


if __name__ == "__main__":
    main()
