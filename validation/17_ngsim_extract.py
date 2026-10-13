"""Extract the NGSIM I-80 events from the complete original trajectories.

Rule (Table 3 of the manuscript): for every vehicle of each recording period, the
bumper-to-bumper TTC to its original Preceding vehicle is computed in every frame; a
vehicle yields one event if its minimum TTC is below 3 s, with a 6-s window (61
frames) centred on the frame of minimum TTC; the event is graded high if that minimum
is below 1.5 s and moderate otherwise. Roles and records are written with
15_ngsim_rebuild.py (released schema, native units ft and ft/s).

Output: analysis/ngsim_v3/{ngsim_high_risk,ngsim_normal_risk}/ngsim_car<ID>_frame_<f0>_to_<f1>.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("rb", os.path.join(os.path.dirname(__file__), "15_ngsim_rebuild.py"))
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

DST = os.path.join(os.path.dirname(__file__), "ngsim_v3")
HALF = 30


def candidates():
    d = rb.ORIG
    d = d.assign(period=np.select([d.global_time < 1113435000000, d.global_time < 1113437560000], [1, 2], 3))
    k = d[["period", "frame_id", "vehicle_id", "local_y", "v_length", "v_vel"]].rename(
        columns={"vehicle_id": "preceding", "local_y": "yp", "v_length": "lp", "v_vel": "vp"})
    m = d.merge(k, on=["period", "frame_id", "preceding"], how="left")
    gap = (m.yp - m.lp) - m.local_y
    dv = m.v_vel - m.vp
    m["ttc"] = np.where((m.preceding > 0) & (dv > 0) & (gap > 0), gap / dv, np.nan)
    m = m[m.ttc < 3]
    i = m.groupby(["period", "vehicle_id"]).ttc.idxmin()
    c = m.loc[i, ["period", "vehicle_id", "frame_id", "global_time", "ttc"]].reset_index(drop=True)
    return c


def build(row):
    period, vid, f, gt, t = row
    f0, f1 = f - HALF, f + HALF
    times = [gt + (fr - f) * 100 for fr in range(f0, f1 + 1)]
    gtv = rb.ORIG.global_time.to_numpy()
    i0, i1 = np.searchsorted(gtv, times[0], "left"), np.searchsorted(gtv, times[-1], "right")
    sub = rb.ORIG.iloc[i0:i1]
    snaps = {tt: g.set_index("vehicle_id") for tt, g in sub.groupby("global_time")}
    keep, ego_roles = {vid}, {}
    for tt in times:
        s = snaps.get(tt)
        if s is None or vid not in s.index:
            continue
        r = rb.roles(s, vid)
        ego_roles[tt] = r
        keep |= {int(v) for v in r.values() if v > 0}
    recs = []
    for tt in times:
        s = snaps.get(tt)
        if s is None:
            continue
        for v in sorted(keep):
            if v not in s.index:
                continue
            o = s.loc[v]
            r = ego_roles[tt] if v == vid and tt in ego_roles else rb.roles(s, v)
            recs.append({"Vehicle_ID": int(v), "Frame_ID": int(o.frame_id), "Lane_ID": int(o.lane_id),
                         "Local_X": float(o.local_x), "Local_Y": float(o.local_y), "v_Length": float(o.v_length),
                         "preceding_id": r["preceding_id"], "Following": r["Following"],
                         "leftPrecedingId": r["leftPrecedingId"], "leftAlongsideId": r["leftAlongsideId"],
                         "leftFollowingId": r["leftFollowingId"], "rightPrecedingId": r["rightPrecedingId"],
                         "rightAlongsideId": r["rightAlongsideId"], "rightFollowingId": r["rightFollowingId"],
                         "Total_Frames": int(o.total_frames), "Global_Time": int(tt), "Global_X": float(o.global_x),
                         "Global_Y": float(o.global_y), "v_Width": float(o.v_width), "v_Class": int(o.v_class),
                         "v_Vel": float(o.v_vel), "v_Acc": float(o.v_acc), "Space_Headway": float(o.space_headway),
                         "Time_Headway": float(o.time_headway), "TTC": rb.ttc(s, v)})
    level = "high" if t < 1.5 else "normal"
    name = f"ngsim_car{int(vid)}_frame_{int(f0)}_to_{int(f1)}.json"
    return dict(name=name, level=level, period=int(period), vehicle=int(vid), ttc_min=float(t), recs=recs)


def main():
    rb.TTC_MODE = "gap"
    rb.load()
    c = candidates()
    print("events", len(c), "high", int((c.ttc < 1.5).sum()))
    with Pool(16) as p:
        res = p.map(build, list(c.itertuples(index=False, name=None)), chunksize=16)
    names = pd.Series([r["name"] for r in res])
    dup = names[names.duplicated(keep=False)]
    print("name collisions across periods:", len(dup))
    log = []
    for r in res:
        name = r["name"]
        if name in set(dup):  # disambiguate the rare same-ID, same-window collisions by period
            name = name.replace("ngsim_car", f"ngsim_p{r['period']}_car")
        folder = os.path.join(DST, f"ngsim_{r['level']}_risk")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, name), "w") as fh:
            json.dump(r["recs"], fh, allow_nan=True)
        log.append(dict(file=name, level=r["level"], period=r["period"], vehicle=r["vehicle"], ttc_min=r["ttc_min"],
                        records=len(r["recs"]), ego_frames=sum(x["Vehicle_ID"] == r["vehicle"] for x in r["recs"])))
    L = pd.DataFrame(log)
    L.to_csv(os.path.join(DST, "extract_log.csv"), index=False)
    print(L.groupby("level").agg(n=("file", "size"), ego_frames=("ego_frames", "median"), records=("records", "median")))


if __name__ == "__main__":
    main()
