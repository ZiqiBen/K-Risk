"""Diagnose the released NGSIM event records against the original I-80 trajectories.

For a sample of events, compares the ego rows of the released JSON with the original
rows of the same vehicle at the same Global_Time, the released preceding_id with the
original Preceding, and the released TTC with TTC recomputed from the original
preceding vehicle under several gap definitions.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from kr_common import HV  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngsim_source")


def load_original():
    p = os.path.join(SRC, "i80.parquet.pkl")
    if os.path.exists(p):
        return pd.read_pickle(p)
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(SRC, "chunk_*.csv")))], ignore_index=True)
    n0 = len(d)
    d = d.drop_duplicates(["vehicle_id", "frame_id", "global_time"])
    print("rows", n0, "after de-duplication", len(d))
    d.to_pickle(p)
    return d


def main():
    d = load_original()
    key = d.set_index(["vehicle_id", "global_time"]).sort_index()
    fs = sorted(glob.glob(os.path.join(HV, "ngsim", "*", "*.json")))
    rng = np.random.default_rng(0)
    rows = []
    for p in rng.choice(fs, 60, replace=False):
        j = pd.DataFrame(json.load(open(p)))
        ego = int(os.path.basename(p).split("car")[1].split("_")[0])
        e = j[j.Vehicle_ID == ego]
        for _, r in e.iloc[::10].iterrows():
            k = (ego, int(r.Global_Time))
            if k not in key.index:
                rows.append(dict(ev=os.path.basename(p), found=0))
                continue
            o = key.loc[k]
            if isinstance(o, pd.DataFrame):
                o = o.iloc[0]
            pre = int(o.preceding)
            rec = dict(ev=os.path.basename(p), found=1, dy=abs(o.local_y - r.Local_Y), dv=abs(o.v_vel - r.v_Vel),
                       dsh=abs(o.space_headway - r.Space_Headway), json_pre=r.preceding_id, orig_pre=pre,
                       ttc_rel=r.TTC)
            kp = (pre, int(r.Global_Time))
            if pre > 0 and kp in key.index:
                q = key.loc[kp]
                if isinstance(q, pd.DataFrame):
                    q = q.iloc[0]
                dvc = o.v_vel - q.v_vel
                rec["ttc_sh"] = o.space_headway / dvc if dvc > 0 else np.inf
                rec["ttc_gap"] = (q.local_y - q.v_length - o.local_y) / dvc if dvc > 0 else np.inf
                rec["pre_dy_vs_sh"] = abs((q.local_y - o.local_y) - o.space_headway)
                # json record of the json preceding vehicle
                jp = j[(j.Vehicle_ID == r.preceding_id) & (j.Global_Time == r.Global_Time)]
                rec["json_pre_dy_vs_sh"] = abs(jp.Local_Y.iloc[0] - r.Local_Y - r.Space_Headway) if len(jp) else np.nan
            rows.append(rec)
    R = pd.DataFrame(rows)
    print(R.describe().round(3).to_string())
    print("json preceding == original preceding:", (R.json_pre == R.orig_pre).mean())
    for c in ("ttc_sh", "ttc_gap"):
        m = np.isfinite(R[c]) & np.isfinite(R.ttc_rel)
        print(c, "median |TTC_released - TTC|", np.median(np.abs(R.ttc_rel[m] - R[c][m])), "n", m.sum())
    print(R.head(12).to_string())


if __name__ == "__main__":
    main()
