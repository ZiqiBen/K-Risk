"""Second, version-independent verification of the NGSIM issue.

(1) Internal consistency of the released JSON alone: lane and position of the vehicle
    listed as preceding versus the ego's own Space_Headway; how the released TTC relates
    to the speeds of that vehicle.
(2) Version check: do the released ego AND neighbour records equal the original rows of
    the same vehicle at the same Global_Time (same data version)?
(3) Pipeline input: the local labelled subsample (trajectory_data/HV/ngsim) - its size
    relative to the full recording and whether its preceding_id equals the original
    Preceding.
"""
import glob, json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from kr_common import HV, RAW

ORIG = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngsim_source", "i80.parquet.pkl"))
KEY = ORIG.set_index(["vehicle_id", "global_time"])[["local_y", "local_x", "v_vel", "lane_id", "preceding", "space_headway"]]
KEY = KEY[~KEY.index.duplicated()]


def one(p):
    j = pd.DataFrame(json.load(open(p)))
    ego = int(os.path.basename(p).split("car")[1].split("_")[0])
    e = j[j.Vehicle_ID == ego].copy()
    pj = j[["Vehicle_ID", "Global_Time", "Local_Y", "Lane_ID", "v_Vel"]].rename(
        columns={"Vehicle_ID": "preceding_id", "Local_Y": "yp", "Lane_ID": "lp", "v_Vel": "vp"})
    m = e.merge(pj, on=["preceding_id", "Global_Time"], how="left")
    m = m[(m.preceding_id > 0) & m.yp.notna()]
    out = dict(frames=len(m))
    if len(m):
        out["same_lane"] = float((m.lp == m.Lane_ID).mean())
        out["pos_consistent"] = float((((m.yp - m.Local_Y) - m.Space_Headway).abs() < 15).mean())
        dv = m.v_Vel - m.vp
        t_json = np.where(dv > 0, m.Space_Headway / dv, np.nan)
        ok = np.isfinite(t_json) & m.TTC.notna()
        out["ttc_matches_json_preceding"] = float((np.abs(t_json[ok] - m.TTC[ok]) < 0.01).mean()) if ok.any() else np.nan
    # version check on every record
    idx = pd.MultiIndex.from_arrays([j.Vehicle_ID.astype(int), j.Global_Time.astype(np.int64)])
    o = KEY.reindex(idx)
    found = o.local_y.notna().to_numpy()
    out["records"] = len(j)
    out["found"] = float(found.mean())
    out["identical"] = float((np.abs(o.local_y.to_numpy()[found] - j.Local_Y.to_numpy()[found]) < 1e-6).mean()) if found.any() else np.nan
    eo = KEY.reindex(pd.MultiIndex.from_arrays([e.Vehicle_ID.astype(int), e.Global_Time.astype(np.int64)]))
    f2 = eo.preceding.notna().to_numpy()
    out["ego_pre_equals_original"] = float((eo.preceding.to_numpy()[f2] == np.where(e.preceding_id.to_numpy()[f2] > 0, e.preceding_id.to_numpy()[f2], 0)).mean()) if f2.any() else np.nan
    return out


def main():
    fs = sorted(glob.glob(os.path.join(HV, "ngsim", "*", "*.json")))
    with Pool(16) as p:
        R = pd.DataFrame(p.map(one, fs, chunksize=16))
    print("released NGSIM files", len(R))
    print(R.describe().round(4).to_string())
    print("files with ALL records found & identical in the original:", ((R.found == 1) & (R.identical == 1)).mean())
    # pipeline input subsample
    sub = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RAW, "ngsim", "*.csv")))])
    s_idx = pd.MultiIndex.from_arrays([sub.Vehicle_ID.astype(int), sub.Global_Time.astype(np.int64)])
    so = KEY.reindex(s_idx)
    print("local labelled NGSIM rows", len(sub), "of", len(ORIG), f"({100 * len(sub) / len(ORIG):.1f}%)",
          "| rows identical to original:", float((np.abs(so.local_y.to_numpy() - sub.Local_Y.to_numpy()) < 1e-6).mean()),
          "| preceding_id == original Preceding:", float((so.preceding.to_numpy() == sub.preceding_id.fillna(0).clip(lower=0).to_numpy()).mean()))
    # are neighbours of the subsample rows present in the subsample?
    pres = set(zip(sub.Vehicle_ID.astype(int), sub.Global_Time.astype(np.int64)))
    op = so.preceding.to_numpy()
    has = np.array([(int(a), int(b)) in pres for a, b in zip(op, sub.Global_Time.astype(np.int64))])
    print("share of subsample rows whose true preceding vehicle is absent from the subsample:", float(1 - has[op > 0].mean()))


if __name__ == "__main__":
    main()
