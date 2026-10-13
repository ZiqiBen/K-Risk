"""Release-ready validation files referenced in the revised manuscript.

release/event_audit_flags.csv  one row per HV event: rule-consistency flag, recorded and
                               reproduced conflicts, independent indicators, smoothed
                               trigger acceleration (CitySim) and NGSIM neighbour
                               consistency
release/llm_audit_flags.csv    one row per annotated event and model
"""
import glob, json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from kr_common import OUT, HV

REL = os.path.join(os.path.dirname(__file__), "release")
A = pd.read_csv(os.path.join(OUT, "event_audit_flags.csv"))
F = pd.read_csv(os.path.join(OUT, "event_features.csv"), low_memory=False)
C = pd.read_csv(os.path.join(OUT, "conflict_scan.csv"))


def ngsim_consistency(p):
    d = pd.DataFrame(json.load(open(p)))
    ego = int(os.path.basename(p).split("car")[1].split("_")[0])
    e = d[d.Vehicle_ID == ego]
    m = e.merge(d[["Vehicle_ID", "Frame_ID", "Local_Y"]].rename(columns={"Vehicle_ID": "preceding_id", "Local_Y": "yp"}),
                on=["preceding_id", "Frame_ID"], how="left")
    m = m[m.preceding_id > 0]
    ok = ((m.yp - m.Local_Y) - m.Space_Headway).abs() < 15
    return os.path.basename(p)[:-5], float(ok.mean()) if len(m) else np.nan


with Pool(16) as pool:
    ng = dict(pool.map(ngsim_consistency, glob.glob(os.path.join(HV, "ngsim", "*", "*.json")), chunksize=16))
X = A.merge(F[["event", "ego_amax_raw", "ego_amax_s", "slot_amax_raw", "slot_amax_s", "min_clear", "pet", "drac",
               "dur_s", "n_frames"]], on="event", how="left")
X = X.merge(C[["event", "n_pred_partners"]], on="event", how="left")
cs = X.src.isin(["expresswayA", "freewayB"]) & (X.level != "extreme")
X["trigger_acc_raw"] = np.where(cs, np.where(X.level == "high", X.ego_amax_raw, X.slot_amax_raw), np.nan)
X["trigger_acc_smoothed"] = np.where(cs, np.where(X.level == "high", X.ego_amax_s, X.slot_amax_s), np.nan)
X["ngsim_neighbour_consistency"] = X.event.map(ng)
cols = ["event", "src", "level", "sub", "trigger_ok", "recorded_conflict_any", "recorded_conflict_ego",
        "predicted_conflict", "pred_conflict", "pred_t_min", "n_pred_partners", "trigger_acc_raw",
        "trigger_acc_smoothed", "min_clear", "pet", "drac", "dur_s", "n_frames", "ngsim_neighbour_consistency"]
X = X[cols].rename(columns={"pred_conflict": "reproduced_conflict_any_level", "predicted_conflict": "reproduced_conflict_extreme",
                            "pred_t_min": "reproduced_time_to_overlap_s", "n_pred_partners": "reproduced_conflict_partners",
                            "min_clear": "min_clearance_m", "pet": "pet_s", "drac": "drac_ms2", "dur_s": "window_s"})
X.to_csv(os.path.join(REL, "event_audit_flags.csv"), index=False, float_format="%.4g")
L = pd.read_csv(os.path.join(OUT, "llm_audit_per_event.csv"))
L.to_csv(os.path.join(REL, "llm_audit_flags.csv"), index=False)
print(len(X), X.ngsim_neighbour_consistency.notna().sum(), len(L))
