"""Shared loaders for the K-Risk rebuttal analyses.

Every released HV event JSON is normalised into one per-frame, per-agent table
with SI units and an oriented bounding box, so that the same indicator code can
run on highD, CitySim (ExpresswayA/FreewayB/extreme), inD, rounD and NGSIM.

Source-specific facts used here (checked against the released files):
  * highD  x, y = upper-left corner of the bounding box (native highD convention,
           verified against dhw = |dx| - length); width = length, height = width.
  * CitySim boundingBox{1..4}{X,Y}m give the four box vertices in metres.
  * inD/rounD xCenter, yCenter, heading (deg), length, width; VRUs carry 0 size.
  * NGSIM  native US customary units (ft, ft/s); Local_Y is the front bumper.
"""
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd

# Default layout: the unzipped Figshare release (K-Risk_data/) and the labelled source
# recordings (trajectory_data/, not redistributed) sit next to this repository.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROOT = os.environ.get("KR_ROOT", os.path.join(os.path.dirname(REPO), "K-Risk_data"))
HV = os.path.join(ROOT, "event_annotations", "HV")
RAW = os.environ.get("KR_RAW", os.path.join(os.path.dirname(REPO), "trajectory_data", "HV"))
OUT = os.environ.get("KR_OUT", os.path.join(HERE, "results"))
FIG = os.environ.get("KR_FIG", os.path.join(HERE, "figures"))

FPS = {"highd": 25.0, "expresswayA": 30.0, "freewayB": 30.0, "ind": 25.0, "round": 25.0, "ngsim": 10.0}
FT = 0.3048
SLOTS = ["preceding", "following", "left_preceding", "left_alongside", "left_following",
         "right_preceding", "right_alongside", "right_following"]
HIGHD_SLOT_COLS = {"preceding": "precedingId", "following": "followingId",
                   "left_preceding": "leftPrecedingId", "left_alongside": "leftAlongsideId",
                   "left_following": "leftFollowingId", "right_preceding": "rightPrecedingId",
                   "right_alongside": "rightAlongsideId", "right_following": "rightFollowingId"}
NGSIM_SLOT_COLS = {"preceding": "preceding_id", "following": "Following",
                   "left_preceding": "leftPrecedingId", "left_alongside": "leftAlongsideId",
                   "left_following": "leftFollowingId", "right_preceding": "rightPrecedingId",
                   "right_alongside": "rightAlongsideId", "right_following": "rightFollowingId"}
DEFAULT_DIMS = {"pedestrian": (0.5, 0.5), "bicycle": (1.8, 0.6), "motorcycle": (2.2, 0.8),
                "car": (4.5, 1.8), "truck_bus": (10.0, 2.5), "van": (5.0, 2.0), "trailer": (12.0, 2.5)}


def resolve(path):
    """Map a stored event path (release-relative, or absolute from another machine) onto ROOT."""
    p = str(path).replace("\\", "/")
    i = p.find("event_annotations/")
    return os.path.join(ROOT, *p[i:].split("/")) if i >= 0 else str(path)


# --------------------------------------------------------------------------- event index
def event_index():
    """Mutually exclusive HV event list, identical to compute_statistics.py:
    de-dup by basename; Extreme = extreme/, High = *_high_risk minus extreme,
    Moderate = *_normal_risk minus (extreme or high)."""
    ext = {}
    for sub in ("ttc_1s", "ttc_2s"):
        for p in sorted(glob.glob(os.path.join(HV, "extreme", sub, "*.json"))):
            b = os.path.basename(p)
            ext.setdefault(b, (p, sub))  # ttc_1s takes precedence when in both
    rows = []
    for b, (p, sub) in ext.items():
        rows.append(dict(event=b[:-5], src=b.split("_")[0], level="extreme", sub=sub, path=p))
    for s in ("highd", "expresswayA", "freewayB", "ind", "round", "ngsim"):
        hi = {os.path.basename(p): p for p in glob.glob(os.path.join(HV, s, f"{s}_high_risk", "*.json"))}
        no = {os.path.basename(p): p for p in glob.glob(os.path.join(HV, s, f"{s}_normal_risk", "*.json"))}
        for b, p in sorted(hi.items()):
            if b not in ext:
                rows.append(dict(event=b[:-5], src=s, level="high", sub="", path=p))
        for b, p in sorted(no.items()):
            if b not in ext and b not in hi:
                rows.append(dict(event=b[:-5], src=s, level="moderate", sub="", path=p))
    return pd.DataFrame(rows)


def ego_of(src, event):
    if src == "highd":
        return int(event.split("_")[2])
    if src in ("expresswayA", "freewayB"):
        return int(re.search(r"car_(\d+)", event).group(1))
    if src in ("ind", "round"):
        return int(re.search(r"ego_(\d+)", event).group(1))
    if src == "ngsim":
        return int(re.search(r"car(\d+)", event).group(1))
    raise ValueError(src)


# --------------------------------------------------------------------------- geometry
def rect(cx, cy, L, W, yaw):
    """4x2 corners of an oriented rectangle (arrays broadcast over frames)."""
    cx, cy, L, W, yaw = (np.asarray(v, float).reshape(-1, 1) for v in (cx, cy, L, W, yaw))
    c, s = np.cos(yaw), np.sin(yaw)
    ux = np.array([[0.5, 0.5, -0.5, -0.5]])
    uy = np.array([[0.5, -0.5, -0.5, 0.5]])
    dx, dy = L * ux, W * uy
    return np.stack([cx + c * dx - s * dy, cy + s * dx + c * dy], axis=-1)


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


# --------------------------------------------------------------------------- loader
def load_event(src, path):
    """Return (agents DataFrame, ego id). Columns: frame,t,id,x,y,vx,vy,speed,acc,yaw,L,W,
    cls,box(4x2 object),drf,ttc,flags...,slot_<role> (ego rows only meaningful)."""
    with open(resolve(path)) as f:
        raw = json.load(f)
    if raw and isinstance(raw[0], dict) and "vehicles" in raw[0]:
        recs = []
        for fr in raw:
            for v in fr["vehicles"]:
                v = dict(v)
                v["frame"] = fr["frame_id"]
                recs.append(v)
    else:
        recs = raw
    d = pd.DataFrame(recs)
    fps = FPS[src]
    out = pd.DataFrame()
    if src == "highd":
        out["frame"] = d["frame"].astype(int)
        out["id"] = d["id"].astype(int)
        L, W = d["width"].astype(float), d["height"].astype(float)
        out["x"], out["y"] = d["x"] + L / 2, d["y"] + W / 2
        out["vx"], out["vy"] = d["xVelocity"], d["yVelocity"]
        out["speed"] = np.hypot(out.vx, out.vy)
        sgn = np.sign(d["xVelocity"]).replace(0, 1)
        out["acc"] = d["xAcceleration"] * sgn
        out["yaw"] = np.where(sgn > 0, 0.0, np.pi)
        out["L"], out["W"] = L, W
        out["cls"] = d["vehicle_class"].str.lower()
        out["drf"] = d["total_risk"]
        out["ttc"] = d["ttc"]
        for c in ("acc_high", "brake_high", "yaw_left", "yaw_right"):
            out[c] = d[c].fillna(False).astype(bool)
        for r, c in HIGHD_SLOT_COLS.items():
            out["slot_" + r] = d[c].where(d[c] > 0)
    elif src in ("expresswayA", "freewayB", "extreme"):
        out["frame"] = d["frame_id"].astype(int)
        out["id"] = d["car_id"].astype(int)
        out["x"], out["y"] = d["car_center_x"], d["car_center_y"]
        out["vx"], out["vy"] = d["vx"], d["vy"]
        out["speed"] = d["speed"]
        out["acc"] = d["acceleration"] if "acceleration" in d else np.nan
        out["yaw"] = np.arctan2(d["vy"], d["vx"])
        out["L"] = d["length"]
        bx = d[[f"boundingBox{i}{a}m" for i in range(1, 5) for a in "XY"]].to_numpy(float).reshape(-1, 4, 2)
        out["W"] = np.hypot(bx[:, 1, 0] - bx[:, 2, 0], bx[:, 1, 1] - bx[:, 2, 1])
        out["cls"] = "car"
        out["drf"] = d.get("risk_value")
        out["ttc"] = d.get("ttc")
        def flag(lbl, hi, mo):
            if lbl in d:
                return d[lbl].fillna(False).astype(bool)
            out_ = pd.Series(False, index=d.index)
            for c in (hi, mo):  # raw CitySim columns in a few extreme files
                if c in d:
                    out_ |= d[c].fillna(False).astype(bool)
            return out_
        out["acc_high"] = flag("acc_label", "acc_high", "acc_moderate")
        out["brake_high"] = flag("brake_label", "brake_high", "brake_moderate")
        out["yaw_left"] = flag("left_turn_label", "left_yaw_high", "left_yaw_moderate")
        out["yaw_right"] = flag("right_turn_label", "right_yaw_high", "right_yaw_moderate")
        if "collision_in_2s" in d:
            out["coll2s"] = d["collision_in_2s"].fillna(False).astype(bool)
            out["coll_id"] = d["collision_id"].map(_num)
            out["coll_t"] = d["Time to Collision"].map(_num)
        for r in SLOTS:
            out["slot_" + r] = d[r + "_id"]
        out["box"] = list(bx)
    elif src in ("ind", "round"):
        out["frame"] = d["frame"].astype(int)
        out["id"] = d["trackId"].astype(int)
        out["x"], out["y"] = d["xCenter"], d["yCenter"]
        out["vx"], out["vy"] = d["xVelocity"], d["yVelocity"]
        out["speed"] = np.hypot(out.vx, out.vy)
        out["acc"] = d["lonAcceleration"]
        out["yaw"] = np.deg2rad(d["heading"])
        cls = d["class"].astype(str).str.lower()
        out["cls"] = cls
        L, W = d["length"].astype(float).copy(), d["width"].astype(float).copy()
        dl = cls.map(lambda c: DEFAULT_DIMS.get(c, (4.5, 1.8))[0])
        dw = cls.map(lambda c: DEFAULT_DIMS.get(c, (4.5, 1.8))[1])
        out["L"] = np.where(L > 0.1, L, dl)
        out["W"] = np.where(W > 0.1, W, dw)
        out["drf"] = d["risk_value"]
        out["ttc"] = np.nan
        for r in SLOTS:
            out["slot_" + r] = d[r + "_id"]
    elif src == "ngsim":
        out["frame"] = d["Frame_ID"].astype(int)
        out["id"] = d["Vehicle_ID"].astype(int)
        L, W = d["v_Length"] * FT, d["v_Width"] * FT
        out["x"] = d["Local_X"] * FT
        out["y"] = d["Local_Y"] * FT - L / 2
        out["vx"], out["vy"] = 0.0, d["v_Vel"] * FT
        out["speed"] = d["v_Vel"] * FT
        out["acc"] = d["v_Acc"] * FT
        out["yaw"] = np.pi / 2
        out["L"], out["W"] = L, W
        out["cls"] = d["v_Class"].map({1: "motorcycle", 2: "car", 3: "truck_bus"}).fillna("car")
        out["drf"] = np.nan
        out["ttc"] = d["TTC"]
        for r, c in NGSIM_SLOT_COLS.items():
            out["slot_" + r] = d[c].where(d[c] > 0)
    if "box" not in out:
        out["box"] = list(rect(out.x.to_numpy(), out.y.to_numpy(),
                               out.L.to_numpy(), out.W.to_numpy(),
                               out.yaw.to_numpy()))
    # records without a position (a small share of some CitySim files) cannot be used
    # for geometry; they are counted by the audit and dropped here
    finite = np.array([np.isfinite(np.asarray(b, float)).all() for b in out["box"]])
    out = out[finite & np.isfinite(out.x.to_numpy(float)) & np.isfinite(out.y.to_numpy(float))].copy()
    out["L"] = out["L"].fillna(4.5)
    out["W"] = out["W"].fillna(1.8)
    out["t"] = (out["frame"] - out["frame"].min()) / fps
    out = out.drop_duplicates(["frame", "id"]).sort_values(["frame", "id"]).reset_index(drop=True)
    return out


def smooth_acc(speed, fps, win_s=0.5):
    """Longitudinal acceleration from the speed series with a centred moving average
    of win_s seconds, robust to single-frame tracking spikes."""
    s = pd.Series(np.asarray(speed, float))
    k = max(3, int(round(win_s * fps)) | 1)
    sm = s.rolling(k, center=True, min_periods=1).mean()
    a = np.gradient(sm.to_numpy(), 1.0 / fps)
    return pd.Series(a).rolling(k, center=True, min_periods=1).mean().to_numpy()


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)
