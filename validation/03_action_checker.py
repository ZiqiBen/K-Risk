"""Action-conditioned conflict checker on the CitySim events (moderate, high, extreme).

Dataset-centric closed-loop validation (R2-5, R2-6, R2-8, R2-9). No LLM and no human
rating is involved. For every event the checker takes the peak-risk frame t*, rolls the
ego forward under each of the five schema actions and every other agent under constant
acceleration and constant yaw rate (CA/CS), and records whether the ego bounding box
overlaps any other box within the horizon. Lane changes towards a side without a legal
lane (lane topology of the released generators) count as unsafe.

Outputs per event:
  unsafe_<action>   1 if that action leads to a box overlap within the horizon
  n_unsafe          number of unsafe actions (0..5)
  human_action      discrete action that best matches the recorded human response
  human_overlap     1 if the recorded human trajectories overlap within the horizon
  human_unsafe      checker verdict for the matched human action
Also evaluates the GPT-4o / GPT-4.1 actions of the 372 annotated events at the frame
that was given to the LLM (the last described frame).

Run: python 03_action_checker.py [--horizon 2.0 --decel 4 --accel 2 --vlat 1.0 --tag base]
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, os.path.dirname(__file__))
from kr_common import FPS, OUT, ROOT, ego_of, event_index, load_event, rect, resolve  # noqa: E402

GEN = os.environ.get("KR_GEN", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTIONS = ["IDLE", "LEFT", "RIGHT", "ACC", "DEC"]


def _lane_dict(name, var):
    spec = importlib.util.spec_from_file_location(name, os.path.join(GEN, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return getattr(m, var)


LANES = {"expresswayA": _lane_dict("generate_expresswayA", "lane_info_expresswayA"),
         "freewayB": _lane_dict("generate_freewayB", "lane_info_freewayB")}


def legal(src, lane, side):
    info = LANES[src].get(lane) if lane is not None else None
    if info is None:
        return True  # unknown lane: do not penalise
    return bool(info.get("left_adjacent_lane" if side == "LEFT" else "right_adjacent_lane"))


def state_at(g, fps, frame, hist_s=0.5):
    """Current kinematic state of one agent with CA/CS rates estimated over hist_s."""
    g = g.sort_values("frame")
    h = g[(g.frame <= frame) & (g.frame > frame - hist_s * fps - 1)]
    cur = g[g.frame == frame]
    if cur.empty or len(h) < 2:
        return None
    c = cur.iloc[0]
    t = (h.frame.to_numpy() - frame) / fps
    sp = h.speed.to_numpy(float)
    vx, vy = h.vx.to_numpy(float), h.vy.to_numpy(float)
    if not (np.isfinite(sp).all() and np.isfinite(vx).all() and np.isfinite(vy).all()):
        # fall back to finite differences of the positions
        dx = np.gradient(h.x.to_numpy(float), t) if len(h) >= 2 else np.zeros(len(h))
        dy = np.gradient(h.y.to_numpy(float), t) if len(h) >= 2 else np.zeros(len(h))
        vx, vy = dx, dy
        sp = np.hypot(dx, dy)
        c = c.copy()
        c["speed"] = sp[-1]
    yaw = np.unwrap(np.arctan2(vy, vx))
    if not (np.isfinite(sp).all() and np.isfinite(yaw).all()):
        return None
    a = np.polyfit(t, sp, 1)[0] if len(h) >= 3 else 0.0
    w = np.polyfit(t, yaw, 1)[0] if len(h) >= 3 and sp.mean() > 1.0 else 0.0
    return dict(x=c.x, y=c.y, v=max(0.0, float(c.speed)), yaw=float(yaw[-1]),
                a=float(np.clip(a, -8, 8)), w=float(np.clip(w, -0.5, 0.5)), L=float(c.L), W=float(c.W))


def roll(s, H, dt, a=None, w=None, vlat=0.0, side=0):
    """CA/CS roll-out; side=+1 left, -1 right lateral motion at vlat (y-south frame)."""
    a = s["a"] if a is None else a
    w = s["w"] if w is None else w
    n = int(round(H / dt))
    x, y, v, yaw = s["x"], s["y"], s["v"], s["yaw"]
    out = []
    lat = 0.0
    for _ in range(n):
        v = max(0.0, v + a * dt)
        yaw = yaw + w * dt
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        lat += vlat * dt
        # left normal in a y-south frame is (sin yaw, -cos yaw)
        out.append((x + side * lat * np.sin(yaw), y - side * lat * np.cos(yaw), yaw))
    return np.array(out)


def boxes(traj, L, W):
    return shapely.polygons(rect(traj[:, 0], traj[:, 1], np.full(len(traj), L),
                                 np.full(len(traj), W), traj[:, 2]))


def check(a, ego, fps, frame, src, lane, P):
    """Return dict action->unsafe for the ego at frame."""
    es = state_at(a[a.id == ego], fps, frame)
    if es is None:
        return None
    others = []
    for oid, g in a[a.id != ego].groupby("id"):
        s = state_at(g, fps, frame)
        if s is None or np.hypot(s["x"] - es["x"], s["y"] - es["y"]) > 120:
            continue
        others.append(boxes(roll(s, P.horizon, P.dt), s["L"], s["W"]))
    res = {}
    for act in ACTIONS:
        if act == "IDLE":
            tr = roll(es, P.horizon, P.dt, a=0.0, w=0.0)
        elif act == "ACC":
            tr = roll(es, P.horizon, P.dt, a=P.accel, w=0.0)
        elif act == "DEC":
            tr = roll(es, P.horizon, P.dt, a=-P.decel, w=0.0)
        else:
            side = 1 if act == "LEFT" else -1
            if not legal(src, lane, act):
                res[act] = 1
                continue
            tr = roll(es, P.horizon, P.dt, a=0.0, w=0.0, vlat=P.vlat, side=side)
        eb = boxes(tr, es["L"], es["W"])
        res[act] = int(any(bool(np.any(shapely.intersects(eb, ob))) for ob in others))
    return res


def human_response(a, ego, fps, frame, H):
    """Discrete action matching the recorded ego motion over H, and recorded overlap."""
    e = a[a.id == ego].sort_values("frame")
    fut = e[(e.frame >= frame) & (e.frame <= frame + H * fps)]
    if len(fut) < H * fps * 0.8:
        return None, None
    c = fut.iloc[0]
    yaw = np.arctan2(c.vy, c.vx)
    dx, dy = fut.x.iloc[-1] - c.x, fut.y.iloc[-1] - c.y
    lat = dx * np.sin(yaw) - dy * np.cos(yaw)  # + = left (y-south frame)
    dv = fut.speed.iloc[-1] - c.speed
    s = H / 2.0  # thresholds were set for a 2-s window
    if lat > 0.9 * s:
        act = "LEFT"
    elif lat < -0.9 * s:
        act = "RIGHT"
    elif dv > 1.0 * s:
        act = "ACC"
    elif dv < -1.0 * s:
        act = "DEC"
    else:
        act = "IDLE"
    # recorded overlap
    o = a[(a.id != ego) & a.frame.isin(fut.frame)]
    if o.empty:
        return act, 0
    eb = {fr: shapely.polygons(np.asarray(b)) for fr, b in zip(fut.frame, fut.box)}
    ob = shapely.polygons(np.stack(o.box.to_numpy()))
    ep = np.array([eb[fr] for fr in o.frame])
    return act, int(np.any(shapely.intersects(ep, ob)))


def peak_frame(a, ego, fps, src, P):
    """Peak-risk frame: frame (with >= horizon of recorded future and 0.5 s history)
    minimising the predicted constant-velocity clearance of the ego over the horizon."""
    e = a[a.id == ego].sort_values("frame")
    fr = e.frame.to_numpy()
    cand = fr[(fr >= fr.min() + 0.5 * fps) & (fr <= fr.max() - P.horizon * fps)]
    if len(cand) == 0:
        return None
    cand = cand[:: max(1, int(fps // 10))]  # 10 Hz search grid
    best, bestd = None, np.inf
    for f in cand:
        es = state_at(e, fps, f)
        if es is None:
            continue
        et = roll(es, P.horizon, 0.2, a=0.0, w=0.0)
        for oid, g in a[(a.id != ego) & (a.frame == f)].groupby("id"):
            s = state_at(a[a.id == oid], fps, f)
            if s is None:
                continue
            ot = roll(s, P.horizon, 0.2)
            d = float(np.min(np.hypot(et[:, 0] - ot[:, 0], et[:, 1] - ot[:, 1])))
            if d < bestd:
                best, bestd = f, d
    return best


def run_event(args):
    row, P = args
    src = row["src"]
    fps = FPS[src]
    try:
        a = load_event(src, row["path"])
        with open(resolve(row["path"])) as fh:
            raw = json.load(fh)
        lanes = {(int(r["frame_id"]), int(r["car_id"])): r.get("lane_id") for r in raw}
        ego = ego_of(src, row["event"])
        f = int(row["frame"]) if row.get("frame") is not None and not pd.isna(row.get("frame")) else peak_frame(a, ego, fps, src, P)
        if f is None:
            return dict(event=row["event"], error="no frame")
        lane = lanes.get((f, ego))
        res = check(a, ego, fps, f, src, lane, P)
        if res is None:
            return dict(event=row["event"], error="no state")
        out = dict(event=row["event"], frame=f, lane=lane)
        out.update({"unsafe_" + k: v for k, v in res.items()})
        out["n_unsafe"] = sum(res.values())
        h, ov = human_response(a, ego, fps, f, P.hwin or P.horizon)
        out["human_action"], out["human_overlap"] = h, ov
        out["human_unsafe"] = res.get(h) if h else np.nan
        return out
    except Exception as exc:  # noqa: BLE001
        return dict(event=row["event"], error=repr(exc))


def llm_rows():
    """372 annotated events: frame given to the LLM and the two models' actions."""
    llm = os.path.join(ROOT, "llm_analysis")
    rows = []
    idx = event_index().set_index("event")
    for fn in sorted(os.listdir(os.path.join(llm, "event_discriptions"))):
        ev = fn.replace("extreme_desc_", "").replace(".txt", "")
        txt = open(os.path.join(llm, "event_discriptions", fn), encoding="utf8").read()
        last = int(re.findall(r"Frame (\d+):", txt)[-1])
        acts = {}
        for m, folder in (("gpt4o", "analysis_gpt4o"), ("gpt41", "analysis_gpt4.1")):
            r = open(os.path.join(llm, folder, "response_" + fn), encoding="utf8").read()
            k = re.findall(r"Final Decision[^0-9]{0,20}(\d)", r) or re.findall(r"####\s*(\d)", r)
            acts[m] = int(k[-1]) if k else None
        rec = idx.loc[ev] if ev in idx.index else None
        if rec is None:
            continue
        rows.append(dict(event=ev, src=rec.src, path=rec.path, level=rec.level, frame=last, **acts))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--decel", type=float, default=4.0)
    ap.add_argument("--accel", type=float, default=2.0)
    ap.add_argument("--vlat", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--hwin", type=float, default=None, help="human-response window [s], default = horizon")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--llm-only", action="store_true")
    P = ap.parse_args()
    if not P.llm_only:
        idx = event_index()
        idx = idx[idx.src.isin(["expresswayA", "freewayB"])].copy()
        idx["frame"] = None
        rows = [(r, P) for r in idx.to_dict("records")]
        with Pool(22) as p:
            res = p.map(run_event, rows, chunksize=8)
        R = idx[["event", "src", "level", "sub"]].merge(pd.DataFrame(res), on="event")
        R.to_csv(os.path.join(OUT, f"checker_{P.tag}.csv"), index=False)
        print(R.groupby("level")[["n_unsafe", "unsafe_IDLE", "human_overlap", "human_unsafe"]].mean())
    L = llm_rows()
    rows = [(r, P) for r in L.to_dict("records")]
    with Pool(22) as p:
        res = p.map(run_event, rows, chunksize=4)
    R = L.merge(pd.DataFrame(res).drop(columns=["frame"], errors="ignore"), on="event")
    R.to_csv(os.path.join(OUT, f"checker_llm_{P.tag}.csv"), index=False)
    amap = {1: "IDLE", 2: "LEFT", 3: "RIGHT", 4: "ACC", 5: "DEC"}
    for m in ("gpt4o", "gpt41"):
        R[m + "_unsafe"] = [r.get("unsafe_" + amap.get(int(r[m]), "IDLE")) if pd.notna(r[m]) else np.nan
                            for r in R.to_dict("records")]
    print(R[["gpt4o_unsafe", "gpt41_unsafe", "unsafe_IDLE", "unsafe_DEC", "n_unsafe"]].mean())


if __name__ == "__main__":
    main()
