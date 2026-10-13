"""Human driver action in the 1 s after the decision frame, judged with the same at-fault checker as the LLMs.

The human action is classified from the full CitySim trajectory of the ego (frames f..f+1 s):
  LEFT/RIGHT  lane_id moves to the left/right adjacent lane, or lateral displacement > LAT m
              toward a side that has a lane (lane topology of the generators)
  ACC/DEC     mean acceleration over the window > +A_TH / < -A_TH (0.17-s averaged end speeds)
  IDLE        otherwise
Its verdict is looked up in results/checker_fault.csv (at-fault collision or unnecessary braking),
exactly as for the LLM decisions, and compared with zero-shot / ICL LLMs and constant policies.

Run: python 23_human_action.py -> results/human_action_1s.csv, results/safety_table.csv
"""
import importlib.util
import os
import re
from math import comb

import numpy as np
import pandas as pd
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
KR = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("KR_ROOT", os.path.join(KR, "K-Risk_data"))
os.environ.setdefault("KR_GEN", os.path.join(KR, "K-Risk_code"))
_spec = importlib.util.spec_from_file_location("chk", os.path.join(HERE, "03_action_checker.py"))
chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)
from kr_common import load_event  # noqa: E402  (03_action_checker put this folder on sys.path)

OUT = os.environ.get("KR_OUT", os.path.join(HERE, "results"))
RAW = {"expresswayA": os.path.join(KR, "expresswayA", "ExpresswayA-{:02d}_updated_with_bbox.csv"),
       "freewayB": os.path.join(KR, "freewayB", "Trajectories", "FreewayB-{:02d}.csv")}
ICL = os.path.join(KR, "K-Risk_data", "llm_analysis_icl")
FPS, WIN, LAT, A_TH = 30, 1.0, 0.5, 0.5
AMAP = {1: "IDLE", 2: "LEFT", 3: "RIGHT", 4: "ACC", 5: "DEC"}
ACTS = list(AMAP.values())


def heading(e, f, w):
    d = e.loc[f - 5:f][["car_center_x", "car_center_y"]].to_numpy()
    h = d[-1] - d[0] if len(d) > 1 else np.zeros(2)
    if np.hypot(*h) < 0.2:  # nearly stopped: take the heading from the window itself
        p = w[["car_center_x", "car_center_y"]].to_numpy()
        h = p[-1] - p[0]
    return np.arctan2(h[1], h[0]), np.hypot(*h) >= 0.2


def human_action_original(e, f, src, lane, H=2.0):
    """Rule of 03_action_checker.human_response: 2-s window, |lat| > 0.9 m, |dv| > 1.0 m/s.
    Heading from positions because the FreewayB trajectories have no vx/vy."""
    w = e.loc[f:f + int(H * FPS)]
    if len(w) < H * FPS * 0.8 or f not in e.index:
        return None, np.nan, np.nan
    yaw, moving = heading(e, f, w)
    p = w[["car_center_x", "car_center_y"]].to_numpy()
    lat = (p[-1] - p[0]) @ np.array([np.sin(yaw), -np.cos(yaw)]) if moving else 0.0  # + = left (y-south)
    dv = w.speed.iloc[-1] - w.speed.iloc[0]
    act = ("LEFT" if lat > 0.9 else "RIGHT" if lat < -0.9 else
           "ACC" if dv > 1.0 else "DEC" if dv < -1.0 else "IDLE")
    return act, lat, dv / H


def human_action(e, f, src, lane):
    """e: ego rows of the full trajectory indexed by frame."""
    n = int(WIN * FPS)
    w = e.loc[f:f + n]
    if len(w) < 0.8 * n or f not in e.index:
        return None, np.nan, np.nan
    hist = e.loc[f - 5:f]
    d = hist[["car_center_x", "car_center_y"]].to_numpy()
    h = d[-1] - d[0] if len(d) > 1 else np.zeros(2)
    if np.hypot(*h) < 0.2:  # nearly stopped: take the heading from the window itself
        h = w[["car_center_x", "car_center_y"]].to_numpy()[-1] - w[["car_center_x", "car_center_y"]].to_numpy()[0]
    yaw = np.arctan2(h[1], h[0])
    disp = w[["car_center_x", "car_center_y"]].to_numpy()[-1] - w[["car_center_x", "car_center_y"]].to_numpy()[0]
    lat = disp @ np.array([np.sin(yaw), -np.cos(yaw)]) if np.hypot(*h) >= 0.2 else 0.0  # + = left (y-south)
    k = 5
    acc = (w.speed.iloc[-k:].mean() - w.speed.iloc[:k].mean()) / ((len(w) - k) / FPS)
    info = chk.LANES[src].get(lane) if lane is not None else None
    l1 = w.lane_id.iloc[-1]
    to = lambda side: info and pd.notna(l1) and int(l1) in (info.get(side + "_adjacent_lane") or [])  # noqa: E731
    if to("left") or (lat > LAT and chk.legal(src, lane, "LEFT")):
        return "LEFT", lat, acc
    if to("right") or (lat < -LAT and chk.legal(src, lane, "RIGHT")):
        return "RIGHT", lat, acc
    return ("ACC" if acc > A_TH else "DEC" if acc < -A_TH else "IDLE"), lat, acc


def human_action_matched(a, ego, f, src, lane, W=1.0):
    """Thresholds halfway between IDLE and the checker's idealised actions over W seconds:
    lateral vlat*W/2, ACC accel*W/2, DEC min(decel*W, v0)/2. Lane changes only toward a side with a lane."""
    P = SimpleNamespace(accel=2.0, decel=4.0, vlat=1.0)
    e = a[a.id == ego].sort_values("frame")
    fut = e[(e.frame >= f) & (e.frame <= f + W * FPS)]
    if len(fut) < W * FPS * 0.8 or fut.frame.iloc[0] != f:
        return None, np.nan, np.nan
    c = fut.iloc[0]
    yaw = np.arctan2(c.vy, c.vx)
    lat = (fut.x.iloc[-1] - c.x) * np.sin(yaw) - (fut.y.iloc[-1] - c.y) * np.cos(yaw)  # + = left (y-south)
    dv = fut.speed.iloc[-1] - c.speed
    if lat > P.vlat * W / 2 and chk.legal(src, lane, "LEFT"):
        return "LEFT", lat, dv
    if lat < -P.vlat * W / 2 and chk.legal(src, lane, "RIGHT"):
        return "RIGHT", lat, dv
    dec_th = min(P.decel * W, c.speed) / 2
    if dv > P.accel * W / 2:
        return "ACC", lat, dv
    if dec_th > 0.05 and dv < -dec_th:
        return "DEC", lat, dv
    return "IDLE", lat, dv


def icl_action(m, e):
    fol = {"gpt4o": "analysis_gpt4o", "gpt41": "analysis_gpt4.1"}[m]
    p = os.path.join(ICL, fol, f"response_extreme_desc_{e}.txt")
    if not os.path.exists(p):
        return None
    r = open(p, encoding="utf-8").read()
    k = re.findall(r"Final Decision[^0-9]{0,20}(\d)", r) or re.findall(r"####\s*(\d)", r)
    return AMAP.get(int(k[-1])) if k else None


def mcnemar(a, b):
    b10, b01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = b10 + b01
    return b10, b01, (min(1.0, 2 * sum(comb(n, k) for k in range(min(b10, b01) + 1)) / 2 ** n) if n else 1.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["scaled1s", "original2s", "json1s", "matched1s"], default="matched1s",
                    help="json1s: 03_action_checker rule scaled to 1 s; matched1s: thresholds halfway to the "
                         "checker's action magnitudes (both from the event JSONs)")
    A = ap.parse_args()
    classify = human_action if A.rule == "scaled1s" else human_action_original
    label = {"scaled1s": "Human (1 s)", "original2s": "Human (2 s, original rule)",
             "json1s": "Human (1 s, original rule scaled)", "matched1s": "Human (1 s, checker-matched)"}[A.rule]
    C = pd.read_csv(os.path.join(OUT, "checker_llm_hwin1.csv"))
    V = pd.read_csv(os.path.join(OUT, "checker_fault.csv")).set_index("event")
    C["track"] = C.event.str.extract(r"track_(\d+)")[0].astype(int)
    C["ego"] = C.event.str.extract(r"car_(\d+)")[0].astype(int)
    rows = []
    for (src, t), g in C.groupby(["src", "track"]) if A.rule in ("scaled1s", "original2s") else []:
        d = pd.read_csv(RAW[src].format(t), usecols=["frame_id", "car_id", "car_center_x", "car_center_y", "speed", "lane_id"])
        d = d[d.car_id.isin(g.ego)].drop_duplicates(["car_id", "frame_id"])
        for r in g.itertuples():
            e = d[d.car_id == r.ego].set_index("frame_id").sort_index()
            lane = int(r.lane) if pd.notna(r.lane) else None
            act, lat, acc = classify(e, int(r.frame), src, lane)
            rows.append(dict(event=r.event, human=act, lat=lat, acc=acc))
    if A.rule == "matched1s":
        for r in C.itertuples():
            act, lat, dv = human_action_matched(load_event(r.src, r.path), r.ego, int(r.frame), r.src,
                                                int(r.lane) if pd.notna(r.lane) else None)
            rows.append(dict(event=r.event, human=act, lat=lat, dv=dv))
    H = pd.DataFrame(rows).set_index("event") if rows else C.set_index("event")[["human_action"]].rename(
        columns={"human_action": "human"})
    H.to_csv(os.path.join(OUT, f"human_action_{A.rule}.csv"))

    sel = set(pd.read_csv(os.path.join(OUT, "icl_selected.csv")).event)
    D = C.set_index("event")[["gpt4o", "gpt41"]].apply(lambda s: s.map(AMAP))
    D.columns = ["GPT-4o zero-shot", "GPT-4.1 zero-shot"]
    D["GPT-4o ICL"] = [icl_action("gpt4o", e) for e in D.index]
    D["GPT-4.1 ICL"] = [icl_action("gpt41", e) for e in D.index]
    D[label] = H.human
    D["Always IDLE"], D["Always DEC"] = "IDLE", "DEC"
    U = pd.DataFrame({c: [V.loc[e, f"unsafe_{a}"] if isinstance(a, str) else np.nan for e, a in D[c].items()]
                      for c in D.columns}, index=D.index)
    test = [e for e in D.index if e not in sel]
    order = [label, "GPT-4o zero-shot", "GPT-4o ICL", "GPT-4.1 zero-shot", "GPT-4.1 ICL", "Always IDLE", "Always DEC"]
    tab = []
    for c in order:
        row = {"policy": c, "n (all 372)": int(U[c].notna().sum()),
               "unsafe all 372": U[c].mean() if c not in ("GPT-4o ICL", "GPT-4.1 ICL") else np.nan,
               "n (test 352)": int(U.loc[test, c].notna().sum()), "unsafe test 352": U.loc[test, c].mean()}
        both = U.loc[test, [label, c]].dropna()
        if c != label:
            b10, b01, p = mcnemar(both[label].values, both[c].values)
            row.update({"vs human: human-only unsafe": b10, "policy-only unsafe": b01, "McNemar p": p})
        row.update({f"{a} %": 100 * (D.loc[test, c].dropna() == a).mean() for a in ACTS})
        tab.append(row)
    T = pd.DataFrame(tab)
    T.to_csv(os.path.join(OUT, f"safety_table_{A.rule}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"human action classified: {H.human.notna().sum()} / {len(H)} | distribution:",
          H.human.value_counts().to_dict())
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
