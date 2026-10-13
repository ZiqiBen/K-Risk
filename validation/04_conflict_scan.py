"""Independent re-implementation of the 2-s trajectory-conflict predictor (CA/CS on
bounding boxes) scanned over every frame (10 Hz grid) of every released HV event.

For each event it reports whether the ego's predicted box overlaps any other agent's
predicted box within 2 s at any frame of the window, the minimum predicted time to
overlap, and the number of distinct agents that individually produce a predicted
overlap with the ego (multi-agent attribution, R1-1). Used to audit the extreme subset
(R2-5 extraction audit) and to compare grades (R2-1).

Run: python 04_conflict_scan.py -> results/conflict_scan.csv
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, os.path.dirname(__file__))
from kr_common import FPS, OUT, ego_of, event_index, load_event  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("chk", os.path.join(os.path.dirname(__file__), "03_action_checker.py"))
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

H, DT = 2.0, 0.1


def scan(row):
    src = row["src"]
    fps = FPS[src]
    try:
        a = load_event(src, row["path"])
        ego = ego_of(src, row["event"])
        e = a[a.id == ego].sort_values("frame")
        fr = e.frame.to_numpy()
        grid = fr[fr >= fr.min() + 0.5 * fps][:: max(1, int(fps // 10))]
        states = {}
        tmin, partners, n_conf_frames = np.inf, set(), 0
        partner_t = {}
        for f in grid:
            es = chk.state_at(e, fps, f)
            if es is None:
                continue
            eb = chk.boxes(chk.roll(es, H, DT), es["L"], es["W"])
            hit = False
            for oid, g in a[(a.id != ego) & (a.frame == f)].groupby("id"):
                s = chk.state_at(a[a.id == oid], fps, f)
                if s is None or np.hypot(s["x"] - es["x"], s["y"] - es["y"]) > 80:
                    continue
                ob = chk.boxes(chk.roll(s, H, DT), s["L"], s["W"])
                inter = shapely.intersects(eb, ob)
                if inter.any():
                    k = int(np.argmax(inter))
                    t = (k + 1) * DT
                    hit = True
                    partners.add(int(oid))
                    partner_t[int(oid)] = min(partner_t.get(int(oid), np.inf), t)
                    tmin = min(tmin, t)
            n_conf_frames += int(hit)
        # simultaneous multi-agent conflict at one frame
        return dict(event=row["event"], pred_conflict=int(np.isfinite(tmin)),
                    pred_t_min=tmin if np.isfinite(tmin) else np.nan,
                    n_pred_partners=len(partners), n_pred_partners_le1s=sum(t <= 1.0 for t in partner_t.values()),
                    conf_frames=n_conf_frames, grid_frames=len(grid))
    except Exception as exc:  # noqa: BLE001
        return dict(event=row["event"], error=repr(exc))


def main():
    idx = event_index()
    with Pool(22) as p:
        res = p.map(scan, idx.to_dict("records"), chunksize=8)
    R = idx[["event", "src", "level", "sub"]].merge(pd.DataFrame(res), on="event")
    R.to_csv(os.path.join(OUT, "conflict_scan.csv"), index=False)
    print(R.groupby("level")[["pred_conflict", "n_pred_partners"]].mean())
    print(R.groupby(["level", "sub"]).pred_t_min.describe())


if __name__ == "__main__":
    main()
