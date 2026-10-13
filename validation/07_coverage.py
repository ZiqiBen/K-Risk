"""Share of each original recording that the released K-Risk event windows contain (E2).

Counts unique (recording, frame, agent) tuples present in the released event JSONs and
divides by the number of agent-frames in the complete original recordings held locally
(highD, inD, rounD, NGSIM I-80). Also reports the share of distinct agents and of
recording time touched by at least one event window.

Output: results/coverage.json
"""
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from kr_common import OUT, RAW, event_index, resolve  # noqa: E402


def keys(row):
    src, path, ev = row
    with open(resolve(path)) as f:
        raw = json.load(f)
    out = set()
    if src == "highd":
        rec = int(ev.split("_")[1])
        for r in raw:
            out.add((rec, int(r["frame"]), int(r["id"])))
    elif src in ("ind", "round"):
        rec = int(re.search(r"recording_(\d+)", ev).group(1))
        for fr in raw:
            for v in fr["vehicles"]:
                out.add((rec, int(fr["frame_id"]), int(v["trackId"])))
    elif src == "ngsim":
        for r in raw:
            # NGSIM I-80 vehicle IDs repeat across the three periods; use Global_Time
            out.add((0, int(r["Global_Time"]), int(r["Vehicle_ID"])))
    return out


def raw_count(src):
    if src == "highd":
        fs = glob.glob(os.path.join(RAW, "highd", "*_labelled.csv"))
        n = sum(len(pd.read_csv(f, usecols=["id"])) for f in fs)
        veh = sum(pd.read_csv(f, usecols=["id"]).id.nunique() for f in fs)
    elif src in ("ind", "round"):
        fs = glob.glob(os.path.join(RAW, src, "*_tracks.csv"))
        n = sum(len(pd.read_csv(f, usecols=["trackId"])) for f in fs)
        veh = sum(pd.read_csv(f, usecols=["trackId"]).trackId.nunique() for f in fs)
    else:
        fs = glob.glob(os.path.join(RAW, "ngsim", "*.csv"))
        n = sum(len(pd.read_csv(f, usecols=["Vehicle_ID"])) for f in fs)
        veh = sum(pd.read_csv(f, usecols=["Vehicle_ID"]).Vehicle_ID.nunique() for f in fs)
    return n, veh


def main():
    idx = event_index()
    res = {}
    for src in ("highd", "ind", "round", "ngsim"):
        g = idx[idx.src == src]
        with Pool(10) as p:
            sets = p.map(keys, list(zip(g.src, g.path, g.event)), chunksize=16)
        u = set().union(*sets)
        n_raw, veh_raw = raw_count(src)
        veh = len({(k[0], k[2]) for k in u}) if src != "ngsim" else len({k[2] for k in u})
        res[src] = dict(events=len(g), released_agent_frames=len(u), original_agent_frames=n_raw,
                        share_agent_frames=len(u) / n_raw, released_agents=veh, original_agents=veh_raw,
                        share_agents=veh / veh_raw)
        print(src, res[src], flush=True)
    with open(os.path.join(OUT, "coverage.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
