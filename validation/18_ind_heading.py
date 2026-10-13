"""Heading difference between the inD ego vehicle and its left-preceding agent.

Used in the response to Reviewer 1, Comment 2 ("the median heading difference
between the ego vehicle and its left-preceding agent is 175 deg"). For every
frame of every released inD event in which the ego vehicle has a left-preceding
agent, the absolute heading difference (deg, wrapped to [0, 180]) is computed.
Prints the median over frames and the median of the per-event medians.
"""
import json, math, os, re, statistics as st, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("KR_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "K-Risk_data")), "event_annotations", "HV", "ind")

frame_diffs, event_medians = [], []
for root, _, files in os.walk(ROOT):
    for f in files:
        m = re.search(r"ego_(\d+)_frame_(\d+)_to_(\d+)", f)
        if not m:
            continue
        ego = float(m.group(1))
        diffs = []
        for fr in json.load(open(os.path.join(root, f))):
            agents = {v["trackId"]: v for v in fr["vehicles"]}
            e = agents.get(ego)
            if not e:
                continue
            lp = e.get("left_preceding_id")
            if lp is None or (isinstance(lp, float) and math.isnan(lp)) or lp not in agents:
                continue
            diffs.append(abs((agents[lp]["heading"] - e["heading"] + 180) % 360 - 180))
        frame_diffs += diffs
        if diffs:
            event_medians.append(st.median(diffs))

print(f"frames {len(frame_diffs)}, median {st.median(frame_diffs):.1f} deg")
print(f"events {len(event_medians)}, median of event medians {st.median(event_medians):.1f} deg")
