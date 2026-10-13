"""Visual judgements for the 100-event spot-check sample (rendered by
09_spotcheck_render.py) and their agreement with the automated audit flags.

Judgements were recorded while inspecting figures/spotcheck/sheet_*.png:
  V_trigger   the labelled trigger (manoeuvre, DRF exceedance, TTC or conflict record)
              is visible in the window
  V_agent     the agent carrying the trigger is plausibly placed relative to the ego
  V_clean     the trigger is not produced by a tracking artefact (jump, single-frame
              spike or periodic jitter)
Output: results/spotcheck_judgements.csv, results/spotcheck_summary.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from kr_common import OUT, wilson  # noqa: E402

S = pd.read_csv(os.path.join(OUT, "spotcheck_sample.csv"))
V = pd.DataFrame({"n": range(1, 101)})
V["V_trigger"], V["V_agent"], V["V_clean"] = 1, 1, 1
ngsim_bad_agent = []  # NGSIM sample redrawn from the current release (09b); all agents correctly placed
artefact = [68, 69, 70, 71, 72, 76, 85, 90, 95]
no_trigger = [79, 97, 99, 100]
V.loc[V.n.isin(ngsim_bad_agent), "V_agent"] = 0
V.loc[V.n.isin(artefact), "V_clean"] = 0
V.loc[V.n.isin(no_trigger), ["V_trigger", "V_agent"]] = 0
V = S.merge(V, on="n")
A = pd.read_csv(os.path.join(OUT, "event_audit_flags.csv"))
V = V.merge(A[["event", "trigger_ok", "kinematic_confirmed", "recorded_conflict_any", "predicted_conflict"]],
            on="event", how="left")
V.to_csv(os.path.join(OUT, "spotcheck_judgements.csv"), index=False)


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in (0, 1))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


out = {"n": len(V)}
for c in ("V_trigger", "V_agent", "V_clean"):
    k = int(V[c].sum())
    out[c] = [k, len(V), list(wilson(k, len(V)))]
allok = (V.V_trigger == 1) & (V.V_agent == 1) & (V.V_clean == 1)
out["all_three"] = [int(allok.sum()), len(V), list(wilson(int(allok.sum()), len(V)))]
out["by_source"] = V.assign(ok=allok).groupby("src").ok.agg(["sum", "size"]).to_dict("index")
# agreement of visual trigger judgement with the automated rule check
m = V.trigger_ok.notna()
out["trigger_agreement"] = dict(agree=float((V.V_trigger[m] == V.trigger_ok[m]).mean()),
                                kappa=kappa(V.V_trigger[m], V.trigger_ok[m]))
# extreme events judged without a visible record: independent predictor
x = V[V.level == "extreme"]
out["extreme_visual_record"] = [int(x.V_trigger.sum()), len(x)]
out["extreme_record_or_predicted"] = [int(((x.recorded_conflict_any == 1) | (x.predicted_conflict == 1)).sum()), len(x)]
# artefact judgement vs automated kinematic confirmation (CitySim moderate/high)
c = V[V.kinematic_confirmed.notna()]
out["artefact_vs_kinematic"] = dict(n=len(c), agree=float((c.V_clean == c.kinematic_confirmed).mean()),
                                    kappa=kappa(c.V_clean, c.kinematic_confirmed),
                                    visual_clean=int(c.V_clean.sum()), kinematic_confirmed=int(c.kinematic_confirmed.sum()))
with open(os.path.join(OUT, "spotcheck_summary.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
