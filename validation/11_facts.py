"""Collects the remaining descriptive numbers quoted in the revision (fact sheet)."""
import glob, json, os, re, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from kr_common import OUT, RAW, ROOT, FPS, event_index

F = pd.read_csv(os.path.join(OUT, "event_features.csv"), low_memory=False)
C = pd.read_csv(os.path.join(OUT, "conflict_scan.csv"))
F = F.merge(C[["event", "pred_conflict", "n_pred_partners"]], on="event")
facts = {}
idx = event_index()
facts["extreme_sub_counts"] = idx[idx.level == "extreme"].groupby(["src", "sub"]).size().to_dict()
facts["extreme_sub_counts"] = {f"{k[0]}|{k[1]}": int(v) for k, v in facts["extreme_sub_counts"].items()}
# window durations
facts["window_s"] = {s: dict(median=float(g.dur_s.median()), p05=float(g.dur_s.quantile(.05)), p95=float(g.dur_s.quantile(.95)))
                     for s, g in F.groupby("src")}
# multi-agent attribution
hd = F[F.src == "highd"]
facts["highd_flagged_slot_agents_ge2"] = float((hd.n_flagged_slot_agents >= 2).mean())
facts["highd_flagged_slot_agents_ge1"] = float((hd.n_flagged_slot_agents >= 1).mean())
cs = F[F.src.isin(["expresswayA", "freewayB"])]
facts["citysim_flagged_slot_agents_ge2"] = float((cs.n_flagged_slot_agents >= 2).mean())
pc = F[F.pred_conflict == 1]
facts["pred_conflict_events_with_ge2_partners"] = {l: float((g.n_pred_partners >= 2).mean()) for l, g in pc.groupby("level")}
x = F[F.level == "extreme"]
facts["extreme_recorded_partners_ge2"] = float((x.n_coll_partners >= 2).mean())
facts["extreme_pred_partners_ge2"] = float((x.n_pred_partners >= 2).mean())
# slot coverage inD/rounD (all events)
for s in ("ind", "round"):
    g = F[F.src == s]
    facts[f"{s}_slots"] = dict(others_per_frame=float(g.others_per_frame.mean()), slot_per_frame=float(g.slot_per_frame.mean()),
                              nonslot_med_dist=float(g.nonslot_med_dist.median()), nonslot_within20m=float(g.nonslot_within20m_per_frame.mean()),
                              vru_per_frame=float(g.vru_per_frame.mean()), closest_in_slot=float(g.closest_in_slot.mean()))
facts["closest_agent_in_slot_by_src"] = {s: float(g.closest_in_slot.mean()) for s, g in F.groupby("src")}
# extreme conflict partner in slot
facts["extreme_partner_in_slot_frac"] = float(x.coll_partner_in_slot_frac[x.coll2s_frames > 0].mean())
# LLM prompt lengths
D = os.path.join(ROOT, "llm_analysis", "event_discriptions")
lens = []
for fn in os.listdir(D):
    t = open(os.path.join(D, fn), encoding="utf8").read()
    fr = list(re.finditer(r"Frame (\d+):", t))
    lens.append(len(t[fr[-1].start():]))
facts["llm_input_chars"] = dict(mean=float(np.mean(lens)), median=float(np.median(lens)), max=int(np.max(lens)), min=int(np.min(lens)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_MESSAGE
facts["system_prompt_chars"] = len(SYSTEM_MESSAGE)
# highD acceleration distribution (complete recordings)
def acc(f):
    d = pd.read_csv(f, usecols=["xAcceleration", "acc_rate_1s"])
    a = d.xAcceleration.abs()
    return len(d), int((a > 3).sum()), int((a > 2).sum()), int((a > 1).sum())
with Pool(12) as p:
    r = np.array(p.map(acc, sorted(glob.glob(os.path.join(RAW, "highd", "*_labelled.csv")))))
n = r[:, 0].sum()
facts["highd_abs_acc_gt3_frac"] = float(r[:, 1].sum() / n)
facts["highd_abs_acc_gt2_frac"] = float(r[:, 2].sum() / n)
facts["highd_abs_acc_gt1_frac"] = float(r[:, 3].sum() / n)
# highD per-recording fitted thresholds (onset of acc/brake/lateral flags)
facts["highd_ttc3_frac"] = None
H = pd.read_csv(os.path.join(OUT, "highd_frame_sets.csv"))
facts["highd_ttc3_frac"] = float(H.ttc3.sum() / H.n.sum())
facts["highd_ttc2_frac"] = float(H.ttc2.sum() / H.n.sum())
K = pd.read_csv(os.path.join(OUT, "sensitivity_checker.csv")).set_index("tag")
facts["checker_base"] = K.loc["base"].to_dict()
json.dump(facts, open(os.path.join(OUT, "facts.json"), "w"), indent=2, default=float)
print(json.dumps(facts, indent=2, default=float))
