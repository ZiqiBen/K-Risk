"""Sensitivity analyses (R2-9, R1-3).

A. Threshold sensitivity of the released grades. Each source's operative grading rule
   is re-applied to the released per-event values with the threshold perturbed in both
   directions; we report the share of events whose grade changes and, for the
   retention criteria, the share of events that would remain.
B. Sensitivity of the action-conditioned checker to its horizon and action magnitudes:
   whether the ordering moderate < high < extreme of the unsafe-action count and of
   the IDLE-unsafe rate is preserved.

Outputs: results/sensitivity_grades.csv, results/sensitivity_checker.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from kr_common import OUT, RAW  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("av", os.path.join(os.path.dirname(__file__), "06_audit_validity.py"))
av = importlib.util.module_from_spec(spec)
spec.loader.exec_module(av)


def pooled_quantile_fn(values, zero_frac=None):
    v = np.sort(np.asarray(values, float))
    if zero_frac is None:
        return lambda p: float(np.quantile(v, p))
    return lambda p: 0.0 if p <= zero_frac else float(np.quantile(v, (p - zero_frac) / (1 - zero_frac)))


def main():
    F = pd.read_csv(os.path.join(OUT, "event_features.csv"), low_memory=False)
    C = pd.read_csv(os.path.join(OUT, "conflict_scan.csv"))
    F = F.merge(C[["event", "pred_t_min"]], on="event", how="left")
    rows = []
    # ---- DRF-graded sources: high <=> event DRF >= pooled P_q of all frames
    S = pd.read_csv(os.path.join(OUT, "highd_frame_sample.csv"))
    qf = {"highd": pooled_quantile_fn(S.total_risk, zero_frac=0.708)}
    for s in ("ind", "round"):
        vals = np.concatenate([pd.read_csv(f, usecols=["risk_value"]).risk_value.dropna().to_numpy()
                               for f in glob.glob(os.path.join(RAW, s, "*_tracks.csv"))])
        qf[s] = pooled_quantile_fn(vals)
    for s in ("highd", "ind", "round"):
        g = F[F.src == s]
        base = g.drf_max >= qf[s](0.99)
        agree = float((base == (g.level == "high")).mean())
        for q in (0.98, 0.985, 0.995, 0.998):
            alt = g.drf_max >= qf[s](q)
            rows.append(dict(src=s, parameter="DRF percentile for high", baseline="top 1%",
                             value=f"top {100 * (1 - q):.1f}%", regraded=float((alt != base).mean()),
                             rule_agreement=agree))
        if s in ("ind", "round"):
            q90 = {}
            for f in glob.glob(os.path.join(RAW, s, "*_tracks.csv")):
                d = pd.read_csv(f, usecols=["recordingId", "risk_value"])
                q90[int(d.recordingId.iloc[0])] = d.risk_value
            rec = g.event.str.extract(r"recording_(\d+)")[0].astype(int)
            for q in (0.875, 0.925, 0.95):
                thr = rec.map({r: v.quantile(q) for r, v in q90.items()})
                rows.append(dict(src=s, parameter="DRF percentile for retention", baseline="top 10%",
                                 value=f"top {100 * (1 - q):.1f}%", retained=float((g.drf_max.to_numpy() >= thr.to_numpy()).mean())))
    # ---- CitySim: hard-manoeuvre magnitude
    for s in ("expresswayA", "freewayB"):
        g = F[(F.src == s) & (F.level != "extreme")]
        ea, sa = g.ego_amax_raw.fillna(0), g.slot_amax_raw.fillna(0)
        base = np.where(ea >= 5, "high", "moderate")
        for thr in (4.5, 5.5, 6.0):
            alt = np.where(ea >= thr, "high", "moderate")
            rows.append(dict(src=s, parameter="|a| for high (ego)", baseline="5.0 m/s2", value=f"{thr} m/s2",
                             regraded=float((alt != base).mean()),
                             retained=float((np.fmax(ea, sa) >= thr).mean()),
                             rule_agreement=float((base == g.level).mean())))
        # smoothed-kinematics variant: grades if only kinematically confirmed triggers count
    # ---- NGSIM: TTC thresholds
    g = F[F.src == "ngsim"]
    base = g.ttc_min_pos < 1.5
    for thr in (1.25, 1.75, 2.0):
        rows.append(dict(src="ngsim", parameter="TTC for high", baseline="1.5 s", value=f"{thr} s",
                         regraded=float(((g.ttc_min_pos < thr) != base).mean()),
                         rule_agreement=float((base == (g.level == "high")).mean())))
    for thr in (2.5, 2.75):
        rows.append(dict(src="ngsim", parameter="TTC for retention", baseline="3.0 s", value=f"{thr} s",
                         retained=float((g.ttc_min_pos < thr).mean())))
    # ---- extreme: conflict horizon of the independent predictor
    g = F[F.level == "extreme"]
    for h in (1.0, 1.5):
        rows.append(dict(src="extreme", parameter="conflict horizon", baseline="2.0 s", value=f"{h} s",
                         retained=float((g.pred_t_min <= h).mean()),
                         note="share of extreme events with a predicted overlap within the horizon"))
    G = pd.DataFrame(rows)
    G.to_csv(os.path.join(OUT, "sensitivity_grades.csv"), index=False)
    print(G.round(3).to_string())
    # ---- B. checker sensitivity
    out = []
    for tag in ("base", "h15", "h30", "d3", "d6", "v075", "v15", "a1", "a3"):
        p = os.path.join(OUT, f"checker_{tag}.csv")
        if not os.path.exists(p):
            continue
        R = pd.read_csv(p)
        R = R[R.n_unsafe.notna()]
        rec = dict(tag=tag, n=len(R))
        groups_n, groups_i = [], []
        for l in ("moderate", "high", "extreme"):
            g = R[R.level == l]
            rec[f"{l}_n_unsafe"] = float(g.n_unsafe.mean())
            rec[f"{l}_idle_unsafe"] = float(g.unsafe_IDLE.mean())
            rec[f"{l}_solvable"] = float((g.n_unsafe < 5).mean())
            rec[f"{l}_discriminative"] = float(((g.n_unsafe > 0) & (g.n_unsafe < 5)).mean())
            groups_n.append(g.n_unsafe.to_numpy(float))
            groups_i.append(g.unsafe_IDLE.to_numpy(float))
        rec["jt_z_n_unsafe"], rec["jt_p_n_unsafe"] = av.jonckheere(groups_n)
        rec["jt_z_idle"], rec["jt_p_idle"] = av.jonckheere(groups_i)
        rec["ordered"] = bool(rec["moderate_n_unsafe"] < rec["high_n_unsafe"] < rec["extreme_n_unsafe"]
                              and rec["moderate_idle_unsafe"] < rec["high_idle_unsafe"] < rec["extreme_idle_unsafe"])
        for a in ("IDLE", "LEFT", "RIGHT", "ACC", "DEC"):
            for l in ("moderate", "high", "extreme"):
                rec[f"{l}_unsafe_{a}"] = float(R[R.level == l][f"unsafe_{a}"].mean())
        out.append(rec)
    K = pd.DataFrame(out)
    K.to_csv(os.path.join(OUT, "sensitivity_checker.csv"), index=False)
    print(K.round(3).T.to_string())


if __name__ == "__main__":
    main()
