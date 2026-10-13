"""Extraction audit (R2-5, R1 general comment) and convergent validity of the risk
grades (R2-1, R2-7, R2-9, R1-3).

Audit: for every released HV event, check that the trigger record that defines the
event in its source is present in the released window, and screen
acceleration-triggered CitySim events for tracking artefacts (raw trigger |a| >= 5 but
smoothed |a| < 3 m s^-2).

Validity: compare grades within each source on indicators that the grading rules do
not use (minimum clearance, PET, DRAC, predicted 2-s conflict from the independent
re-implementation). Effect size = Cliff's delta oriented so that > 0 means the higher
grade is riskier; 95% CI by percentile bootstrap (2,000 resamples); two-sided
Mann-Whitney U with Holm correction; Jonckheere-Terpstra trend for the three CitySim
grades.

Outputs: results/audit.json, results/validity_effects.csv, results/validity.json,
         results/event_audit_flags.csv
"""
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm

sys.path.insert(0, os.path.dirname(__file__))
from kr_common import OUT, RAW, resolve, wilson  # noqa: E402

REL = {"precedingId": "preceding", "followingId": "following", "leftPrecedingId": "left_preceding",
       "leftAlongsideId": "left_alongside", "leftFollowingId": "left_following",
       "rightPrecedingId": "right_preceding", "rightAlongsideId": "right_alongside",
       "rightFollowingId": "right_following"}


def highd_relation_check(path):
    """True if the agent in the named relation slot shows the named behaviour flag in
    some frame while it occupies that slot."""
    ev = os.path.basename(path)[:-5]
    m = re.match(r"highd_\d+_(\d+)_(\w+?Id)_(\w+?)_frame", ev)
    ego, rel, beh = int(m.group(1)), m.group(2), m.group(3)
    with open(resolve(path)) as f:
        d = pd.DataFrame(json.load(f))
    e = d[d.id == ego][["frame", rel]]
    e = e[e[rel] > 0]
    fl = d.set_index(["frame", "id"])[beh]
    key = pd.MultiIndex.from_arrays([e.frame.to_numpy(), e[rel].astype(int).to_numpy()])
    v = fl.reindex(key).fillna(False).to_numpy().astype(bool)
    return ev, bool(v.any())


def cliffs_delta(x, y):
    """P(X > Y) - P(X < Y) via ranks (x, y 1-D arrays without NaN)."""
    nx, ny = len(x), len(y)
    allv = np.concatenate([x, y])
    ranks = pd.Series(allv).rank().to_numpy()
    rx = ranks[:nx].sum()
    u = rx - nx * (nx + 1) / 2
    return 2 * u / (nx * ny) - 1


def boot_delta(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(cliffs_delta(rng.choice(x, len(x)), rng.choice(y, len(y))))
    return np.percentile(out, [2.5, 97.5])


def stratum_of(src, ev):
    if src == "highd":
        return ev.split("_")[1]
    if src in ("expresswayA", "freewayB"):
        return re.search(r"track_(\d+)", ev).group(1)
    if src in ("ind", "round"):
        return re.search(r"recording_(\d+)", ev).group(1)
    return "all"


def ngsim_period(path):
    """NGSIM I-80 recording period (1-3) from the Global_Time of the first released record."""
    with open(resolve(path)) as fh:
        t = json.load(fh)[0]["Global_Time"]
    return str(1 if t < 1113435000000 else 2 if t < 1113437560000 else 3)


def stratified_delta(df, col, sgn, hi, lo, n_boot=1000, seed=0):
    """Recording-stratified Cliff's delta: within-stratum deltas pooled with weights
    n_hi * n_lo (the number of pairwise comparisons); bootstrap resamples events within
    strata."""
    rng = np.random.default_rng(seed)
    strata = []
    for _, g in df.groupby("stratum"):
        a = g.loc[g.level == hi, col].dropna().to_numpy(float) * sgn
        b = g.loc[g.level == lo, col].dropna().to_numpy(float) * sgn
        if len(a) >= 3 and len(b) >= 3:
            strata.append((a, b))
    if not strata:
        return np.nan, np.nan, np.nan, 0

    def pooled(st):
        w = np.array([len(a) * len(b) for a, b in st], float)
        d = np.array([cliffs_delta(a, b) for a, b in st])
        return float((w * d).sum() / w.sum())

    est = pooled(strata)
    bs = []
    for _ in range(n_boot):
        bs.append(pooled([(rng.choice(a, len(a)), rng.choice(b, len(b))) for a, b in strata]))
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(strata)


def jonckheere(groups):
    """Jonckheere-Terpstra test for increasing trend across ordered groups (normal approx.)."""
    k = len(groups)
    J = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            u = mannwhitneyu(groups[j], groups[i], alternative="two-sided").statistic
            J += u
    n = np.array([len(g) for g in groups])
    N = n.sum()
    mu = (N ** 2 - (n ** 2).sum()) / 4
    var = (N ** 2 * (2 * N + 3) - (n ** 2 * (2 * n + 3)).sum()) / 72
    z = (J - mu) / np.sqrt(var)
    return float(z), float(2 * norm.sf(abs(z)))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    F = pd.read_csv(os.path.join(OUT, "event_features.csv"), low_memory=False)
    C = pd.read_csv(os.path.join(OUT, "conflict_scan.csv"))
    F = F.merge(C[["event", "pred_conflict", "pred_t_min", "n_pred_partners"]], on="event", how="left")
    audit = {}
    flags = F[["event", "src", "level", "sub"]].copy()
    # ---------------- highD: named relation + behaviour present
    hd = F[F.src == "highd"]
    with Pool(20) as p:
        hres = dict(p.map(highd_relation_check, hd.path.tolist(), chunksize=32))
    flags["trigger_ok"] = np.nan
    m = flags.src == "highd"
    flags.loc[m, "trigger_ok"] = flags.loc[m, "event"].map(hres).astype(float)
    # ---------------- CitySim moderate/high: hard manoeuvre >= 5 m/s^2 by ego or slot agent
    cs = F.src.isin(["expresswayA", "freewayB"]) & (F.level != "extreme")
    trig = np.fmax(F.ego_amax_raw.fillna(0), F.slot_amax_raw.fillna(0)) >= 5
    flags.loc[cs, "trigger_ok"] = trig[cs].astype(float)
    # grade rule: high <=> the ego itself performs it
    g_rule = np.where(F.ego_amax_raw.fillna(0) >= 5, "high", "moderate")
    audit["citysim_grade_rule_agreement"] = float((g_rule[cs] == F.level[cs]).mean())
    # artefact screen on the triggering agent: confirmed if smoothed |a| >= 3
    conf = np.where(F.level == "high", F.ego_amax_s, F.slot_amax_s) >= 3
    flags.loc[cs, "kinematic_confirmed"] = conf[cs].astype(float)
    # ---------------- inD / rounD: DRF >= P90 of the recording
    for s in ("ind", "round"):
        q = {}
        for f in sorted(glob.glob(os.path.join(RAW, s, "*_tracks.csv"))):
            d = pd.read_csv(f, usecols=["recordingId", "risk_value"])
            q[int(d.recordingId.iloc[0])] = d.risk_value.quantile(0.9)
        ms = F.src == s
        rec = F.loc[ms, "event"].str.extract(r"recording_(\d+)")[0].astype(int)
        flags.loc[ms, "trigger_ok"] = (F.loc[ms, "drf_max"].to_numpy() >= rec.map(q).to_numpy()).astype(float)
    # ---------------- NGSIM: TTC < 3 s, high <=> TTC < 1.5 s
    mn = F.src == "ngsim"
    flags.loc[mn, "trigger_ok"] = (F.loc[mn, "ttc_min_pos"] < 3).astype(float)
    audit["ngsim_grade_rule_agreement"] = float(((F.loc[mn, "ttc_min_pos"] < 1.5) == (F.loc[mn, "level"] == "high")).mean())
    # ---------------- extreme: recorded conflict and independent re-check
    mx = F.level == "extreme"
    rec_any = F.coll_rows_any.fillna(0) > 0
    rec_ego = (F.coll2s_frames.fillna(0) > 0) | (F.ego_is_partner.fillna(0) > 0)
    flags.loc[mx, "recorded_conflict_ego"] = rec_ego[mx].astype(float)
    flags.loc[mx, "recorded_conflict_any"] = rec_any[mx].astype(float)
    flags.loc[mx, "predicted_conflict"] = F.loc[mx, "pred_conflict"].astype(float)
    flags.loc[mx, "trigger_ok"] = (rec_any | (F.pred_conflict == 1))[mx].astype(float)
    flags["pred_conflict"] = F.pred_conflict
    flags["pred_t_min"] = F.pred_t_min
    flags.to_csv(os.path.join(OUT, "event_audit_flags.csv"), index=False)
    # summaries
    by = {}
    for (s, l), g in flags.groupby(["src", "level"]):
        k, n = int(g.trigger_ok.sum()), int(g.trigger_ok.notna().sum())
        r = dict(n=int(len(g)), trigger_ok=k, trigger_n=n, trigger_rate=k / n if n else None,
                 ci=list(wilson(k, n)) if n else None)
        if "kinematic_confirmed" in g and g.kinematic_confirmed.notna().any():
            kk, nn = int(g.kinematic_confirmed.sum()), int(g.kinematic_confirmed.notna().sum())
            r.update(kinematic_confirmed=kk / nn, kinematic_ci=list(wilson(kk, nn)))
        if l == "extreme":
            for c in ("recorded_conflict_ego", "recorded_conflict_any", "predicted_conflict"):
                r[c] = float(g[c].mean())
            r["recorded_or_predicted"] = float(((g.recorded_conflict_any == 1) | (g.predicted_conflict == 1)).mean())
            r["ego_recorded_or_predicted"] = float(((g.recorded_conflict_ego == 1) | (g.predicted_conflict == 1)).mean())
        by[f"{s}|{l}"] = r
    audit["by_source_level"] = by
    x = flags[flags.level == "extreme"]
    audit["extreme_overall"] = dict(
        n=len(x), recorded_ego=float(x.recorded_conflict_ego.mean()), recorded_any=float(x.recorded_conflict_any.mean()),
        predicted=float(x.predicted_conflict.mean()),
        recorded_or_predicted=float(((x.recorded_conflict_any == 1) | (x.predicted_conflict == 1)).mean()),
        neither=int(((x.recorded_conflict_any != 1) & (x.predicted_conflict != 1)).sum()))
    allok = flags.trigger_ok.dropna()
    audit["overall_trigger_rate"] = [int(allok.sum()), int(len(allok))]
    # ---------------- convergent validity
    F["pet_c"] = F.pet.fillna(F.dur_s)  # no co-location within the window -> censored at window length
    F["pred_t_c"] = F.pred_t_min.fillna(3.0)  # no predicted overlap within 2 s -> censored above horizon
    ind = [("min_clear", -1, "Minimum clearance"), ("pet_c", -1, "PET"), ("drac", 1, "DRAC"),
           ("pred_conflict", 1, "Predicted 2-s conflict")]
    F["stratum"] = [stratum_of(s_, e_) for s_, e_ in zip(F.src, F.event)]
    mn = F.src == "ngsim"
    F.loc[mn, "stratum"] = [ngsim_period(p_) for p_ in F.loc[mn, "path"]]
    rows = []
    for s in ("highd", "ind", "round", "ngsim", "expresswayA", "freewayB"):
        g = F[F.src == s]
        pairs = [("high", "moderate")]
        if s in ("expresswayA", "freewayB"):
            pairs += [("extreme", "high"), ("extreme", "moderate")]
        for hi, lo in pairs:
            for col, sgn, name in ind:
                a = g.loc[g.level == hi, col].dropna().to_numpy(float) * sgn
                b = g.loc[g.level == lo, col].dropna().to_numpy(float) * sgn
                if len(a) < 10 or len(b) < 10:
                    continue
                dlt = cliffs_delta(a, b)
                lo_ci, hi_ci = boot_delta(a, b, n=1000)
                p = mannwhitneyu(a, b, alternative="two-sided").pvalue
                sd, slo, shi, ns = stratified_delta(g, col, sgn, hi, lo)
                rows.append(dict(src=s, contrast=f"{hi}>{lo}", indicator=name, delta=dlt, ci_lo=lo_ci, ci_hi=hi_ci,
                                 sdelta=sd, sci_lo=slo, sci_hi=shi, n_strata=ns,
                                 p=p, n_hi=len(a), n_lo=len(b),
                                 med_hi=float(np.median(a * sgn)), med_lo=float(np.median(b * sgn))))
    E = pd.DataFrame(rows)
    E["p_holm"] = holm(E.p.to_numpy())
    E.to_csv(os.path.join(OUT, "validity_effects.csv"), index=False)
    V = {}
    for s in ("expresswayA", "freewayB"):
        g = F[F.src == s]
        for col, sgn, name in ind:
            groups = [g.loc[g.level == l, col].dropna().to_numpy(float) * sgn for l in ("moderate", "high", "extreme")]
            z, p = jonckheere(groups)
            V[f"{s}|{name}"] = dict(z=z, p=p, medians=[float(np.median(x_ * sgn)) for x_ in groups])
    V["pred_conflict_rate"] = {f"{s}|{l}": [int(g.pred_conflict.sum()), int(g.pred_conflict.notna().sum())]
                               for (s, l), g in F.groupby(["src", "level"])}
    with open(os.path.join(OUT, "validity.json"), "w") as f:
        json.dump(V, f, indent=2)
    with open(os.path.join(OUT, "audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit, indent=2))
    print(E.round(3).to_string())
    print(json.dumps(V, indent=2))


if __name__ == "__main__":
    main()
