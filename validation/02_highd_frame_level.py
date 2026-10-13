"""Frame-level analysis on the complete highD recordings (60 tracks, all vehicles).

Answers R1-3: how the driver risk field (DRF) and TTC complement each other, and
how rare the frames that satisfy the joint screening conditions are in routine
traffic.

Definitions (per recording, as in the extraction):
  DRF10 / DRF1   ego DRF in the top 10% / 1% of all frames of that recording
  TTC defined    ttc > 0 (closing on a preceding vehicle); otherwise undefined
  BEH            behaviour flag (acc_high, brake_high, yaw_left, yaw_right) on the
                 ego or on any of its eight slot agents in the same frame

Outputs: results/highd_frame_level.json, results/highd_frame_sets.csv
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from kr_common import OUT, RAW  # noqa: E402

COLS = ["frame", "id", "total_risk", "ttc", "precedingId", "followingId", "leftPrecedingId",
        "leftAlongsideId", "leftFollowingId", "rightPrecedingId", "rightAlongsideId",
        "rightFollowingId", "acc_high", "brake_high", "yaw_left", "yaw_right", "xVelocity"]
SLOTC = COLS[4:12]


def one(path):
    d = pd.read_csv(path, usecols=COLS)
    for c in ("acc_high", "brake_high", "yaw_left", "yaw_right"):
        d[c] = d[c].fillna(False).astype(bool)
    d["beh_self"] = d[["acc_high", "brake_high", "yaw_left", "yaw_right"]].any(axis=1)
    q90, q99 = d.total_risk.quantile([0.9, 0.99]).to_numpy()
    flag = d.set_index(["frame", "id"]).beh_self
    beh = d.beh_self.to_numpy().copy()
    for c in SLOTC:
        key = pd.MultiIndex.from_arrays([d.frame.to_numpy(), d[c].to_numpy()])
        beh |= flag.reindex(key).fillna(False).to_numpy().astype(bool)
    d["beh"] = beh
    d["drf10"] = d.total_risk >= q90
    d["drf1"] = d.total_risk >= q99
    d["ttc_def"] = d.ttc > 0
    d["ttc5"] = d.ttc_def & (d.ttc < 5)
    d["ttc3"] = d.ttc_def & (d.ttc < 3)
    d["ttc2"] = d.ttc_def & (d.ttc < 2)
    n = len(d)
    r = dict(track=os.path.basename(path)[6:8], n=n, q90=float(q90), q99=float(q99),
             drf_zero=int((d.total_risk == 0).sum()))
    for k in ("drf10", "drf1", "ttc_def", "ttc5", "ttc3", "ttc2", "beh"):
        r[k] = int(d[k].sum())
    m = d.drf10
    r["drf10_ttc_undef"] = int((m & ~d.ttc_def).sum())
    r["drf10_ttc_gt5"] = int((m & d.ttc_def & (d.ttc >= 5)).sum())
    r["drf10_ttc5"] = int((m & d.ttc5).sum())
    r["ttc5_not_drf10"] = int((d.ttc5 & ~m).sum())
    r["drf10_beh"] = int((m & d.beh).sum())
    r["drf10_beh_ttc5"] = int((m & d.beh & d.ttc5).sum())
    r["drf1_beh"] = int((d.drf1 & d.beh).sum())
    # 3-set Venn counts (DRF10, TTC5, BEH)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                r[f"venn_{a}{b}{c}"] = int(((d.drf10 == a) & (d.ttc5 == b) & (d.beh == c)).sum())
    # Spearman between DRF and closing intensity 1/TTC (0 when TTC undefined)
    inv = np.where(d.ttc_def, 1.0 / d.ttc.clip(lower=1e-3), 0.0)
    pos = d.total_risk > 0
    rho_all = spearmanr(d.total_risk[pos], inv[pos]).correlation if pos.sum() > 10 else np.nan
    both = pos & d.ttc_def
    rho_def = spearmanr(d.total_risk[both], inv[both]).correlation if both.sum() > 10 else np.nan
    r["rho_drfpos"] = float(rho_all)
    r["rho_both_def"] = float(rho_def)
    r["n_both_def"] = int(both.sum())
    # subsample for pooled correlation and plots
    s = d[pos].sample(min(4000, int(pos.sum())), random_state=0)[["total_risk", "ttc"]]
    s["q90"], s["q99"], s["track"] = q90, q99, r["track"]
    return r, s


def main():
    files = sorted(glob.glob(os.path.join(RAW, "highd", "highd_*_labelled.csv")))
    with Pool(8) as p:
        res = p.map(one, files)
    rows = pd.DataFrame([r for r, _ in res])
    samp = pd.concat([s for _, s in res])
    rows.to_csv(os.path.join(OUT, "highd_frame_sets.csv"), index=False)
    samp.to_csv(os.path.join(OUT, "highd_frame_sample.csv"), index=False)
    tot = rows.drop(columns=["track"]).sum(numeric_only=True)
    N = tot.n
    summ = dict(
        n_tracks=len(rows), n_frames=int(N),
        drf_zero_frac=tot.drf_zero / N,
        drf10_frac=tot.drf10 / N, ttc5_frac=tot.ttc5 / N, ttc_def_frac=tot.ttc_def / N,
        beh_frac=tot.beh / N,
        drf10_ttc_undef_frac=tot.drf10_ttc_undef / tot.drf10,
        drf10_ttc_gt5_frac=tot.drf10_ttc_gt5 / tot.drf10,
        drf10_ttc_missed_frac=(tot.drf10_ttc_undef + tot.drf10_ttc_gt5) / tot.drf10,
        ttc5_not_drf10_frac=tot.ttc5_not_drf10 / tot.ttc5,
        joint_drf10_beh_frac=tot.drf10_beh / N,
        joint_drf10_beh_ttc5_frac=tot.drf10_beh_ttc5 / N,
        joint_drf1_beh_frac=tot.drf1_beh / N,
        rho_drfpos_median=float(rows.rho_drfpos.median()),
        rho_drfpos_iqr=[float(rows.rho_drfpos.quantile(0.25)), float(rows.rho_drfpos.quantile(0.75))],
        rho_both_def_median=float(rows.rho_both_def.median()),
        rho_both_def_iqr=[float(rows.rho_both_def.quantile(0.25)), float(rows.rho_both_def.quantile(0.75))],
        venn={k: int(tot[k]) for k in tot.index if k.startswith("venn_")},
    )
    with open(os.path.join(OUT, "highd_frame_level.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
