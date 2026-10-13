"""Zero-shot vs. ICL (failure-case prompt) safety on the events not used as ICL cases.

Verdicts come from results/checker_fault.csv (21_failure_cases.py): at-fault collision or
unnecessary braking within the 2-s replay. Paired comparison per model with an exact McNemar test.

Run: python 22_eval_icl.py --icl-dir <stage folder of 13_regenerate_llm.py --icl ...>
"""
import argparse
import os
import re
from math import comb

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("KR_OUT", os.path.join(HERE, "results"))
LLM = os.path.join(os.environ.get("KR_ROOT", os.path.join(os.path.abspath(os.path.join(HERE, "..", "..")), "K-Risk_data")),
                   "llm_analysis")
AMAP = {1: "IDLE", 2: "LEFT", 3: "RIGHT", 4: "ACC", 5: "DEC"}
FOLDERS = {"gpt4o": "analysis_gpt4o", "gpt41": "analysis_gpt4.1"}


def decision(path):
    if not os.path.exists(path):
        return None
    r = open(path, encoding="utf-8").read()
    k = re.findall(r"Final Decision[^0-9]{0,20}(\d)", r) or re.findall(r"####\s*(\d)", r)
    return AMAP.get(int(k[-1])) if k else None


def mcnemar(a, b):
    b10, b01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = b10 + b01
    return b10, b01, (min(1.0, 2 * sum(comb(n, k) for k in range(min(b10, b01) + 1)) / 2 ** n) if n else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icl-dir", help="single ICL condition (label 'ICL')")
    ap.add_argument("--icl", nargs="+", default=[], help="label=dir pairs, e.g. ICL-20=... ICL-40=...")
    ap.add_argument("--selected", default=os.path.join(OUT, "icl_selected.csv"),
                    help="events excluded from the test set (the largest ICL case set)")
    ap.add_argument("--zs-dir", default=LLM)
    ap.add_argument("--tag", default="")
    P = ap.parse_args()
    conds = [("zero-shot", P.zs_dir)] + ([("ICL", P.icl_dir)] if P.icl_dir else []) + [
        tuple(x.split("=", 1)) for x in P.icl]
    V = pd.read_csv(os.path.join(OUT, "checker_fault.csv")).set_index("event")
    sel = pd.read_csv(P.selected)
    test = [e for e in V.index if e not in set(sel.event)]
    # events of the same recording whose window overlaps an ICL case (shared scene)
    pat = re.compile(r"(\w+?)_track_(\d+)_car_\d+_frame_(\d+)_to_(\d+)")
    win = [pat.search(e).groups() for e in sel.event]
    def shared(e):
        s, t, a, b = pat.search(e).groups()
        return any(s == s2 and t == t2 and int(a) <= int(b2) and int(a2) <= int(b) for s2, t2, a2, b2 in win)
    rows = []
    for m, fol in FOLDERS.items():
        for e in test:
            fn = f"response_extreme_desc_{e}.txt"
            for cond, root in conds:
                act = decision(os.path.join(root, fol, fn))
                rows.append(dict(model=m, event=e, cond=cond, action=act, shared=shared(e),
                                 unsafe=V.loc[e, f"unsafe_{act}"] if act else np.nan,
                                 any=V.loc[e, f"any_{act}"] if act else np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, f"icl_eval{P.tag}.csv"), index=False)
    print(f"test events {len(test)} (ICL cases excluded: {len(sel)}) | sharing a scene with an ICL case: "
          f"{int(pd.Series([shared(e) for e in test]).sum())}")
    print("missing / unparsable:", R[R.action.isna()].groupby(["model", "cond"]).size().to_dict())
    print(f"baselines on test: always IDLE {V.loc[test, 'unsafe_IDLE'].mean():.3f} | always DEC {V.loc[test, 'unsafe_DEC'].mean():.3f}")
    for name, sub in (("all test", R), ("no shared scene", R[~R.shared])):
        print(f"\n[{name}]")
        for m in FOLDERS:
            W = sub[sub.model == m].pivot(index="event", columns="cond", values="unsafe").dropna()
            print(f"  {m}: n={len(W)} unsafe " + " | ".join(f"{c} {W[c].mean():.3f}" for c, _ in conds))
            names = [c for c, _ in conds]
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    b10, b01, p = mcnemar(W[a].values, W[b].values)
                    print(f"     {a} -> {b}: fixed {b10}, broken {b01}, McNemar p={p:.4f}")
    print("\naction distribution (all test):")
    print(R.groupby(["model", "cond"]).action.value_counts(normalize=True).unstack().round(3).to_string())


if __name__ == "__main__":
    main()
