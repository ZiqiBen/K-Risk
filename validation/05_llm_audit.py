"""Automated audit of the 372 LLM annotations (GPT-4o and GPT-4.1), R2-3/R2-4/R1-7.

All checks are deterministic and run against the exact text that was sent to the
model (the last described frame, as in llm_test.ipynb::extract_last_frame_text).

  entity grounding     every vehicle ID cited in the response exists in the input
  numeric grounding    every operand of every TTC expression appears in the input
  arithmetic           the stated TTC equals the value implied by its own operands
  direction            sign of the stated TTC agrees with whether the pair is actually
                       closing (positions projected on the ego heading); the prompt
                       formula assumes motion towards +x
  decision relevance   direction errors that flip the prompt's 5-s pruning rule
  rule compliance      the final action is admissible under the system-prompt pruning
                       rules evaluated with direction-correct TTCs and lane legality
  attribution recall   share of vehicles named in the rule-based safety reminders that
                       the response explicitly analyses
  agreement            GPT-4o vs GPT-4.1 final actions (Cohen's kappa)

Output: results/llm_audit.json, results/llm_audit_per_event.csv
"""
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from kr_common import OUT, ROOT, wilson  # noqa: E402

DESC = os.path.join(ROOT, "llm_analysis", "event_discriptions")
# responses are read from KR_LLM_ROOT (default: the release); outputs carry KR_TAG
LLM = os.environ.get("KR_LLM_ROOT", os.path.join(ROOT, "llm_analysis"))
TAG = os.environ.get("KR_TAG", "")
MODELS = {"gpt4o": "analysis_gpt4o", "gpt41": "analysis_gpt4.1"}
ROLE_TXT = {
    "driving directly ahead in your lane": "preceding",
    "driving behind in your lane": "following",
    "ahead in your adjacent left lane": "left_preceding",
    "behind in your adjacent left lane": "left_following",
    "laterally aligned in your adjacent left lane": "left_alongside",
    "ahead in your adjacent right lane": "right_preceding",
    "behind in your adjacent right lane": "right_following",
    "laterally aligned in your adjacent right lane": "right_alongside",
}
N = r"(-?\d+\.?\d*)"
TTC_EXPR = re.compile(r"\(\s*" + N + r"\s*[-−]\s*" + N + r"\s*\)\s*/\s*\(\s*" + N + r"\s*[-−]\s*" + N
                      + r"\s*\)((?:\s*(?:=|≈)\s*\**\s*\(?-?\d+\.?\d*\)?\s*(?:/\s*\(?-?\d+\.?\d*\)?)?)+)")
NUM = re.compile(r"-?\d+\.\d+")


def read(p):
    with open(p, encoding="utf8") as f:
        return f.read()


def last_frame(text):
    fr = list(re.finditer(r"Frame (\d+):", text))
    return text[fr[-1].start():], int(fr[-1].group(1))


def parse_frame(lf):
    ego = re.search(r"Ego Vehicle (\d+) is currently traveling in lane (\d+), classified as a ([a-z-]+) direction"
                    r".*?legal maneuvers for this lane include: ([^\n]*)\n.*?coordinates \((-?[\d.]+), (-?[\d.]+)\), "
                    r"moving at a speed of ([\d.]+) m/s.*?heading angle is (-?[\d.]+) degrees", lf, re.S)
    E = dict(id=int(ego.group(1)), lane=int(ego.group(2)), dir=ego.group(3), legal=ego.group(4).lower(),
             x=float(ego.group(5)), y=float(ego.group(6)), v=float(ego.group(7)), h=float(ego.group(8)))
    V = {}
    for m in re.finditer(r"Vehicle (\d+) is ([a-z ]+?): lane (\d+).*?located at \((-?[\d.]+), (-?[\d.]+)\), "
                         r"moving at a speed of ([\d.]+) m/s.*?heading angle is (-?[\d.]+) degrees", lf, re.S):
        role = ROLE_TXT.get(m.group(2).strip())
        V[int(m.group(1))] = dict(role=role, x=float(m.group(4)), y=float(m.group(5)), v=float(m.group(6)),
                                  h=float(m.group(7)))
    rem = lf.split("Safety Reminders:")[1] if "Safety Reminders:" in lf else ""
    rem_ids = set(int(i) for i in re.findall(r"vehicle (\d+)", rem, re.I))
    ttc_warn = "Time-To-Collision" in rem or "collision course" in rem or "collision risk" in rem
    return E, V, rem_ids, ttc_warn, rem


def true_ttc(E, v):
    """Direction-aware TTC of a longitudinal neighbour (inf if not closing)."""
    hx, hy = math.cos(math.radians(E["h"])), math.sin(math.radians(E["h"]))
    lon = (v["x"] - E["x"]) * hx + (v["y"] - E["y"]) * hy  # + = ahead of ego
    if v["role"] and "preceding" in v["role"]:
        gap, close = lon, E["v"] - v["v"]
    elif v["role"] and "following" in v["role"]:
        gap, close = -lon, v["v"] - E["v"]
    else:
        return None
    if gap > 0 and close > 1e-6:
        return gap / close
    return math.inf


def admissible(E, V, ttcs):
    """Action set allowed by the system-prompt pruning rules (direction-correct TTC)."""
    allowed = {1, 2, 3, 4, 5}
    if "left" not in E["legal"] or "(illegal)" in E["legal"].split("left")[-1][:15]:
        allowed.discard(2)
    if "right" not in E["legal"]:
        allowed.discard(3)
    imminent_ahead = False
    for vid, v in V.items():
        r, t = v["role"], ttcs.get(vid)
        if r == "preceding" and t is not None and t < 5:
            allowed -= {1, 4}
            imminent_ahead = True
        if r in ("left_preceding", "left_following") and t is not None and t < 5:
            allowed.discard(2)
        if r in ("right_preceding", "right_following") and t is not None and t < 5:
            allowed.discard(3)
        if r == "left_alongside":
            allowed.discard(2)
        if r == "right_alongside":
            allowed.discard(3)
    for vid, v in V.items():
        if v["role"] == "following" and ttcs.get(vid) is not None and ttcs[vid] < 5 and not imminent_ahead:
            allowed.discard(5)
    return allowed

SUB = re.compile(r"(?<![\d.])(-?\d+\.?\d*)\s*[-\u2212]\s*(-?\d+\.?\d*)\s*=\s*\**\s*(-?\d+\.?\d*)")
DIV = re.compile(r"(?<![\d.])(-?\d+\.?\d*)\s*/\s*\(?\s*(-?\d+\.?\d*)\s*\)?\s*(?:=|\u2248)\s*\**\s*(-?\d+\.?\d*)")
NEG = re.compile(r"not closing|no ttc|ttc is not defined|ttc is undefined|undefined|not defined|not approaching|"
                 r"moving away|no closing|not applicable|no collision risk|no risk of collision|negative|diverging|"
                 r"opening|increasing gap|n/a", re.I)


def statements(resp, V, ttcs, in_nums):
    """Format-agnostic checks of the numerical reasoning in one response."""
    ar_n = ar_ok = op_n = op_ok = 0
    for mt in SUB.finditer(resp):
        x, y, z = map(float, mt.groups())
        ar_n += 1
        ar_ok += abs((x - y) - z) <= max(0.02, 0.01 * abs(z))
        op_n += 1
        op_ok += all(any(abs(v - q) <= 0.011 for q in in_nums) for v in (x, y))
    for mt in DIV.finditer(resp):
        x, y, z = map(float, mt.groups())
        if y == 0:
            continue
        ar_n += 1
        ar_ok += abs(x / y - z) <= max(0.05, 0.02 * abs(z)) or abs(abs(x / y) - abs(z)) <= max(0.05, 0.02 * abs(z))
    for mt in TTC_EXPR.finditer(resp):
        a1, b1, c1, d1 = map(float, mt.groups()[:4])
        steps = re.findall(r"(?:=|\u2248)\s*\**\s*\(?(-?\d+\.?\d*)\)?\s*(?:/\s*\(?(-?\d+\.?\d*)\)?)?", mt.group(5))
        if not steps or c1 == d1:
            continue
        num, den = steps[-1]
        if den and float(den) == 0:
            continue
        stated = float(num) / float(den) if den else float(num)
        implied = (a1 - b1) / (c1 - d1)
        ar_n += 1
        ar_ok += abs(abs(implied) - abs(stated)) <= max(0.1, 0.03 * abs(implied))
        op_n += 1
        op_ok += all(any(abs(v - q) <= 0.011 for q in in_nums) for v in (a1, b1, c1, d1))
    # direction: one verdict per longitudinal neighbour, read from its block of text
    pos = []
    for vid in V:
        for mm in re.finditer(r"(?<![\d.])%d(?![\d.])" % vid, resp):
            pos.append((mm.start(), vid))
    pos.sort()
    first = {}
    for p_, vid in pos:
        first.setdefault(vid, p_)
    heads = sorted((p_, vid) for vid, p_ in first.items())
    d_n = d_ok = flips = 0
    for k, (p_, vid) in enumerate(heads):
        t_true = ttcs.get(vid)
        if t_true is None:
            continue
        end = heads[k + 1][0] if k + 1 < len(heads) else len(resp)
        block = resp[p_:min(end, p_ + 1200)]
        vals = [float(v) for v in re.findall(r"TTC[^\n]{0,80}?(?:=|\u2248)\s*\**\s*(-?\d+\.?\d*)\s*\**\s*(?:s\b|sec|second)", block)]
        neg = bool(NEG.search(block))
        if vals:
            v = vals[-1]
            llm_closing = v > 0 and not (neg and v <= 0)
            llm_rule = 0 < v < 5
        elif neg:
            llm_closing, llm_rule = False, False
        else:
            continue
        true_closing = math.isfinite(t_true)
        d_n += 1
        d_ok += int(llm_closing == true_closing)
        flips += int(llm_rule != (t_true < 5))
    return dict(ar_n=ar_n, ar_ok=ar_ok, op_n=op_n, op_ok=op_ok, d_n=d_n, d_ok=d_ok, flips=flips)


def final_action(resp):
    m = (re.findall(r"Final Decision[^0-9]{0,20}(\d)", resp) or re.findall(r"####\s*(\d)", resp)
         or re.findall(r"Action ID[^0-9]{0,6}(\d)", resp))
    return int(m[-1]) if m else None


def audit():
    files = sorted(os.listdir(DESC))
    per = []
    agg = {m: Counter() for m in MODELS}
    for fn in files:
        lf, fno = last_frame(read(os.path.join(DESC, fn)))
        E, V, rem_ids, ttc_warn, rem = parse_frame(lf)
        ttcs = {vid: true_ttc(E, v) for vid, v in V.items()}
        allowed = admissible(E, V, ttcs)
        in_nums = [float(x) for x in NUM.findall(lf)]
        rec = dict(event=fn.replace("extreme_desc_", "")[:-4], src=fn.split("_")[2], dir=E["dir"],
                   n_neighbours=len(V), n_reminder_ids=len(rem_ids), ttc_warning=int(ttc_warn),
                   admissible="".join(map(str, sorted(allowed))))
        for m, folder in MODELS.items():
            resp = read(os.path.join(LLM, folder, "response_" + fn))
            a = final_action(resp)
            rec[m + "_action"] = a
            cited = set(int(i) for i in re.findall(r"(?<![\d.])(\d{2,6})(?![\d.])", resp)) & (set(V) | {E["id"]} | set(
                int(i) for i in re.findall(r"[Vv]ehicle (\d+)", resp)))
            cited_veh = set(int(i) for i in re.findall(r"[Vv]ehicle\s*\(?(\d{2,6})", resp)) | set(
                int(i) for i in re.findall(r"\((\d{3,6})\)", resp))
            known = set(V) | {E["id"]}
            agg[m]["ids"] += len(cited_veh)
            agg[m]["ids_ok"] += len(cited_veh & known)
            # attribution recall
            if rem_ids:
                agg[m]["rem"] += len(rem_ids)
                agg[m]["rem_hit"] += len(rem_ids & cited_veh)
            # TTC expressions
            n_expr = n_ar = n_op = n_dir = n_dir_ok = n_flip = 0
            for mt in TTC_EXPR.finditer(resp):
                a1, b1, c1, d1 = map(float, mt.groups()[:4])
                steps = re.findall(r"(?:=|≈)\s*\**\s*\(?(-?\d+\.?\d*)\)?\s*(?:/\s*\(?(-?\d+\.?\d*)\)?)?", mt.group(5))
                if not steps:
                    continue
                num, den = steps[-1]
                if den and float(den) == 0:
                    continue
                stated = float(num) / float(den) if den else float(num)
                n_expr += 1
                n_op += all(any(abs(v - y) <= 0.011 for y in in_nums) for v in (a1, b1, c1, d1))
                if c1 != d1:
                    implied = (a1 - b1) / (c1 - d1)
                    n_ar += abs(implied - stated) <= max(0.1, 0.03 * abs(implied)) or abs(abs(implied) - abs(stated)) <= max(0.1, 0.03 * abs(implied))
                # identify the vehicle the expression refers to (last ID before it)
                pre = resp[max(0, mt.start() - 400):mt.start()]
                ids = [int(i) for i in re.findall(r"(\d{3,6})", pre) if int(i) in V]
                if not ids:
                    continue
                vid = ids[-1]
                t_true = ttcs.get(vid)
                if t_true is None:
                    continue
                n_dir += 1
                llm_closing = stated > 0
                true_closing = math.isfinite(t_true)
                n_dir_ok += int(llm_closing == true_closing)
                llm_rule = 0 < stated < 5
                true_rule = t_true < 5
                n_flip += int(llm_rule != true_rule)
            st = statements(resp, V, ttcs, in_nums)
            rec.update({m + "_" + k: v for k, v in st.items()})
            rec.update({m + "_ttc_expr": n_expr, m + "_ttc_operands_ok": n_op, m + "_ttc_arith_ok": n_ar,
                        m + "_ttc_dir_n": n_dir, m + "_ttc_dir_ok": n_dir_ok, m + "_ttc_rule_flip": n_flip,
                        m + "_ids": len(cited_veh), m + "_ids_ok": len(cited_veh & known),
                        m + "_rem_recall": (len(rem_ids & cited_veh) / len(rem_ids)) if rem_ids else np.nan,
                        m + "_admissible": int(a in allowed) if (a and allowed) else np.nan,
                        m + "_legal": int(not ((a == 2 and 2 not in admissible(E, {}, {})) or (a == 3 and 3 not in admissible(E, {}, {})))) if a else np.nan})
        per.append(rec)
    P = pd.DataFrame(per)
    P.to_csv(os.path.join(OUT, f"llm_audit_per_event{TAG}.csv"), index=False)
    S = {"n_events": len(P), "direction_counts": P.dir.value_counts().to_dict(),
         "ttc_warning_in_input": int(P.ttc_warning.sum()), "events_with_reminders": int((P.n_reminder_ids > 0).sum())}
    for m in MODELS:
        e = P[m + "_ttc_expr"].sum()
        dn = P[m + "_ttc_dir_n"].sum()
        S[m] = dict(
            actions=P[m + "_action"].value_counts().sort_index().to_dict(),
            entity_grounding=[int(P[m + "_ids_ok"].sum()), int(P[m + "_ids"].sum())],
            ttc_expressions=int(e),
            operands_grounded=float(P[m + "_ttc_operands_ok"].sum() / e) if e else None,
            arithmetic_correct=float(P[m + "_ttc_arith_ok"].sum() / e) if e else None,
            events_with_arith_error=int((P[m + "_ttc_arith_ok"] < P[m + "_ttc_expr"]).sum()),
            arithmetic_ops=int(P[m + "_ar_n"].sum()),
            arithmetic_ops_correct=float(P[m + "_ar_ok"].sum() / max(1, P[m + "_ar_n"].sum())),
            operands_all=int(P[m + "_op_n"].sum()),
            operands_all_grounded=float(P[m + "_op_ok"].sum() / max(1, P[m + "_op_n"].sum())),
            closing_verdicts=int(P[m + "_d_n"].sum()),
            closing_correct=float(P[m + "_d_ok"].sum() / max(1, P[m + "_d_n"].sum())),
            closing_correct_by_dir={d: float(g[m + "_d_ok"].sum() / max(1, g[m + "_d_n"].sum())) for d, g in P.groupby("dir")},
            rule_flip_rate_all=float(P[m + "_flips"].sum() / max(1, P[m + "_d_n"].sum())),
            events_with_flip=int((P[m + "_flips"] > 0).sum()),
            direction_n=int(dn),
            direction_correct=float(P[m + "_ttc_dir_ok"].sum() / dn) if dn else None,
            direction_correct_by_dir={d: float(g[m + "_ttc_dir_ok"].sum() / max(1, g[m + "_ttc_dir_n"].sum()))
                                      for d, g in P.groupby("dir")},
            rule_flip_rate=float(P[m + "_ttc_rule_flip"].sum() / dn) if dn else None,
            events_with_rule_flip=int((P[m + "_ttc_rule_flip"] > 0).sum()),
            admissible_rate=float(P[m + "_admissible"].mean()),
            admissible_ci=list(wilson(int(P[m + "_admissible"].sum()), int(P[m + "_admissible"].notna().sum()))),
            legal_rate=float(P[m + "_legal"].mean()),
            reminder_recall=float(P[m + "_rem_recall"].mean()),
        )
    pairs = P.dropna(subset=["gpt4o_action", "gpt41_action"])
    po = float((pairs.gpt4o_action == pairs.gpt41_action).mean())
    ca, cb = Counter(pairs.gpt4o_action), Counter(pairs.gpt41_action)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / len(pairs) ** 2
    S["agreement"] = dict(n=len(pairs), observed=po, kappa=(po - pe) / (1 - pe))
    # bootstrap CI of kappa
    rng = np.random.default_rng(0)
    ks = []
    A, B = pairs.gpt4o_action.to_numpy(), pairs.gpt41_action.to_numpy()
    for _ in range(2000):
        i = rng.integers(0, len(A), len(A))
        a, b = A[i], B[i]
        po_ = (a == b).mean()
        cats = np.union1d(a, b)
        pe_ = sum((a == c).mean() * (b == c).mean() for c in cats)
        ks.append((po_ - pe_) / (1 - pe_))
    S["agreement"]["kappa_ci"] = [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))]
    S["admissible_set_size"] = P.admissible.str.len().value_counts().sort_index().to_dict()
    with open(os.path.join(OUT, f"llm_audit{TAG}.json"), "w") as f:
        json.dump(S, f, indent=2, default=str)
    print(json.dumps(S, indent=2, default=str))


if __name__ == "__main__":
    audit()
