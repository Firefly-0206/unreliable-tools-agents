"""POST HOC analyses of the belief-elicitation forks (not in PLAN_be_belief_elicit.md; written 2026-11-22 after the frozen
analysis be_analysis.py had been run and after an independent check of its results). Offline, no model call.

  P1 stated value of calling vs the exact ONE-STEP value VOI1 (stored in the points as a reference, no reading planned)
     and vs the exact multi-step value VOI, own JSON: replica tool at U, and all tools at all points
  P2 rank agreement of stated values with VOI1 at U and O, and of VOI1 with VOI themselves (no model involved)
  P3 fork C: share of replies whose action is the argmax of the same reply's stated values (stop = 0)
  P4 flash sensitivity: B3 without repaired rows; without U points whose logged turn also called a replica later;
     B4 agreement with the logged first call anywhere among fork A's calls; B5 with a missing C action counted as
     'no replica', and with replica = any call of the turn
  P5 terra: share of fork-B replies with all-null values, by label and by calls already made

  python be_posthoc.py  ->  results/be_posthoc.txt, results/be_posthoc.json
"""
import json, os
from collections import defaultdict
import numpy as np
from env import HERE, SCENARIOS, load_episodes
from be_analysis import RUNS, load_rows, spearman, ci
import be_points

SRC = {k: v[0] for k, v in be_points.LOGS.items()}
NEAR = 0.06


def turn_calls(model):
    """pid -> list of tool indices called in the logged turn (fc only; text turns have one action)."""
    eps = load_episodes(); recs = be_points.load(SRC[model]); out = {}
    for i, r in recs.items():
        msgs = r["transcript"]; ep = eps[i]; names = [t["name"] for t in SCENARIOS[ep["scenario"]]["tools"]]
        i0 = next(k for k, m in enumerate(msgs) if m["role"] == "user" and m.get("content") in be_points.ACKS)
        turns = [k for k in range(i0 + 1, len(msgs)) if msgs[k]["role"] == "assistant"]
        for t, k in enumerate(turns):
            out[f"{ep['id']}#{t}"] = [names.index(n) for n in (c.get("name") or c.get("function", {}).get("name") for c in (msgs[k].get("tool_calls") or [])) if n in names]
    return out


def calls_of(row, ep):
    names = [t["name"] for t in SCENARIOS[ep["scenario"]]["tools"]]
    return [names.index(c["name"]) for c in (row.get("tool_calls") or []) if c["name"] in names]


def C(vals):
    v = [x for x in vals if x[0] is not None and np.isfinite(x[0])]         # spearman can be nan when VOI ties exactly
    return ci([x[0] for x in v], [x[1] for x in v], [x[2] for x in v]) if v else None


def fmt(d):
    if d is None: return "-"
    return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  n={d['n']}"


def analyse(model):
    eps = load_episodes(); rows = load_rows(os.path.join(HERE, "logs", RUNS[model]))
    pts = [json.loads(l) for l in open(os.path.join(HERE, "results", f"be_points_{model}.jsonl"), encoding="utf-8")]
    pts = [p for p in pts if p["subset"]]; S = {}
    tc = turn_calls(model) if model == "flash" else {}
    P1u, P1u1, closer, near1, nearm, ntools = [], [], [], 0, 0, 0
    rho1 = {"U": [], "O": []}; rhoenv = {"U": [], "O": []}
    argmaxC = {"U": [], "all": []}; sens = defaultdict(list); nulls = defaultdict(lambda: [0, 0])
    for p in pts:
        key = (p["episode"], p["scenario"]); H = p["json"] if p.get("parse_ok", True) else None
        A, B, Cc = rows.get((p["pid"], "A")), rows.get((p["pid"], "B")), rows.get((p["pid"], "C"))
        jr = str(p["true"]["best"][1]); ep = eps[p["episode"]]
        if B and B.get("belief_found") and H:
            val = B["belief"]["value"]
            for j, x in val.items():
                if j in H["voi1"]:
                    ntools += 1; near1 += abs(x - H["voi1"][j]) < NEAR; nearm += abs(x - H["voi"][j]) < NEAR
            if p["label"] == "U" and jr in val:
                P1u.append((val[jr] - H["voi"][jr], *key)); P1u1.append((val[jr] - H["voi1"][jr], *key))
                closer.append((float(abs(val[jr] - H["voi1"][jr]) < abs(val[jr] - H["voi"][jr])), *key))
                excl_rep = B.get("belief_repaired"); later_rep = any(j in p["cache"] for j in tc.get(p["pid"], [])[1:])
                if not excl_rep: sens["B3 rep_bias_json | U, repaired rows excluded"].append((val[jr] - H["voi"][jr], *key))
                if not later_rep: sens["B3 rep_bias_json | U, turns with a later replica call excluded"].append((val[jr] - H["voi"][jr], *key))
            js = [j for j in val if j in H["voi1"]]
            if p["label"] in rho1 and len(js) >= 3:
                rho1[p["label"]].append((spearman([val[j] for j in js], [H["voi1"][j] for j in js]), *key))
        if H and p["label"] in rhoenv:
            js = sorted(H["voi1"]);
            if len(js) >= 3: rhoenv[p["label"]].append((spearman([H["voi1"][j] for j in js], [H["voi"][j] for j in js]), *key))
        if Cc and Cc.get("belief_found") and Cc.get("action") is not None:
            v = Cc["belief"]["value"]
            if v:
                mx = max(v.values()); best = {int(j) for j, x in v.items() if abs(x - mx) <= 1e-12}   # every tool tied at the maximum
                act = Cc["action"]
                hit = float(act[0] == "probe" and act[1] in best) if mx > 0 else float(act[0] == "stop")
                argmaxC["all"].append((hit, *key))
                if p["label"] == "U": argmaxC["U"].append((hit, *key))
        if model == "flash" and A:
            la = p["logged"]; ac = calls_of(A, ep)
            anyhit = float(la in [["probe", j] for j in ac]) if la[0] == "probe" else float(A["action"] == la)
            sens["B4 agree | all, logged first call anywhere among fork A's calls"].append((anyhit, *key))
            if p["label"] == "U":
                arep = float(A["action"] is not None and A["action"][0] == "probe" and A["action"][1] in p["cache"])
                crep = float(Cc is not None and Cc.get("action") is not None and Cc["action"][0] == "probe" and Cc["action"][1] in p["cache"])
                sens["B5 replica C-A | U, missing C action = no replica (all U)"].append((crep - arep, *key))
                aany = float(any(j in p["cache"] for j in ac)); cany = float(any(j in p["cache"] for j in calls_of(Cc, ep))) if Cc else 0.0
                sens["B5 replica C-A | U, replica = any call of the turn, missing = no"].append((cany - aany, *key))
                sens["fork C gave no action | U"].append((float(Cc is None or Cc.get("action") is None), *key))
        if model == "terra" and B and B.get("belief_found") and p["affordable"]:
            allnull = float(len(B["belief"]["value"]) == 0)
            for g in (p["label"], f"calls_before {min(p['calls_before'], 3)}{'+' if p['calls_before'] >= 3 else ''}"):
                nulls[g][0] += allnull; nulls[g][1] += 1
    S["P1 stated - VOI1 (one-step) | U replica"] = C(P1u1)
    S["P1 stated - VOI (multi-step) | U replica"] = C(P1u)
    S["P1 share closer to VOI1 than to VOI | U replica"] = C(closer)
    S[f"P1 all tools, all points: within {NEAR} of VOI1 / of VOI / tools"] = f"{near1} / {nearm} / {ntools}"
    for lab in ("U", "O"):
        S[f"P2 rho(stated, VOI1) | {lab}"] = C(rho1[lab]); S[f"P2 rho(VOI1, VOI), no model | {lab}"] = C(rhoenv[lab])
    S["P3 fork C action = argmax of own stated values | all"] = C(argmaxC["all"]); S["P3 ... | U"] = C(argmaxC["U"])
    for k, v in sens.items(): S["P4 " + k] = C(v)
    for g in sorted(nulls): S[f"P5 all-null values | {g}"] = f"{int(nulls[g][0])}/{nulls[g][1]}"
    return S


def main():
    out, L = {}, ["POST HOC (not in the frozen plan); episode-weighted means, nested bootstrap 95%, own-JSON references"]
    for m in RUNS:
        S = analyse(m); out[m] = S; L += ["", f"== {m}"]
        L += [f"  {k:72s} {fmt(v) if isinstance(v, dict) or v is None else v}" for k, v in S.items()]
    txt = "\n".join(L); print(txt)
    json.dump(out, open(os.path.join(HERE, "results", "be_posthoc.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "be_posthoc.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
