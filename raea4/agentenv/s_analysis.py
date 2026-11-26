"""
Layered supply of exact quantities inside the closed loop: analysis (PLAN_s_supply.md). No model call.

  python s_analysis.py                 # the three model runs
  python s_analysis.py --mock FILE     # any log of the S conditions on the S episodes (code check only)
  python s_analysis.py --selftest      # a reader that takes the lowest supplied score: KS3 must reach the oracle

Episodes: persistent arm of the X sweep at fault rates 0.10 and 0.30 (50). Conditions, all on the stated model (K1):
K1 nothing; KC checklist without numbers; KS1 exact joint posterior; KS2 + next-report probabilities;
KS2v + one-step-ahead scores; KS3 = KS2 + exact scores. Trajectories are scored under the true model.

Per condition: replica first-call share (model, oracle), regret, under-use, over-use, verdict regret, calls, and the
share of steps whose action is exactly optimal. Contrasts, paired by episode, nested bootstrap (scenarios, then
draws), 95%:
  L   adjacent layers  KC-K1, KS1-KC, KS2-KS1, KS2v-KS2, KS3-KS2v (and KS3-KS2, KS3-K1): regret, first-call share
  G   model minus oracle first-call share and under-use regret, per condition
Output: results/s_analysis.json, results/s_analysis.txt
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_x.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_x.jsonl")
import argparse, json, re
import numpy as np
from env import HERE, Env, SCENARIOS, load_episodes, load_truth, build_model, first_user_cond
from oracle import score_trajectory, qvalues
from x_analysis import parts, ci, fmt, REP

RUNS = {"flash": ("flash (non-thinking)", "s_ds_chat_fc.jsonl"), "terra": ("gpt-6.1-terra (medium)", "s_codex_terra.jsonl"),
        "sonnet": ("Sonnet 5.5 (high)", "s_claude_s55.jsonl")}
CONDS = ["K1", "KC", "KS1", "KS2", "KS2v", "KS3"]
PAIRS = [("KC", "K1"), ("KS1", "KC"), ("KS2", "KS1"), ("KS2v", "KS2"), ("KS3", "KS2v"), ("KS3", "KS2"), ("KS3", "K1")]


def s_ids(eps):
    return sorted(i for i, e in eps.items() if e["arm"] == "XP" and any(abs(e["phi"] - p) < 1e-9 for p in (0.10, 0.30)))


def load(path):
    recs = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] not in CONDS or r["status"].startswith("api_error") or r.get("sample", 0) != 0: continue
        recs[(r["episode"], r["condition"])] = r
    return recs


def row(ep, log, truth):
    d = parts(ep, log, truth); s = score_trajectory(ep, log, truth)
    d["opt_steps"] = float(np.mean([st["adv"] < 1e-9 for st in s["steps"]])); return d


def analyse(recs, eps, truth):
    ids = s_ids(eps); missing = [(i, c) for i in ids for c in CONDS if (i, c) not in recs]
    assert not missing, f"{len(missing)} runs missing, e.g. {missing[:3]}"
    first = {}
    for i in ids:
        M = build_model(eps[i]["params"]); q = qvalues(M, M.prior.copy(), float(M.budget)); first[i] = float(min(q, key=q.get) == ("probe", REP))
    R = {(i, c): row(eps[i], recs[(i, c)]["log"], truth[i]) for i in ids for c in CONDS}
    dr = [eps[i]["pair"] for i in ids]; sc = [eps[i]["scenario"] for i in ids]
    C = lambda v: ci(v, dr, sc)
    out = dict(n=len(ids), no_submit={c: int(sum(recs[(i, c)]["status"] != "submitted" for i in ids)) for c in CONDS},
               invalid={c: int(sum(sum(x["type"] == "invalid" for x in recs[(i, c)]["log"]) for i in ids)) for c in CONDS},
               oracle_first=float(np.mean(list(first.values()))), cells={}, contrasts={})
    for c in CONDS:
        cell = {m: C([R[(i, c)][m] for i in ids]) for m in ["first", "regret", "under", "over", "verdict", "calls", "rep_calls", "opt_steps"]}
        for p in (0.10, 0.30):
            sub = [i for i in ids if abs(eps[i]["phi"] - p) < 1e-9]
            cell[f"first@{p:.2f}"] = float(np.mean([R[(i, c)]["first"] for i in sub])); cell[f"regret@{p:.2f}"] = float(np.mean([R[(i, c)]["regret"] for i in sub]))
        out["cells"][c] = cell
        out["contrasts"][f"G first model-oracle|{c}"] = C([R[(i, c)]["first"] - first[i] for i in ids])
        out["contrasts"][f"G under|{c}"] = cell["under"]
    for a, b in PAIRS:
        for m in ["regret", "first", "under"]:
            out["contrasts"][f"L {m} {a}-{b}"] = C([R[(i, a)][m] - R[(i, b)][m] for i in ids])
    return out


def follower(ep, cond):
    """A reader that always takes the action with the lowest score in the Decision aid (KS2v, KS3)."""
    sc = SCENARIOS[ep["scenario"]]; env = Env(ep, cond, "text"); text = first_user_cond(ep, cond)
    for _ in range(20):
        line = [l for l in text.split("\n") if l.startswith("- Expected final score")][-1].split("): ")[-1].rstrip(".")
        opts = [(float(x.rsplit(" ", 1)[1]), x.rsplit(" ", 1)[0]) for x in line.split("; ")]
        act = min(opts)[1]
        if act.startswith("submit "): env.submit(act[7:]); break
        text = env.call(act[5:])
    return env.log


def selftest(eps, truth):
    ids = s_ids(eps)
    for cond in ("KS3", "KS2v"):
        reg = [score_trajectory(eps[i], follower(eps[i], cond), truth[i])["regret"] for i in ids]
        print(f"selftest {cond}: lowest-score reader, mean regret {np.mean(reg):.4f}, max {np.max(reg):.4f} (n={len(ids)})")
        if cond == "KS3": assert np.mean(reg) < 0.01, "KS3 scores do not lead to the oracle"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--runs", nargs="*", default=list(RUNS)); a = ap.parse_args()
    eps, truth = load_episodes(), load_truth(); out = {}; L = []
    if a.selftest: selftest(eps, truth); return
    jobs = [("mock", ("mock", a.mock))] if a.mock else [(k, RUNS[k]) for k in a.runs]
    for key, (label, fn) in jobs:
        o = analyse(load(fn if a.mock else os.path.join(HERE, "logs", fn)), eps, truth); o["label"] = label; out[key] = o
        L += ["", f"== {label}   n={o['n']} per condition  oracle first {100 * o['oracle_first']:.0f}%  no submit {o['no_submit']}  invalid {o['invalid']}",
              f"{'cond':6s}{'first':>7s}{'@.10':>6s}{'@.30':>6s}{'regret':>8s}{'under':>8s}{'over':>8s}{'verdict':>9s}{'calls':>7s}{'rep':>6s}{'opt steps':>11s}"]
        for c, x in o["cells"].items():
            L.append(f"{c:6s}{100 * x['first']['mean']:6.0f}%{100 * x['first@0.10']:5.0f}%{100 * x['first@0.30']:5.0f}%{x['regret']['mean']:8.2f}{x['under']['mean']:8.2f}"
                     f"{x['over']['mean']:8.2f}{x['verdict']['mean']:9.2f}{x['calls']['mean']:7.2f}{x['rep_calls']['mean']:6.2f}{100 * x['opt_steps']['mean']:10.0f}%")
        for name, d in o["contrasts"].items():
            L.append(f"  {name:28s} {fmt(d, pct='first' in name)}" + ("  (percentage points)" if "first" in name else ""))
    txt = "\n".join(L); print(txt)
    if not a.mock:
        json.dump(out, open(os.path.join(HERE, "results", "s_analysis.json"), "w"), indent=1)
        open(os.path.join(HERE, "results", "s_analysis.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
