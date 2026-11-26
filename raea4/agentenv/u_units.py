"""
Post hoc (added 2026-11-18 after a mock review; no plan fixed in advance; no model call): the loss in agent units.

For every K2e episode of a run, three executors act in the same world with the same pre-drawn tool outputs:
  model    the LLM's own trajectory (K2e log)
  planner  the exact DP planner on the model's stored JSON (K2r; replayed here)
  oracle   the exact DP planner on the true model
and each trajectory's realised outcome is recorded against the TRUE hidden state: correct verdict, false-benign
("severe": benign verdict when the state is not benign), other wrong verdict, escalation, query spend, terminal loss,
realised loss (terminal + spend) and exact regret. Paired differences model - oracle and model - planner use the
nested bootstrap (scenarios, then episodes; 95%).

Sets: main  CORE episodes of the decomposition runs (domain 1, n=100 per run)
      held  held-out CORE episodes (confirm_* logs, domain 1)
      d2    second domain, DN part (run with RAEA_DOMAIN=2)
      d3    third domain, DN part (run with RAEA_DOMAIN=3)

  python u_units.py main|held          RAEA_DOMAIN=2 python u_units.py d2          RAEA_DOMAIN=3 python u_units.py d3
Output: results/u_units_<set>.json, .txt
"""
import os, sys
SET = sys.argv[1] if len(sys.argv) > 1 else "main"
if SET == "d2": os.environ.setdefault("RAEA_EPISODES", "episodes_d2.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_d2.jsonl")
if SET == "d3": os.environ.setdefault("RAEA_EPISODES", "episodes_d3.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_d3.jsonl")
import json
from multiprocessing import Pool
import numpy as np
WORKERS = int(os.environ.get("U_WORKERS", "7"))
import env as E
from env import HERE, Env, CORE, load_episodes, load_truth, N_TOOLS
from oracle import score_trajectory, true_model
import k2decomp as KD
from nestboot import nested_ci

assert SET in ("main", "held", "d2", "d3")
assert (SET in ("d2", "d3")) == (os.environ.get("RAEA_DOMAIN", "1") != "1"), "set RAEA_DOMAIN=2/3 for d2/d3 only"
P = lambda *a: os.path.join(HERE, *a)
LOGS = {
    "main": {"flash": "ds_chat_fc_k2decomp.jsonl", "terra": "codex_terra_medium_k2decomp.jsonl", "terra_high": "codex_terra_high.jsonl", "sonnet": "claude_s55.jsonl"},
    "held": {"flash": "confirm_ds_chat_fc_k2e.jsonl", "terra": "confirm_codex_terra_k2e.jsonl", "sonnet": "confirm_claude_s55_k2e.jsonl"},
    "d2": {"flash": "d2_ds_chat_fc_k2e.jsonl", "terra": "d2_codex_terra_k2e.jsonl", "sonnet": "d2_claude_s55_k2e.jsonl"},
    "d3": {"flash": "d3_ds_chat_fc_k2e.jsonl", "terra": "d3_codex_terra_k2e.jsonl", "sonnet": "d3_claude_s55_k2e.jsonl"},
}[SET]
EPS = {"main": "episodes.jsonl", "held": "episodes.jsonl", "d2": "episodes_d2.jsonl", "d3": "episodes_d3.jsonl"}[SET]
TRU = {"main": "ground_truth.jsonl", "held": "ground_truth.jsonl", "d2": "ground_truth_d2.jsonl", "d3": "ground_truth_d3.jsonl"}[SET]
METRICS = ["correct", "severe", "wrong_other", "esc", "cost", "dloss", "realised", "regret"]


def load(fn):
    recs = {}
    for line in open(P("logs", fn), encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != "K2e" or r["status"].startswith("api_error") or r.get("sample", 0) != 0: continue
        recs[r["episode"]] = r                      # a later successful rerun replaces an earlier row (as in score.py)
    return recs


def oracle_play(ep):
    M = true_model(ep); env = Env(ep, "K2e", "text"); b = M.prior.copy(); verdicts = env.sc["states"] + ["escalate"]
    for _ in range(4 * N_TOOLS + 4):
        if env.done: break
        a = M.V(b, env.budget)[1]
        if a[0] == "stop": env.submit(verdicts[a[1]]); break
        env.call(env.names[a[1]]); b = M.update(b, a[1], env.hist[-1][1])
    return env.log


def planner_play(ep, spec):
    env = Env(ep, "K2e", "text"); KD.plan_and_play(ep, spec, env); return env.log


def outcome(ep, log, truth):
    s = score_trajectory(ep, log, truth); st = truth["s"]; K = E.K
    return dict(correct=float(s["correct"]), severe=float(s["severe"]), esc=float(s["esc"]),
                wrong_other=float(not s["correct"] and not s["severe"] and not s["esc"]),
                cost=s["cost"], dloss=s["dloss"], realised=s["realised"], regret=s["regret"], V0=s["V0"], forced=float(s["forced"]))


def one(args):
    ep, tr, r = args
    spec = (r.get("elicit") or {}).get("spec")
    m = outcome(ep, r["log"], tr); o = outcome(ep, oracle_play(ep), tr)
    p = outcome(ep, planner_play(ep, spec), tr) if spec else None
    import oracle as _O; _O._MODELS.clear()                 # bound memory (second/third domain models are large)
    return dict(id=ep["id"], sc=ep["scenario"], m=m, o=o, p=p)


def ci(v, ids, scl):
    m, lo, hi = nested_ci(v, ids, scl); return dict(mean=float(np.mean(v)), lo=float(lo), hi=float(hi))


def main():
    eps, truth = load_episodes(P(EPS)), load_truth(P(TRU))
    out, L = dict(set=SET, runs={}), [f"Loss in agent units, set {SET} (post hoc; realised outcomes against the true state; model / planner on the model's JSON / oracle)"]
    for key, fn in LOGS.items():
        recs = load(fn)
        if SET == "main": ids = sorted(i for i, r in recs.items() if r["family"] in CORE)
        elif SET == "held": ids = sorted(i for i, r in recs.items() if r["family"] in CORE)
        else: ids = sorted(i for i in recs if eps[i].get("part") == "DN")
        with Pool(WORKERS) as pool:
            rows = pool.map(one, [(eps[i], truth[i], recs[i]) for i in ids], chunksize=1)
        idl = [d["id"] for d in rows]; scl = [d["sc"] for d in rows]; pr = [d for d in rows if d["p"]]
        o = dict(n=len(rows), n_planner=len(pr), V0=float(np.mean([d["m"]["V0"] for d in rows])), forced=float(np.mean([d["m"]["forced"] for d in rows])),
                 mean={x: {w: float(np.mean([d[w][x] for d in (pr if w == "p" else rows)])) for w in ("m", "p", "o")} for x in METRICS},
                 diff_mo={x: ci([d["m"][x] - d["o"][x] for d in rows], idl, scl) for x in METRICS},
                 diff_mp={x: ci([d["m"][x] - d["p"][x] for d in pr], [d["id"] for d in pr], [d["sc"] for d in pr]) for x in METRICS})
        out["runs"][key] = o
        L += ["", f"== {key}  ({fn}; n={o['n']}, planner n={o['n_planner']}; oracle expected loss {o['V0']:.2f}; forced endings {100 * o['forced']:.0f}%)",
              f"  {'metric':12s}{'model':>8s}{'planner':>9s}{'oracle':>8s}   model-oracle [95%]        model-planner [95%]"]
        for x in METRICS:
            sc = 100 if x in ("correct", "severe", "wrong_other", "esc") else 1; f = ".0f" if sc == 100 else ".2f"
            a, b = o["diff_mo"][x], o["diff_mp"][x]
            L.append(f"  {x:12s}{sc * o['mean'][x]['m']:8{f}}{sc * o['mean'][x]['p']:9{f}}{sc * o['mean'][x]['o']:8{f}}   "
                     f"{sc * a['mean']:+6{f}} [{sc * a['lo']:+.2f}, {sc * a['hi']:+.2f}]   {sc * b['mean']:+6{f}} [{sc * b['lo']:+.2f}, {sc * b['hi']:+.2f}]")
    txt = "\n".join(L); print(txt)
    json.dump(out, open(P("results", f"u_units_{SET}.json"), "w"), indent=1); open(P("results", f"u_units_{SET}.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
