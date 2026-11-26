"""K2d / K2dp: a decision-value planner with conditionally independent checks on the model's own JSON. NO LLM calls.
Post hoc (added in response to a review; not part of any plan fixed before the runs).

Recent language-to-solver systems for information seeking (DIVA, arXiv:2610.32181) let the LLM write a small decision
model, treat the checks as independent given the state, value each remaining check by its expected decision value net
of cost, acquire the best check while that value is positive, and remove a check from the candidate set once acquired;
a variant replaces the one-step value by exact dynamic programming over the remaining checks. This script applies that
valuation rule to the observation-model JSON each model wrote in K2e, in the same worlds as K2e / K2v / K2r:

  K2d   one-step net decision value, checks independent given the status, each tool at most once
  K2dp  exact dynamic programme over the remaining checks under the same independent model, each tool at most once

"Independent given the status" = every fault the JSON declares, persistent or not, is applied to each call separately
with the written rate (the marginal-matched single-call law of the JSON). It is an adaptation of the valuation rule to
this task (three statuses, four verdict actions, the model's K2e JSON), NOT a run of the DIVA system: its elicitation
prompts, risk-form answers, shrinkage and uncertainty-aware variant are not reproduced.

Reference rows computed the same way: K2e (model acts), K2v (one-step, dependence as written, repeats allowed),
K2r (exact DP, dependence as written, repeats allowed).

  python k2d_summary.py   ->  results/k2d_summary.{json,txt}
"""
import json, os
import numpy as np
from env import HERE, CORE, K as NS, LOSS, BUDGET, Env, load_episodes, load_truth      # env puts the parent folder on sys.path
from sim import Model
from nestboot import nested_ci, paired_ci
import k2decomp as K2D
import k2v_summary as KV
from score import per_run

N = K2D.N_TOOLS


def independent_model(ep, spec):
    """The recovered model of k2decomp.build_recovered_model with every declared fault made per-call."""
    par = ep["params"]; tools = [None] * N
    for g in spec["groups"]:
        faulty = g["shared_fault"] and g["phi"] > 0
        for j in g["tools"]:
            t = dict(kind=par["tools"][j].get("kind", "sym"), q=spec["accuracy"][j], cost=par["tools"][j]["cost"], group=None, femit=None)
            if faulty: t["marg_phi"] = g["phi"]; t["femit"] = np.array(g["emission"])
            tools[j] = t
    return Model(NS, np.array(par["prior"]), tools, {}, LOSS, BUDGET)


def one_step(M, b, B, used):
    best, a = M.term(b); out = (best, ("stop", a))
    for j, t in enumerate(M.tools):
        if j in used or t["cost"] > B + 1e-9: continue
        q = t["cost"] + sum(p * M.term(M.update(b, j, o))[0] for o, p in enumerate(M.pred(b, j)) if p > 1e-12)
        if q < out[0] - 1e-9: out = (q, ("probe", j))
    return out


def dp(M, b, B, used, memo):
    k = (tuple(np.round(b, 9)), round(B, 6), used)
    if k in memo: return memo[k]
    best, a = M.term(b); out = (best, ("stop", a))
    for j, t in enumerate(M.tools):
        if j in used or t["cost"] > B + 1e-9: continue
        q = t["cost"] + sum(p * dp(M, M.update(b, j, o), B - t["cost"], used | {j}, memo)[0] for o, p in enumerate(M.pred(b, j)) if p > 1e-12)
        if q < out[0] - 1e-9: out = (q, ("probe", j))
    memo[k] = out; return out


def play(ep, rec, cond):
    spec = rec["elicit"]["spec"]; env = Env(ep, "K2e", rec["protocol"]); M = independent_model(ep, spec)
    b = M.prior.copy(); used = frozenset(); verdicts = env.sc["states"] + ["escalate"]; memo = {}
    for _ in range(N + 2):
        if env.done: break
        a = (one_step(M, b, env.budget, used) if cond == "K2d" else dp(M, b, env.budget, used, memo))[1]
        if a[0] == "stop": env.submit(verdicts[a[1]]); break
        j = a[1]; env.call(env.names[j])
        if env.log[-1]["type"] != "call": break
        used = used | {j}; nb = M.update(b, j, env.hist[-1][1]); b = nb if nb is not None else M.prior.copy()
    return dict(rec, condition=cond, status="submitted" if env.done else "turn_cap", verdict=env.verdict, log=env.log)


def main():
    eps, truth = load_episodes(), load_truth(); sc = {i: e["scenario"] for i, e in eps.items()}
    out, L = {}, []
    for name, files in KV.RUNS.items():
        recs = KV.load(files)
        ids = sorted(e for e in eps if eps[e]["family"] in CORE and all((e, c) in recs for c in KV.CONDS)); assert len(ids) == 100, (name, len(ids))
        reg = {c: {e: per_run(recs[(e, c)], eps[e], truth[e])["regret"] for e in ids} for c in KV.CONDS}
        calls = {}
        for c in ("K2d", "K2dp"):
            pl = {e: play(eps[e], recs[(e, "K2e")], c) for e in ids}
            reg[c] = {e: per_run(pl[e], eps[e], truth[e])["regret"] for e in ids}
            calls[c] = float(np.mean([sum(x["type"] == "call" for x in pl[e]["log"]) for e in ids]))
        m = dict(n=len(ids), calls=calls, per_episode={c: reg[c] for c in ("K2d", "K2dp")})
        for c in ("K2e", "K2v", "K2d", "K2dp", "K2r"):
            mu, lo, hi = nested_ci([reg[c][e] for e in ids], ids, [sc[e] for e in ids]); m[c] = dict(mean=mu, lo=lo, hi=hi)
        for a_, b_ in (("K2d", "K2e"), ("K2dp", "K2e"), ("K2r", "K2d"), ("K2r", "K2dp"), ("K2d", "K2v")):
            d, lo, hi, n = paired_ci(reg[a_], reg[b_], sc); m[f"{a_}-{b_}"] = dict(diff=d, lo=lo, hi=hi, n=n)
        out[name] = m
        L.append(f"{name:11s} K2e {m['K2e']['mean']:.2f}  K2v {m['K2v']['mean']:.2f}  K2d {m['K2d']['mean']:.2f}  K2dp {m['K2dp']['mean']:.2f}  K2r {m['K2r']['mean']:.2f}"
                 + "".join(f"  | {k} {m[k]['diff']:+.2f} [{m[k]['lo']:+.2f},{m[k]['hi']:+.2f}]" for k in ("K2d-K2e", "K2dp-K2e", "K2r-K2d", "K2r-K2dp")))
    json.dump(out, open(os.path.join(HERE, "results", "k2d_summary.json"), "w"), indent=1)
    txt = "\n".join(L); print(txt); open(os.path.join(HERE, "results", "k2d_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
