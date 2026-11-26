"""Offline checks of the third domain (no model calls). Usage: python d3_selftest.py [--n 12]

1. Prompt rendering: system prompt (K2e) and tool schemas build for every scenario; outputs render for every value.
2. Doc round trip: k2decomp.parse_docs recovers every documented number from the K2e prompt.
3. Perfect description: the JSON a perfect doc reader would write, parsed and planned on (K2r), reproduces the exact
   oracle's value and first action, and its replay has zero trajectory regret.
4. Structure is used: under the exact oracle, how often the sub-question tools (figure / source check) and the second
   outlet are called (realised on the episode's pre-drawn outputs).
"""
import os
os.environ["RAEA_DOMAIN"] = "3"
os.environ.setdefault("RAEA_EPISODES", "episodes_d3.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_d3.jsonl")
import argparse, json
import numpy as np
from env import Env, SCENARIOS, N_TOOLS, K, system_prompt, tool_schemas, load_episodes, load_truth, build_model
from oracle import score_trajectory, qvalues
import k2decomp as kd


def oracle_play(ep):
    env = Env(ep, "K2e"); M = build_model(ep["params"]); b = M.prior.copy(); verdicts = env.sc["states"] + ["escalate"]
    while not env.done:
        a = M.V(b, env.budget)[1]
        if a[0] == "stop": env.submit(verdicts[a[1]]); break
        env.call(env.names[a[1]]); b = M.update(b, a[1], env.hist[-1][1])
    return env


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=12); a = ap.parse_args()
    eps, truth = load_episodes(), load_truth()
    ids = [i for i in eps if i.startswith("DN")][: a.n]
    for sc in SCENARIOS:                                             # 1. rendering
        ep = next(e for e in eps.values() if e["scenario"] == sc)
        sp = system_prompt(ep, "K2e", "fc"); assert "observation_model" in sp and len(tool_schemas(ep)) == N_TOOLS + 1
        env = Env(ep, "K2e")
        for j, d in enumerate(env.sc["tools"]):
            assert len(d["values"]) == (2 if d.get("kind") else K), (sc, j)
            for o in range(len(d["values"])): json.loads(env.render(j, o).split("\n")[0])
    print("1. rendering ok for", len(SCENARIOS), "scenarios")
    gaps, use = [], np.zeros(N_TOOLS)
    for i in ids:
        ep = eps[i]; par = ep["params"]; sp = system_prompt(ep, "K2e", "fc")
        docs = kd.parse_docs(ep, sp)                                 # 2. round trip
        for j, (d, t) in enumerate(zip(docs, par["tools"])):
            assert abs(d["acc"] - t["q"]) < 1e-9 and d["cost"] == t["cost"], (i, j, d, t)
            assert d["cache"] == (t["src"] in ("cache", "feed")), (i, j, d)
        assert abs(docs[0]["phi"] - round(par["phi"], 2)) < 0.006 and docs[0]["backend"] == docs[1]["backend"], docs[:2]
        js = kd.doc_reader_model(ep, sp)                             # 3. perfect description
        spec = kd.normalize_model(ep, js, docs); assert not spec["defaults"] and not spec["fallback"], spec
        Mt, Mr = build_model(par), kd.build_recovered_model(ep, spec)
        Vt, Vr = Mt.V(Mt.prior.copy(), float(Mt.budget)), Mr.V(Mr.prior.copy(), float(Mr.budget))
        gaps.append(abs(Vt[0] - Vr[0])); assert Vt[1] == Vr[1], (i, Vt, Vr)
        env = Env(ep, "K2e"); kd.plan_and_play(ep, spec, env)
        r = score_trajectory(ep, env.log, truth[i]); assert abs(r["regret"]) < 1e-6, (i, r["regret"])
        env = oracle_play(ep)                                        # 4. structure is used
        for k, _ in env.hist: use[k] += 1
    print(f"2. doc round trip ok on {len(ids)} episodes")
    print(f"3. perfect-JSON planner: max |V_true - V_json| = {max(gaps):.2e}; zero replay regret on all {len(ids)}")
    names = [d["name"] for d in SCENARIOS[next(iter(SCENARIOS))]["tools"]]
    print("4. oracle calls per episode by tool:", ", ".join(f"{n} {u / len(ids):.2f}" for n, u in zip(names, use)))


if __name__ == "__main__":
    main()
