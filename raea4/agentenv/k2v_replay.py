"""K2v: same-JSON replay with a MYOPIC planner. NO LLM calls.

Identical to k2e_replay.py (K2r) except for the executor: instead of the exact DP on the recovered model, a one-step
value-of-information planner (oracle.voi1, the policy behind the Oracle-VOI1 baseline) acts on the observation-model JSON
the LLM wrote in K2e. At each step it compares stopping now (best terminal action under its belief) with every affordable
single probe followed by the best terminal action, and takes the cheaper; it never looks two probes ahead. Its belief is the
exact Bayes update under the recovered model. Same episode (parameters, hidden state, pre-drawn output table), so K2v is a
deterministic function of (JSON, episode).

Question it answers: is the K2e-to-K2r drop only "an optimal planner beats an LLM", or does even a simple myopic decision
rule on the model's own JSON beat the model acting on it?

  python k2v_replay.py logs\\claude_s55.jsonl --out logs\\claude_s55_k2v_replay.jsonl"""
import argparse, json, os, time
from env import Env, load_episodes
import k2decomp as K
from oracle import voi1

def play_voi1(ep, spec, env):
    M = K.build_recovered_model(ep, spec); M.srcmap = list(range(len(M.tools)))   # srcmap only read when forbid is set
    b = M.prior.copy(); verdicts = env.sc["states"] + ["escalate"]
    for _ in range(4 * K.N_TOOLS + 4):
        if env.done: break
        a = voi1(M, b, env.budget, frozenset())
        if a[0] == "stop": env.submit(verdicts[a[1]]); break
        j = a[1]; env.call(env.names[j])
        if env.log[-1]["type"] != "call": break
        nb = M.update(b, j, env.hist[-1][1]); b = nb if nb is not None else M.prior.copy()
    return M

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("--out", required=True)
    ap.add_argument("--src-cond", default="K2e"); ap.add_argument("--dst-cond", default="K2v"); a = ap.parse_args()
    assert not os.path.exists(a.out), "refusing to overwrite " + a.out
    eps = load_episodes(); src = {}
    for line in open(a.log, encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != a.src_cond or r["status"].startswith("api_error"): continue
        src[(r["episode"], r.get("sample", 0))] = r
    n = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for (eid, smp), r in sorted(src.items()):
            assert r.get("elicit"), eid
            ep = eps[eid]; env = Env(ep, a.dst_cond, r["protocol"]); play_voi1(ep, r["elicit"]["spec"], env)
            el = dict(r["elicit"], planner="one-step VOI on the recovered model, replayed offline from " + a.src_cond)
            out = dict(episode=eid, family=r["family"], scenario=r["scenario"], condition=a.dst_cond, model=r["model"],
                       protocol=r["protocol"], status="submitted" if env.done else "turn_cap", verdict=env.verdict, log=env.log,
                       n_api=0, elicit=el, derived_from=a.src_cond, models_returned=r.get("models_returned", []),
                       finish_reasons=[], tokens=dict(prompt=0, completion=0, reasoning=0, cache_hit=0, cache_miss=0), latency=0.0,
                       max_tokens=r.get("max_tokens"), max_turns=r.get("max_turns"), api=[], transcript=r["transcript"][:3],
                       time=time.strftime("%Y-%m-%d %H:%M:%S"), sample=smp)
            fo.write(json.dumps(out, ensure_ascii=False) + "\n"); n += 1
    print(f"wrote {n} {a.dst_cond} episodes to {a.out}")

if __name__ == "__main__":
    main()
