"""K2r: same-JSON executor replay. NO LLM calls.

For every K2e episode the observation-model JSON that the LLM wrote (stored spec in the K2e record) is handed to the exact DP
planner (k2decomp.plan_and_play), which then chooses every call and the verdict in the SAME episode. Same JSON, same episode
(parameters, hidden state, and the pre-drawn output table ep['table']); only the executor changes (LLM -> exact planner).
Unlike k2p_replay.py this script de-duplicates rows the way score.py does (later successful row of the same
(episode, condition, sample) wins, api_error rows ignored), asserts one source row per episode, and stores how the planner's
trajectory relates to the LLM's (replay field).

Determinism: Env.call returns ep['table'][tool][n-th call of that tool] (common random numbers drawn when the episodes were
generated); the planner is deterministic given spec + prior/costs/loss/budget. So K2r is a deterministic function of
(JSON, episode); nothing is sampled anew. The planner chooses other tools/stops at other times than the LLM, so the observation
sequences differ whenever the action sequences differ; what is shared is the world (state + table), not the evidence seen.

  python k2e_replay.py logs\\codex_terra_medium_k2decomp.jsonl --out logs\\codex_terra_k2e_replay.jsonl"""
import argparse, json, os, time
from env import Env, load_episodes
import k2decomp as K

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("--out", required=True)
    ap.add_argument("--src-cond", default="K2e"); ap.add_argument("--dst-cond", default="K2r"); a = ap.parse_args()
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
            ep = eps[eid]; env = Env(ep, a.dst_cond, r["protocol"]); K.plan_and_play(ep, r["elicit"]["spec"], env)
            llm_seq = [(e["j"], e["o"]) for e in r["log"] if e["type"] == "call"]
            pl_seq = [(e["j"], e["o"]) for e in env.log if e["type"] == "call"]
            llm_v = next((e["a"] for e in r["log"] if e["type"] == "submit"), None)
            rp = dict(llm_calls=[j for j, _ in llm_seq], planner_calls=[j for j, _ in pl_seq], same_call_sequence=llm_seq == pl_seq,
                      same_tool_sequence=[j for j, _ in llm_seq] == [j for j, _ in pl_seq], llm_verdict=llm_v, planner_verdict=env.verdict,
                      same_verdict=llm_v == env.verdict, deterministic="outputs read from ep['table'] by (tool, n-th call); no resampling")
            el = dict(r["elicit"], planner="exact DP on the recovered model, replayed offline from " + a.src_cond)
            out = dict(episode=eid, family=r["family"], scenario=r["scenario"], condition=a.dst_cond, model=r["model"],
                       protocol=r["protocol"], status="submitted" if env.done else "turn_cap", verdict=env.verdict, log=env.log,
                       n_api=0, elicit=el, derived_from=a.src_cond, replay=rp, models_returned=r.get("models_returned", []),
                       finish_reasons=[], tokens=dict(prompt=0, completion=0, reasoning=0, cache_hit=0, cache_miss=0), latency=0.0,
                       max_tokens=r.get("max_tokens"), max_turns=r.get("max_turns"), api=[], transcript=r["transcript"][:3],
                       time=time.strftime("%Y-%m-%d %H:%M:%S"), sample=smp)
            fo.write(json.dumps(out, ensure_ascii=False) + "\n"); n += 1
    print(f"wrote {n} {a.dst_cond} episodes to {a.out}")

if __name__ == "__main__":
    main()
