"""Offline K2p: replay the exact planner on the observation models an LLM already wrote in K2e (or K1e) runs. No LLM calls.

K2e and K2p have the identical system prompt and first user message, so the model JSON of a K2e run is a draw from exactly the
distribution K2p would sample; planning from it gives a K2p episode for free (the actor, not the recovery, is replaced).

  python k2p_replay.py logs\\k2e.jsonl --out logs\\k2p_replay.jsonl [--src-cond K2e --dst-cond K2p]
Records carry derived_from=<src cond>, n_api=0 and zero tokens (the LLM cost was paid by the source run)."""
import argparse, json, os, time
from env import Env, HERE, load_episodes
import k2decomp as K

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("--out", required=True)
    ap.add_argument("--src-cond", default="K2e"); ap.add_argument("--dst-cond", default="K2p"); a = ap.parse_args()
    eps = load_episodes(); n = 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for line in open(a.log, encoding="utf-8"):
            r = json.loads(line)
            if r["condition"] != a.src_cond or not r.get("elicit"): continue
            ep = eps[r["episode"]]; env = Env(ep, a.dst_cond, r["protocol"]); K.plan_and_play(ep, r["elicit"]["spec"], env)
            el = dict(r["elicit"], planner="exact DP on the recovered model, replayed offline from " + a.src_cond)
            out = dict(episode=r["episode"], family=r["family"], scenario=r["scenario"], condition=a.dst_cond, model=r["model"],
                       protocol=r["protocol"], status="submitted" if env.done else "turn_cap", verdict=env.verdict, log=env.log,
                       n_api=0, elicit=el, derived_from=a.src_cond, models_returned=r.get("models_returned", []), finish_reasons=[],
                       tokens=dict(prompt=0, completion=0, reasoning=0, cache_hit=0, cache_miss=0), latency=0.0,
                       max_tokens=r.get("max_tokens"), max_turns=r.get("max_turns"), api=[], transcript=r["transcript"][:3],
                       time=time.strftime("%Y-%m-%d %H:%M:%S"), sample=r.get("sample", 0))
            fo.write(json.dumps(out, ensure_ascii=False) + "\n"); n += 1
    print(f"wrote {n} {a.dst_cond} episodes to {a.out}")

if __name__ == "__main__":
    main()
