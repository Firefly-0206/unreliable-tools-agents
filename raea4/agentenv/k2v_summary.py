"""Summary of the myopic same-JSON replay (K2v, k2v_replay.py) against K2e (LLM acts on its JSON) and K2r (exact DP on the
same JSON), on CORE episodes; paired nested-bootstrap CIs (scenarios > episodes), as in k2r_struct_subset.py.
Reference: Oracle-VOI1 = the same one-step planner on the TRUE model (results/baselines.json), regret against Oracle-DP.

python k2v_summary.py  ->  results/k2v_summary.{json,txt}"""
import json, os
import numpy as np
from env import HERE, CORE, FAMILIES, load_episodes, load_truth
from nestboot import nested_ci, paired_ci
import k2decomp as K2D
from score import per_run

RUNS = {"flash": ["logs/ds_chat_fc_k2decomp.jsonl", "logs/ds_chat_fc_k2e_replay.jsonl", "logs/ds_chat_fc_k2v_replay.jsonl"],
        "terra": ["logs/codex_terra_medium_k2decomp.jsonl", "logs/codex_terra_k2e_replay.jsonl", "logs/codex_terra_k2v_replay.jsonl"],
        "terra_high": ["logs/codex_terra_high.jsonl", "logs/codex_terra_high_k2e_replay.jsonl", "logs/codex_terra_high_k2v_replay.jsonl"],
        "sonnet": ["logs/claude_s55.jsonl", "logs/claude_s55_k2e_replay.jsonl", "logs/claude_s55_k2v_replay.jsonl"]}
CONDS = ("K2e", "K2v", "K2r")

def load(files):
    recs = {}
    for f in files:
        for line in open(os.path.join(HERE, f), encoding="utf-8"):
            r = json.loads(line)
            if r["status"].startswith("api_error") or r["condition"] not in CONDS: continue
            recs[(r["episode"], r["condition"])] = r
    return recs

def main():
    eps, truth = load_episodes(), load_truth()
    sc = {i: e["scenario"] for i, e in eps.items()}
    base = json.load(open(os.path.join(HERE, "results", "baselines.json"), encoding="utf-8"))["per_episode"]
    out, lines = {}, []
    for name, files in RUNS.items():
        recs = load(files)
        ids_all = sorted(e for e in eps if all((e, c) in recs for c in CONDS))
        reg = {c: {e: per_run(recs[(e, c)], eps[e], truth[e])["regret"] for e in ids_all} for c in CONDS}
        struct = {e: (not recs[(e, "K2e")]["elicit"]["spec"]["fallback"]) and
                     K2D.score_recovery(eps[e], recs[(e, "K2e")]["elicit"]["spec"])["struct_exact"] == 1.0 for e in ids_all}
        res = {}
        for sub, fams, keep in [("CORE", CORE, lambda e: True), ("CORE_struct_exact", CORE, lambda e: struct[e])] + \
                               [(f, [f], lambda e: True) for f in FAMILIES if any(eps[e]["family"] == f for e in ids_all)]:
            s = [e for e in ids_all if eps[e]["family"] in fams and keep(e)]
            if not s: continue
            m = {"n": len(s)}
            for c in CONDS:
                mu, lo, hi = nested_ci([reg[c][e] for e in s], s, [sc[e] for e in s]); m[c] = dict(mean=mu, lo=lo, hi=hi)
            ov = [base[e]["Oracle-VOI1"]["total"] - base[e]["Oracle-DP"]["total"] for e in s]
            mu, lo, hi = nested_ci(ov, s, [sc[e] for e in s]); m["OracleVOI1_true"] = dict(mean=mu, lo=lo, hi=hi)
            for a_, b_ in (("K2v", "K2e"), ("K2v", "K2r"), ("K2r", "K2e")):
                d, lo, hi, n = paired_ci({e: reg[a_][e] for e in s}, {e: reg[b_][e] for e in s}, sc)
                m[f"{a_}-{b_}"] = dict(diff=d, lo=lo, hi=hi, n=n)
            m["K2v_le_K2e_share"] = float(np.mean([reg["K2v"][e] <= reg["K2e"][e] + 1e-9 for e in s]))
            res[sub] = m
        out[name] = res
        lines.append(f"{name}")
        for sub, m in res.items():
            lines.append(f"  {sub:18s} n={m['n']:3d}  K2e {m['K2e']['mean']:.3f}  K2v {m['K2v']['mean']:.3f} [{m['K2v']['lo']:.3f},{m['K2v']['hi']:.3f}]"
                         f"  K2r {m['K2r']['mean']:.3f}  | K2v-K2e {m['K2v-K2e']['diff']:+.3f} [{m['K2v-K2e']['lo']:+.3f},{m['K2v-K2e']['hi']:+.3f}]"
                         f"  K2v-K2r {m['K2v-K2r']['diff']:+.3f} [{m['K2v-K2r']['lo']:+.3f},{m['K2v-K2r']['hi']:+.3f}]"
                         f"  | OracleVOI1(true) {m['OracleVOI1_true']['mean']:.3f}  K2v<=K2e {100*m['K2v_le_K2e_share']:.0f}%")
    json.dump(out, open(os.path.join(HERE, "results", "k2v_summary.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "k2v_summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
