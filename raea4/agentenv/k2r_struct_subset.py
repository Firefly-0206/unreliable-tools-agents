"""
K2e (LLM acts on its own JSON) vs K2r (exact planner acts on the same JSON) on the CORE episodes whose K2e JSON
was written by the model (no full fallback) and has the exact tool grouping.  Also counts, per model, how many
K2e episodes produced a usable JSON and how many of those have the exact grouping.

python k2r_struct_subset.py  ->  results/k2r_struct_subset.{json,txt}
"""
import json, os
from collections import defaultdict
import numpy as np
from env import HERE, CORE, load_episodes, load_truth
from nestboot import nested_ci, paired_ci
import k2decomp as K2D
from score import per_run

RUNS = {"flash": ["logs/ds_chat_fc_k2decomp.jsonl", "logs/ds_chat_fc_k2e_replay.jsonl"],
        "terra": ["logs/codex_terra_medium_k2decomp.jsonl", "logs/codex_terra_k2e_replay.jsonl"],
        "terra_high": ["logs/codex_terra_high.jsonl", "logs/codex_terra_high_k2e_replay.jsonl"],
        "sonnet": ["logs/claude_s55.jsonl", "logs/claude_s55_k2e_replay.jsonl"]}

def load(files):
    recs = {}
    for f in files:
        for line in open(os.path.join(HERE, f), encoding="utf-8"):
            r = json.loads(line)
            if r["status"].startswith("api_error") or r["family"] not in CORE or r["condition"] not in ("K2e", "K2r"): continue
            recs[(r["episode"], r["condition"])] = r       # later successful rerun replaces an earlier row
    return recs

def main():
    eps, truth = load_episodes(), load_truth()
    sc = {i: e["scenario"] for i, e in eps.items()}
    out, lines = {}, []
    for name, files in RUNS.items():
        recs = load(files)
        e2 = {e: r for (e, c), r in recs.items() if c == "K2e"}
        rr = {e: r for (e, c), r in recs.items() if c == "K2r"}
        ids = sorted(set(e2) & set(rr))
        reg = {c: {e: per_run(src[e], eps[e], truth[e])["regret"] for e in ids} for c, src in [("K2e", e2), ("K2r", rr)]}
        info = {}
        for e in ids:
            sp = e2[e]["elicit"]["spec"]
            rec = K2D.score_recovery(eps[e], sp)
            info[e] = dict(written=not sp["fallback"], struct=rec["struct_exact"] == 1.0)
        n_written = sum(v["written"] for v in info.values())
        n_struct_written = sum(v["written"] and v["struct"] for v in info.values())
        n_struct_all = sum(v["struct"] for v in info.values())
        res = dict(n=len(ids), n_written=n_written, n_struct_written=n_struct_written, n_struct_incl_fallback=n_struct_all)
        for sub, keep in [("all", lambda v: True), ("written_struct_exact", lambda v: v["written"] and v["struct"]),
                          ("not_written_struct_exact", lambda v: not (v["written"] and v["struct"]))]:
            s = [e for e in ids if keep(info[e])]
            if not s: res[sub] = dict(n=0); continue
            m = {}
            for c in ("K2e", "K2r"):
                mu, lo, hi = nested_ci([reg[c][e] for e in s], s, [sc[e] for e in s]); m[c] = dict(mean=mu, lo=lo, hi=hi)
            d, lo, hi, n = paired_ci({e: reg["K2r"][e] for e in s}, {e: reg["K2e"][e] for e in s}, sc)
            m["K2r_minus_K2e"] = dict(diff=d, lo=lo, hi=hi, n=n); m["n"] = len(s)
            res[sub] = m
        out[name] = res
        lines.append(f"{name}: CORE n={res['n']}  usable JSON {n_written}/{res['n']}  exact grouping among usable "
                     f"{n_struct_written}/{n_written}  exact grouping incl. fallback {n_struct_all}/{res['n']}")
        for sub in ("all", "written_struct_exact", "not_written_struct_exact"):
            m = res[sub]
            if not m.get("n"): continue
            lines.append(f"  {sub:26s} n={m['n']:3d}  K2e {m['K2e']['mean']:.3f} [{m['K2e']['lo']:.2f},{m['K2e']['hi']:.2f}]"
                         f"  K2r {m['K2r']['mean']:.3f} [{m['K2r']['lo']:.3f},{m['K2r']['hi']:.3f}]"
                         f"  K2r-K2e {m['K2r_minus_K2e']['diff']:+.3f} [{m['K2r_minus_K2e']['lo']:+.2f},{m['K2r_minus_K2e']['hi']:+.2f}]")
    json.dump(out, open(os.path.join(HERE, "results", "k2r_struct_subset.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "k2r_struct_subset.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
