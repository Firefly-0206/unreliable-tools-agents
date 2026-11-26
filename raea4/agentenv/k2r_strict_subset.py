"""
K2e (LLM acts on its own JSON) vs K2r (exact planner on the same JSON) on STRICT subsets of the CORE episodes.
Offline, post hoc (added 2026-11-14 in response to a review): no model call, existing logs only.

  as_written : the parser changed nothing: no full fallback, no documented default filled in, nothing clipped,
               no parser note (unknown / duplicate tool, malformed group).
  as_written_correct : as_written, and the written model is right: exact tool grouping, persistence right,
               and EVERY parameter separately within TOL (maximum, not mean, error): each tool's fault rate, each
               tool's accuracy, each entry of the stale-emission vector.  TOL = 0.02 (prompt numbers are rounded).
               (First version, 2026-11-14, used mean errors; the subsets are identical under both definitions.)

python k2r_strict_subset.py  ->  results/k2r_strict_subset.{json,txt}
"""
import json, os
from env import HERE, CORE, load_episodes, load_truth
from nestboot import nested_ci, paired_ci
import k2decomp as K2D
from score import per_run

TOL = 0.02
RUNS = {"flash": ["logs/ds_chat_fc_k2decomp.jsonl", "logs/ds_chat_fc_k2e_replay.jsonl"],
        "terra": ["logs/codex_terra_medium_k2decomp.jsonl", "logs/codex_terra_k2e_replay.jsonl"],
        "terra_high": ["logs/codex_terra_high.jsonl", "logs/codex_terra_high_k2e_replay.jsonl"],
        "sonnet": ["logs/claude_s55.jsonl", "logs/claude_s55_k2e_replay.jsonl"],
        "confirm_flash": ["logs/confirm_ds_chat_fc_k2e.jsonl", "logs/confirm_ds_chat_fc_k2e_replay.jsonl"],
        "confirm_terra": ["logs/confirm_codex_terra_k2e.jsonl", "logs/confirm_codex_terra_k2e_replay.jsonl"],
        "confirm_sonnet": ["logs/confirm_claude_s55_k2e.jsonl", "logs/confirm_claude_s55_k2e_replay.jsonl"]}

def load(files):
    recs = {}
    for f in files:
        for line in open(os.path.join(HERE, f), encoding="utf-8"):
            r = json.loads(line)
            if r["status"].startswith("api_error") or r["family"] not in CORE or r["condition"] not in ("K2e", "K2r"): continue
            recs[(r["episode"], r["condition"])] = r       # later successful rerun replaces an earlier row
    return recs

def flags(ep, sp):
    rec = K2D.score_recovery(ep, sp)
    as_written = (not sp["fallback"]) and not sp["defaults"] and not sp["clipped"] and not sp["notes"]
    # every parameter separately (maximum, not mean, error): each tool's accuracy, each tool's fault rate, and each entry
    # of the stale-emission vector of the replica group when the truth has a fault
    tr = K2D.truth_spec(ep); n = len(tr["accuracy"])
    acc_max = max(abs(a - b) for a, b in zip(sp["accuracy"], tr["accuracy"]))
    phi_max = max(abs(a - b) for a, b in zip(K2D._phi_eff(sp), K2D._phi_eff(tr)))
    em_max = 0.0
    for gt in tr["groups"]:
        if gt["shared_fault"] and gt["phi"] > 0:
            for j in gt["tools"]:
                g = next(x for x in sp["groups"] if j in x["tools"])
                em_max = max(em_max, max(abs(a - b) for a, b in zip(g["emission"], gt["emission"])) if g["shared_fault"] else 1.0)
    correct = (rec["struct_exact"] == 1.0 and rec["persistent_ok"] in (None, 1.0) and rec["fault_present_ok"] == 1.0
               and acc_max <= TOL + 1e-12 and phi_max <= TOL + 1e-12 and em_max <= TOL + 1e-12)
    return dict(as_written=as_written, as_written_correct=as_written and correct, acc_max=acc_max, phi_max=phi_max, em_max=em_max)

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
        info = {e: flags(eps[e], e2[e]["elicit"]["spec"]) for e in ids}
        res = dict(n=len(ids), tol=TOL)
        lines.append(f"{name}: CORE n={len(ids)}")
        for sub in ("as_written", "as_written_correct"):
            s = [e for e in ids if info[e][sub]]
            if not s: res[sub] = dict(n=0); lines.append(f"  {sub:20s} n=  0"); continue
            m = {}
            for c in ("K2e", "K2r"):
                mu, lo, hi = nested_ci([reg[c][e] for e in s], s, [sc[e] for e in s]); m[c] = dict(mean=mu, lo=lo, hi=hi)
            m["K2r_max"] = max(reg["K2r"][e] for e in s)
            d, lo, hi, n = paired_ci({e: reg["K2r"][e] for e in s}, {e: reg["K2e"][e] for e in s}, sc)
            m["K2r_minus_K2e"] = dict(diff=d, lo=lo, hi=hi, n=n); m["n"] = len(s)
            res[sub] = m
            lines.append(f"  {sub:20s} n={m['n']:3d}  K2e {m['K2e']['mean']:.3f} [{m['K2e']['lo']:.2f},{m['K2e']['hi']:.2f}]"
                         f"  K2r {m['K2r']['mean']:.3f} (max {m['K2r_max']:.3f})"
                         f"  K2r-K2e {d:+.3f} [{lo:+.2f},{hi:+.2f}]")
        out[name] = res
    json.dump(out, open(os.path.join(HERE, "results", "k2r_strict_subset.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "k2r_strict_subset.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
