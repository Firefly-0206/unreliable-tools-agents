"""
Description versus execution loss in the second domain, split by JSON structure (post hoc, no model call).
  python d2_de_split.py [--workers 4] [--runs flash terra sonnet]

Uses d2_analysis.work (same scoring and same-JSON replay). Per episode, K2r regret estimates the description loss D of
the stored JSON (Prop. decomp: K2r = D(Mhat) in expectation) and K2e - K2r the execution loss E. Reported on DN and on
all 200 episodes, for JSONs with the correct fault structure (json_ok, as in d2_analysis) and the rest.
Output: results/d2_de_split.json, results/d2_de_split.txt
"""
import os
os.environ["RAEA_DOMAIN"] = "2"
os.environ.setdefault("RAEA_EPISODES", "episodes_d2.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_d2.jsonl")
import argparse, json
from multiprocessing import Pool
import numpy as np
from d2_analysis import work, ci, RUNS, P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runs", nargs="*", default=list(RUNS)); a = ap.parse_args()
    from env import load_episodes, load_truth
    EP, TR = load_episodes(P("episodes_d2.jsonl")), load_truth(P("ground_truth_d2.jsonl"))
    out, L = {}, ["Second domain: description loss D (K2r) and execution loss E (K2e - K2r) by JSON structure (post hoc)", ""]
    for key in a.runs:
        label, stem = RUNS[key]; recs = {}
        for line in open(P("logs", f"d2_{stem}_k2e.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["condition"] == "K2e" and not r["status"].startswith("api_error") and r.get("sample", 0) == 0: recs[r["episode"]] = r
        ids = sorted(EP); assert all(i in recs for i in ids)
        with Pool(a.workers) as pool: res = dict(pool.imap_unordered(work, [(EP[i], TR[i], recs[i]) for i in ids], chunksize=1))
        R = [dict(res[i], id=i, part=EP[i]["part"], draw=EP[i]["pair"], sc=EP[i]["scenario"]) for i in ids]
        o = {}
        for part, sel in [("DN", lambda d: d["part"] == "DN"), ("all", lambda d: True)]:
            for js, jsel in [("all", lambda d: True), ("struct_ok", lambda d: d["json_ok"] == 1.0), ("struct_wrong", lambda d: d["json_ok"] != 1.0)]:
                rows = [d for d in R if sel(d) and jsel(d)]
                if not rows: continue
                C = lambda f: ci([f(d) for d in rows], [d["draw"] for d in rows], [d["sc"] for d in rows]) if len(rows) > 1 else dict(mean=float(f(rows[0])), lo=None, hi=None, n=1)
                o[f"{part}|{js}"] = dict(n=len(rows), K2e=C(lambda d: d["m"]["regret"]), D=C(lambda d: d["r"]["regret"]),
                                         E=C(lambda d: d["m"]["regret"] - d["r"]["regret"]))
                x = o[f"{part}|{js}"]; f = lambda c: f"{c['mean']:.3f}" + (f" [{c['lo']:.3f},{c['hi']:.3f}]" if c.get("lo") is not None else "")
                L.append(f"{label:24s} {part:3s} {js:12s} n={len(rows):3d}  K2e {f(x['K2e'])}  D=K2r {f(x['D'])}  E=K2e-K2r {f(x['E'])}")
        out[key] = o; L.append("")
        json.dump(out, open(P("results", "d2_de_split.json"), "w"), indent=1)
    open(P("results", "d2_de_split.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
