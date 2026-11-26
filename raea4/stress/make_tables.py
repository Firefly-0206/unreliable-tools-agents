"""Render stress/results/S*.json as markdown. Writes TABLES.md (every cell, every policy) and prints compact
regret tables (DP / VOI-1 / VOI-3 regret per agent) used in SUMMARY.md."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
f2 = lambda x: f"{x:.2f}"

def agents(pol):
    return [p[3:-1] for p in pol if p.startswith("DP[")]

def full(exp, res):
    L = [f"## {exp}\n"]
    for fam, cells in res.items():
        for par, c in cells.items():
            P = c["policies"]; n = next(iter(P.values()))["n"]
            L.append(f"### {exp} {fam} param={par} (n={n} instances; Oracle-DP total {P['Oracle-DP']['total']:.2f})\n")
            L.append("| policy | regret [95% CI] | severe | esc | calls | repeat-src calls | probe cost |")
            L.append("|---|---|---|---|---|---|---|")
            for p, r in P.items():
                L.append(f"| {p} | {f2(r['regret'])} [{f2(r['lo'])}, {f2(r['hi'])}] | {r['severe']:.3f} | {r['esc']:.3f} "
                         f"| {f2(r['calls'])} | {f2(r['repeat'])} | {f2(r['cost'])} |")
            if c["diag"]:
                L.append("\ndiagnostics: " + "; ".join(f"{k}={v['mean']:.3g} (max {v['max']:.3g}, n={v['n_valid']})"
                                                       for k, v in c["diag"].items() if v))
            L.append("")
    return "\n".join(L)

def compact(exp, res):
    L = []
    for fam, cells in res.items():
        first = next(iter(cells.values()))["policies"]; ag = agents(first)
        L.append(f"\n**{exp} {fam}** regret DP / VOI-1 / VOI-3 (DP 95% CI)\n")
        L.append("| param | " + " | ".join(ag) + " |"); L.append("|---" * (len(ag) + 1) + "|")
        for par, c in cells.items():
            P = c["policies"]; row = []
            for a in ag:
                d = P[f"DP[{a}]"]; v = P[f"VOI1[{a}]"]; v3 = P.get(f"VOI3[{a}]")
                row.append(f"{f2(d['regret'])} / {f2(v['regret'])}" + (f" / {f2(v3['regret'])}" if v3 else "")
                           + f" [{f2(d['lo'])},{f2(d['hi'])}]")
            L.append(f"| {par} | " + " | ".join(row) + " |")
        L.append(f"\n**{exp} {fam}** DP policies: severe / esc / calls / repeat-src calls\n")
        L.append("| param | Oracle-DP | " + " | ".join(a for a in ag if a != "true") + " |")
        L.append("|---" * (len(ag) + 1) + "|")
        for par, c in cells.items():
            P = c["policies"]
            g = lambda r: f"{r['severe']:.3f} / {r['esc']:.2f} / {r['calls']:.2f} / {r['repeat']:.2f}"
            L.append(f"| {par} | {g(P['Oracle-DP'])} | " + " | ".join(g(P[f'DP[{a}]']) for a in ag if a != "true") + " |")
    return "\n".join(L)

if __name__ == "__main__":
    exps = sys.argv[1:] or ["S1", "S2", "S3", "S4", "S5"]
    out = ["# Stress tests: full tables\n", "regret = expected total loss minus Oracle-DP of the TRUE environment, "
           "paired per instance, 95% cluster-bootstrap CI over instances. severe/esc are probabilities; calls and "
           "repeat-source calls are expectations.\n"]
    for e in exps:
        p = os.path.join(HERE, "results", f"{e}.json")
        if not os.path.exists(p): continue
        res = json.load(open(p)); out.append(full(e, res)); print(compact(e, res))
    open(os.path.join(HERE, "TABLES.md"), "w", encoding="utf-8").write("\n".join(out))
