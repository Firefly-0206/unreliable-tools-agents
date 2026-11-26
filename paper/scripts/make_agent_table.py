"""Exact non-LLM baselines of the closed-loop environment -> LaTeX tables.

Reads  raea4/agentenv/results/baselines.json   (by_family, twin_contrasts)
Writes paper/tables/agent_baselines.tex        (main text: regret, 6 families)
       paper/tables/agent_baselines_ci.tex     (appendix: regret with 95% nested-bootstrap CI)
       paper/tables/agent_twins.tex            (appendix: twin contrasts)
       paper/tables/agent_oracle_behaviour.tex (appendix: oracle behaviour)

    python paper/scripts/make_agent_table.py
"""
import json
import os

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "raea4", "agentenv", "results", "baselines.json")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables")
os.makedirs(OUT, exist_ok=True)

J = json.load(open(SRC, encoding="utf-8"))
BF = J["by_family"]
FAMS = ["PERSIST", "SHARED", "MIXED", "VALUE", "TRANSIENT_P", "TRANSIENT"]
FAM_TEX = {"PERSIST": "PERSIST", "SHARED": "SHARED", "MIXED": "MIXED", "VALUE": "VALUE",
           "TRANSIENT_P": r"TRANS.\,P", "TRANSIENT": "TRANS."}
# (json key, label). Labels distinguish discarding existing reports from forbidding future calls.
ROWS = [
    ("Oracle-VOI1", r"Oracle, one-step VOI"),
    ("FaultUnaware-DP", r"Fault-unaware ($\varphi{=}0$) planner"),
    ("MMI-DP", r"MMI planner"),
    ("MMI-VOI1", r"MMI, one-step VOI"),
    ("Oracle+ForbidCache-DP", r"Forbid repeat cache calls"),
    ("Oracle+ForbidAny-DP", r"Forbid any repeat call"),
    ("Oracle+DiscardCache-aware-DP", r"Discard repeat cache reports, aware"),
    ("Oracle+DiscardCache-unaware-DP", r"Discard repeat cache reports, unaware"),
    ("Fixed2-A", r"Fixed: tool A twice"),
    ("Probe-once", r"Probe once"),
    ("Prior-only", r"Prior only"),
]


def f2(x):
    return "0" if abs(x) < 5e-3 else f"{x:.2f}"


def reg(fam, pol, k="regret"):
    return BF[fam][pol][k]


def main_table():
    L = []
    L.append(r"\begin{tabular}{@{}l" + "r" * len(FAMS) + r"@{}}")
    L.append(r"\toprule")
    L.append("Policy (regret vs.\\ oracle) & " + " & ".join(FAM_TEX[f] for f in FAMS) + r" \\")
    L.append(r"\midrule")
    L.append(r"Oracle DP, expected loss & " + " & ".join(f2(BF[f]["Oracle-DP"]["total"]) for f in FAMS) + r" \\")
    L.append(r"\midrule")
    for k, lab in ROWS:
        L.append(lab + " & " + " & ".join(f2(reg(f, k)) for f in FAMS) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L) + "\n"


def ci_table():
    L = []
    L.append(r"\begin{tabular}{@{}l" + "r" * len(FAMS) + r"@{}}")
    L.append(r"\toprule")
    L.append("Policy & " + " & ".join(FAM_TEX[f] for f in FAMS) + r" \\")
    L.append(r"\midrule")
    L.append(r"Oracle DP, expected loss & " + " & ".join(f2(BF[f]["Oracle-DP"]["total"]) for f in FAMS) + r" \\")
    L.append(r"\midrule")
    for k, lab in [("Oracle-DP", None)] + ROWS:
        if lab is None:
            continue
        cells = []
        for f in FAMS:
            r = BF[f][k]
            cells.append(f"{f2(r['regret'])} [{f2(r['regret_lo'])}, {f2(r['regret_hi'])}]")
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L) + "\n"


def twin_table():
    T = J["twin_contrasts"]
    pols = [("MMI-DP", "MMI planner"), ("FaultUnaware-DP", r"Fault-unaware planner"),
            ("Oracle-VOI1", "Oracle, one-step VOI"),
            ("Oracle+ForbidAny-DP", "Forbid any repeat"), ("Oracle+ForbidCache-DP", "Forbid repeat cache calls"),
            ("Fixed2-A", "Fixed: tool A twice"), ("Probe-once", "Probe once")]
    L = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
         r"Policy & SHARED $-$ TRANS. & PERSIST $-$ TRANS.\,P \\", r"\midrule"]
    for k, lab in pols:
        cells = []
        for pair in ["SHARED-minus-TRANSIENT", "PERSIST-minus-TRANSIENT_P"]:
            r = T[f"{pair}|{k}"]
            cells.append(f"{r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]".replace("-", "$-$"))
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def behaviour_table():
    cols = [("calls", "calls"), ("rep_cache", "rep.\\ cache"), ("rep_log", "rep.\\ log"),
            ("used_cache", r"P(cache)"), ("probe_used", r"P(probe)"), ("severe", "severe"), ("esc", "escal.")]
    L = [r"\begin{tabular}{@{}l" + "r" * len(cols) + r"@{}}", r"\toprule",
         "Family & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for f in FAMS:
        o = BF[f]["Oracle-DP"]
        L.append(FAM_TEX[f] + " & " + " & ".join(f"{o[k]:.2f}" if k not in ("severe", "esc") else f"{o[k]:.3f}" for k, _ in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


for name, fn in [("agent_baselines", main_table), ("agent_baselines_ci", ci_table),
                 ("agent_twins", twin_table), ("agent_oracle_behaviour", behaviour_table)]:
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as fh:
        fh.write("% generated by scripts/make_agent_table.py from raea4/agentenv/results/baselines.json; do not edit\n")
        fh.write(fn())
    print("wrote", name)

