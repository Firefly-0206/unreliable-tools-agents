"""tables/nums_u.tex, tables/agent_u.tex, tables/agent_h.tex from raea4/agentenv/results/u_units_{main,held,d2,d3}.json,
r_robust.json and h_horizon_{core,x,m}.json (u_units.py, r_robust.py, h_horizon.py; post hoc, added after review). No model call.

    python paper/scripts/make_u_outputs.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(os.path.dirname(ROOT), "raea4", "agentenv", "results")
J = lambda f: json.load(open(os.path.join(R, f), encoding="utf-8"))
NL, BS, MINUS = chr(10), chr(92), "$-$"
SETS = [("main", "Main", "first domain"), ("held", "Held", "held-out"), ("d2", "Dtwo", "second domain"), ("d3", "Dthree", "third domain")]
RUNS = [("flash", "Flash", "flash"), ("terra", "Terra", "terra"), ("sonnet", "Sonnet", "Sonnet")]
mac = []
GN = {1: "One", 4: "Four"}


def M(name, val, src): mac.append(BS + "providecommand{" + BS + "u" + name + "}{" + str(val) + "}% src: " + src)
def pc(x): return f"{100 * x:.0f}"
def sg(x, nd=2):
    s = f"{abs(x):.{nd}f}"; return (MINUS if x < 0 and float(s) != 0 else ("+" if float(s) != 0 else "")) + s
def cis(d, scale=1, nd=2):
    if scale == 100: return f"{sg(100 * d['mean'], 0)} [{sg(100 * d['lo'], 0)}, {sg(100 * d['hi'], 0)}]"
    return f"{sg(d['mean'], nd)} [{sg(d['lo'], nd)}, {sg(d['hi'], nd)}]"


# ---- loss in agent units
rows = []
for sk, stag, slab in SETS:
    if not os.path.exists(os.path.join(R, f"u_units_{sk}.json")):
        print("WARNING: missing", f"u_units_{sk}.json"); continue
    U = J(f"u_units_{sk}.json")["runs"]
    for i, (rk, rtag, rlab) in enumerate(RUNS):
        o = U[rk]; m = o["mean"]; src = f"u_units_{sk}.json:runs.{rk}"
        for x, X in (("correct", "Correct"), ("wrong_other", "Wrong"), ("severe", "Severe"), ("esc", "Esc")):
            M(f"{stag}{rtag}{X}", pc(m[x]["m"]), src + f".mean.{x}.m"); M(f"{stag}{rtag}{X}Oracle", pc(m[x]["o"]), src + f".mean.{x}.o")
        M(f"{stag}{rtag}Cost", f"{m['cost']['m']:.2f}", src + ".mean.cost.m"); M(f"{stag}{rtag}CostOracle", f"{m['cost']['o']:.2f}", src + ".mean.cost.o")
        M(f"{stag}{rtag}CostDiffCI", cis(o["diff_mo"]["cost"]), src + ".diff_mo.cost")
        M(f"{stag}{rtag}CorrectDiffCI", cis(o["diff_mo"]["correct"], 100), src + ".diff_mo.correct")
        M(f"{stag}{rtag}RealisedDiffCI", cis(o["diff_mo"]["realised"]), src + ".diff_mo.realised")
        rows.append(f"{slab if i == 0 else ''} & {rlab} & {pc(m['correct']['m'])} / {pc(m['correct']['p'])} / {pc(m['correct']['o'])} & "
                    f"{pc(m['severe']['m'])} / {pc(m['severe']['o'])} & {pc(m['wrong_other']['m'])} / {pc(m['wrong_other']['o'])} & "
                    f"{pc(m['esc']['m'])} / {pc(m['esc']['o'])} & {m['cost']['m']:.2f} / {m['cost']['o']:.2f} & "
                    f"{cis(o['diff_mo']['correct'], 100)} & {cis(o['diff_mo']['cost'])} & {cis(o['diff_mo']['realised'])} " + BS * 2)
    if sk != SETS[-1][0]: rows.append(BS + "midrule")
tab_u = (BS + "begin{tabular}{@{}llcccccccc@{}}" + NL + BS + "toprule" + NL
         + " & & Correct (\\%) & False benign & Other wrong & Escalated & Spend & \\multicolumn{3}{c}{Model $-$ oracle [95\\% CI]} " + BS * 2 + NL
         + "\\cmidrule(l){8-10}" + NL
         + "Set & Run & model / planner / oracle & model / oracle & model / oracle & model / oracle & model / oracle & correct (pts) & spend & realized loss " + BS * 2 + NL
         + BS + "midrule" + NL + NL.join(rows) + NL + BS + "bottomrule" + NL + BS + "end{tabular}" + NL)

# ---- ambiguity-averse planner
RB = J("r_robust.json")["runs"]
for rk, rtag, _ in RUNS:
    o = RB[rk]; src = f"r_robust.json:runs.{rk}"
    M(f"Rob{rtag}Delta", f"{o['delta']:g}", src + ".delta"); M(f"Rob{rtag}DeltaIndep", f"{o['delta_indep']:g}", src + ".delta_indep")
    M(f"Rob{rtag}PhiThree", f"{min(.95, .3 + o['delta']):.2f}", src + ".delta (+0.3)"); M(f"Rob{rtag}PhiOne", f"{min(.95, .1 + o['delta']):.2f}", src + ".delta (+0.1)")
    for k, K in (("fit_persistent", "Fit"), ("oracle_fit", "OracleFit"), ("independent", "Indep"), ("oracle_indep", "OracleIndep"), ("randomised", "Rand")):
        M(f"Rob{rtag}{K}", pc(o[k]), src + "." + k)
    M(f"Rob{rtag}Regret", f"{o['regret']:.2f}", src + ".regret"); M(f"Rob{rtag}RegretObs", f"{o['regret_obs']:.2f}", src + ".regret_obs")

# ---- horizon
hrows = []
for sk, slab in (("core", "first domain (CORE)"), ("x", "fault-rate sweep"), ("m", "matched pairs")):
    H = J(f"h_horizon_{sk}.json")["runs"]
    for rk, rtag, rlab in RUNS:
        o = H[rk]
        for g in (1, 4):
            for part in ("first", "later"):
                c = o.get(f"all|g{g}|{part}")
                if not c: continue
                hrows.append(f"{slab} & {rlab} & {g} & {part} ({c['n']}) & " + " & ".join(f"{100 * c['agree'][k]:.0f}" for k in ("1", "2", "3", "4", "inf")) + " " + BS * 2)
        if sk == "m":
            for part in ("first", "later"):
                c = o[f"MZ|g1|{part}"]
                hrows.append(f"matched pairs, fault-free only & {rlab} & 1 & {part} ({c['n']}) & " + " & ".join(f"{100 * c['agree'][k]:.0f}" for k in ("1", "2", "3", "4", "inf")) + " " + BS * 2)
        hrows.append(BS + "addlinespace")
        src = f"h_horizon_{sk}.json:runs.{rk}"
        for g in (1, 4):
            for part in ("first", "later"):
                c = o.get(f"all|g{g}|{part}")
                if c:
                    M(f"H{sk.capitalize()}{rtag}G{GN[g]}{part.capitalize()}KOne", f"{100 * c['agree']['1']:.0f}", src + f".all|g{g}|{part}.agree.1")
                    M(f"H{sk.capitalize()}{rtag}G{GN[g]}{part.capitalize()}KInf", f"{100 * c['agree']['inf']:.0f}", src + f".all|g{g}|{part}.agree.inf")
                    M(f"H{sk.capitalize()}{rtag}G{GN[g]}{part.capitalize()}Max", f"{100 * max(c['agree'].values()):.0f}", src + f".all|g{g}|{part}.agree max")
        if sk == "m":
            c = o["MZ|g1|later"]; M(f"HMZ{rtag}LaterMax", f"{100 * max(c['agree'].values()):.0f}", src + ".MZ|g1|later.agree max")
            M(f"HMZ{rtag}LaterKOne", f"{100 * c['agree']['1']:.0f}", src + ".MZ|g1|later.agree.1"); M(f"HMZ{rtag}LaterKInf", f"{100 * c['agree']['inf']:.0f}", src + ".MZ|g1|later.agree.inf")
if hrows and hrows[-1] == BS + "addlinespace": hrows.pop()
tab_h = (BS + "begin{tabular}{@{}lllcrrrrr@{}}" + NL + BS + "toprule" + NL
         + "Set & Run & $g$ & Decisions ($n$) & $k{=}1$ & 2 & 3 & 4 & $\\infty$ " + BS * 2 + NL + BS + "midrule" + NL + NL.join(hrows) + NL
         + BS + "bottomrule" + NL + BS + "end{tabular}" + NL)

head = "% generated by scripts/make_u_outputs.py from raea4/agentenv/results/u_units_*.json, r_robust.json, h_horizon_*.json; do not edit" + NL
open(os.path.join(ROOT, "tables", "nums_u.tex"), "w", encoding="utf-8").write(head + NL.join(mac) + NL)
open(os.path.join(ROOT, "tables", "agent_u.tex"), "w", encoding="utf-8").write(head + tab_u)
open(os.path.join(ROOT, "tables", "agent_h.tex"), "w", encoding="utf-8").write(head + tab_h)
print(f"{len(mac)} macros")
