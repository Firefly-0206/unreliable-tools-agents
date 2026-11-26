"""figs/d2_tracking.pdf, tables/agent_d2_cells.tex, tables/agent_d2_contrasts.tex, tables/nums_d2.tex from
raea4/agentenv/results/d2_analysis.json (d2_analysis.py; plan PLAN_d2_domain.md). No model call.

    python paper/scripts/make_d2_outputs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(os.path.join(os.path.dirname(ROOT), "raea4", "agentenv", "results", "d2_analysis.json"), encoding="utf-8"))
RUNS = [(k, t, l, c, m) for k, t, l, c, m in [("flash", "Flash", "flash", "#D55E00", "o"), ("terra", "Terra", "terra (medium)", "#0072B2", "s"),
                                              ("sonnet", "Sonnet", "Sonnet 5.5 (high)", "#009E73", "^")] if k in A]
PHIS = [0.10, 0.30, 0.50, 0.70]; PN = {0.10: "Ten", 0.30: "Thirty", 0.50: "Fifty", 0.70: "Seventy"}; NL, BS, MINUS = chr(10), chr(92), "$-$"


def num(x, plus=False, nd=2):
    s = f"{abs(x):.{nd}f}"
    if float(s) == 0: return s
    return (MINUS if x < 0 else ("+" if plus else "")) + s
def ci(d, plus=False, k=1, nd=2): return f"{num(k * d['mean'], plus, nd)} [{num(k * d['lo'], plus, nd)}, {num(k * d['hi'], plus, nd)}]"
def pc(x): return f"{100 * x:.0f}"


plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
                     "legend.frameon": False, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
fig, ax = plt.subplots(1, 1, figsize=(3.1, 1.55), constrained_layout=True); k0 = RUNS[0][0]
ax.plot(PHIS, [100 * A[k0]["cells"][f"DX|{p:.2f}"]["oracle_rep"] for p in PHIS], color="black", ls="--", lw=1.1, marker="D", ms=2.8, label="exact oracle", zorder=2)
for i, (key, tag, lab, col, mk) in enumerate(RUNS):
    c = [A[key]["cells"][f"DX|{p:.2f}"]["first_rep"] for p in PHIS]; off = (i - (len(RUNS) - 1) / 2) * 0.012; m = [100 * x["mean"] for x in c]
    ax.errorbar([p + off for p in PHIS], m, yerr=[[a - 100 * x["lo"] for a, x in zip(m, c)], [100 * x["hi"] - a for a, x in zip(m, c)]],
                color=col, marker=mk, ms=3.2, lw=1.0, elinewidth=0.7, capsize=1.3, label=lab, zorder=3)
ax.set_xticks(PHIS); ax.set_xlabel(r"replica fault rate $\varphi$", labelpad=1); ax.set_ylabel("first call is the replica tool (%)", labelpad=1)
ax.set_ylim(-5, 105); ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis="y", lw=0.3, color="0.88")
ax.set_xlim(0.05, 1.08); ax.legend(loc="center right", handlelength=1.6, labelspacing=0.25, borderaxespad=0.0)
out = os.path.join(ROOT, "figs", "d2_tracking.pdf"); fig.savefig(out); fig.savefig(out[:-4] + ".png", dpi=200)

mac, rows, crow = [], [], []
def M(name, val, src): mac.append(BS + "providecommand{" + BS + "dd" + name + "}{" + str(val) + "}% src: d2_analysis.json:" + src)


CN = [("R1 K2r-K2e|DN", "Gap", 1), ("R1 K2r-K2e|all", "GapAll", 1), ("regret|DN", "Regret", 1), ("R2 first replica model-oracle|DX 0.10,0.30", "LowGap", 100),
      ("R2 under|DX 0.10,0.30", "LowUnder", 1), ("R3 first replica 0.10-0.70|DX", "Track", 100), ("R3 over at 0.70|DX", "HighOver", 1),
      ("R4 under|DN", "Under", 1), ("R4 verdict|DN", "Verdict", 1), ("first replica model-oracle|DN", "NatGap", 100)]
for key, tag, lab, _, _ in RUNS:
    o = A[key]; D = o["contrasts"]
    M(tag + "Usable", o["usable"], f"{key}.usable"); M(tag + "NoSubmit", o["no_submit"], f"{key}.no_submit"); M(tag + "PassShare", pc(o["pass_share_DN"]), f"{key}.pass_share_DN")
    for cn, c in o["cells"].items():
        n = "Nat" if cn == "DN" else PN[float(cn.split("|")[1])]
        M(f"{tag}{n}First", pc(c["first_rep"]["mean"]), f"{key}.cells[{cn}].first_rep"); M(f"{tag}{n}Regret", num(c["regret"]["mean"]), f"{key}.cells[{cn}].regret")
        M(f"{tag}{n}Planner", num(c["K2r"]["mean"]), f"{key}.cells[{cn}].K2r"); M(f"{tag}{n}Verdict", num(c["verdict"]["mean"]), f"{key}.cells[{cn}].verdict")
        M(f"{tag}{n}JsonOk", pc(c["json_ok"]), f"{key}.cells[{cn}].json_ok")
        fm, fo = c["first_model"], c["first_oracle"]
        rows.append(f"{lab if cn == 'DN' else ''} & {'DN (drawn)' if cn == 'DN' else 'DX ' + cn.split('|')[1]} & {pc(c['first_rep']['mean'])} & {pc(c['oracle_rep'])} & {pc(c['planner_rep'])} & "
                    f"{pc(fm[1])} / {pc(fo[1])} & {pc(fm[2] + fm[3])} / {pc(fo[2] + fo[3])} & {pc(fm[4])} / {pc(fo[4])} & {num(c['regret']['mean'])} & {num(c['K2r']['mean'])} & "
                    f"{num(c['under']['mean'])} & {num(c['over']['mean'])} & {num(c['verdict']['mean'])} & {num(c['calls']['mean'])} & {pc(c['json_ok'])} " + BS * 2)
    rows.append(BS + "addlinespace")
    for name, mn, k in CN:
        if name in D: M(tag + mn, ci(D[name], k == 100 or mn.startswith("Gap"), k, 0 if k == 100 else 2), f"{key}.{name}")
    f = lambda n, k=100, plus=True: ci(D[n], plus, k, 0 if k == 100 else 2)
    crow.append(f"{lab} & {f(CN[0][0], 1)} & {f(CN[1][0], 1)} & {f(CN[3][0])} & {f(CN[4][0], 1, False)} & {f(CN[5][0])} & {f(CN[6][0], 1, False)} & {pc(o['pass_share_DN'])} & {f(CN[8][0], 1, False)} " + BS * 2)
for cn, c in A[k0]["cells"].items():
    n = "Nat" if cn == "DN" else PN[float(cn.split("|")[1])]; M(f"Oracle{n}First", pc(c["oracle_rep"]), f"cells[{cn}].oracle_rep")
AG = os.path.join(os.path.dirname(ROOT), "raea4", "agentenv")
_orc = json.load(open(os.path.join(AG, "d2_oracle.json"), encoding="utf-8")); _v = [v["V0"] for k, v in _orc.items() if k.startswith("DN-")]
_ol = sum(_v) / len(_v); M("OracleLoss", num(_ol), "d2_oracle.json: mean V0 over DN")
for key, tag, lab, _, _ in RUNS: M(tag + "RelLoss", pc(A[key]["contrasts"]["regret|DN"]["mean"] / _ol), f"{key}.regret|DN / mean oracle V0 (DN)")
hdr = "% generated by scripts/make_d2_outputs.py from raea4/agentenv/results/d2_analysis.json; do not edit"
t1 = [hdr, BS + "begin{tabular}{@{}llrrrcccrrrrrrr@{}}", BS + "toprule",
      " & & " + BS + "multicolumn{3}{c}{first: replica A (" + BS + "%)} & " + BS + "multicolumn{3}{c}{first action, model / oracle (" + BS + "%)} & " + BS + "multicolumn{5}{c}{regret per episode} & & " + BS * 2
      + BS + "cmidrule(lr){3-5}" + BS + "cmidrule(lr){6-8}" + BS + "cmidrule(lr){9-13}",
      "Run & Episodes & model & oracle & planner & feed B & log or primary & probe & K2e & K2r & under-use & over-use & verdict & calls & JSON correct " + BS + "% " + BS * 2,
      BS + "midrule"] + rows[:-1] + [BS + "bottomrule", BS + "end{tabular}"]
t2 = [hdr, BS + "begin{tabular}{@{}lcccccccc@{}}", BS + "toprule",
      " & " + BS + "multicolumn{2}{c}{R1: K2r$-$K2e} & " + BS + "multicolumn{2}{c}{R2: fault rates 0.10, 0.30} & " + BS + "multicolumn{2}{c}{R3} & " + BS + "multicolumn{2}{c}{R4 (DN)} " + BS * 2
      + BS + "cmidrule(lr){2-3}" + BS + "cmidrule(lr){4-5}" + BS + "cmidrule(lr){6-7}" + BS + "cmidrule(lr){8-9}",
      "Run & DN & all & replica first, model$-$oracle (pp) & under-use & replica first, 0.10$-$0.70 (pp) & over-use at 0.70 & passed-over share (" + BS + "%) & verdict regret " + BS * 2,
      BS + "midrule"] + crow + [BS + "bottomrule", BS + "end{tabular}"]
for name, t in (("agent_d2_cells.tex", t1), ("agent_d2_contrasts.tex", t2), ("nums_d2.tex", [hdr] + mac)):
    open(os.path.join(ROOT, "tables", name), "w", encoding="utf-8").write(NL.join(t) + NL)
print(len(mac), "macros;", out)
