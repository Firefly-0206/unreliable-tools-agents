"""Figure figs/x_tracking.pdf, tables/agent_x_cells.tex, tables/agent_x_contrasts.tex and tables/nums_x.tex from
raea4/agentenv/results/x_analysis.json (x_analysis.py; plan PLAN_x_dependence.md). No model call.

    python paper/scripts/make_x_outputs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(os.path.join(os.path.dirname(ROOT), "raea4", "agentenv", "results", "x_analysis.json"), encoding="utf-8"))
RUNS = [(k, t, l, c, m) for k, t, l, c, m in [("flash", "Flash", "flash", "#D55E00", "o"), ("terra", "Terra", "terra (medium)", "#0072B2", "s"),
                                              ("sonnet", "Sonnet", "Sonnet 5.5 (high)", "#009E73", "^")] if k in A]
PHIS = [0.10, 0.30, 0.50, 0.70]; ARMS = [("XP", "persistent fault"), ("XI", "independent fault")]
NL, BS, MINUS = chr(10), chr(92), "$-$"


def num(x, plus=False, nd=2):
    s = f"{abs(x):.{nd}f}"
    if float(s) == 0: return s
    return (MINUS if x < 0 else ("+" if plus else "")) + s


def ci(d, plus=False, k=1, nd=2): return f"{num(k * d['mean'], plus, nd)} [{num(k * d['lo'], plus, nd)}, {num(k * d['hi'], plus, nd)}]"
def pc(x): return f"{100 * x:.0f}"


# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
                     "legend.frameon": False, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
fig, axes = plt.subplots(1, 2, figsize=(4.3, 1.55), sharey=True, constrained_layout=True)
for ax, (arm, title) in zip(axes, ARMS):
    key0 = RUNS[0][0]
    ax.plot(PHIS, [100 * A[key0]["cells"][f"{arm}|{p:.2f}"]["first_oracle"] for p in PHIS], color="black", ls="--", lw=1.1, marker="D", ms=2.8,
            label="exact oracle", zorder=2)
    for i, (key, tag, lab, col, mk) in enumerate(RUNS):
        c = [A[key]["cells"][f"{arm}|{p:.2f}"]["first_model"] for p in PHIS]; off = (i - (len(RUNS) - 1) / 2) * 0.012
        m = [100 * x["mean"] for x in c]
        ax.errorbar([p + off for p in PHIS], m, yerr=[[a - 100 * x["lo"] for a, x in zip(m, c)], [100 * x["hi"] - a for a, x in zip(m, c)]],
                    color=col, marker=mk, ms=3.2, lw=1.0, elinewidth=0.7, capsize=1.3, label=lab, zorder=3)
    ax.set_title(title, fontsize=7, pad=2); ax.set_xticks(PHIS); ax.set_xlabel(r"fault rate $\varphi$", labelpad=1)
    ax.set_ylim(-5, 105); ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis="y", lw=0.3, color="0.88")
axes[0].set_ylabel("first call is the replica tool (%)", labelpad=1)
axes[1].legend(loc="upper right", handlelength=1.6, labelspacing=0.25, borderaxespad=0.0)
out = os.path.join(ROOT, "figs", "x_tracking.pdf"); fig.savefig(out); fig.savefig(out[:-4] + ".png", dpi=200)

# ------------------------------------------------------------------ tables and macros
mac, rows, crow = [], [], []


def M(name, val, src): mac.append(BS + "providecommand{" + BS + "x" + name + "}{" + str(val) + "}% src: x_analysis.json:" + src)


PN = {0.10: "Ten", 0.30: "Thirty", 0.50: "Fifty", 0.70: "Seventy"}
for key, tag, lab, _, _ in RUNS:
    o = A[key]; D = o["contrasts"]
    M(tag + "Usable", o["usable"], f"{key}.usable"); M(tag + "NoSubmit", o["no_submit"], f"{key}.no_submit")
    for arm, an in ARMS:
        for p in PHIS:
            c = o["cells"][f"{arm}|{p:.2f}"]; pre = f"{tag}{arm}{PN[p]}"
            M(pre + "First", pc(c["first_model"]["mean"]), f"{key}.cells[{arm}|{p:.2f}].first_model")
            M(pre + "Regret", num(c["regret"]["mean"]), f"{key}.cells[{arm}|{p:.2f}].regret")
            M(pre + "Under", num(c["under"]["mean"]), f"{key}.cells[{arm}|{p:.2f}].under")
            M(pre + "Planner", num(c["K2r"]["mean"]), f"{key}.cells[{arm}|{p:.2f}].K2r")
            M(pre + "Over", num(c["over"]["mean"]), f"{key}.cells[{arm}|{p:.2f}].over")
            first = (lab if (arm, p) == ("XP", 0.10) else "")
            rows.append(f"{first} & {an if p == 0.10 else ''} & {p:.2f} & {pc(c['first_model']['mean'])} & {pc(c['first_oracle'])} & {pc(c['first_planner'])} & "
                        f"{num(c['regret']['mean'])} & {num(c['K2r']['mean'])} & {num(c['under']['mean'])} & {num(c['over']['mean'])} & {num(c['verdict']['mean'])} & "
                        f"{num(c['calls']['mean'])} & {pc(c['json_persistent'])} " + BS * 2)
        M(f"{tag}{arm}LowFirstGap", ci(D[f"D1 first model-oracle|{arm}"], True, 100, 0), f"{key}.D1 first model-oracle|{arm} (points)")
        M(f"{tag}{arm}LowUnder", ci(D[f"D1 under|{arm}"]), f"{key}.D1 under|{arm}")
        M(f"{tag}{arm}Track", ci(D[f"D2 first 0.10-0.70|{arm}"], True, 100, 0), f"{key}.D2 first 0.10-0.70|{arm} (points)")
        M(f"{tag}{arm}Gap", ci(D[f"D3 K2r-K2e|{arm}"], True), f"{key}.D3 K2r-K2e|{arm}")
        M(f"{tag}{arm}RegretAll", num(D[f"regret|{arm}"]["mean"]), f"{key}.regret|{arm}")
        M(f"{tag}{arm}JsonPersist", pc(sum(o["cells"][f"{arm}|{p:.2f}"]["json_persistent"] for p in PHIS) / 4), f"{key}.cells[{arm}|*].json_persistent mean")
    M(tag + "ArmFirst", ci(D["D1 first XP-XI"], True, 100, 0), f"{key}.D1 first XP-XI (points)")
    M(tag + "ArmUnder", ci(D["D1 under XP-XI"], True), f"{key}.D1 under XP-XI")
    rows.append(BS + "addlinespace")
    crow.append(f"{lab} & {ci(D['D1 first model-oracle|XP'], True, 100, 0)} & {ci(D['D1 first model-oracle|XI'], True, 100, 0)} & {ci(D['D1 first XP-XI'], True, 100, 0)} & "
                f"{ci(D['D1 under XP-XI'], True)} & {ci(D['D2 first 0.10-0.70|XP'], True, 100, 0)} & {ci(D['D2 first 0.10-0.70|XI'], True, 100, 0)} & "
                f"{ci(D['D3 K2r-K2e|XP'], True)} & {ci(D['D3 K2r-K2e|XI'], True)} " + BS * 2)
key0 = RUNS[0][0]
for arm, _ in ARMS:
    for p in PHIS: M(f"Oracle{arm}{PN[p]}First", pc(A[key0]["cells"][f"{arm}|{p:.2f}"]["first_oracle"]), f"cells[{arm}|{p:.2f}].first_oracle")
hdr = "% generated by scripts/make_x_outputs.py from raea4/agentenv/results/x_analysis.json; do not edit"
t1 = [hdr, BS + "begin{tabular}{@{}llrrrrrrrrrrr@{}}", BS + "toprule",
      " & & & " + BS + "multicolumn{3}{c}{first call replica (" + BS + "%)} & " + BS + "multicolumn{5}{c}{regret per episode} & & " + BS * 2 + BS + "cmidrule(lr){4-6}" + BS + "cmidrule(lr){7-11}",
      "Run & Arm & $" + BS + "varphi$ & model & oracle & planner & K2e & K2r & under-use & over-use & verdict & calls & JSON persistent " + BS + "% " + BS * 2, BS + "midrule"]
t1 += rows[:-1] + [BS + "bottomrule", BS + "end{tabular}"]
t2 = [hdr, BS + "begin{tabular}{@{}lcccccccc@{}}", BS + "toprule",
      " & " + BS + "multicolumn{4}{c}{D1: rates 0.10, 0.30} & " + BS + "multicolumn{2}{c}{D2: tracking} & " + BS + "multicolumn{2}{c}{D3: K2r$-$K2e} " + BS * 2
      + BS + "cmidrule(lr){2-5}" + BS + "cmidrule(lr){6-7}" + BS + "cmidrule(lr){8-9}",
      "Run & m$-$o, pers. & m$-$o, indep. & pers.$-$indep. & under-use & pers. & indep. & pers. & indep. " + BS * 2,
      BS + "midrule"] + crow + [BS + "bottomrule", BS + "end{tabular}"]
for name, t in (("agent_x_cells.tex", t1), ("agent_x_contrasts.tex", t2), ("nums_x.tex", [hdr] + mac)):
    open(os.path.join(ROOT, "tables", name), "w", encoding="utf-8").write(NL.join(t) + NL)
print(len(mac), "macros;", out)
