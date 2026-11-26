"""figs/w_flip.pdf, tables/agent_w_cells.tex, tables/agent_w_contrasts.tex, tables/nums_w.tex from
raea4/agentenv/results/w_analysis.json (w_analysis.py; plan PLAN_w_flip.md). No model call.

    python paper/scripts/make_w_outputs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(os.path.join(os.path.dirname(ROOT), "raea4", "agentenv", "results", "w_analysis.json"), encoding="utf-8"))
RUNS = [(k, t, l, c, m) for k, t, l, c, m in [("flash", "Flash", "flash", "#D55E00", "o"), ("terra", "Terra", "terra (medium)", "#0072B2", "s"),
                                              ("sonnet", "Sonnet", "Sonnet 5.5 (high)", "#009E73", "^")] if k in A]
LV = [0.55, 0.65, 0.75, 0.85, 0.95]; NL, BS, MINUS = chr(10), chr(92), "$-$"
LN = {0.55: "FiftyFive", 0.65: "SixtyFive", 0.75: "SeventyFive", 0.85: "EightyFive", 0.95: "NinetyFive"}


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
fig, axes = plt.subplots(1, 2, figsize=(4.3, 1.55), sharey=True, constrained_layout=True)
k0 = RUNS[0][0]
for ax, (fk, ok, title) in zip(axes, [("first_rep", "oracle_rep", "opens with the replica tool"), ("first_log", "oracle_log", "opens with the log tool")]):
    ax.plot(LV, [100 * A[k0]["cells"][f"A|{q:.2f}"][ok] for q in LV], color="black", ls="--", lw=1.1, marker="D", ms=2.8, label="exact oracle", zorder=2)
    for i, (key, tag, lab, col, mk) in enumerate(RUNS):
        c = [A[key]["cells"][f"A|{q:.2f}"][fk] for q in LV]; off = (i - (len(RUNS) - 1) / 2) * 0.008; m = [100 * x["mean"] for x in c]
        ax.errorbar([q + off for q in LV], m, yerr=[[a - 100 * x["lo"] for a, x in zip(m, c)], [100 * x["hi"] - a for a, x in zip(m, c)]],
                    color=col, marker=mk, ms=3.2, lw=1.0, elinewidth=0.7, capsize=1.3, label=lab, zorder=3)
    ax.set_title(title, fontsize=7, pad=2); ax.set_xticks(LV); ax.set_xlabel("accuracy of the log tool", labelpad=1)
    ax.set_ylim(-5, 105); ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis="y", lw=0.3, color="0.88")
axes[0].set_ylabel("share of first calls (%)", labelpad=1); axes[1].legend(loc="lower right", handlelength=1.6, labelspacing=0.25, borderaxespad=0.6)
out = os.path.join(ROOT, "figs", "w_flip.pdf"); fig.savefig(out); fig.savefig(out[:-4] + ".png", dpi=200)

mac, rows, crow = [], [], []
def M(name, val, src): mac.append(BS + "providecommand{" + BS + "w" + name + "}{" + str(val) + "}% src: w_analysis.json:" + src)


CN = [("W1 first replica model-oracle|A 0.55,0.65", "LowGap", 100), ("W1 under replica|A 0.55,0.65", "LowUnder", 1), ("W2 first replica 0.55-0.95|A", "Track", 100),
      ("W3 first log model-oracle|A 0.85,0.95", "HighLogGap", 100), ("W3 first replica|A 0.85,0.95", "HighRep", 100), ("W3 under log|A 0.85,0.95", "HighLogUnder", 1),
      ("W4 first replica model-oracle|B 0.75", "BSeventyFiveGap", 100), ("W4 first replica model-oracle|B 0.65", "BSixtyFiveGap", 100), ("W4 first replica drawn-0.65|B", "BTrack", 100),
      ("W5 K2r-K2e|all", "Gap", 1), ("W6 first-action match, as-if|all", "AsifMatch", 100), ("W6 first-action match, oracle|all", "OracleMatch", 100),
      ("W6 first-action match, as-if minus oracle|all", "AsifDiff", 100)]
for key, tag, lab, _, _ in RUNS:
    o = A[key]; D = o["contrasts"]
    M(tag + "Usable", o["usable"], f"{key}.usable"); M(tag + "NoSubmit", o["no_submit"], f"{key}.no_submit"); M(tag + "G", f"{o['g']:g}", f"{key}.g")
    for cn, c in o["cells"].items():
        ax, q = cn.split("|"); n = ax + LN[float(q)]
        M(f"{tag}{n}Rep", pc(c["first_rep"]["mean"]), f"{key}.cells[{cn}].first_rep"); M(f"{tag}{n}Log", pc(c["first_log"]["mean"]), f"{key}.cells[{cn}].first_log")
        M(f"{tag}{n}Regret", num(c["regret"]["mean"]), f"{key}.cells[{cn}].regret"); M(f"{tag}{n}Verdict", num(c["verdict"]["mean"]), f"{key}.cells[{cn}].verdict")
        rows.append(f"{lab if cn == 'A|0.55' else ''} & {'A: log' if cn == 'A|0.55' else ('B: replica' if cn == 'B|0.75' else '')} & {q} & {pc(c['first_rep']['mean'])} & {pc(c['oracle_rep'])} & {pc(c['asif_rep'])} & "
                    f"{pc(c['first_log']['mean'])} & {pc(c['oracle_log'])} & {pc(c['asif_log'])} & {pc(c['first_other'])} & {pc(c['first_stop'])} & {num(c['regret']['mean'])} & {num(c['K2r']['mean'])} & "
                    f"{num(c['under_rep']['mean'])} & {num(c['under_log']['mean'])} & {num(c['over_rep']['mean'])} & {num(c['verdict']['mean'])} " + BS * 2)
    rows.append(BS + "addlinespace")
    for name, mn, k in CN:
        if name in D: M(tag + mn, ci(D[name], k == 100 or mn == "Gap", k, 0 if k == 100 else 2), f"{key}.{name}")
    w = o["w6_disagree"]; M(tag + "DisN", w["n"], f"{key}.w6_disagree.n"); M(tag + "DisAsif", pc(w["model_with_asif"]), f"{key}.w6_disagree"); M(tag + "DisOracle", pc(w["model_with_oracle"]), f"{key}.w6_disagree")
    f = lambda n, k=100, plus=True: ci(D[n], plus, k, 0 if k == 100 else 2)
    crow.append(f"{lab} & {f(CN[0][0])} & {f(CN[2][0])} & {f(CN[3][0])} & {f(CN[4][0], 100, False)} & {f(CN[6][0])} & {f(CN[7][0])} & {f(CN[9][0], 1)} & "
                f"{pc(D[CN[10][0]]['mean'])} / {pc(D[CN[11][0]]['mean'])} & {f(CN[12][0])} " + BS * 2)
for cn, c in A[k0]["cells"].items():
    ax, q = cn.split("|"); M(f"Oracle{ax}{LN[float(q)]}Rep", pc(c["oracle_rep"]), f"cells[{cn}].oracle_rep"); M(f"Oracle{ax}{LN[float(q)]}Log", pc(c["oracle_log"]), f"cells[{cn}].oracle_log")
hdr = "% generated by scripts/make_w_outputs.py from raea4/agentenv/results/w_analysis.json; do not edit"
t1 = [hdr, BS + "begin{tabular}{@{}llrrrrrrrrrrrrrrr@{}}", BS + "toprule",
      " & & & " + BS + "multicolumn{3}{c}{first: replica (" + BS + "%)} & " + BS + "multicolumn{3}{c}{first: log (" + BS + "%)} & & & " + BS + "multicolumn{6}{c}{regret per episode} " + BS * 2
      + BS + "cmidrule(lr){4-6}" + BS + "cmidrule(lr){7-9}" + BS + "cmidrule(lr){12-17}",
      "Run & Axis & accuracy & model & oracle & as-if & model & oracle & as-if & other (" + BS + "%) & none (" + BS + "%) & K2e & K2r & under-use replica & under-use log & over-use replica & verdict " + BS * 2,
      BS + "midrule"] + rows[:-1] + [BS + "bottomrule", BS + "end{tabular}"]
t2 = [hdr, BS + "begin{tabular}{@{}lccccccccc@{}}", BS + "toprule",
      " & W1 & W2 & " + BS + "multicolumn{2}{c}{W3 (log accuracy 0.85, 0.95)} & " + BS + "multicolumn{2}{c}{W4: replica, model$-$oracle} & W5 & " + BS + "multicolumn{2}{c}{W6: first action} " + BS * 2
      + BS + "cmidrule(lr){4-5}" + BS + "cmidrule(lr){6-7}" + BS + "cmidrule(lr){9-10}",
      "Run & replica, model$-$oracle & replica, 0.55$-$0.95 & log, model$-$oracle & replica share & acc.\\ 0.75 & acc.\\ 0.65 & K2r$-$K2e & match: as-if / oracle (" + BS + "%) & as-if$-$oracle " + BS * 2,
      BS + "midrule"] + crow + [BS + "bottomrule", BS + "end{tabular}"]
for name, t in (("agent_w_cells.tex", t1), ("agent_w_contrasts.tex", t2), ("nums_w.tex", [hdr] + mac)):
    open(os.path.join(ROOT, "tables", name), "w", encoding="utf-8").write(NL.join(t) + NL)
print(len(mac), "macros;", out)
