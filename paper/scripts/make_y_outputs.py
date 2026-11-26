"""tables/agent_y_a.tex, tables/agent_y_b.tex and tables/nums_y.tex from raea4/agentenv/results/y_analysis.json
(y_analysis.py; plan PLAN_y_surface.md). No model call.

    python paper/scripts/make_y_outputs.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(os.path.join(os.path.dirname(ROOT), "raea4", "agentenv", "results", "y_analysis.json"), encoding="utf-8"))
RUNS = [(k, t, l) for k, t, l in [("flash", "Flash", "flash"), ("terra", "Terra", "terra (medium)"), ("sonnet", "Sonnet", "Sonnet 5.5 (high)")] if k in A]
NL, BS, MINUS = chr(10), chr(92), "$-$"
ARM = [("P", "persistent"), ("I", "independent"), ("all", "both")]


def num(x, plus=False, nd=2):
    s = f"{abs(x):.{nd}f}"
    if float(s) == 0: return s
    return (MINUS if x < 0 else ("+" if plus else "")) + s


def ci(d, plus=False, k=1, nd=2): return f"{num(k * d['mean'], plus, nd)} [{num(k * d['lo'], plus, nd)}, {num(k * d['hi'], plus, nd)}]"
def pc(x): return f"{100 * x:.0f}"


mac, ra, rb = [], [], []
def M(name, val, src): mac.append(BS + "providecommand{" + BS + "y" + name + "}{" + str(val) + "}% src: y_analysis.json:" + src)


for key, tag, lab in RUNS:
    a, b = A[key]["A"], A[key]["B"]; D = a["contrasts"]
    M(tag + "Usable", a["usable"], f"{key}.A.usable"); M(tag + "NoSubmit", a["no_submit"] + b["no_submit"], f"{key}.A+B.no_submit")
    M(tag + "JsonOk", pc(a["json_ok"]), f"{key}.A.json_ok")
    for arm, an in ARM:
        cells = [a["cells"][f"{arm}|{p:.2f}"] for p in (0.10, 0.30)] if arm != "all" else [c for c in a["cells"].values()]
        fy = sum(c["first_y"]["mean"] for c in cells) / len(cells); fx = sum(c["first_x"]["mean"] for c in cells) / len(cells)
        fo = sum(c["first_oracle"] for c in cells) / len(cells)
        T = "All" if arm == "all" else arm
        M(f"{tag}{T}FirstRand", pc(fy), f"{key}.A.cells first_y mean ({arm})"); M(f"{tag}{T}FirstOrig", pc(fx), f"{key}.A.cells first_x mean ({arm})")
        M(f"{tag}{T}Gap", ci(D[f"A1 first model-oracle, randomised|{arm}"], True, 100, 0), f"{key}.A1 first model-oracle, randomised|{arm}")
        M(f"{tag}{T}Under", ci(D[f"A1 under, randomised|{arm}"]), f"{key}.A1 under, randomised|{arm}")
        M(f"{tag}{T}SurfFirst", ci(D[f"A2 first randomised-original|{arm}"], True, 100, 0), f"{key}.A2 first randomised-original|{arm}")
        M(f"{tag}{T}SurfUnder", ci(D[f"A2 under randomised-original|{arm}"], True), f"{key}.A2 under randomised-original|{arm}")
        M(f"{tag}{T}SurfRegret", ci(D[f"A2 regret randomised-original|{arm}"], True), f"{key}.A2 regret randomised-original|{arm}")
        M(f"{tag}{T}Planner", ci(D[f"A3 K2r-K2e, randomised|{arm}"], True), f"{key}.A3 K2r-K2e, randomised|{arm}")
        ra.append(f"{lab if arm == 'P' else ''} & {an} & {pc(fy)} & {pc(fx)} & {pc(fo)} & {ci(D[f'A1 first model-oracle, randomised|{arm}'], True, 100, 0)} & "
                  f"{ci(D[f'A2 first randomised-original|{arm}'], True, 100, 0)} & {ci(D[f'A1 under, randomised|{arm}'])} & "
                  f"{ci(D[f'A2 under randomised-original|{arm}'], True)} & {ci(D[f'A3 K2r-K2e, randomised|{arm}'], True)} " + BS * 2)
    ra.append(BS + "addlinespace")
    for p, v in a["by_replica_position"].items(): M(f"{tag}Pos{['', 'One', 'Two', 'Three', 'Four'][int(p)]}", pc(v["first"]), f"{key}.A.by_replica_position[{p}]")
    for c in ("P|0.10", "P|0.30", "I|0.10", "I|0.30"):
        n = c[0] + ("Ten" if c.endswith("10") else "Thirty")
        M(f"{tag}{n}FirstRand", pc(a["cells"][c]["first_y"]["mean"]), f"{key}.A.cells[{c}].first_y"); M(f"{tag}{n}FirstOrig", pc(a["cells"][c]["first_x"]["mean"]), f"{key}.A.cells[{c}].first_x")
        M(f"{tag}{n}Verdict", num(a["cells"][c]["verdict_y"]["mean"]), f"{key}.A.cells[{c}].verdict_y")
    M(tag + "RepFirst", pc(b["first_all"]["mean"]), f"{key}.B.first_all"); M(tag + "RepGap", ci(b["first_minus_oracle"], True, 100, 0), f"{key}.B.first_minus_oracle")
    M(tag + "RepFirstZero", pc(b["first_sample0"]), f"{key}.B.first_sample0"); M(tag + "RepFirstNew", pc(b["first_new"]), f"{key}.B.first_new")
    M(tag + "RepUnder", ci(b["under"]), f"{key}.B.under"); M(tag + "RepRegret", ci(b["regret"]), f"{key}.B.regret"); M(tag + "RepVerdict", num(b["verdict"]["mean"]), f"{key}.B.verdict")
    M(tag + "RepUnanimous", pc(b["unanimous"]), f"{key}.B.unanimous"); M(tag + "RepMajoritySkip", pc(b["majority_skip"]), f"{key}.B.majority_skip")
    M(tag + "RepIcc", num(b["icc_first"]), f"{key}.B.icc_first")
    M(tag + "RepSdWithin", num(b["regret_sd_within"]), f"{key}.B.regret_sd_within"); M(tag + "RepSdBetween", num(b["regret_sd_between"]), f"{key}.B.regret_sd_between")
    h = b["count_hist"]
    rb.append(f"{lab} & {pc(b['first_sample0'])} & {pc(b['first_new'])} & {pc(b['oracle_first'])} & {ci(b['first_minus_oracle'], True, 100, 0)} & "
              + " & ".join(str(h[str(c)]) for c in range(6)) + f" & {num(b['icc_first'])} & {ci(b['under'])} & {num(b['regret_sd_within'])} / {num(b['regret_sd_between'])} " + BS * 2)
hdr = "% generated by scripts/make_y_outputs.py from raea4/agentenv/results/y_analysis.json; do not edit"
ta = [hdr, BS + "begin{tabular}{@{}llrrrccccc@{}}", BS + "toprule",
      " & & " + BS + "multicolumn{3}{c}{first call replica (" + BS + "%)} & " + BS + "multicolumn{2}{c}{first-call share (pp)} & " + BS + "multicolumn{2}{c}{under-use regret} & " + BS * 2
      + BS + "cmidrule(lr){3-5}" + BS + "cmidrule(lr){6-7}" + BS + "cmidrule(lr){8-9}",
      "Run & Arm & rand. & orig. & oracle & rand.$-$oracle & rand.$-$orig. & rand. & rand.$-$orig. & K2r$-$K2e, rand. " + BS * 2, BS + "midrule"] + ra[:-1] + [BS + "bottomrule", BS + "end{tabular}"]
tb = [hdr, BS + "begin{tabular}{@{}lrrrcrrrrrrccc@{}}", BS + "toprule",
      " & " + BS + "multicolumn{4}{c}{first call replica} & " + BS + "multicolumn{6}{c}{instances with $k$ of 5 samples opening with it} & & & " + BS * 2
      + BS + "cmidrule(lr){2-5}" + BS + "cmidrule(lr){6-11}",
      "Run & first sample (" + BS + "%) & new samples (" + BS + "%) & oracle (" + BS + "%) & all$-$oracle (pp) & 0 & 1 & 2 & 3 & 4 & 5 & ICC & under-use & regret SD within / between " + BS * 2,
      BS + "midrule"] + rb + [BS + "bottomrule", BS + "end{tabular}"]
for name, t in (("agent_y_a.tex", ta), ("agent_y_b.tex", tb), ("nums_y.tex", [hdr] + mac)):
    open(os.path.join(ROOT, "tables", name), "w", encoding="utf-8").write(NL.join(t) + NL)
print(len(mac), "macros")
