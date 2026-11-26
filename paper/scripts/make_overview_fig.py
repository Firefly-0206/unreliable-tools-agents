"""Figure 1 (figs/overview.pdf): the same-JSON executor replay, its headline result and where the lost regret sits.

Reads only stored result files (no model call, no re-scoring):
  raea4/agentenv/results/k2v_summary.json   CORE regret of K2e / K2v / K2r per run, nested-bootstrap 95% CI
  raea4/agentenv/results/k2e_anatomy.json   split of K2e regret by kind of lossy step (agentenv/k2e_anatomy.py)
  raea4/agentenv/results/m_analysis.json    matched fault / fault-free tasks (agentenv/m_analysis.py): first-step loss,
                                            regret and suboptimal-first share per arm, nested-bootstrap 95% CI

    python paper/scripts/make_overview_fig.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(os.path.dirname(ROOT), "raea4", "agentenv", "results")
KV = json.load(open(os.path.join(RES, "k2v_summary.json"), encoding="utf-8"))
AN = json.load(open(os.path.join(RES, "k2e_anatomy.json"), encoding="utf-8"))
MA = json.load(open(os.path.join(RES, "m_analysis.json"), encoding="utf-8"))
NL = chr(10)

RUNS = [("flash", "flash"), ("terra", "terra" + NL + "(medium)"), ("terra_high", "terra" + NL + "(high)"),
        ("sonnet", "Sonnet" + NL + "(high)")]
LLM, VOI, DP = "#D55E00", "#E69F00", "#0072B2"      # Okabe-Ito: who executes
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
    "legend.frameon": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
fig = plt.figure(figsize=(5.5, 1.72))

# ------------------------------------------------------------------ (a) design
ax = fig.add_axes([0.0, 0.04, 0.255, 0.84]); ax.set_xlim(-0.1, 10.1); ax.set_ylim(-0.1, 10); ax.axis("off")


def box(x, y, w, h, text, fc="white", ec="0.25", fs=5.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)


def arrow(p, q, col="0.25"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=col, shrinkA=0, shrinkB=0))


box(1.4, 8.5, 7.2, 1.3, "tool documentation only", fc="0.95")
arrow((5, 8.5), (5, 7.75))
box(0.9, 6.0, 8.2, 1.7, "LLM writes its observation" + NL + "model as JSON (stored)", fc="#F3E9D2")
arrow((2.6, 6.0), (2.0, 4.75), LLM); arrow((5, 6.0), (5, 4.75), VOI); arrow((7.4, 6.0), (8.0, 4.75), DP)
box(0.0, 3.0, 3.15, 1.7, "the LLM" + NL + "acts (K2e)", ec=LLM)
box(3.45, 3.0, 3.1, 1.7, "1-step VOI" + NL + "acts (K2v)", ec=VOI)
box(6.85, 3.0, 3.15, 1.7, "exact DP" + NL + "acts (K2r)", ec=DP)
arrow((1.6, 3.0), (3.0, 1.75), LLM); arrow((5, 3.0), (5, 1.75), VOI); arrow((8.4, 3.0), (7.0, 1.75), DP)
box(0.0, 0.0, 10.0, 1.7, "same world and pre-drawn tool outputs;" + NL + "regret against the exact oracle", fc="0.95")
fig.text(0.0, 0.95, "a", fontsize=8, fontweight="bold", va="center")

# ------------------------------------------------------------------ (b) headline
ax = fig.add_axes([0.335, 0.2, 0.19, 0.6])
for off, cond, col, mk, lab in ((-0.22, "K2e", LLM, "o", "LLM"), (0.0, "K2v", VOI, "^", "1-step VOI"),
                                (0.22, "K2r", DP, "s", "exact DP")):
    xs, m, lo, hi = [], [], [], []
    for i, (key, _) in enumerate(RUNS):
        d = KV[key]["CORE"][cond]
        xs.append(i + off); m.append(d["mean"]); lo.append(d["mean"] - d["lo"]); hi.append(d["hi"] - d["mean"])
    ax.errorbar(xs, m, yerr=[lo, hi], fmt=mk, color=col, ms=3.4, mew=0.8, elinewidth=0.8, capsize=1.5, capthick=0.7,
                label=lab, zorder=3)
ax.set_yscale("symlog", linthresh=0.1, linscale=0.55)
ax.set_ylim(-0.02, 6.5); ax.set_yticks([0, 0.1, 0.3, 1, 3]); ax.set_yticklabels(["0", "0.1", "0.3", "1", "3"])
ax.minorticks_off()
ax.set_xticks(range(len(RUNS))); ax.set_xticklabels(["flash", "terra" + NL + "(med.)", "terra" + NL + "(high)", "Sonnet" + NL + "(high)"], fontsize=5.6); ax.set_xlim(-0.55, len(RUNS) - 0.45)
ax.set_ylabel("regret per episode (95% CI)", labelpad=1); ax.grid(axis="y", lw=0.3, color="0.88", zorder=0)
ax.legend(loc="lower center", ncol=3, handletextpad=0.0, columnspacing=0.4, borderaxespad=0.0, bbox_to_anchor=(0.44, 1.03))
fig.text(0.272, 0.95, "b", fontsize=8, fontweight="bold", va="center")

# ------------------------------------------------------------------ (c) anatomy
ax = fig.add_axes([0.632, 0.44, 0.13, 0.4])
PARTS = [("pass_other", "replica skipped," + NL + "other source probed", "#0072B2"),
         ("pass_stop", "replica skipped," + NL + "stopped early", "#56B4E9"),
         ("over", "probe not worth" + NL + "its cost", "#E69F00"),
         ("verdict", "wrong verdict", "#D55E00"),
         ("other", "other", "0.75")]
ys = list(range(len(RUNS)))[::-1]
left = [0.0] * len(RUNS)
for pk, lab, col in PARTS:
    vals = []
    for key, _ in RUNS:
        a = AN[key]; p = a["pass"]["K2e|CORE"]; reg = a["regret"]
        v = dict(pass_other=p["pass_cache_other"], pass_stop=p["pass_cache_stop"], over=a["over"], verdict=a["verdict"])
        v["other"] = reg - sum(v.values())
        assert v["other"] > -1e-9
        vals.append(100 * v[pk] / reg)
    ax.barh(ys, vals, left=left, color=col, edgecolor="white", lw=0.4, height=0.64, label=lab, zorder=3)
    left = [a + b for a, b in zip(left, vals)]
for y, (key, _) in zip(ys, RUNS):
    ax.text(102, y, f"{AN[key]['regret']:.2f}", va="center", ha="left", fontsize=6.2)
ax.text(101, len(RUNS) - 0.55, "regret", va="bottom", ha="left", fontsize=6.2)
ax.set_yticks(ys); ax.set_yticklabels([l.replace(NL, " ") for _, l in RUNS], fontsize=6)
ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("share of the LLM's regret (%)", labelpad=1)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.42), ncol=3, columnspacing=0.4, handlelength=0.8, handletextpad=0.25,
          labelspacing=0.3, fontsize=6.0)
fig.text(0.545, 0.95, "c", fontsize=8, fontweight="bold", va="center")

# ------------------------------------------------------------------ (d) matched fault / fault-free tasks
ax = fig.add_axes([0.87, 0.2, 0.13, 0.6])
F1, LT = "0.25", "0.72"
for g, (key, name) in enumerate((("terra", "terra"), ("sonnet", "Sonnet"))):
    for k, arm in enumerate(("MF", "MZ")):
        c = MA[key]["cells"][arm]; x = 3 * g + k; l1 = c["loss1"]["mean"]; r = c["regret"]["mean"]
        ax.bar(x, l1, color=F1, width=0.78, zorder=3, label="first-step loss" if x == 0 else None)
        ax.bar(x, r - l1, bottom=l1, color=LT, width=0.78, zorder=3, label="later regret" if x == 0 else None)
        ax.errorbar(x, r, yerr=[[r - c["regret"]["lo"]], [c["regret"]["hi"] - r]], fmt="none", ecolor="0.1", elinewidth=0.7,
                    capsize=1.3, capthick=0.6, zorder=4)
        ax.text(x, c["regret"]["hi"] + 0.03, f"{100 * c['subopt']['mean']:.0f}%", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks([0, 1, 3, 4]); ax.set_xticklabels(["F", "N"] * 2, fontsize=5.8)
for g, name in enumerate(("terra", "Sonnet")):
    ax.text(3 * g + 0.5, -0.17, name, ha="center", va="top", fontsize=6, transform=ax.get_xaxis_transform())
ax.set_xlim(-0.9, 4.6); ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "", "0.5", "", "1"])
ax.set_ylabel("regret per episode", labelpad=1); ax.grid(axis="y", lw=0.3, color="0.88", zorder=0)
ax.legend(loc="lower center", ncol=1, handlelength=0.9, handletextpad=0.3, borderaxespad=0.0, bbox_to_anchor=(0.45, 1.0), fontsize=5.4,
          labelspacing=0.2)
fig.text(0.815, 0.95, "d", fontsize=8, fontweight="bold", va="center")

out = os.path.join(ROOT, "figs", "overview.pdf")
fig.savefig(out); fig.savefig(out[:-4] + ".png", dpi=200)
print("wrote", out)
