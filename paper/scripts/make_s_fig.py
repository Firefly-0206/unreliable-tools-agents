"""Figure figs/s_supply.pdf: regret per condition of the layered-supply experiment (raea4/agentenv/results/s_analysis.json).
Same style as figs/x_tracking.pdf (scripts/make_x_outputs.py); placed next to it in the main text."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(os.path.join(os.path.dirname(ROOT), "raea4", "agentenv", "results", "s_analysis.json"), encoding="utf-8"))
RUNS = [("flash", "flash", "#D55E00", "o"), ("terra", "terra (medium)", "#0072B2", "s"), ("sonnet", "Sonnet 5.5 (high)", "#009E73", "^")]
CONDS = [("K1", "none"), ("KC", "check-\nlist"), ("KS1", "+poste-\nrior"), ("KS2", "+next\nreport"), ("KS2v", "+1-step\nvalues"), ("KS3", "exact\nvalues")]

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 5.6, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
                     "legend.frameon": False, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
fig, ax = plt.subplots(figsize=(2.35, 1.55), constrained_layout=True)
xs = list(range(len(CONDS)))
for i, (key, lab, col, mk) in enumerate(RUNS):
    c = [A[key]["cells"][k]["regret"] for k, _ in CONDS]; m = [x["mean"] for x in c]; off = (i - 1) * 0.09
    ax.errorbar([x + off for x in xs], m, yerr=[[a - x["lo"] for a, x in zip(m, c)], [x["hi"] - a for a, x in zip(m, c)]],
                color=col, marker=mk, ms=3.0, lw=1.0, elinewidth=0.7, capsize=1.3, label=lab, zorder=3)
ax.set_yscale("symlog", linthresh=0.5, linscale=1.0); ax.set_ylim(-0.04, 7)
ax.set_yticks([0, 0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(["0", "0.25", "0.5", "1", "2", "4"]); ax.minorticks_off()
ax.set_xticks(xs); ax.set_xticklabels([l for _, l in CONDS], linespacing=0.9)
ax.set_title("supplied before every action", fontsize=7, pad=2); ax.set_ylabel("regret per episode", labelpad=1)
ax.grid(axis="y", lw=0.3, color="0.88")
out = os.path.join(ROOT, "figs", "s_supply.pdf"); fig.savefig(out); fig.savefig(out[:-4] + ".png", dpi=200)
print(out)
