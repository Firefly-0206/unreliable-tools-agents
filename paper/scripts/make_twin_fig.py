"""Main-text figure figs/twin_gap.pdf: closed-loop regret under dependent faults vs their phi = 0 twins, K2 decomposition.

Reads tables/k2decomp_levels.json (written by scripts/make_agent_llm_tables.py: pooled regret levels re-scored from
raea4/agentenv/logs/{codex_terra_medium,ds_chat_fc}{,_k2decomp}.jsonl with the exact oracle, nested bootstrap 95% CI over
scenarios/episodes). Fault = PERSIST + SHARED (50 episodes); twin = TRANSIENT_P + TRANSIENT (50 episodes, same parameters, phi = 0).
    python paper/scripts/make_twin_fig.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = json.load(open(os.path.join(ROOT, "tables", "k2decomp_levels.json"), encoding="utf-8"))["levels"]
CONDS = ["K1", "K2", "K2a", "K2e", "K2p"]
XL = ["K1", "K2", "K2a", "K2e", "K2p"]
MODELS = [("terra", "gpt-6.1-terra"), ("flash", "flash (non-thinking)")]
FAULT, TWIN = "#D55E00", "#0072B2"          # Okabe-Ito vermillion / blue

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.55), constrained_layout=True)
for ax, (tag, lab) in zip(axes, MODELS):
    for off, grp, col, mk, fill, name in ((-0.13, "FaultPS", FAULT, "o", True, r"fault families (PERSIST, SHARED)"),
                                          (0.13, "TwinPS", TWIN, "s", False, r"$\varphi=0$ twins (TRANS., TRANS. P)")):
        xs = [i + off for i in range(len(CONDS))]
        m = [J[f"{tag}|{grp}|{c}"]["mean"] for c in CONDS]
        lo = [J[f"{tag}|{grp}|{c}"]["lo"] for c in CONDS]
        hi = [J[f"{tag}|{grp}|{c}"]["hi"] for c in CONDS]
        ax.errorbar(xs, m, yerr=[[a - b for a, b in zip(m, lo)], [b - a for a, b in zip(m, hi)]], fmt=mk, color=col, ms=4.2,
                    mfc=col if fill else "white", mew=1.0, elinewidth=0.9, capsize=1.8, capthick=0.8, label=name, zorder=3)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels(XL)
    ax.set_xlim(-0.55, len(CONDS) - 0.45)
    ax.set_ylim(-0.04, 2.15) if tag == "terra" else ax.set_ylim(-0.2, 5.8)
    ax.set_ylabel("regret (mean, 95% CI)")
    ax.set_title(lab, fontsize=7.5, loc="left", pad=2)
    ax.grid(axis="y", lw=0.3, color="0.85", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(3.5, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.text(4.0, ax.get_ylim()[1] * 0.97, "exact\nplanner", ha="center", va="top", fontsize=6, color="0.35")
axes[0].legend(loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.1, bbox_to_anchor=(0.0, 1.0), ncol=1)
os.makedirs(os.path.join(ROOT, "figs"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "figs", "twin_gap.pdf"))
fig.savefig(os.path.join(ROOT, "figs", "twin_gap.png"), dpi=200)
print("wrote figs/twin_gap.pdf, .png")

