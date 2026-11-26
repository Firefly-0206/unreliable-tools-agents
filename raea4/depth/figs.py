"""Figures for D1-D5 (PDF, Okabe-Ito colours). Usage: python figs.py"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.3})
OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
MOD = {"true": ("Exact", OI[0], "o"), "mmi": ("MMI", OI[6], "s"), "fit_wrongmech": ("Fitted, wrong mechanism", OI[7], "^"),
       "fault_unaware": ("Fault-unaware", OI[5], "D"), "split_true": ("Split shared", OI[3], "v")}
PL = ["VOI-1", "VOI-2", "VOI-3", "VOI-4", "DP"]
FAMS = ["RETRY", "SHARED"]
L = lambda e: json.load(open(f"results/{e}.json"))

def err(s):
    return [[s["regret"] - s["lo"]], [s["hi"] - s["regret"]]]

def fig_d1():
    d = L("D1")
    fig, ax = plt.subplots(2, 2, figsize=(6.6, 4.6), constrained_layout=True)
    for c, fam in enumerate(FAMS):
        r6 = d[fam]["by_budget"]["6"]
        for m, (lab, col, mk) in MOD.items():
            if fam == "RETRY" and m == "split_true": continue
            y = [r6[f"{m}|{p}"] for p in PL]
            ax[0, c].errorbar(range(5), [s["regret"] for s in y], yerr=np.hstack([err(s) for s in y]),
                              color=col, marker=mk, ms=4, capsize=2, label=lab)
            for ls, p in [("-", "DP"), ("--", "VOI-1")]:
                ys = [d[fam]["by_budget"][str(B)][f"{m}|{p}"]["regret"] for B in (2, 4, 6, 8)]
                ax[1, c].plot([2, 4, 6, 8], ys, ls=ls, color=col, marker=mk, ms=3.5)
        ax[0, c].set_xticks(range(5)); ax[0, c].set_xticklabels(PL)
        ax[0, c].set_title(f"{fam}: regret vs planning depth (B = 6)"); ax[0, c].set_ylabel("regret vs Oracle-DP")
        ax[1, c].set_title(f"{fam}: vs budget (solid DP, dashed VOI-1)")
        ax[1, c].set_xlabel("budget B"); ax[1, c].set_ylabel("regret vs Oracle-DP"); ax[1, c].set_xticks([2, 4, 6, 8])
    h, l = ax[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False)
    fig.savefig("figs/D1_depth_budget.pdf"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), constrained_layout=True)
    for c, fam in enumerate(FAMS):
        r6 = d[fam]["by_budget"]["6"]
        for m, (lab, col, mk) in MOD.items():
            if fam == "RETRY" and m == "split_true": continue
            ax[c].plot(range(5), [r6[f"{m}|{p}"]["optimism"] for p in PL], color=col, marker=mk, ms=4, label=lab)
            ax[c].plot(range(5), [r6[f"{m}|{p}"]["repeat"] for p in PL], color=col, marker=mk, ms=3, ls=":", alpha=.7)
        ax[c].set_xticks(range(5)); ax[c].set_xticklabels(PL); ax[c].axhline(0, color="0.6", lw=.6)
        ax[c].set_title(f"{fam} (B = 6): optimism gap (solid), repeat calls (dotted)")
        ax[c].set_ylabel("true - own predicted loss / calls")
    ax[0].legend(frameon=False, fontsize=6.5)
    fig.savefig("figs/D1_optimism.pdf"); plt.close(fig)

def fig_d2():
    d = L("D2"); modes = ["neither", "discard", "forbid", "both", "group+forbid"]
    lab = {"neither": "neither", "discard": "naive discard", "forbid": "forbid",
           "both": "both (= consistent discard)", "group+forbid": "group by source + forbid"}
    cols = [OI[0], OI[1], OI[2], OI[3], OI[7]]
    ms = [("true", "Exact"), ("fit", "Fitted\ndep."), ("fit_indep", "Fitted\nindep."), ("mmi", "MMI"),
          ("fault_unaware", "Fault-\nunaware")]
    fig, ax = plt.subplots(2, 2, figsize=(6.8, 4.4), constrained_layout=True, sharey="row")
    for c, fam in enumerate(FAMS):
        for r, setting in enumerate(["fresh", "inherited"]):
            s = d[fam][setting]; w = 0.16
            for k, mode in enumerate(modes):
                y = [s[f"DP[{m}]+{mode}"] for m, _ in ms]
                ax[r, c].bar(np.arange(len(ms)) + (k - 2) * w, [x["regret"] for x in y], w, color=cols[k],
                             yerr=np.hstack([err(x) for x in y]), capsize=1.5, error_kw=dict(lw=.6), label=lab[mode])
            ax[r, c].set_xticks(range(len(ms))); ax[r, c].set_xticklabels([n for _, n in ms])
            ax[r, c].set_title(f"{fam}, {'fresh episode' if setting == 'fresh' else 'inherited repeat history'} (DP)")
            ax[r, c].set_ylabel("regret vs Oracle-DP")
    ax[0, 0].legend(frameon=False, title="dedup operation", title_fontsize=7)
    fig.savefig("figs/D2_discard_vs_forbid.pdf"); plt.close(fig)

def fig_d3():
    d = L("D3"); ns = [30, 100, 300]
    ser = [("DP[dep@{}]", "Dependence model (fitted)", OI[0], "o"), ("DP[indep@{}]", "Independent (fitted)", OI[6], "s"),
           ("DP[indep@{}]+first-report", "Independent + first report per source\n(= forbid repeats; true grouping)", OI[2], "D"),
           ("DP[indep@{}]+first-report(per-tool)", "Independent + first report per tool", OI[1], "^"),
           ("DP[symq-nofault@{}]", "Symmetric-q, no faults (fitted)", OI[5], "v")]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), constrained_layout=True)
    for c, fam in enumerate(FAMS):
        for k, lab, col, mk in ser:
            if fam == "RETRY" and "per-tool" in k: continue        # identical to per-source in RETRY
            y = [d[fam][k.format(n)] for n in ns]
            ax[c].errorbar(ns, [s["regret"] for s in y], yerr=np.hstack([err(s) for s in y]), color=col, marker=mk,
                           ms=4, capsize=2, label=lab)
        ax[c].axhline(d[fam]["DP[fault_unaware]"]["regret"], color=OI[5], ls="--", lw=.8)
        ax[c].set_xscale("log"); ax[c].set_xticks(ns); ax[c].set_xticklabels(ns)
        ax[c].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[c].set_xlabel("labelled calibration episodes n"); ax[c].set_ylabel("regret vs Oracle-DP")
        ax[c].set_title(f"{fam} (dashed: fault-unaware DP, true q)")
    ax[1].legend(frameon=False, fontsize=6)
    fig.savefig("figs/D3_same_data.pdf"); plt.close(fig)

def fig_d4():
    d = L("D4")
    pols = ["Oracle-DP", "VOI-1[exact]", "DP[MMI]", "DP[fault-unaware]", "DP[oracle dedup]", "DP[fitted dep]",
            "DP[fitted indep]", "DP[fitted indep]+first-report", "DP[fitted indep]+first-report(per-tool)",
            "Fixed-3[exact]", "Thr-0.9[exact]"]
    mk = ["o", "o", "s", "D", "P", "^", "v", "<", ">", "X", "*"]
    col = [OI[0], "0.55", OI[6], OI[5], OI[3], OI[7], OI[1], "#8C510A", "#5AAE61", OI[2], OI[4]]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.1), constrained_layout=True)
    for c, fam in enumerate(FAMS):
        for i, p in enumerate(pols):
            ss = [d[fam][f"sev{v}_esc4"][p] for v in (1, 3, 10, 30)]
            ax[c].plot([s["esc"] for s in ss], [s["severe"] for s in ss], color=col[i], lw=.8, alpha=.7)
            ax[c].scatter([s["esc"] for s in ss], [s["severe"] for s in ss], s=[10 + 18 * s["calls"] for s in ss],
                          marker=mk[i], facecolor="none" if i == 1 else col[i], edgecolor=col[i], lw=.7, label=p, zorder=3)
        ax[c].set_xlabel("escalation rate"); ax[c].set_ylabel("severe-error rate")
        ax[c].set_title(f"{fam}, c_esc = 4: λ_sev/λ_err = 1, 3, 10, 30\n(ratio 1 at upper left; size = expected calls)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False, markerscale=.7)
    fig.savefig("figs/D4_tradeoff.pdf"); plt.close(fig)

def fig_d5():
    d = L("D5"); ref = "fault-unaware DP (true q, no data)"
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), constrained_layout=True)
    for c, fam in enumerate(FAMS):
        for k, n in enumerate([30, 100, 300]):
            e = d[fam][str(n)][ref]
            ax[c].plot([0, 4, 30], [e[f"breakeven_episodes_clabel{cl}"] or np.nan for cl in (0, 4, 30)],
                       marker="o", ms=4, color=OI[k + 5 if k < 3 else 0], label=f"n = {n}")
        ax[c].set_yscale("log"); ax[c].set_xlabel("label cost per calibration episode")
        ax[c].set_ylabel("break-even deployment episodes"); ax[c].set_title(f"{fam}: fitted dependence model vs fault-unaware DP")
    ax[0].legend(frameon=False)
    fig.savefig("figs/D5_breakeven.pdf"); plt.close(fig)

if __name__ == "__main__":
    for f in [fig_d1, fig_d2, fig_d3, fig_d4, fig_d5]:
        try: f(); print("ok", f.__name__)
        except FileNotFoundError as e: print("skip", f.__name__, e)
