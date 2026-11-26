"""Paper figures. Reads only verified result files under raea4 (and recomputes the worked
example exactly with raea4/sim.py). Writes PDFs to paper/figs/.

    python paper/scripts/make_figs.py [--png]

--png also renders every PDF to PNG (PyMuPDF) next to it, for visual checking.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RAEA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "raea4")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
sys.path.insert(0, RAEA)
import sim  # noqa: E402  (read-only import)

# ------------------------------------------------------------------ style ----
W = 5.5  # text width (in)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "errorbar.capsize": 1.5,
})

# One colour per model, used in every figure (Okabe-Ito + two Tol colours; colour-blind safe).
C = {
    "exact": "#000000",          # exact dependence model (true parameters)
    "mmi": "#D55E00",            # marginal-matched independence
    "fault_unaware": "#0072B2",  # phi = 0
    "split": "#009E73",          # split shared (per-tool flags)
    "wrongmech": "#997700",      # fitted, wrong fault mechanism
    "fit_dep": "#CC79A7",        # fitted dependence model
    "fit_indep": "#E69F00",      # fitted independent model
    "forbid": "#56B4E9",         # forbid future repeats / first report per source
    "hybrid": "#882255",         # exact current belief + independent future reports
    "heuristic": "#888888",      # Fixed-3, Thr-0.9
}
LBL = {
    "exact": "exact", "mmi": "MMI", "fault_unaware": "fault-unaware", "split": "split shared",
    "wrongmech": "fitted, wrong mechanism", "fit_dep": "fitted dependence", "fit_indep": "fitted independent",
    "forbid": "first report / forbid", "hybrid": "hybrid planner",
}


def load(rel):
    with open(os.path.join(RAEA, rel)) as fh:
        return json.load(fh)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)
    return p


def panel_tag(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


# ====================================================== Fig. 1: mechanism ====
LOSS = sim.loss_matrix(3, 10, 30, 4)          # lambda_err = 10, lambda_sev = 30, c_esc = 4
TAU = 1 - 4 / 30                               # tau* = 1 - c_esc / lambda_sev = 0.867
PRIOR = np.array([.5, .25, .25])


def diag_spec(phi, q, prior=PRIOR, B=4):
    """Same construction as raea4/theory/check_belief.py diag_model: source A (group g, stuck-at-benign
    f = 0.97) and a fault-free verifier V (q = 0.8, cost 2)."""
    tools = [dict(kind="sym", q=q, cost=1, group="g", femit=sim.STUCK(3)), dict(kind="sym", q=0.8, cost=2)]
    return dict(family="D", K=3, prior=np.asarray(prior, float), tools=tools, groups={"g": phi}, loss=LOSS, budget=B)


def benign_run(M, n_max):
    b = M.prior.copy(); out = [M.marg_s(b)[0]]
    for _ in range(n_max):
        b = M.update(b, 0, 0); out.append(M.marg_s(b)[0])
    return np.array(out)


MECH_STYLE = {  # colour + marker + linestyle, so lines stay distinguishable in grayscale
    "exact": dict(color=C["exact"], marker="o", ls="-", lw=1.4, ms=3.6),
    "mmi": dict(color=C["mmi"], marker="s", ls="--", lw=1.2, ms=3.6),
    "fault_unaware": dict(color=C["fault_unaware"], marker="^", ls=":", lw=1.4, ms=4.0),
}


def _regimes(nmax):
    runs = {}
    for phi, q in [(0.3, 0.9), (0.05, 0.75)]:
        sp = diag_spec(phi, q)
        ex = benign_run(sim.build(sp), nmax)
        runs[(phi, q)] = dict(exact=ex, mmi=benign_run(sim.mmi(sp), nmax), fault_unaware=benign_run(sim.fault_unaware(sp), nmax),
                              first=np.r_[ex[0], np.full(nmax, ex[1])])   # first report only: later reports ignored
    return runs


def _post_panel(ax, r, n, phi, q, first=False, legend=True, tau_label=True):
    ax.axhline(TAU, color="0.45", lw=0.8, ls=(0, (4, 2)), zorder=0)
    for k in ("fault_unaware", "mmi", "exact"):
        ax.plot(n, r[k], label=LBL[k], **MECH_STYLE[k])
    if first:
        ax.plot(n[1:], r["first"][1:], "-.", color=C["forbid"], lw=1.2, label="first report only")
    ax.set_xlim(-0.3, n[-1] + 0.3); ax.set_ylim(0.45, 1.03); ax.set_xticks(range(0, n[-1] + 1, 2))
    ax.set_xlabel("identical benign reports $n$")
    ax.set_ylabel(r"$P(s{=}\mathrm{benign})$")
    if tau_label:
        ax.text(n[-1] + 0.2, TAU - 0.015, "\u03c4*", ha="right", va="top", fontsize=7.5, color="0.25")
    ax.text(0.5, 1.0, rf"$\varphi={phi:g},\ q={q:g}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
    if legend:
        ax.legend(loc="lower right", handlelength=2.4, labelspacing=0.25, borderaxespad=0.2)


def fig_mechanism():
    """Main-text Fig. 1 (two panels, drawn at its printed width: fonts are the printed sizes) and the
    appendix version with both regimes (mechanism_app.pdf)."""
    nmax = 8; n = np.arange(nmax + 1)
    runs = _regimes(nmax)
    # ---- appendix: both regimes + the first-report-only line ----
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1), gridspec_kw=dict(wspace=0.32))
    for k, ((phi, q), r) in enumerate(runs.items()):
        _post_panel(axes[k], r, n, phi, q, first=True, legend=(k == 0), tau_label=(k == 0))
        panel_tag(axes[k], "ab"[k], x=-0.02)
    save(fig, "mechanism_app.pdf")
    # ---- main text: (a) regime a only; (b) anticipated posterior after one more report ----
    fig = plt.figure(figsize=(W, 1.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.45], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    (phi, q), r = next(iter(runs.items()))
    _post_panel(ax, r, n, phi, q)
    panel_tag(ax, "a", x=-0.02)
    cb = load("theory/check_belief.json")["pilot_requery"]
    REG = {"frequent": (.3, .9), "moderate": (.15, .85), "rare": (.05, .75)}
    STN = {"healthy": 0, "degraded": 1, "down": 2}
    short = {"healthy": "h", "degraded": "d", "down": "x"}
    ax = fig.add_subplot(gs[1])
    ax.axvline(TAU, color="0.45", lw=0.8, ls=(0, (4, 2)), zorder=0)
    ylab = []; mmi_post = []
    for i, r in enumerate(cb):
        phi, q = REG[r["regime"]]
        Mm = sim.mmi(diag_spec(phi, q)); bm = Mm.prior.copy()
        for o in r["hist"]: bm = Mm.update(bm, 0, STN[o])
        pm = Mm.marg_s(Mm.update(bm, 0, 0))[0]; mmi_post.append(pm)
        y = len(cb) - 1 - i
        xs = [r["p_healthy"], r["post_repeat_exact"], r["post_repeat_hybrid"], pm]
        ax.plot([min(xs), max(xs)], [y, y], color="0.8", lw=1.0, zorder=0)
        ax.plot(r["p_healthy"], y, "o", mfc="white", mec=C["exact"], mew=1.0, ms=5.5, zorder=3)
        ax.plot(r["post_repeat_exact"], y, "o", color=C["exact"], ms=5.0, zorder=3)
        ax.plot(r["post_repeat_hybrid"], y, "D", color=C["hybrid"], ms=4.6, zorder=4)
        ax.plot(pm, y, "s", color=C["mmi"], ms=4.6, zorder=2)
        act = {"query_V": "V", "query_A": "A", "escalate": "esc"}
        ax.text(1.02, y, f"{act[r['best_exact']]} / {act[r['best_hybrid']]}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7)
        ylab.append(rf"$\varphi$={phi:g}, " + "".join(short[h] for h in r["hist"]))
    ax.text(1.02, len(cb) - 0.5, "best action\nexact / hybrid", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=7, color="0.2", linespacing=1.0)
    ax.set_yticks(range(len(cb))); ax.set_yticklabels(ylab[::-1], fontsize=7)
    ax.set_ylim(-0.6, len(cb) + 0.1); ax.set_xlim(0.68, 1.0)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel(r"$P(\mathrm{healthy})$ after one more benign report")
    ax.tick_params(axis="y", length=0)
    ax.text(TAU + 0.004, -0.55, "\u03c4*", fontsize=7.5, color="0.25", ha="left", va="bottom")
    panel_tag(ax, "b", x=-0.47)
    hh = [Line2D([], [], ls="", marker="o", mfc="white", mec="k", mew=1.0, ms=5.5), Line2D([], [], ls="", marker="o", color="k", ms=5.0),
          Line2D([], [], ls="", marker="D", color=C["hybrid"], ms=4.6), Line2D([], [], ls="", marker="s", color=C["mmi"], ms=4.6)]
    ax.legend(hh, ["current (exact)", "exact, after repeat", LBL["hybrid"], "MMI, after repeat"], loc="lower left",
              bbox_to_anchor=(-0.02, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.9, labelspacing=0.2, borderaxespad=0.2)
    save(fig, "mechanism.pdf")
    return runs, mmi_post


# ========================================================= Fig. 2: depth ====
D1_MODELS = [("mmi", "mmi"), ("wrongmech", "fit_wrongmech"), ("fault_unaware", "fault_unaware"),
             ("split", "split_true"), ("exact", "true")]


def fig_depth(ab=False):
    d1 = load("depth/results/D1.json")
    fig, axes = plt.subplots(2, 2, figsize=(W, 3.45), gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.55, wspace=0.22))
    depths = ["VOI-1", "VOI-2", "VOI-3", "VOI-4", "DP"]; x = np.arange(5)
    for c, fam in enumerate(["RETRY", "SHARED"]):
        ax = axes[0, c]; bb = d1[fam]["by_budget"]["6"]
        for k, key in D1_MODELS:
            if fam == "RETRY" and k == "split": continue           # identical to exact in RETRY (no shared backend)
            reg = [bb[f"{key}|{d}"]["regret"] for d in depths]
            ax.plot(x, reg, "-o", color=C[k], ms=2.8, label=LBL[k], zorder=3 if k == "exact" else 2)
            for i in (0, 4):
                e = bb[f"{key}|{depths[i]}"]
                ax.errorbar(x[i], e["regret"], yerr=[[e["regret"] - e["lo"]], [e["hi"] - e["regret"]]], color=C[k],
                            lw=0.8, capsize=1.8, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(depths); ax.set_ylim(-0.08, 2.95)
        ax.set_xlabel("planning depth ($B=6$)", labelpad=1.5)
        ax.text(0.5, 0.30, fam, transform=ax.transAxes, ha="center", va="center", fontsize=8, fontweight="bold")
        if c == 0: ax.set_ylabel("regret vs Oracle-DP")
        panel_tag(ax, "ab"[c])
        ax = axes[1, c]; Bs = ["2", "4", "6", "8"]; xb = [int(b) for b in Bs]
        for k, key in D1_MODELS:
            if fam == "RETRY" and k == "split": continue
            ax.plot(xb, [d1[fam]["by_budget"][b][f"{key}|DP"]["regret"] for b in Bs], "-o", color=C[k], ms=2.5)
            if k != "exact":
                ax.plot(xb, [d1[fam]["by_budget"][b][f"{key}|VOI-1"]["regret"] for b in Bs], ":", color=C[k], lw=1.0)
        ax.set_xticks(xb); ax.set_ylim(-0.08, 2.95); ax.set_xlabel("budget $B$", labelpad=1.5)
        if c == 0: ax.set_ylabel("regret")
        panel_tag(ax, "cd"[c])
    hs = [Line2D([], [], color=C[k], marker="o", ms=2.8) for k, _ in D1_MODELS]
    if not ab:
        hs += [Line2D([], [], color="0.3", marker="o", ms=2.5), Line2D([], [], color="0.3", ls=":")]
    fig.legend(hs, [LBL[k] for k, _ in D1_MODELS] + ([] if ab else ["DP (bottom row)", "VOI-1 (bottom row)"]), loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=4,
               handlelength=1.8, columnspacing=1.2, borderaxespad=0)
    save(fig, "depth_ab.pdf" if ab else "depth.pdf")


DEPTH_STYLE = {"mmi": ("s", "-"), "wrongmech": ("^", "--"), "fault_unaware": ("v", "-."), "split": ("D", ":"), "exact": ("o", "-")}


def fig_depth_ab():
    """Main-text Fig. 2: top row of fig_depth, drawn at its printed width (no scaling in LaTeX)."""
    d1 = load("depth/results/D1.json")
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.6), gridspec_kw=dict(wspace=0.22))
    depths = ["VOI-1", "VOI-2", "VOI-3", "VOI-4", "DP"]; x = np.arange(5)
    for c, fam in enumerate(["RETRY", "SHARED"]):
        ax = axes[c]; bb = d1[fam]["by_budget"]["6"]
        for k, key in D1_MODELS:
            if fam == "RETRY" and k == "split": continue
            mk, ls = DEPTH_STYLE[k]
            reg = [bb[f"{key}|{d}"]["regret"] for d in depths]
            ax.plot(x, reg, ls=ls, marker=mk, color=C[k], ms=3.8, lw=1.5 if k == "exact" else 1.3, label=LBL[k],
                    zorder=3 if k == "exact" else 2)
            for i in (0, 4):
                e = bb[f"{key}|{depths[i]}"]
                ax.errorbar(x[i], e["regret"], yerr=[[e["regret"] - e["lo"]], [e["hi"] - e["regret"]]], color=C[k],
                            lw=0.9, capsize=2.0, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(depths); ax.set_ylim(-0.1, 3.0); ax.set_yticks([0, 1, 2, 3])
        ax.set_xlabel("planning depth ($B=6$)", labelpad=1.5)
        ax.text(0.5, 0.34, fam, transform=ax.transAxes, ha="center", va="center", fontsize=8, fontweight="bold")
        if c == 0: ax.set_ylabel("regret vs Oracle-DP")
        panel_tag(ax, "ab"[c], x=-0.02 if c else -0.14)
    hs = [Line2D([], [], color=C[k], marker=DEPTH_STYLE[k][0], ls=DEPTH_STYLE[k][1], ms=3.8, lw=1.3) for k, _ in D1_MODELS]
    fig.legend(hs, [LBL[k] for k, _ in D1_MODELS], loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=5,
               handlelength=2.4, columnspacing=1.0, handletextpad=0.4, borderaxespad=0)
    save(fig, "depth_ab.pdf")


# ======================================================= Fig. 3: history ====
D2_MODELS = [("exact", "true"), ("fit_dep", "fit"), ("fit_indep", "fit_indep"), ("mmi", "mmi"), ("fault_unaware", "fault_unaware")]
D2_VARS = [("neither", "keep all, may repeat", "white", ""), ("forbid", "forbid future repeats", "0.80", ""),
           ("both", "discard history + forbid", "0.45", ""), ("group+forbid", "pool per source + forbid", "0.15", "")]


def fig_history():
    d2 = load("depth/results/D2.json")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    for c, fam in enumerate(["RETRY", "SHARED"]):
        ax = axes[c]; inh = d2[fam]["inherited"]
        vars_ = D2_VARS + ([("group+forbid(per-tool)", "pool per tool + forbid", "0.15", "////")] if fam == "SHARED" else [])
        nv = len(vars_); wbar = 0.8 / 5
        for i, (k, key) in enumerate(D2_MODELS):
            for j, (vk, _, fc, hatch) in enumerate(vars_):
                e = inh[f"DP[{key}]+{vk}"]; xpos = i + (j - (nv - 1) / 2) * wbar
                ax.bar(xpos, e["regret"], wbar, color=fc, edgecolor=C[k], lw=0.9, hatch=hatch, zorder=2)
                ax.errorbar(xpos, e["regret"], yerr=[[e["regret"] - e["lo"]], [e["hi"] - e["regret"]]], color="0.25",
                            lw=0.6, capsize=0, zorder=3)
        ax.set_xticks(range(len(D2_MODELS)))
        ax.set_xticklabels([LBL[k].replace("fitted dependence", "fitted\ndep.").replace("fitted independent", "fitted\nindep.").replace("fault-unaware", "fault-\nunaware") for k, _ in D2_MODELS], fontsize=6.8)
        for t, (k, _) in zip(ax.get_xticklabels(), D2_MODELS): t.set_color(C[k] if k != "exact" else "k")
        ax.tick_params(axis="x", length=0)
        ax.text(0.02, 0.98, fam + (" (history A,A,A)" if fam == "RETRY" else " (history A,B,A)"), transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, fontweight="bold")
        ax.set_xlim(-0.55, len(D2_MODELS) - 0.45)
        if c == 0: ax.set_ylabel("regret vs Oracle-DP (DP)")
        ax.yaxis.grid(True, color="0.92", lw=0.5, zorder=0); ax.set_axisbelow(True)
        panel_tag(ax, "ab"[c])
    hs = [Patch(fc=fc, ec="0.2", lw=0.7, hatch=h) for _, _, fc, h in D2_VARS + [("", "", "0.15", "////")]]
    ls = [lab for _, lab, _, _ in D2_VARS] + ["pool per tool + forbid (SHARED)"]
    fig.legend(hs, ls, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=5, handlelength=1.4, columnspacing=1.0,
               borderaxespad=0, fontsize=6.6)
    save(fig, "history.pdf")


# ====================================================== Fig. 4: tradeoff ====
D4_POL = [("Oracle-DP", "exact", "*", "full", "Oracle-DP"), ("VOI-1[exact]", "exact", "o", "none", "VOI-1, exact"),
          ("DP[oracle dedup]", "forbid", "^", "full", "exact + forbid"),
          ("DP[fitted dep]", "fit_dep", "o", "full", "fitted dependence"),
          ("DP[fitted indep]+first-report", "forbid", "s", "full", "fitted indep. + first report / source"),
          ("DP[fitted indep]+first-report(per-tool)", "forbid", "s", "none", "fitted indep. + first report / tool"),
          ("DP[fitted indep]", "fit_indep", "o", "full", "fitted independent"),
          ("DP[MMI]", "mmi", "o", "full", "MMI"), ("DP[fault-unaware]", "fault_unaware", "o", "full", "fault-unaware"),
          ("Fixed-3[exact]", "heuristic", "P", "full", "Fixed-3"), ("Thr-0.9[exact]", "heuristic", "X", "full", "Thr-0.9")]


def fig_tradeoff():
    d4 = load("depth/results/D4.json")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), sharey=True, gridspec_kw=dict(wspace=0.08))
    sz = lambda calls: 6 + 22 * calls
    for c, fam in enumerate(["RETRY", "SHARED"]):
        ax = axes[c]
        # Oracle-DP across the 12 loss settings: the achievable trade-off as the loss changes
        pts = sorted((v["Oracle-DP"]["esc"], v["Oracle-DP"]["severe"]) for v in d4[fam].values())
        ax.plot(*zip(*pts), ".", color="0.6", ms=3, zorder=0)
        main = d4[fam]["sev3_esc4"]
        for key, ck, m, fill, _ in D4_POL:
            e = main[key]
            ax.scatter(e["esc"], e["severe"], s=sz(e["calls"]), marker=m, c=C[ck] if fill == "full" else "white",
                       edgecolors=C[ck], linewidths=0.8, zorder=3, alpha=0.95)
        ax.set_xlim(-0.04, 1.03); ax.set_ylim(-0.006, 0.165)
        ax.set_xlabel("escalation rate")
        if c == 0: ax.set_ylabel("severe-error rate")
        ax.text(0.98, 0.98, fam, transform=ax.transAxes, ha="right", va="top", fontsize=8, fontweight="bold")
        panel_tag(ax, "ab"[c])
    axes[0].text(0.60, 0.04, "grey dots: Oracle-DP,\n12 loss settings", fontsize=6.3, color="0.45", ha="left", va="bottom", linespacing=0.95)
    hs = [Line2D([], [], ls="", marker=m, ms=4.5, mfc=C[ck] if fill == "full" else "white", mec=C[ck], mew=0.8)
          for _, ck, m, fill, _ in D4_POL]
    fig.legend(hs, [lab for *_, lab in D4_POL], loc="center left", bbox_to_anchor=(0.995, 0.55), ncol=1,
               handlelength=1.0, labelspacing=0.55)
    sh = [plt.scatter([], [], s=sz(cl), c="white", edgecolors="0.3", lw=0.7) for cl in (0.5, 1.5, 3.0)]
    fig.legend(sh, ["0.5", "1.5", "3"], title="calls / episode", title_fontsize=6.5, loc="upper left",
               bbox_to_anchor=(0.995, 0.20), ncol=3, handletextpad=0.1, columnspacing=0.6, fontsize=6.5)
    save(fig, "tradeoff.pdf")


# ======================================================== Fig. 5: stress ====
def fig_stress():
    s1, s3, s4 = load("stress/results/S1.json"), load("stress/results/S3.json"), load("stress/results/S4.json")
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.85), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1, 1], wspace=0.42))
    fk = ["0.5", "0.7", "0.85", "q", "0.95", "0.97", "0.99"]
    fx = [0.5, 0.7, 0.85, 0.90, 0.95, 0.97, 0.99]                  # 'q' drawn between 0.85 and 0.95 (per-instance q)
    for c, fam in enumerate(["RETRY", "SHARED"]):
        ax = axes[c]
        ax.axvline(0.90, color="0.8", lw=0.6, zorder=0)
        for k, pk, extra in (("mmi", "mmi", True), ("fault_unaware", "fault_unaware", False), ("split", "split_shared", True)):
            if fam == "RETRY" and k == "split": continue
            P = [s1[fam][f]["policies"] for f in fk]
            dp = [p[f"DP[{pk}]"] for p in P]
            ax.plot(fx, [e["regret"] for e in dp], "-o", color=C[k], ms=2.3)
            ax.fill_between(fx, [e["lo"] for e in dp], [e["hi"] for e in dp], color=C[k], alpha=0.15, lw=0)
            if extra:
                ax.plot(fx, [p[f"VOI1[{pk}]"]["regret"] for p in P], ":", color=C[k], lw=1.0)
        ax.set_xticks([0.5, 0.7, 0.9]); ax.set_xticklabels(["0.5", "0.7", "$f{=}q$"])
        ax.set_xlim(0.47, 1.01); ax.set_ylim(-0.05, 2.85)
        ax.set_xlabel("fault emission $f$", labelpad=1.5)
        ax.text(0.03, 0.98, "S1 " + fam, transform=ax.transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold")
        if c == 0: ax.set_ylabel("regret vs Oracle-DP")
        panel_tag(ax, "ab"[c])
    # S3: partial sharing rho (SHARED)
    ax = axes[2]; rk = ["0.0", "0.25", "0.5", "0.75", "1.0"]; rx = [float(r) for r in rk]
    for k, pk in (("exact", "assume_shared"), ("split", "assume_split"), ("mmi", "mmi"), ("forbid", "oracle_dedup")):
        e = [s3["SHARED"][r]["policies"][f"DP[{pk}]"] for r in rk]
        ax.plot(rx, [v["regret"] for v in e], marker="o", color=C[k], ms=2.3, ls="--" if k == "forbid" else "-")
        ax.fill_between(rx, [v["lo"] for v in e], [v["hi"] for v in e], color=C[k], alpha=0.15, lw=0)
    ax.set_xticks([0, 0.5, 1]); ax.set_ylim(-0.05, 2.85); ax.set_xlabel(r"sharing $\rho$", labelpad=1.5)
    ax.text(0.03, 0.98, "S3", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold")
    panel_tag(ax, "c")
    # S4: recovery rate r (RETRY, no onset)
    ax = axes[3]; ck = ["(0.0, 0.0)", "(0.1, 0.0)", "(0.3, 0.0)", "(0.5, 0.0)"]; cx = [0, .1, .3, .5]
    for k, pk, ls in (("exact", "static_fault", "-"), ("mmi", "mmi", "-"), ("mmi", "mmi_t", (0, (2, 1))),
                      ("fault_unaware", "fault_unaware", "-"), ("forbid", "oracle_dedup", "--")):
        e = [s4["RETRY"][cc]["policies"][f"DP[{pk}]"] for cc in ck]
        ax.plot(cx, [v["regret"] for v in e], marker="o", color=C[k], ms=2.3, ls=ls)
        ax.fill_between(cx, [v["lo"] for v in e], [v["hi"] for v in e], color=C[k], alpha=0.15, lw=0)
    ax.set_xticks(cx[::1]); ax.set_xticklabels(["0", ".1", ".3", ".5"]); ax.set_ylim(-0.05, 2.85)
    ax.set_xlabel("recovery rate $r$", labelpad=1.5)
    ax.text(0.97, 0.98, "S4 RETRY", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, fontweight="bold")
    panel_tag(ax, "d")
    hs = [Line2D([], [], color=C["exact"]), Line2D([], [], color=C["mmi"]), Line2D([], [], color=C["mmi"], ls=(0, (2, 1))),
          Line2D([], [], color=C["fault_unaware"]), Line2D([], [], color=C["split"]), Line2D([], [], color=C["forbid"], ls="--"),
          Line2D([], [], color="0.3"), Line2D([], [], color="0.3", ls=":")]
    ls = ["exact model of the assumed structure", "MMI", r"MMI$_t$ (S4)", "fault-unaware", "split / assume split",
          "forbid future repeats", "DP", "VOI-1"]
    fig.legend(hs, ls, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4, handlelength=2.0, columnspacing=1.2,
               borderaxespad=0)
    save(fig, "stress.pdf")


def to_png(paths):
    import pymupdf
    for p in paths:
        doc = pymupdf.open(p)
        doc[0].get_pixmap(dpi=220).save(p[:-4] + ".png")
        doc.close()


if __name__ == "__main__":
    runs, mmi_post = fig_mechanism()
    for (phi, q), r in runs.items():
        print(f"phi={phi} q={q}: " + "; ".join(f"{k} " + " ".join(f"{v:.3f}" for v in r[k][:5]) for k in r))
    print("pilot items, MMI P(healthy) after one more benign report:", [round(v, 3) for v in mmi_post])
    fig_depth(); fig_depth_ab(); fig_history(); fig_tradeoff(); fig_stress()
    if "--png" in sys.argv:
        to_png([os.path.join(OUT, f) for f in ("mechanism.pdf", "depth.pdf", "history.pdf", "tradeoff.pdf", "stress.pdf")])




