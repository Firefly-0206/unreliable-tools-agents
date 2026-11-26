"""
Exploratory, post hoc (no plan fixed in advance; no model call). Companion to z_asif.py (App. D.8).

Question (raised by an external review, 2026-11-18): is the under-use of the replica tool better described by a planner that
plans exactly but with DIFFERENT STAKES or BELIEFS than the stated ones -- a larger severe (false-benign) loss, a different
escalation cost, a different wrong-verdict loss, all losses scaled (equivalently all query costs AND the budget scaled down), or a prior
tilted toward / away from the benign state -- instead of a planner with inflated replica fault odds (g)?

Each family is one parameter on a grid that contains the true value (multiplier 1 = the exact oracle). The as-if planner
uses the altered loss matrix / prior to choose every action; trajectories are scored under the TRUE model.
Fit rule as in z_asif.py: per run, the multiplier is chosen on the PERSISTENT arm of the fault-rate sweep X (100 episodes) to
maximise first-call agreement (replica tool or not); ties go to the value closest to 1. Checks not used for the fit:
the independent arm (100), the randomised surface Y (100), the five-sample instances (20); regret is never fitted.
A joint severe x escalation grid is also fitted (2 parameters) as an upper bound for the loss route.

  python z_asif_loss.py            # X / Y / repeats   -> results/z_asif_loss.json, .txt
  python z_asif_loss.py --m        # fault-free matched twins (M): fitted variants applied unchanged -> results/z_asif_loss_m.json, .txt
"""
import os, sys
M_MODE = "--m" in sys.argv
if M_MODE:
    os.environ.setdefault("RAEA_EPISODES", "episodes_m.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_m.jsonl")
else:
    os.environ.setdefault("RAEA_EPISODES", "episodes_x.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_x.jsonl")
import copy, json, itertools
from multiprocessing import Pool
import numpy as np
from env import HERE, Env, load_episodes, load_truth, ERR, SEV, ESC, K, BUDGET, N_TOOLS
from sim import Model, loss_matrix
import k2decomp as KD
from x_analysis import parts, REP

P = lambda *a: os.path.join(HERE, *a)
RUNS = {"flash": ("flash (non-thinking)", "ds_chat_fc"), "terra": ("gpt-6.1-terra (medium)", "codex_terra"), "sonnet": ("Sonnet 5.5 (high)", "claude_s55")}
B_SAMPLES = [0, 1, 2, 3, 4]
PHIS = [0.10, 0.30, 0.50, 0.70]
FAMILIES = {
    "g":   [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64],            # replica fault odds x g (z_asif.py)
    "sev": [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8],                  # severe (false-benign) loss x m
    "esc": [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5],                   # escalation loss x m (2.5 = wrong-verdict loss)
    "err": [0.5, 0.75, 1, 1.5, 2, 3],                                 # wrong (non-severe) verdict loss x m
    "lsc": [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8],                     # every loss x m  (= every query cost and the budget / m)
    "pri": [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8],                         # prior odds of the benign state x h
}
JOINT = [("sev", "esc")]


def variant(fam, v):
    d = dict(g=1.0, sev=1.0, esc=1.0, err=1.0, lsc=1.0, pri=1.0); d[fam] = float(v); return d


def all_variants():
    vs = {("g", 1): variant("g", 1)}
    for fam, grid in FAMILIES.items():
        for v in grid: vs[(fam, v)] = variant(fam, v)
    for a, b in JOINT:
        for va, vb in itertools.product(FAMILIES[a], FAMILIES[b]):
            d = variant(a, va); d[b] = float(vb); vs[(a + "x" + b, (va, vb))] = d
    return vs


def key(d):
    return tuple(sorted(d.items()))


def build(ep, spec, prior, loss):
    """KD.build_recovered_model with the prior and loss matrix replaced (same tools, groups, budget)."""
    par = ep["params"]; tools = [None] * N_TOOLS; groups = {}
    for gi, g in enumerate(spec["groups"]):
        faulty = g["shared_fault"] and g["phi"] > 0
        for j in g["tools"]:
            t = dict(kind=par["tools"][j].get("kind", "sym"), q=spec["accuracy"][j], cost=par["tools"][j]["cost"], group=None, femit=None)
            if faulty and g["persistent"]:
                t["group"] = f"g{gi}"; t["femit"] = np.array(g["emission"]); groups[f"g{gi}"] = g["phi"]
            elif faulty:
                t["marg_phi"] = g["phi"]; t["femit"] = np.array(g["emission"])
            tools[j] = t
    return Model(K, np.asarray(prior, float), tools, groups, loss, BUDGET)


def play(args):
    ep, truth, d = args
    spec = copy.deepcopy(KD.truth_spec(ep))
    for grp in spec["groups"]:
        if REP in grp["tools"] and grp["phi"] > 0:
            phi = min(0.95, grp["phi"] + d.get("dphi", 0.0)); g = d["g"]; grp["phi"] = g * phi / (1 - phi + g * phi)   # dphi: r_robust.py only
    L = loss_matrix(K, ERR * d["err"], SEV * d["sev"], ESC * d["esc"]) * d["lsc"]
    pr = np.array(ep["params"]["prior"], float); h = d["pri"]; p0 = h * pr[0] / (h * pr[0] + 1 - pr[0])
    pr = np.concatenate([[p0], pr[1:] * (1 - p0) / max(1 - pr[0], 1e-12)])
    M = build(ep, spec, pr, L)
    env = Env(ep, "K2e", "text"); b = M.prior.copy(); verdicts = env.sc["states"] + ["escalate"]
    for _ in range(4 * N_TOOLS + 4):
        if env.done: break
        a = M.V(b, env.budget)[1]
        if a[0] == "stop": env.submit(verdicts[a[1]]); break
        env.call(env.names[a[1]])
        if env.log[-1]["type"] != "call": break
        nb = M.update(b, a[1], env.hist[-1][1]); b = nb if nb is not None else M.prior.copy()
    calls = [x["j"] for x in env.log if x["type"] == "call"]
    p = parts(ep, env.log, truth)
    sub = [x for x in env.log if x["type"] == "submit"]
    p["escalated"] = float(bool(sub) and sub[-1].get("verdict") == "escalate")
    p["first_j"] = calls[0] if calls else -1
    return ep["id"], key(d), p


def load(path, samples):
    recs = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != "K2e" or r["status"].startswith("api_error") or r.get("sample", 0) not in samples: continue
        recs[(r["episode"], r.get("sample", 0))] = r
    return recs


def solve(eps, truth, vs, workers=12):
    jobs = [(eps[i], truth[i], d) for i in sorted(eps) for d in {key(d): d for d in vs.values()}.values()]
    pred = {}
    with Pool(workers) as pool:
        for n, (i, k, p) in enumerate(pool.imap_unordered(play, jobs, chunksize=4)):
            pred[(i, k)] = p
            if n % 2000 == 0: print(f"  {n}/{len(jobs)}", flush=True)
    return pred


def closest(vals, score):
    """argmax score, ties -> value closest to 1 in log terms (the true model)."""
    def dist(v):
        v = v if isinstance(v, tuple) else (v,); return sum(abs(np.log(x)) for x in v)
    return max(vals, key=lambda v: (round(score(v), 9), -dist(v)))


def main_x():
    EX, TX = load_episodes(P("episodes_x.jsonl")), load_truth(P("ground_truth_x.jsonl")); EY = load_episodes(P("episodes_y.jsonl"))
    vs = all_variants(); ids = sorted(EX)
    print(f"{len(vs)} variants x {len(ids)} episodes"); pred = solve(EX, TX, vs)
    arm = lambda a: [i for i in ids if EX[i]["arm"] == a]
    cell = lambda a, p: [i for i in ids if EX[i]["arm"] == a and abs(EX[i]["phi"] - p) < 1e-9]
    fams = list(FAMILIES) + [a + "x" + b for a, b in JOINT]
    grid = {f: [v for (ff, v) in vs if ff == f] for f in fams}
    pv = lambda f, v, i: pred[(i, key(vs[(f, v)]))]
    out = dict(families={f: [list(v) if isinstance(v, tuple) else v for v in grid[f]] for f in fams}, runs={}); L = []
    # what each family DOES: replica first-call share and calls, by fault rate (persistent arm), at each grid value
    L.append("As-if planner behaviour, persistent arm: replica first-call % at fault rate 0.1/0.3/0.5/0.7 | mean calls | escalation %")
    out["behaviour"] = {}
    for f in FAMILIES:
        for v in grid[f]:
            fs = [100 * np.mean([pv(f, v, i)["first"] for i in cell("XP", p)]) for p in PHIS]
            cl = np.mean([pv(f, v, i)["calls"] for i in arm("XP")]); es = 100 * np.mean([pv(f, v, i)["escalated"] for i in arm("XP")])
            out["behaviour"][f"{f}={v}"] = dict(first=fs, calls=float(cl), esc=float(es))
            L.append(f"  {f:4s}={v:<7.3g}" + "".join(f"{x:5.0f}" for x in fs) + f"   | {cl:4.2f} | {es:4.0f}")
    for rk, (label, stem) in RUNS.items():
        X = load(P("logs", f"x_{stem}_k2e.jsonl"), [0]); Y = load(P("logs", f"y_{stem}_k2e.jsonl"), [0])
        R = dict(X); R.update(load(P("logs", f"xrep_{stem}_k2e.jsonl"), B_SAMPLES[1:]))
        obs = {i: parts(EX[i], X[(i, 0)]["log"], TX[i]) for i in ids}
        yobs = {EY[j]["x_id"]: float([x["j"] for x in Y[(j, 0)]["log"] if x["type"] == "call"][:1] == [REP]) for j in EY}
        bids = sorted({i for (i, s) in R if s > 0})
        share = {i: float(np.mean([[x["j"] for x in R[(i, s)]["log"] if x["type"] == "call"][:1] == [REP] for s in B_SAMPLES])) for i in bids}
        o = dict(label=label, obs=dict(first={f"{a}|{p:.2f}": float(np.mean([obs[i]["first"] for i in cell(a, p)])) for a in ("XP", "XI") for p in PHIS},
                                       calls=float(np.mean([obs[i]["calls"] for i in arm("XP")])), regret=float(np.mean([obs[i]["regret"] for i in ids])),
                                       esc=float(np.mean([any(x["type"] == "submit" and x.get("verdict") == "escalate" for x in X[(i, 0)]["log"]) for i in arm("XP")]))), fits={})
        L += ["", f"== {label}: observed replica first-call % (persistent 0.1/0.3/0.5/0.7): "
              + " ".join(f"{100 * o['obs']['first'][f'XP|{p:.2f}']:.0f}" for p in PHIS) + f"; calls {o['obs']['calls']:.2f}; regret {o['obs']['regret']:.2f}",
              "  family     fitted      fit(pers.)  indep.  randomised  repeats   regret as-if   first % at 0.1/0.3/0.5/0.7   calls"]
        for f in fams:
            m = lambda v, sel: float(np.mean([pv(f, v, i)["first"] == obs[i]["first"] for i in sel]))
            vh = closest(grid[f], lambda v: m(v, arm("XP")))
            rec = dict(value=list(vh) if isinstance(vh, tuple) else vh, fit_persistent=m(vh, arm("XP")), independent=m(vh, arm("XI")),
                       randomised=float(np.mean([pv(f, vh, i)["first"] == yobs[i] for i in yobs])),
                       repeats_majority=float(np.mean([(share[i] > 0.5) == bool(pv(f, vh, i)["first"]) for i in bids])),
                       regret=float(np.mean([pv(f, vh, i)["regret"] for i in ids])),
                       first=[float(np.mean([pv(f, vh, i)["first"] for i in cell("XP", p)])) for p in PHIS],
                       calls=float(np.mean([pv(f, vh, i)["calls"] for i in arm("XP")])),
                       esc=float(np.mean([pv(f, vh, i)["escalated"] for i in arm("XP")])),
                       profile={str(v): m(v, arm("XP")) for v in grid[f]})
            o["fits"][f] = rec
            vs_ = ",".join(f"{x:.3g}" for x in (vh if isinstance(vh, tuple) else (vh,)))
            L.append(f"  {f:9s}{vs_:>10s}{100 * rec['fit_persistent']:11.0f}{100 * rec['independent']:9.0f}{100 * rec['randomised']:11.0f}"
                     f"{100 * rec['repeats_majority']:9.0f}{rec['regret']:13.2f}      " + " ".join(f"{100 * x:4.0f}" for x in rec["first"]) + f"{rec['calls']:12.2f}")
        L.append(f"  (oracle = every family at 1; 'never replica' persistent-arm match {100 * np.mean([obs[i]['first'] == 0 for i in arm('XP')]):.0f}%)")
        out["runs"][rk] = o
    txt = "\n".join(L); print(txt)
    json.dump(out, open(P("results", "z_asif_loss.json"), "w"), indent=1); open(P("results", "z_asif_loss.txt"), "w", encoding="utf-8").write(txt + "\n")


def main_m():
    """Fault-free twins: apply each run's fitted variants (from results/z_asif_loss.json) unchanged; compare regret and
    the repeat-call pattern with the run's own on MZ (and MF)."""
    EM, TM = load_episodes(P("episodes_m.jsonl")), load_truth(P("ground_truth_m.jsonl"))
    fit = json.load(open(P("results", "z_asif_loss.json"))); vs = all_variants(); ids = sorted(EM)
    want = {}
    for rk, o in fit["runs"].items():
        for f, rec in o["fits"].items():
            v = tuple(rec["value"]) if isinstance(rec["value"], list) else rec["value"]; want[(f, v)] = vs[(f, v)]
    want[("g", 1)] = vs[("g", 1)]
    pred = solve(EM, TM, want)
    arm = lambda a: [i for i in ids if EM[i]["arm"] == a] if "arm" in EM[ids[0]] else [i for i in ids if i.startswith(a)]
    out, L = {}, ["Fault-free matched twins (M): regret and calls of the as-if variants FITTED ON X, applied unchanged"]
    orc = lambda a, f: float(np.mean([pred[(i, key(vs[('g', 1)]))][f] for i in arm(a)]))
    for rk, (label, stem) in RUNS.items():
        recs = load(P("logs", f"m_{stem}_k2e.jsonl"), [0]); o = {}
        obs = {i: parts(EM[i], recs[(i, 0)]["log"], TM[i]) for i in ids if (i, 0) in recs}
        L += ["", f"== {label}", "  variant          arm   regret  calls  repeat-calls(tool 0)  first=tool0 %"]
        for a in ("MF", "MZ"):
            sel = [i for i in arm(a) if i in obs]
            o[f"obs|{a}"] = dict(regret=float(np.mean([obs[i]["regret"] for i in sel])), calls=float(np.mean([obs[i]["calls"] for i in sel])),
                                 rep=float(np.mean([obs[i]["rep_calls"] for i in sel])), first=float(np.mean([obs[i]["first"] for i in sel])))
            L.append(f"  {'observed':16s} {a}  {o[f'obs|{a}']['regret']:7.2f}{o[f'obs|{a}']['calls']:7.2f}{o[f'obs|{a}']['rep']:12.2f}{100 * o[f'obs|{a}']['first']:16.0f}")
        for f, rec in fit["runs"][rk]["fits"].items():
            v = tuple(rec["value"]) if isinstance(rec["value"], list) else rec["value"]; k = key(vs[(f, v)])
            for a in ("MF", "MZ"):
                sel = [i for i in arm(a) if i in obs]
                r = dict(regret=float(np.mean([pred[(i, k)]["regret"] for i in sel])), calls=float(np.mean([pred[(i, k)]["calls"] for i in sel])),
                         rep=float(np.mean([pred[(i, k)]["rep_calls"] for i in sel])), first=float(np.mean([pred[(i, k)]["first"] for i in sel])))
                o[f"{f}|{a}"] = dict(value=rec["value"], **r)
                L.append(f"  {f + '=' + str(rec['value']):16s} {a}  {r['regret']:7.2f}{r['calls']:7.2f}{r['rep']:12.2f}{100 * r['first']:16.0f}")
        out[rk] = o
    L += ["", f"oracle: MF calls {orc('MF', 'calls'):.2f} repeat {orc('MF', 'rep_calls'):.2f}; MZ calls {orc('MZ', 'calls'):.2f} repeat {orc('MZ', 'rep_calls'):.2f}"]
    txt = "\n".join(L); print(txt)
    json.dump(out, open(P("results", "z_asif_loss_m.json"), "w"), indent=1); open(P("results", "z_asif_loss_m.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main_m() if M_MODE else main_x()
