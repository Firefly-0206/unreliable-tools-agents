"""
Surface randomisation and repeated generation at fault rates 0.10 and 0.30: analysis (PLAN_y_surface.md). No model call.

  python y_analysis.py                       # the three model runs
  python y_analysis.py --mock YLOG REPLOG    # code check on mock logs

Part A (episodes_y.jsonl; 100 episodes per run, one sample): the X episodes at 0.10 and 0.30 with tools listed in a
random order and renamed tool_a..tool_d. Compared, paired by episode, with the original-surface sample of the X run.
Part B (episodes_x.jsonl; persistent arm, draws 000-009, rates 0.10 and 0.30 = 20 instances): four further samples per
instance with the original surface; with the X run's sample that is five per instance.
All regret is against the true model. Intervals: nested bootstrap (scenarios, then parameter draws; all samples of a
draw stay together), 95%.
Output: results/y_analysis.json, results/y_analysis.txt
"""
import os
os.environ.setdefault("RAEA_EPISODES", "episodes_x.jsonl"); os.environ.setdefault("RAEA_TRUTH", "ground_truth_x.jsonl")
import argparse, json
from collections import defaultdict
import numpy as np
from env import HERE, load_episodes, load_truth, build_model
from oracle import qvalues
from nestboot import nested_ci
from x_analysis import parts, replay, REP

RUNS = {"flash": ("flash (non-thinking)", "ds_chat_fc"), "terra": ("gpt-6.1-terra (medium)", "codex_terra"), "sonnet": ("Sonnet 5.5 (high)", "claude_s55")}
PHIS = [0.10, 0.30]; ARMS = ["P", "I"]; B_DRAWS = [f"X-{k:03d}" for k in range(10)]; B_SAMPLES = [0, 1, 2, 3, 4]
P = lambda *a: os.path.join(HERE, *a)


def load(path, samples):
    recs = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != "K2e" or r["status"].startswith("api_error") or r.get("sample", 0) not in samples: continue
        recs[(r["episode"], r.get("sample", 0))] = r
    return recs


def ci(v, ids, scl):
    m, lo, hi = nested_ci(v, ids, scl); return dict(mean=float(np.mean(v)), lo=float(lo), hi=float(hi), n=len(set(ids)))


def oracle_first(ep):
    M = build_model(ep["params"]); q = qvalues(M, M.prior.copy(), float(M.budget)); return float(min(q, key=q.get) == ("probe", REP))


def analyse(ylog, xlog, replog, EX, TX, EY, TY):
    out = {}
    # ------------------------------------------------------------ part A
    Y = load(ylog, [0]); X = load(xlog, [0]); miss = [i for i in EY if (i, 0) not in Y]
    assert not miss, f"part A: {len(miss)} episodes missing, e.g. {miss[:3]}"
    R = []
    for i in sorted(EY):
        ep, r = EY[i], Y[(i, 0)]; xi = ep["x_id"]; g = [x for x in r["elicit"]["spec"]["groups"] if REP in x["tools"]][0]
        calls = [x["j"] for x in r["log"] if x["type"] == "call"]
        R.append(dict(arm=ep["arm"][1], phi=ep["phi"], draw=ep["pair"], sc=ep["base_scenario"], pos=ep["replica_position"],
                      y=parts(ep, r["log"], TY[i]), x=parts(EX[xi], X[(xi, 0)]["log"], TX[xi]), r=parts(ep, replay(ep, r["elicit"]["spec"]), TY[i]),
                      of=oracle_first(ep), usable=float(bool(r["elicit"]["parse_ok"])), no_submit=float(r["status"] != "submitted"),
                      json_ok=float(len(g["tools"]) == 1 and bool(g["shared_fault"]) and bool(g["persistent"]) == (ep["arm"] == "XP")),
                      first_pos=(ep["order"].index(calls[0]) + 1) if calls else 0))
    C = lambda rows, f: ci([f(d) for d in rows], [d["draw"] for d in rows], [d["sc"] for d in rows])
    A = dict(n=len(R), usable=int(sum(d["usable"] for d in R)), no_submit=int(sum(d["no_submit"] for d in R)),
             json_ok=float(np.mean([d["json_ok"] for d in R])), cells={}, contrasts={})
    for arm in ARMS:
        for p in PHIS:
            rows = [d for d in R if d["arm"] == arm and abs(d["phi"] - p) < 1e-9]
            A["cells"][f"{arm}|{p:.2f}"] = dict(n=len(rows), first_y=C(rows, lambda d: d["y"]["first"]), first_x=C(rows, lambda d: d["x"]["first"]),
                                                first_oracle=float(np.mean([d["of"] for d in rows])), first_planner=float(np.mean([d["r"]["first"] for d in rows])),
                                                under_y=C(rows, lambda d: d["y"]["under"]), under_x=C(rows, lambda d: d["x"]["under"]),
                                                regret_y=C(rows, lambda d: d["y"]["regret"]), regret_x=C(rows, lambda d: d["x"]["regret"]),
                                                K2r=C(rows, lambda d: d["r"]["regret"]), verdict_y=C(rows, lambda d: d["y"]["verdict"]),
                                                over_y=C(rows, lambda d: d["y"]["over"]), calls_y=C(rows, lambda d: d["y"]["calls"]))
    for tag, rows in [("P", [d for d in R if d["arm"] == "P"]), ("I", [d for d in R if d["arm"] == "I"]), ("all", R)]:
        D = A["contrasts"]
        D[f"A1 first model-oracle, randomised|{tag}"] = C(rows, lambda d: d["y"]["first"] - d["of"])
        D[f"A1 under, randomised|{tag}"] = C(rows, lambda d: d["y"]["under"])
        D[f"A1 first model-oracle, original|{tag}"] = C(rows, lambda d: d["x"]["first"] - d["of"])
        D[f"A2 first randomised-original|{tag}"] = C(rows, lambda d: d["y"]["first"] - d["x"]["first"])
        D[f"A2 under randomised-original|{tag}"] = C(rows, lambda d: d["y"]["under"] - d["x"]["under"])
        D[f"A2 regret randomised-original|{tag}"] = C(rows, lambda d: d["y"]["regret"] - d["x"]["regret"])
        D[f"A3 K2r-K2e, randomised|{tag}"] = C(rows, lambda d: d["r"]["regret"] - d["y"]["regret"])
    A["by_replica_position"] = {str(p): dict(n=sum(d["pos"] == p for d in R), first=float(np.mean([d["y"]["first"] for d in R if d["pos"] == p])),
                                             oracle=float(np.mean([d["of"] for d in R if d["pos"] == p]))) for p in (1, 2, 3, 4)}
    A["first_call_list_position"] = {str(p): float(np.mean([d["first_pos"] == p for d in R])) for p in (0, 1, 2, 3, 4)}
    out["A"] = A
    # ------------------------------------------------------------ part B
    ids = [i for i in sorted(EX) if EX[i]["arm"] == "XP" and EX[i]["pair"] in B_DRAWS and any(abs(EX[i]["phi"] - p) < 1e-9 for p in PHIS)]
    S = dict(X); S.update(load(replog, B_SAMPLES[1:])); miss = [(i, s) for i in ids for s in B_SAMPLES if (i, s) not in S]
    assert not miss, f"part B: {len(miss)} runs missing, e.g. {miss[:3]}"
    rows = []
    for i in ids:
        of = oracle_first(EX[i])
        for s in B_SAMPLES:
            rows.append(dict(id=i, s=s, phi=EX[i]["phi"], draw=EX[i]["pair"], sc=EX[i]["scenario"], of=of, **parts(EX[i], S[(i, s)]["log"], TX[i]),
                             no_submit=float(S[(i, s)]["status"] != "submitted")))
    Cb = lambda rs, f: ci([f(d) for d in rs], [d["draw"] for d in rs], [d["sc"] for d in rs])
    cnt = {i: int(sum(d["first"] for d in rows if d["id"] == i)) for i in ids}
    g = np.array([[d["first"] for d in rows if d["id"] == i] for i in ids]); n, k = g.shape
    msb = k * g.mean(1).var(ddof=1); msw = g.var(axis=1, ddof=1).mean()
    reg = np.array([[d["regret"] for d in rows if d["id"] == i] for i in ids])
    B = dict(instances=len(ids), samples=len(B_SAMPLES), no_submit=int(sum(d["no_submit"] for d in rows)),
             first_all=Cb(rows, lambda d: d["first"]), first_minus_oracle=Cb(rows, lambda d: d["first"] - d["of"]),
             under=Cb(rows, lambda d: d["under"]), regret=Cb(rows, lambda d: d["regret"]), verdict=Cb(rows, lambda d: d["verdict"]),
             first_sample0=float(np.mean([d["first"] for d in rows if d["s"] == 0])), first_new=float(np.mean([d["first"] for d in rows if d["s"] > 0])),
             oracle_first=float(np.mean([d["of"] for d in rows])),
             count_hist={str(c): int(sum(v == c for v in cnt.values())) for c in range(len(B_SAMPLES) + 1)},
             unanimous=float(np.mean([v in (0, len(B_SAMPLES)) for v in cnt.values()])),
             majority_skip=float(np.mean([v <= len(B_SAMPLES) // 2 for v in cnt.values()])),
             icc_first=float((msb - msw) / (msb + (k - 1) * msw)) if msb + (k - 1) * msw > 0 else float("nan"),
             regret_sd_within=float(np.sqrt(reg.var(axis=1, ddof=1).mean())), regret_sd_between=float(reg.mean(1).std(ddof=1)),
             by_phi={f"{p:.2f}": dict(first=float(np.mean([d["first"] for d in rows if abs(d["phi"] - p) < 1e-9])),
                                      under=float(np.mean([d["under"] for d in rows if abs(d["phi"] - p) < 1e-9])),
                                      regret=float(np.mean([d["regret"] for d in rows if abs(d["phi"] - p) < 1e-9]))) for p in PHIS})
    out["B"] = B
    return out


def fmt(d, pct=False):
    k = 100 if pct else 1; f = (lambda x: f"{k * x:+.0f}") if pct else (lambda x: f"{x:+.3f}")
    return f"{f(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock", nargs=2); ap.add_argument("--runs", nargs="*", default=list(RUNS)); a = ap.parse_args()
    EX, TX = load_episodes(P("episodes_x.jsonl")), load_truth(P("ground_truth_x.jsonl"))
    EY, TY = load_episodes(P("episodes_y.jsonl")), load_truth(P("ground_truth_y.jsonl"))
    jobs = [("mock", ("mock", None))] if a.mock else [(k, RUNS[k]) for k in a.runs]; out, L = {}, []
    for key, (label, stem) in jobs:
        if a.mock: o = analyse(a.mock[0], a.mock[1], a.mock[1], EX, TX, EY, TY)
        else: o = analyse(P("logs", f"y_{stem}_k2e.jsonl"), P("logs", f"x_{stem}_k2e.jsonl"), P("logs", f"xrep_{stem}_k2e.jsonl"), EX, TX, EY, TY)
        o["label"] = label; out[key] = o; A, B = o["A"], o["B"]
        L += ["", f"== {label}", f"Part A  n={A['n']}  usable JSON {A['usable']}  no submit {A['no_submit']}  replica group written correctly {100 * A['json_ok']:.0f}%",
              f"{'cell':8s}{'first: rand.':>13s}{'orig.':>7s}{'oracle':>8s}{'planner':>9s}{'under r.':>10s}{'under o.':>10s}{'regret r.':>11s}{'regret o.':>11s}{'K2r':>7s}{'verdict':>9s}{'over':>7s}"]
        for cn, c in A["cells"].items():
            L.append(f"{cn:8s}{100 * c['first_y']['mean']:12.0f}%{100 * c['first_x']['mean']:6.0f}%{100 * c['first_oracle']:7.0f}%{100 * c['first_planner']:8.0f}%"
                     f"{c['under_y']['mean']:10.2f}{c['under_x']['mean']:10.2f}{c['regret_y']['mean']:11.2f}{c['regret_x']['mean']:11.2f}{c['K2r']['mean']:7.2f}"
                     f"{c['verdict_y']['mean']:9.2f}{c['over_y']['mean']:7.2f}")
        for name, d in A["contrasts"].items(): L.append(f"  {name:44s} {fmt(d, pct='first' in name)}" + ("  (pp)" if "first" in name else ""))
        L.append("  first-call replica share by list position of the replica tool: " + ", ".join(f"{p}: {100 * v['first']:.0f}% (n={v['n']})" for p, v in A["by_replica_position"].items()))
        L.append("  list position of the first call (0 = no call): " + ", ".join(f"{p}: {100 * v:.0f}%" for p, v in A["first_call_list_position"].items()))
        L += [f"Part B  {B['instances']} instances x {B['samples']} samples  no submit {B['no_submit']}",
              f"  first-call replica share {fmt(B['first_all'], True)} pp; oracle {100 * B['oracle_first']:.0f}%; model-oracle {fmt(B['first_minus_oracle'], True)} pp",
              f"  sample 0 (X run) {100 * B['first_sample0']:.0f}%, new samples {100 * B['first_new']:.0f}%",
              f"  under-use {fmt(B['under'])}   regret {fmt(B['regret'])}   verdict {fmt(B['verdict'])}",
              f"  instances by number of samples (of {B['samples']}) opening with the replica tool: {B['count_hist']}",
              f"  unanimous instances {100 * B['unanimous']:.0f}%; majority does not open with it {100 * B['majority_skip']:.0f}%; ICC of first-call choice {B['icc_first']:.2f}",
              f"  regret SD within instance {B['regret_sd_within']:.2f}, between instance means {B['regret_sd_between']:.2f}; by fault rate {B['by_phi']}"]
    txt = "\n".join(L); print(txt)
    if not a.mock:
        json.dump(out, open(P("results", "y_analysis.json"), "w"), indent=1)
        open(P("results", "y_analysis.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
