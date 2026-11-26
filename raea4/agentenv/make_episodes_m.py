"""Episodes of the difficulty-matched fault / fault-free experiment (PLAN_m_matched.md).

For each accepted parameter draw (PERSIST structure: replica tool 0, primary-database tool, log tool 2, live probe):
  MF  persistent fault on the replica tool with rate 0.30, accuracies as drawn
  MZ  no fault at all (rate 0); the accuracies of tool 0 and of the log tool are chosen on a grid so that the first
      decision matches MF's under the exact oracle:
        - the best first action is tool 0 in both
        - the second-best first action is the same
        - the one-step value-of-information rule picks the best first action in both or in neither
        - |margin(MZ) - margin(MF)| <= 0.02, margin = Q(second-best first action) - Q(best)
        - |V(MZ) - V(MF)| <= 0.30, V = the oracle's expected total loss
      among the grid points that satisfy all five, the one with the smallest margin difference (ties: smallest value
      difference, then smallest change of the two accuracies).
A draw with no admissible grid point, or whose MF does not open with tool 0, is skipped and the next draw is taken, until
N pairs are accepted; the number skipped is printed and stored. Same prior, costs, other accuracies, scenario and
uniforms within a pair. Writes episodes_m.jsonl, ground_truth_m.jsonl (hidden; read by code only) and m_matching.json.
Usage: python make_episodes_m.py [--n 50] [--seed 2031] [--block 150] [--workers 14]   (about 20 minutes)
"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np
from env import HERE, SCENARIO_LIST, BUDGET, sample_params, sample_world, build_model, ticket_id
from oracle import qvalues, voi1

REP, LOG = 0, 2
QC = [round(0.60 + 0.01 * i, 2) for i in range(33)]      # tool 0 accuracy, 0.60 .. 0.92
QL = [round(0.55 + 0.01 * i, 2) for i in range(31)]      # log tool accuracy, 0.55 .. 0.85
TOL_MARGIN, TOL_VALUE = 0.02, 0.30


def feat(par):
    M = build_model(par); M.srcmap = list(range(len(M.tools))); b = M.prior.copy(); q = qvalues(M, b, float(BUDGET))
    acts = {k: v for k, v in q.items() if k[0] == "probe"}; acts[("stop",)] = min(v for k, v in q.items() if k[0] == "stop")
    o = sorted(acts.items(), key=lambda kv: kv[1]); v1 = voi1(M, b, float(BUDGET), frozenset()); v1 = ("stop",) if v1[0] == "stop" else tuple(v1)
    return dict(best=list(o[0][0]), second=list(o[1][0]), margin=float(o[1][1] - o[0][1]), value=float(o[0][1]), myopic_ok=bool(v1 == tuple(o[0][0])))


def match(base, f):
    best = None
    for qc in QC:
        for ql in QL:
            p = dict(base, phi=0.0, tools=[dict(t) for t in base["tools"]]); p["tools"][REP]["q"] = qc; p["tools"][LOG]["q"] = ql
            g = feat(p)
            if g["best"] != ["probe", REP] or g["second"] != f["second"] or g["myopic_ok"] != f["myopic_ok"]: continue
            dm, dv = abs(g["margin"] - f["margin"]), abs(g["value"] - f["value"])
            if dm > TOL_MARGIN or dv > TOL_VALUE: continue
            key = (round(dm, 6), round(dv, 6), abs(qc - base["tools"][REP]["q"]) + abs(ql - base["tools"][LOG]["q"]))
            if best is None or key < best[0]: best = (key, p, g)
    return best


def one_draw(s_par):
    base = sample_params("PERSIST", np.random.default_rng(s_par))
    mf = dict(base, family="MF", phi=0.30, stuck="benign", tools=[dict(t) for t in base["tools"]]); f = feat(mf)
    m = match(dict(mf, family="MZ"), f) if f["best"] == ["probe", REP] else None
    return None if m is None else (mf, f, m[1], m[2])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed", type=int, default=2031)
    ap.add_argument("--block", type=int, default=150); ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args(); eps, truth, seen, info, skipped, k, acc = [], [], set(), [], 0, 0, 0
    # the matching of each draw depends on that draw only, so a block of draws is searched in parallel and the pairs are
    # then accepted in draw order: the result is the same as a serial search
    children = [c.spawn(2) for c in np.random.SeedSequence([a.seed, 0]).spawn(a.block)]      # (parameters, world) per draw
    with Pool(a.workers) as pool: found = pool.map(one_draw, [c[0] for c in children], chunksize=1)
    assert sum(x is not None for x in found) >= a.n, "block too small: raise --block"
    for child, hit in zip(children, found):
        if acc >= a.n: break
        k += 1
        if hit is None: skipped += 1; continue
        s_world = child[1]; mf, f, mz, g = hit
        for fam, par, ft in (("MF", mf, f), ("MZ", mz, g)):
            eid = f"{fam}-{acc:03d}"; M = build_model(par); w, table = sample_world(par, np.random.default_rng(s_world), M)
            salt = 0; t = ticket_id(eid)
            while t in seen: salt += 1; t = ticket_id(eid, salt)
            seen.add(t)
            eps.append(dict(id=eid, ticket=t, family=fam, scenario=SCENARIO_LIST[acc % len(SCENARIO_LIST)], pair=f"M-{acc:03d}", arm=fam,
                            phi=par["phi"], params=par, table=table))
            truth.append(dict(id=eid, **w, V0=float(M.V(M.prior.copy(), float(BUDGET))[0])))
        info.append(dict(pair=f"M-{acc:03d}", MF=f, MZ=g, q_rep=[mf["tools"][REP]["q"], mz["tools"][REP]["q"]], q_log=[mf["tools"][LOG]["q"], mz["tools"][LOG]["q"]]))
        acc += 1
    for name, rows in (("episodes_m.jsonl", eps), ("ground_truth_m.jsonl", truth)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as fh:
            for r in rows: fh.write(json.dumps(r) + "\n")
    S = lambda arm, key: [float(i[arm][key]) for i in info]
    summ = dict(n_pairs=acc, draws=k, skipped=skipped, margin_MF=float(np.mean(S("MF", "margin"))), margin_MZ=float(np.mean(S("MZ", "margin"))),
                max_margin_diff=float(max(abs(i["MF"]["margin"] - i["MZ"]["margin"]) for i in info)),
                value_MF=float(np.mean(S("MF", "value"))), value_MZ=float(np.mean(S("MZ", "value"))),
                myopic_ok=float(np.mean(S("MF", "myopic_ok"))), second_is_log=float(np.mean([i["MF"]["second"] == ["probe", LOG] for i in info])),
                q_rep_MF=float(np.mean([i["q_rep"][0] for i in info])), q_rep_MZ=float(np.mean([i["q_rep"][1] for i in info])),
                q_log_MF=float(np.mean([i["q_log"][0] for i in info])), q_log_MZ=float(np.mean([i["q_log"][1] for i in info])))
    json.dump(dict(summary=summ, pairs=info), open(os.path.join(HERE, "m_matching.json"), "w"), indent=1)
    print(f"wrote {len(eps)} episodes; {summ}")


if __name__ == "__main__":
    main()
