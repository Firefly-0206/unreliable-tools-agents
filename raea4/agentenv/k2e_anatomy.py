"""
Anatomy of the describe-execute gap (offline, no model call).  python k2e_anatomy.py

For every K2e CORE episode of the four closed-loop runs, the exact per-step oracle advantages of the model's own
trajectory (oracle.score_trajectory, the scorer used everywhere else) are split by the kind of step:

  over    probe taken where the oracle would stop            (buying evidence that is not worth its cost)
  wrong   probe taken where the oracle would probe another tool
  early   stop where the oracle would still probe             (stop-timing part, adv_stoptime)
  verdict wrong terminal action given the belief at the stop  (adv_verdict)

The four parts sum to the episode's regret (asserted).

Second split, by what the oracle wanted at the step (also sums to the regret):
  pass_cache  the oracle would probe a replica (cache-backed, fault-capable) tool and the model does something else:
              pass_cache_other = it probes a tool of another source, pass_cache_stop = it stops
  rest        every other lossy step (over-gathering, wrong tool within a source, verdict, ...)
Reported for K2e (model writes the JSON first) and K1 (observation model stated), CORE and the three fault
families / the fault-free twin separately.  Also reported: the share of over-gathering probes that
repeat a source already called, calls per episode for the model and for the exact planner on the same JSON (K2r),
the oracle's expected loss V0 on the same episodes, and the share of episodes with zero regret.
CIs: nested bootstrap over scenarios, then episodes (nestboot.nested_ci), as in score.py.
Output: results/k2e_anatomy.json, results/k2e_anatomy.txt
"""
import json, os
from collections import defaultdict
import numpy as np
from env import HERE, CORE, load_episodes, load_truth, srcmap
from oracle import score_trajectory
from nestboot import nested_ci

RUNS = [  # (key, label, K2e log, K2r replay log, K1 log)
    ("flash", "flash (non-thinking)", "ds_chat_fc_k2decomp.jsonl", "ds_chat_fc_k2e_replay.jsonl", "ds_chat_fc.jsonl"),
    ("terra", "gpt-6.1-terra (medium)", "codex_terra_medium_k2decomp.jsonl", "codex_terra_k2e_replay.jsonl", "codex_terra_medium.jsonl"),
    ("terra_high", "gpt-6.1-terra (high)", "codex_terra_high.jsonl", "codex_terra_high_k2e_replay.jsonl", "codex_terra_high.jsonl"),
    ("sonnet", "Sonnet 5.5 (high)", "claude_s55.jsonl", "claude_s55_k2e_replay.jsonl", "claude_s55.jsonl"),
]
PARTS = ["over", "wrong", "early", "verdict"]
PARTS2 = ["pass_cache_other", "pass_cache_stop", "rest"]
FAULT = ["PERSIST", "SHARED", "MIXED"]


def load(fn, cond):
    recs = {}
    for line in open(os.path.join(HERE, "logs", fn), encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != cond or r["family"] not in CORE or r["status"].startswith("api_error"):
            continue
        recs[r["episode"]] = r  # a later successful rerun replaces an earlier row (as in score.py)
    return recs


def split(s, src):
    d = dict.fromkeys(PARTS + PARTS2, 0.0); n_over = n_over_rep = n_probe = 0
    for st in s["steps"]:
        o_cache = st["oracle"][0] == "probe" and src[st["oracle"][1]] == "c"
        if st["action"][0] == "probe":
            if o_cache and src[st["action"][1]] != "c": d["pass_cache_other"] += st["adv"]
            else: d["rest"] += st["adv"]
        else:
            d["pass_cache_stop" if o_cache else "rest"] += st["adv_stoptime"]; d["rest"] += st["adv_verdict"]
        if st["action"][0] == "probe":
            n_probe += 1
            if st["oracle"][0] == "stop":
                d["over"] += st["adv"]; n_over += 1; n_over_rep += bool(st["repeat_src"])
            else:
                d["wrong"] += st["adv"]
        else:
            d["early"] += st["adv_stoptime"]; d["verdict"] += st["adv_verdict"]
    assert abs(sum(d[p] for p in PARTS) - s["regret"]) < 1e-9 and abs(sum(d[p] for p in PARTS2) - s["regret"]) < 1e-9
    d["pass_cache"] = d["pass_cache_other"] + d["pass_cache_stop"]
    d.update(regret=s["regret"], n_over=n_over, n_over_rep=n_over_rep, calls=n_probe, V0=s["V0"],
             zero=float(s["regret"] < 1e-9), severe=float(s["severe"]))
    return d


def main():
    eps, truth = load_episodes(), load_truth()
    out = {}; lines = []
    hdr = (f"{'run':24s}{'n':>4s}{'regret':>8s}" + "".join(f"{p:>9s}" for p in PARTS) +
           f"{'over%':>7s}{'rep/over':>10s}{'calls':>7s}{'K2r calls':>10s}{'V0':>6s}{'zero%':>7s}{'severe%':>8s}")
    lines.append(hdr)
    for key, label, f_e, f_r, f_1 in RUNS:
        e, r = load(f_e, "K2e"), load(f_r, "K2r")
        ids = sorted(e); assert len(ids) == 100, (key, len(ids))
        sp = lambda rec, i: split(score_trajectory(eps[i], rec[i]["log"], truth[i]), srcmap(eps[i]["params"]))
        rows = [sp(e, i) for i in ids]
        rrows = [sp(r, i) for i in ids]
        scl = [eps[i]["scenario"] for i in ids]
        res = dict(label=label, n=len(ids))
        for m in ["regret"] + PARTS + ["calls", "zero", "severe", "V0"]:
            v = [x[m] for x in rows]; mean, lo, hi = nested_ci(v, ids, scl)
            res[m] = float(np.mean(v)); res[m + "_lo"] = float(lo); res[m + "_hi"] = float(hi)
        for p in PARTS:
            res[p + "_share"] = res[p] / res["regret"] if res["regret"] > 0 else None
        res["n_over"] = int(sum(x["n_over"] for x in rows)); res["n_over_rep"] = int(sum(x["n_over_rep"] for x in rows))
        res["k2r_calls"] = float(np.mean([x["calls"] for x in rrows])); res["k2r_regret"] = float(np.mean([x["regret"] for x in rrows]))
        d = [a["calls"] - b["calls"] for a, b in zip(rows, rrows)]; mean, lo, hi = nested_ci(d, ids, scl)
        res["calls_minus_k2r"] = float(np.mean(d)); res["calls_minus_k2r_lo"] = float(lo); res["calls_minus_k2r_hi"] = float(hi)
        fam = defaultdict(list)
        for i, x in zip(ids, rows): fam[eps[i]["family"]].append(x)
        res["by_family"] = {f: {m: float(np.mean([x[m] for x in v])) for m in ["regret"] + PARTS + ["calls"]} for f, v in fam.items()}
        k1 = load(f_1, "K1"); assert sorted(k1) == ids, key
        res["pass"] = {}
        for cond, rw in [("K2e", rows), ("K1", [sp(k1, i) for i in ids])]:
            for sub, fams in [("CORE", CORE), ("FAULT", FAULT), ("TWIN", ["TRANSIENT"])]:
                sel = [k for k, i in enumerate(ids) if eps[i]["family"] in fams]
                q = dict(n=len(sel))
                for m in ["regret", "pass_cache", "pass_cache_other", "pass_cache_stop", "rest"]:
                    v = [rw[k][m] for k in sel]; mean, lo, hi = nested_ci(v, [ids[k] for k in sel], [scl[k] for k in sel])
                    q[m] = float(np.mean(v)); q[m + "_lo"] = float(lo); q[m + "_hi"] = float(hi)
                q["pass_cache_share"] = q["pass_cache"] / q["regret"] if q["regret"] > 0 else None
                res["pass"][cond + "|" + sub] = q
        # use of the replica (cache-backed) tools: model on its own JSON (K2e), exact planner on that JSON (K2r), model with
        # the observation model stated (K1); share of episodes with any replica call / whose first call is a replica tool
        res["use"] = {}
        for cond, rec in [("K2e", e), ("K2r", r), ("K1", k1)]:
            for sub, fams in [("FAULT", FAULT), ("TWIN", ["TRANSIENT"])]:
                sel = [i for i in ids if eps[i]["family"] in fams]; used, first, ncache, nother = [], [], [], []
                for i in sel:
                    src = srcmap(eps[i]["params"]); calls = [src[x["j"]] for x in rec[i]["log"] if x["type"] == "call"]
                    used.append(float("c" in calls)); first.append(float(bool(calls) and calls[0] == "c"))
                    ncache.append(calls.count("c")); nother.append(len(calls) - calls.count("c"))
                q = dict(n=len(sel), cache_calls=float(np.mean(ncache)), other_calls=float(np.mean(nother)))
                for m, v in [("used_cache", used), ("first_cache", first)]:
                    mean, lo, hi = nested_ci(v, sel, [eps[i]["scenario"] for i in sel])
                    q[m] = float(np.mean(v)); q[m + "_lo"] = float(lo); q[m + "_hi"] = float(hi)
                res["use"][cond + "|" + sub] = q
        out[key] = res
        lines.append(f"{label:24s}{res['n']:4d}{res['regret']:8.3f}" + "".join(f"{res[p]:9.3f}" for p in PARTS) +
                     f"{100 * res['over_share']:7.0f}{res['n_over_rep']:>5d}/{res['n_over']:<4d}{res['calls']:7.2f}"
                     f"{res['k2r_calls']:10.2f}{res['V0']:6.2f}{100 * res['zero']:7.0f}{100 * res['severe']:8.0f}")
        for f, v in res["by_family"].items():
            lines.append(f"    {f:20s}    {v['regret']:8.3f}" + "".join(f"{v[p]:9.3f}" for p in PARTS) + f"{'':17s}{v['calls']:7.2f}")
    lines.append("")
    lines.append(f"{'run':24s}{'cond|subset':>12s}{'n':>4s}{'regret':>8s}{'pass_cache [95% CI]':>24s}{'= other':>9s}{'+ stop':>8s}{'rest':>8s}{'share%':>8s}")
    for key, *_ in RUNS:
        for k, q in out[key]["pass"].items():
            lines.append(f"{out[key]['label']:24s}{k:>12s}{q['n']:4d}{q['regret']:8.3f}{q['pass_cache']:8.3f} [{q['pass_cache_lo']:.2f}, {q['pass_cache_hi']:.2f}]"
                         f"{q['pass_cache_other']:9.3f}{q['pass_cache_stop']:8.3f}{q['rest']:8.3f}"
                         + (f"{100 * q['pass_cache_share']:8.0f}" if q["pass_cache_share"] is not None else f"{'-':>8s}"))
    lines.append("")
    lines.append(f"{'run':24s}{'cond|subset':>12s}{'n':>4s}{'any replica call % [95% CI]':>30s}{'first call replica %':>22s}{'replica calls':>15s}{'other calls':>13s}")
    for key, *_ in RUNS:
        for k, q in out[key]["use"].items():
            lines.append(f"{out[key]['label']:24s}{k:>12s}{q['n']:4d}{100 * q['used_cache']:14.0f} [{100 * q['used_cache_lo']:.0f}, {100 * q['used_cache_hi']:.0f}]"
                         f"{100 * q['first_cache']:22.0f}{q['cache_calls']:15.2f}{q['other_calls']:13.2f}")
    txt = "\n".join(lines); print(txt)
    json.dump(out, open(os.path.join(HERE, "results", "k2e_anatomy.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "k2e_anatomy.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
