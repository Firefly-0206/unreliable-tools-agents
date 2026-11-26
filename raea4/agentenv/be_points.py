"""Decision points of the belief-elicitation experiment (PLAN_be_belief_elicit.md). Offline, no model call.

Source: the K2e CORE episodes of the decomposition runs (domain 1, n = 100 per model; the logs of k2e_anatomy.py and
u_units.py). Every model turn after the observation-model JSON is a decision; its prompt is the logged transcript up
to that turn (all three backends are stateless: each turn re-sends the full visible transcript, and flash's
reasoning_content, when present, is part of it). A decision is a POINT if its logged action is a valid call or submit
(fc: a turn with several tool calls is one point, its first call being the logged action; n_parallel records the
count; the later calls of that turn advance the beliefs but are not points). parse_ok records whether the episode's
JSON was usable (otherwise its 'json' reference is the documented default model and is excluded from JSON readings).
P(replica stale) is 0 where the model has no fault group for the replica, None where a written JSON gives it only a
per-call fault. For each point, from the logged reports before it:

  true model (oracle)  status posterior, P(replica stale), next-report law of every affordable tool, exact Q* of every
                       action, V*, R = best stop value, multi-step value of calling tool j  VOI_j = R - Q*(probe j)
                       (cost included; negative when not worth it), one-step value VOI1_j = R - (c_j + E_o R(b_o))
  model's JSON         the same quantities under the model's own written observation model (k2decomp)
  label                O  the logged action is optimal (advantage <= 1e-9)
                       U  otherwise, the oracle would query a replica (cache-backed) tool and the model does something
                          else (another source or stop) = k2e_anatomy pass_cache
                       R  every other step
Sonnet subset (fixed here, seed 2036): every U point plus as many O points drawn at random, at most 150 in total.

  python be_points.py   ->  results/be_points_<model>.jsonl  (one row per point, with the prompt messages)
                            results/be_points_summary.txt
"""
import json, os
import numpy as np
from env import HERE, CORE, SCENARIOS, ELICIT_ACK, load_episodes, tool_schemas, srcmap
from oracle import qvalues, p_stale, true_model
import k2decomp as K2D

LOGS = {"flash": ("ds_chat_fc_k2decomp.jsonl", "fc"), "terra": ("codex_terra_medium_k2decomp.jsonl", "text"),
        "sonnet": ("claude_s55.jsonl", "text")}
ACKS = (ELICIT_ACK, "No observation model recorded. Now take your first action.")
SONNET_CAP, SEED = 150, 2036


def load(fn):
    recs = {}
    for line in open(os.path.join(HERE, "logs", fn), encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != "K2e" or r["status"].startswith("api_error") or r.get("sample", 0) != 0 or r["family"] not in CORE: continue
        recs[r["episode"]] = r                      # a later successful rerun replaces an earlier row (as in score.py)
    return recs


def stale_of(M, b, j=0):
    """P(the fault group of tool j is stale) under model M (None if tool j has no persistent fault flag)."""
    g = M.tools[j].get("group")
    if M.tools[j].get("marg_phi") is not None: return None     # per-call fault in a written JSON: no session flag
    if g not in M.G: return 0.0                                 # never stale (no fault group, e.g. TRANSIENT: phi = 0)
    k = M.G.index(g); return float(sum(w for w, h in zip(b, M.H) if h[1][k]))


def quantities(M, b, B):
    q = qvalues(M, b, B); V = min(q.values()); R = min(v for k, v in q.items() if k[0] == "stop")
    voi, voi1, pred = {}, {}, {}
    for (kind, j), v in q.items():
        if kind != "probe": continue
        p = M.pred(b, j); pred[j] = [float(x) for x in p]; voi[j] = R - v
        e = sum(pk * M.term(M.update(b, j, o))[0] for o, pk in enumerate(p) if pk > 1e-12)
        voi1[j] = R - (M.tools[j]["cost"] + e)
    best = min(q, key=q.get)
    return dict(post=[float(x) for x in M.marg_s(b)], stale=stale_of(M, b), V=V, R=R, best=[best[0], int(best[1])],
                q={f"{k[0]}:{k[1]}": v for k, v in q.items()}, voi={str(j): v for j, v in voi.items()},
                voi1={str(j): v for j, v in voi1.items()}, pred={str(j): v for j, v in pred.items()})


def points_of(ep, r, proto):
    msgs = r["transcript"]; log = r["log"]
    i0 = next(i for i, m in enumerate(msgs) if m["role"] == "user" and m.get("content") in ACKS)
    turns = [i for i in range(i0 + 1, len(msgs)) if msgs[i]["role"] == "assistant"]
    spec = r["elicit"]["spec"]; Mt = true_model(ep); Mh = K2D.build_recovered_model(ep, spec)
    src = srcmap(ep["params"]); cache = {j for j, s in src.items() if s == "c"}
    bt, bh, B = Mt.prior.copy(), Mh.prior.copy(), float(Mt.budget); li = 0; out = []
    for t, i in enumerate(turns):
        if li >= len(log): break                                  # turns after the episode ended
        n_entries = max(1, len(msgs[i].get("tool_calls") or [])) if proto == "fc" else 1
        entries = log[li:li + n_entries]; li += n_entries
        e = entries[0]
        valid = e["type"] in ("call", "submit")       # fc turn with several calls: the point is its first call
        if valid:
            T, H = quantities(Mt, bt, B), quantities(Mh, bh, B)
            act = ["probe", e["j"]] if e["type"] == "call" else ["stop", e["a"]]
            adv = T["q"][f"{act[0]}:{act[1]}"] - T["V"]
            o_cache = T["best"][0] == "probe" and T["best"][1] in cache
            m_cache = act[0] == "probe" and act[1] in cache
            label = "O" if adv <= 1e-9 else ("U" if o_cache and not m_cache else "R")
            out.append(dict(pid=f"{ep['id']}#{t}", episode=ep["id"], scenario=ep["scenario"], family=ep["family"], turn=t,
                            protocol=proto, messages=msgs[:i], logged=act, adv=adv, label=label, budget=B,
                            calls_before=sum(x["type"] == "call" for x in log[:li - n_entries]), true=T, json=H,
                            cache=sorted(cache), affordable=sorted(int(j) for j in T["voi"]), n_parallel=n_entries,
                            parse_ok=bool(r["elicit"]["parse_ok"])))
        for x in entries:                                          # advance beliefs with the logged reports
            if x["type"] == "call":
                bt = Mt.update(bt, x["j"], x["o"]); nb = Mh.update(bh, x["j"], x["o"]); bh = nb if nb is not None else Mh.prior.copy()
                B -= Mt.tools[x["j"]]["cost"]
        if any(x["type"] == "submit" for x in entries): break
    return out


def main():
    eps = load_episodes(); L = ["Decision points (be_points.py)"]
    for key, (fn, proto) in LOGS.items():
        recs = load(fn); pts = []
        for i in sorted(recs): pts += points_of(eps[i], recs[i], proto)
        lab = [p["label"] for p in pts]
        if key == "sonnet":
            rng = np.random.default_rng(SEED); U = [p for p in pts if p["label"] == "U"]; O = [p for p in pts if p["label"] == "O"]
            nU = min(len(U), SONNET_CAP // 2); keepU = sorted(rng.choice(len(U), nU, replace=False)) if nU < len(U) else range(len(U))
            keepO = sorted(rng.choice(len(O), min(len(O), SONNET_CAP - nU), replace=False))
            sub = {U[k]["pid"] for k in keepU} | {O[k]["pid"] for k in keepO}
            for p in pts: p["subset"] = p["pid"] in sub
        else:
            for p in pts: p["subset"] = True
        with open(os.path.join(HERE, "results", f"be_points_{key}.jsonl"), "w", encoding="utf-8") as f:
            for p in pts: f.write(json.dumps(p) + "\n")
        n_sub = sum(p["subset"] for p in pts)
        L.append(f"{key:7s} episodes {len(recs)}  points {len(pts)}  U {lab.count('U')}  O {lab.count('O')}  R {lab.count('R')}  "
                 f"in run subset {n_sub} (U {sum(p['subset'] and p['label'] == 'U' for p in pts)}, O {sum(p['subset'] and p['label'] == 'O' for p in pts)})")
    txt = "\n".join(L); print(txt); open(os.path.join(HERE, "results", "be_points_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
