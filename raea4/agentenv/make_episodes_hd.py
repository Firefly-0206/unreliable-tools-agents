"""Episodes of the hard-description experiment (PLAN_hd_hard_description.md). No model call.

50 parameter draws, first domain, PERSIST tools (replica tool 0 cost 1, primary-database tool 1 cost 2, log tool 2
cost 1, live probe 3 cost 4), persistent benign-stuck fault on the replica. Prior as in PERSIST (numbers, shown).
Fault rate: drawn uniformly inside one of three word ranges (env.HD_PHI_BINS), 17 / 17 / 16 draws.
Accuracy of each tool: drawn uniformly inside one of the word ranges allowed for that tool (env.HD_ACC_BINS):
  replica and probe: highly reliable or usually right;  primary and logs: usually right or noisy.
Every number is rounded to 2 decimals (the ranges have 2-decimal ends, so rounding stays inside the range).
Selection, as in the X sweep (oracle opens with the replica tool in 100% of draws at 0.10/0.30 and 0% at 0.70): the
accuracies and the fault rate of a draw are redrawn (prior kept) until the exact oracle opens with the replica tool
(rare and occasional draws) or does not (frequent draws); the number of tries is stored per episode ('draw_tries').

Three arms per draw, same parameters, hidden status, fault flag and pre-drawn outputs (common random numbers):
  HN  numbers (default documentation, as in every earlier first-domain experiment)
  HQ  words, persistence stated (env.hd_doc 'hq')
  HI  words, persistence left to inference (env.hd_doc 'hi')
Checks done here (assertions): the three arms share every parameter and output; HQ/HI tool docs contain no rate (no percent,
decimal or probability; their other digits are those of the default entry); the words read back (k2decomp.parse_docs) give the true word of every rate; HN docs read back give the
true numbers; the oracle's first action is stored per episode ('oracle_first').
Usage: python make_episodes_hd.py -> episodes_hd.jsonl, ground_truth_hd.jsonl
"""
import glob, json, os, re
import numpy as np
from env import (HERE, SCENARIO_LIST, SCENARIOS, BUDGET, HD_PHI_BINS, HD_ACC_BINS, sample_params, sample_world, build_model,
                 ticket_id, tool_doc, system_prompt, register_surface)
from oracle import qvalues
import k2decomp as K2D

SEED, N = 2035, 50
PHI_WORDS = ["rare"] * 17 + ["occasional"] * 17 + ["frequent"] * 16
ACC_ALLOWED = [["highly reliable", "usually right"], ["usually right", "noisy"], ["usually right", "noisy"],
               ["highly reliable", "usually right"]]
ARMS = [("HN", None), ("HQ", "hq"), ("HI", "hi")]


def u2(rng, lo, hi):
    x = float(np.round(rng.uniform(lo, hi), 2)); assert lo - 1e-9 <= x <= hi + 1e-9; return x


def word_of(x, bins):
    w = [k for k, (lo, hi) in bins.items() if lo - 1e-9 <= x <= hi + 1e-9]; assert len(w) == 1, (x, w); return w[0]


def main():
    seen = set()
    for f in glob.glob(os.path.join(HERE, "episodes*.jsonl")):
        if os.path.basename(f) != "episodes_hd.jsonl":
            seen |= {json.loads(l)["ticket"] for l in open(f, encoding="utf-8")}
    root = np.random.SeedSequence([SEED, 0]); s_order, *children = root.spawn(N + 1)
    phi_words = list(np.random.default_rng(s_order).permutation(PHI_WORDS))
    eps, truth, first = [], [], {w: [] for w in HD_PHI_BINS}
    for k, child in enumerate(children):
        s_par, s_world, s_words = child.spawn(3); rw = np.random.default_rng(s_words)
        base = sample_params("PERSIST", np.random.default_rng(s_par))
        pw = str(phi_words[k]); want = pw != "frequent"; tries = 0
        while True:     # redraw words and rates until the oracle's first action is as required (see the docstring)
            tries += 1; assert tries <= 200, k
            aw = [str(rw.choice(a)) for a in ACC_ALLOWED]
            tools = [dict(t, q=u2(rw, *HD_ACC_BINS[aw[j]])) for j, t in enumerate(base["tools"])]
            par0 = dict(base, phi=u2(rw, *HD_PHI_BINS[pw]), stuck="benign", tools=tools)
            M = build_model(par0); q = qvalues(M, M.prior.copy(), float(M.budget)); of = min(q, key=q.get)
            if (of == ("probe", 0)) == want: break
        assert [t["src"] for t in tools] == ["cache", "primary", "logs", "live"] and [t["cost"] for t in tools] == [1, 2, 1, 4]
        first[pw].append(tries)
        words = dict(phi=pw, acc=aw)
        tables = []
        for fam, style in ARMS:
            eid = f"{fam}-{k:03d}"; par = dict(par0, family=fam, tools=[dict(t) for t in tools])
            Mf = build_model(par); w, table = sample_world(par, np.random.default_rng(s_world), Mf); tables.append(table)
            salt = 0; t = ticket_id(eid)
            while t in seen: salt += 1; t = ticket_id(eid, salt)
            seen.add(t)
            e = dict(id=eid, ticket=t, family=fam, scenario=SCENARIO_LIST[k % len(SCENARIO_LIST)], pair=f"HD-{k:03d}", arm=fam,
                     phi=par["phi"], phi_word=pw, words=words, params=par, table=table,
                     oracle_first=[of[0], int(of[1])], draw_tries=tries)
            if style: e["doc_style"] = style
            eps.append(e); truth.append(dict(id=eid, **w, V0=float(Mf.V(Mf.prior.copy(), float(BUDGET))[0])))
            # checks on the documentation
            text = system_prompt(e, "K2e", "text"); docs = K2D.parse_docs(e, text)
            for j in range(4):
                dj = tool_doc(e, j)
                if style:
                    assert not re.search(r"\d\s*%|\d\.\d|probabilit|\bper cent\b|\bout of\b", dj), dj
                    hn = tool_doc(dict(e, doc_style=None), j)        # same entry, default style: digits differ only in rates
                    assert re.findall(r"\d+", dj) == [x for x in re.findall(r"\d+", hn) if x in re.findall(r"\d+", dj)], (dj, hn)
                    assert word_of(docs[j]["acc"], HD_ACC_BINS) == aw[j] and docs[j]["vague"]
                    assert aw[j] in dj
                else:
                    assert abs(docs[j]["acc"] - tools[j]["q"]) < 1e-9 and not docs[j].get("vague")
            if style:
                assert word_of(docs[0]["phi"], HD_PHI_BINS) == pw and docs[0]["cache"] and not any(d["cache"] for d in docs[1:])
                assert ("Replica staleness incidents: " in text) == (style == "hq")
                assert ("restarted only between on-call sessions" in text) == (style == "hi")
            else:
                assert abs(docs[0]["phi"] - par["phi"]) < 0.005 + 1e-9    # HN states the rate as a whole percent
        assert tables[0] == tables[1] == tables[2]
    for name, rows in (("episodes_hd.jsonl", eps), ("ground_truth_hd.jsonl", truth)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(eps)} episodes ({N} draws x {len(ARMS)} arms)")
    for w, v in first.items():
        print(f"  {w:11s} n={len(v):2d}  draws tried per accepted draw: mean {np.mean(v):.2f}, max {max(v)}")
    acc = {}
    for e in eps[::3]:
        for j, w in enumerate(e["words"]["acc"]): acc.setdefault(j, []).append(w)
    for j, v in acc.items(): print(f"  tool {j}: " + ", ".join(f"{w} {v.count(w)}" for w in HD_ACC_BINS))
    hn = [len(system_prompt(e, "K2e", "text").split()) for e in eps if e["arm"] == "HN"]
    hq = [len(system_prompt(e, "K2e", "text").split()) for e in eps if e["arm"] == "HQ"]
    print(f"  system prompt words: HN {np.mean(hn):.0f}, HQ {np.mean(hq):.0f}")


if __name__ == "__main__":
    main()
