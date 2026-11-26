"""Forks of logged decision points (PLAN_be_belief_elicit.md). Each fork is ONE extra model turn on the logged prompt
of a decision point (results/be_points_<model>.jsonl); nothing is executed in the environment.

  A  the original prompt, unchanged; the action is recorded (reproducibility of the logged action)
  B  the original prompt + BELIEF_REQUEST: beliefs and values as one JSON object, no action
  C  the original prompt + BELIEF_REQUEST + "then act": the JSON, then the next action

  python be_run.py --model flash|terra|sonnet --forks A B [C] [--workers 4]     (real API / CLI calls)
  python be_run.py --model flash --forks A B C --mock exact|noisy|random --out FILE   (offline check)
Resumable: (point, fork) pairs already in the output are skipped; rows with api_error are retried on the next run.
Only points with subset=True are run (all points for flash and terra; the fixed Sonnet subset).
Output: logs/be_<stem>.jsonl
"""
import argparse, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from env import HERE, SCENARIOS, load_episodes, tool_schemas
from run import deepseek_backend, codex_or, Budget, parse_text_action, Fatal

MODELS = {"flash": ("deepseek-chat", "ds_chat_fc"), "terra": ("codex:gpt-6.1-terra@medium", "codex_terra"),
          "sonnet": ("claude:claude-sonnet-5-5@high", "claude_s55")}
lock = threading.Lock()


def belief_request(ep, pt, act):
    sc = SCENARIOS[ep["scenario"]]; st = sc["states"]; T = sc["tools"]
    tl = "; ".join(f"{T[j]['name']} (values: {', '.join(T[j]['values'])})" for j in pt["affordable"])
    s = ("Before your next action, report your current beliefs as one JSON object"
         + (", then take your next action as usual" if act else " and nothing else; take no action in this reply") + ". Format:\n"
         '{"belief": {"status": {"<status>": <probability>, ...}, "replica_stale": <probability>, '
         '"next_report": {"<tool>": {"<value>": <probability>, ...}, ...}, "value_of_calling": {"<tool>": <number>, ...}}}\n'
         f"status: the probability of each status ({', '.join(st)}) given all results so far. "
         "replica_stale: the probability that the read replica is stale (serving a frozen snapshot) in this session. "
         f"next_report and value_of_calling: for each tool you can still afford ({tl}). next_report gives the probability "
         "of each value that tool's next call would return. value_of_calling is how much lower your expected score (tool "
         "costs plus verdict loss, as defined above) would be if you called that tool next and then acted as well as "
         "possible, compared with submitting your best verdict now; it is negative if calling the tool is not worth its cost.")
    if act and pt["protocol"] == "text":
        s += " After the JSON, end your reply with your action as one JSON object, as before."
    return s


def find_belief(text):
    return find_belief_r(text)[0]


def find_belief_r(text):
    """(belief, repaired). Amendment after the smoke test (PLAN): if no complete {"belief": ...} object parses, the text
    from the last '{"belief"' is retried with 1-3 closing braces appended (repaired=True)."""
    dec = json.JSONDecoder(); best = None
    for m in re.finditer(r"\{", text or ""):
        try: obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError: continue
        if isinstance(obj, dict) and isinstance(obj.get("belief"), dict): best = obj["belief"]
    if best is not None or not text: return best, False
    k = max(text.rfind('{"belief"'), text.rfind('{ "belief"'))
    if k < 0: return None, False
    for n in (1, 2, 3):
        try: obj, _ = dec.raw_decode(text[k:].rstrip().rstrip("`").rstrip() + "}" * n)
        except json.JSONDecodeError: continue
        if isinstance(obj, dict) and isinstance(obj.get("belief"), dict): return obj["belief"], True
    return None, False


def parse_belief(ep, pt, bel):
    """Stated belief -> indices. Unknown names are dropped (recorded); probability vectors are kept unnormalised (the
    analysis normalises and records the raw sum)."""
    sc = SCENARIOS[ep["scenario"]]; st = sc["states"]; T = sc["tools"]; names = [t["name"] for t in T]; notes = []
    def num(x):
        if isinstance(x, bool): return None
        try: v = float(x.strip().rstrip("%")) if isinstance(x, str) else float(x) if isinstance(x, (int, float)) else None
        except ValueError: return None
        return v if v is not None and np.isfinite(v) else None
    out = dict(status=None, stale=None, next_report={}, value={}, notes=notes)
    if not isinstance(bel, dict): return out
    s = bel.get("status")
    if isinstance(s, dict):
        v = [0.0] * len(st); ok = True
        for k, x in s.items():
            a = st.index(k) if k in st else next((jj for t in T for jj, val in enumerate(t["values"]) if val == k), None)
            if a is None or num(x) is None: notes.append(f"status {k!r}"); ok = False; continue
            v[a] += num(x)
        out["status"] = v if sum(v) > 0 else None
    out["stale"] = num(bel.get("replica_stale"))
    for key, dst in (("next_report", "next_report"), ("value_of_calling", "value")):
        d = bel.get(key)
        if not isinstance(d, dict): continue
        for tn, x in d.items():
            if tn not in names: notes.append(f"{key} tool {tn!r}"); continue
            j = names.index(tn)
            if dst == "value":
                if num(x) is not None: out["value"][str(j)] = num(x)
            elif isinstance(x, dict):
                v = [0.0] * len(T[j]["values"])
                for val, p in x.items():
                    if val in T[j]["values"] and num(p) is not None: v[T[j]["values"].index(val)] += num(p)
                    else: notes.append(f"next_report {tn}.{val!r}")
                if sum(v) > 0: out["next_report"][str(j)] = v
    return out


def parse_action(ep, pt, r):
    sc = SCENARIOS[ep["scenario"]]; names = [t["name"] for t in sc["tools"]]; verdicts = sc["states"] + ["escalate"]
    if pt["protocol"] == "fc":
        tc = r.get("tool_calls") or []
        if not tc: return None                          # several calls: the first is the action (as for the points)
        if tc[0]["name"] == "submit":
            try: v = json.loads(tc[0]["arguments"] or "{}").get("verdict")
            except json.JSONDecodeError: return None
            v = str(v).strip().lower() if v is not None else ""
            return ["stop", verdicts.index(v)] if v in verdicts else None
        return ["probe", names.index(tc[0]["name"])] if tc[0]["name"] in names else None
    a = parse_text_action(r.get("content"))
    if a is None: return None
    if a[0] == "submit":
        v = str(a[1]).strip().lower() if a[1] is not None else ""
        return ["stop", verdicts.index(v)] if v in verdicts else None
    return ["probe", names.index(a[1])] if a[1] in names else None


def mock_reply(kind, ep, pt, fork, seed=0):
    """exact: the oracle's action and the exact JSON-model quantities; noisy: those plus noise; random: anything."""
    rng = np.random.default_rng([seed, abs(hash((pt["pid"], fork))) % 2**31])
    sc = SCENARIOS[ep["scenario"]]; st = sc["states"]; T = sc["tools"]; Q = pt["json"]
    def jit(v, s): v = np.clip(np.array(v) + rng.normal(0, s, len(v)), 1e-3, None); return (v / v.sum()).tolist()
    noise = {"exact": 0.0, "noisy": 0.08, "random": 0.5}[kind]
    act = pt["logged"] if kind != "random" else (["probe", int(rng.choice(pt["affordable"]))] if pt["affordable"] and rng.random() < .6 else ["stop", 0])
    bel = dict(status={st[k]: p for k, p in enumerate(jit(Q["post"], noise) if noise else Q["post"])},
               replica_stale=(Q["stale"] if Q["stale"] is not None else 0.0) + (rng.normal(0, noise) if noise else 0),
               next_report={T[int(j)]["name"]: {T[int(j)]["values"][o]: p for o, p in enumerate(jit(v, noise) if noise else v)} for j, v in Q["pred"].items()},
               value_of_calling={T[int(j)]["name"]: v + (rng.normal(0, 3 * noise) if noise else 0) for j, v in Q["voi"].items()})
    text = ""
    if fork in ("B", "C"): text = "Mock.\n" + json.dumps({"belief": bel})
    tcs = []
    if fork in ("A", "C"):
        if pt["protocol"] == "fc":
            nm = T[act[1]]["name"] if act[0] == "probe" else "submit"
            args = json.dumps({sc["arg"]: "x"}) if act[0] == "probe" else json.dumps({"verdict": (st + ["escalate"])[act[1]]})
            tcs = [dict(id="m", name=nm, arguments=args)]
        else:
            text += ("\n" + json.dumps({"action": "call", "tool": T[act[1]]["name"]}) if act[0] == "probe"
                     else "\n" + json.dumps({"action": "submit", "verdict": (st + ["escalate"])[act[1]]}))
    return dict(model=f"mock-{kind}", content=text, reasoning=None, tool_calls=tcs, finish_reason="stop",
                usage=dict(prompt_tokens=0, completion_tokens=0), latency=0.0)


def done_keys(path):
    if not os.path.exists(path): return set()
    s = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if not r["status"].startswith("api_error"): s.add((r["pid"], r["fork"]))
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(MODELS)); ap.add_argument("--forks", nargs="+", default=["A", "B"])
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--mock", choices=["exact", "noisy", "random"]); ap.add_argument("--out")
    ap.add_argument("--max-api-calls", type=int, default=None); ap.add_argument("--limit", type=int, default=None, help="first N points only (smoke test)")
    a = ap.parse_args(); model, stem = MODELS[a.model]; eps = load_episodes()
    pts = [json.loads(l) for l in open(os.path.join(HERE, "results", f"be_points_{a.model}.jsonl"), encoding="utf-8")]
    pts = [p for p in pts if p["subset"]][: a.limit]
    out = a.out or os.path.join(HERE, "logs", f"be_{stem}.jsonl"); done = done_keys(out)
    jobs = [(p, f) for p in pts for f in a.forks if (p["pid"], f) not in done]
    print(f"{len(jobs)} forks to run ({len(done)} already in {out})", flush=True)
    llm = None
    if not a.mock:
        cap = Budget(a.max_api_calls)
        llm = codex_or(None, cap) if model.startswith(("codex:", "claude:")) else codex_or(deepseek_backend(cap), cap)

    def work(job):
        p, fork = job; ep = eps[p["episode"]]; msgs = list(p["messages"])
        if fork in ("B", "C"): msgs.append({"role": "user", "content": belief_request(ep, p, fork == "C")})
        tools = tool_schemas(ep) if (p["protocol"] == "fc" and fork in ("A", "C")) else None
        mt = (1500 if fork == "A" else 3000) if model == "deepseek-chat" else 32768
        row = dict(pid=p["pid"], episode=p["episode"], label=p["label"], fork=fork, model=model if not a.mock else f"mock-{a.mock}",
                   protocol=p["protocol"], time=time.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            r = mock_reply(a.mock, ep, p, fork) if a.mock else llm(model, msgs, tools, mt, None)
        except Fatal:
            raise
        except Exception as e:  # noqa: BLE001
            row.update(status=f"api_error: {type(e).__name__}: {str(e)[:200]}")
        else:
            bel, rep = find_belief_r(r.get("content")) if fork in ("B", "C") else (None, False)
            row.update(status="ok", content=r.get("content"), reasoning=r.get("reasoning"), tool_calls=r.get("tool_calls"),
                       finish_reason=r.get("finish_reason"), usage=r.get("usage"), latency=r.get("latency"),
                       n_calls=len(r.get("tool_calls") or []),
                       action=parse_action(ep, p, r) if fork in ("A", "C") else None,
                       belief_found=bel is not None, belief_repaired=rep, belief=parse_belief(ep, p, bel) if bel is not None else None,
                       acted_in_B=(fork == "B" and (bool(r.get("tool_calls")) or parse_text_action(r.get("content")) is not None)))
        with lock:
            with open(out, "a", encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(work, jobs):
            print(f"{r['pid']} {r['fork']}: {r['status'][:60]} action={r.get('action')} belief={r.get('belief_found')}", flush=True)


if __name__ == "__main__":
    main()
