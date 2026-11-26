"""Provider-agnostic runner for bench v2 (resumable JSONL output).

  python bench/run.py --provider mock     --model mock --conds all --samples 1            (= --dry-run)
  python bench/run.py --provider deepseek --model deepseek-chat --conds P0,P1 --samples 3 --max-tokens 8192
  python bench/run.py --provider deepseek --model deepseek-reasoner --conds P0 --max-tokens 4096 --tag cap4k
  python bench/run.py --export bench/export/haiku_prompts.jsonl --conds all --samples 2
  python bench/run.py --ingest answers.jsonl --model claude-haiku-x --out bench/results/haiku.jsonl
  python bench/run.py --estimate-cost --model deepseek-reasoner --conds all --samples 1

Run key = (prompt id, model, sample, max_tokens, tag, temperature as sent, hash of --extra-body); the default
output file also carries the tag. A key with an ok=True record (the API answered) is never redone, whether or
not its answer parsed (truncation under a fixed cap is an outcome). --retry-unparsed re-asks unparsed keys;
every record carries `attempt` (0 = first record of its key), retries are appended, and score.py scores the
FIRST attempt and reports retries separately. ok=False records are kept and scored as failed answers.
The DeepSeek key is read from DEEPSEEK_API_KEY or ../../api.txt at call time and never written anywhere.
"""
import argparse, hashlib, json, os, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core                                         # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
import codex_backend                                 # noqa: E402

URL = "https://api.deepseek.com/chat/completions"
KEYFILE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "api.txt")
PRICES = {"flash": (0.15, 0.60), "pro": (0.66, 1.98)}          # USD per 1M tokens (input, output); x2 at peak
OUT_TOKENS = {"deepseek-chat": 2800, "deepseek-reasoner": 31000, "deepseek-v4-pro": 25000, "mock": 1000}
THINKING = {"deepseek-reasoner", "deepseek-v4-pro"}
FINAL_NUDGE = ("The calculator is no longer available. Using the results you already have, give your final answer "
               "now, ending with the one-line JSON {\"p_benign\": ..., \"action\": ...}.")
lock = threading.Lock()
api_calls = [0]

def price_of(model):
    return PRICES["pro"] if "pro" in model else PRICES["flash"]

def resolve(path, default):
    """Alternate task files (held-out confirmation set): absolute, relative to the cwd, or relative to bench/."""
    if not path: return os.path.join(HERE, default)
    return path if os.path.isabs(path) or os.path.exists(path) else os.path.join(HERE, path)

def load_items(path=None):
    return {json.loads(l)["id"]: json.loads(l) for l in open(resolve(path, "items.jsonl"), encoding="utf-8")}

def load_prompts(conds, path=None):
    P = [json.loads(l) for l in open(resolve(path, "prompts.jsonl"), encoding="utf-8")]
    return [p for p in P if conds is None or p["cond"] in conds]

def api_key():
    k = os.environ.get("DEEPSEEK_API_KEY")
    if not k and os.path.exists(KEYFILE):
        m = re.search(r"sk-[A-Za-z0-9]+", open(KEYFILE, encoding="utf-8").read())
        k = m.group(0) if m else None
    if not k: sys.exit("No DeepSeek key: set DEEPSEEK_API_KEY or provide api.txt")
    return k

# ------------------------------------------------------------------ providers ------------------------
class BudgetExhausted(Exception):
    pass

def deepseek_call(a, messages, tools=None, tool_choice=None):
    with lock:
        if a.max_api_calls is not None and api_calls[0] >= a.max_api_calls: raise BudgetExhausted()
        api_calls[0] += 1
    body = {"model": a.model, "messages": messages, "max_tokens": a.max_tokens}
    if a.temperature is not None and a.model not in THINKING: body["temperature"] = a.temperature
    if tools: body["tools"] = tools
    if tool_choice: body["tool_choice"] = tool_choice
    if a.extra_body: body.update(json.loads(a.extra_body))
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
        return json.loads(r.read().decode())

def codex_call(a, messages):
    """Codex CLI (no function calling, so T1 is not run with this provider). Same response shape as DeepSeek."""
    with lock:
        if a.max_api_calls is not None and api_calls[0] >= a.max_api_calls: raise BudgetExhausted()
        api_calls[0] += 1
    r = codex_backend.call(a.model, messages, timeout=a.timeout)
    msg = {"role": "assistant", "content": r["content"], "reasoning_content": None}
    return {"model": r["model"], "choices": [{"message": msg, "finish_reason": "stop"}], "usage": r["usage"],
            "codex_tool_items": r["tool_items"]}

def mock_call(a, messages, it, cond, sample, tools=None, tool_choice=None):
    """Deterministic fake model: mostly optimal, sometimes the fault-unaware choice, sometimes junk."""
    r = int(hashlib.sha256(f"{it['id']}|{cond}|{sample}|{len(messages)}".encode()).hexdigest(), 16)
    u = (r % 10000) / 10000; u2 = ((r >> 20) % 10000) / 10000
    n_in = sum(len(str(m.get("content") or "")) for m in messages) // 4
    n_out = 400 + (r >> 40) % 3000
    usage = {"prompt_tokens": n_in, "completion_tokens": min(n_out, a.max_tokens),
             "completion_tokens_details": {"reasoning_tokens": 0}}
    if tools and tool_choice != "none" and len(messages) == 1:
        msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function",
               "function": {"name": "calculate", "arguments": json.dumps({"expression": "0.5*0.9**2/(0.5*0.9**2+0.5*0.05**2)"})}}]}
        return {"model": "mock", "choices": [{"message": msg, "finish_reason": "tool_calls"}], "usage": usage}
    if cond == "H1":
        st = core.true_struct(it)
        if u < 0.25: st["sources"] = [{"tools": [t for t in it["tools"] if t != "V"], "fault_prob": it["phi"], "fault_persists": True}]
        if 0.25 <= u < 0.35: st["sources"][0]["fault_persists"] = not st["sources"][0]["fault_persists"]
        text = "Model of the tools:\n<model>" + json.dumps(st) + "</model>"
    else:
        if u < 0.55: act = it["best_action"]; p = it["p_benign"] + (u2 - 0.5) * 0.02
        elif u < 0.8: act = it["naive"]["fault_unaware"]["action"]; p = it["naive"]["fault_unaware"]["p_benign"]
        else: act = sorted(it["Q"])[r % len(it["Q"])]; p = u2
        text = f"Working...\n{{\"p_benign\": {min(max(p, 0), 1):.3f}, \"action\": \"{core.to_scen(it, act)}\"}}"
        if u2 < 0.04: text = "I would escalate this, probability about 0.8."
    fin = "stop"
    if cond != "H1" and 0.04 <= u2 < 0.06:       # truncated answer that quotes the old format template
        text, fin = 'I must end with: {"p_benign": 0.00, "action": "escalate"}. Let me compute P(A stale | ', "length"
    if n_out > a.max_tokens: text, fin = text[: len(text) // 3], "length"
    return {"model": "mock", "choices": [{"message": {"role": "assistant", "content": text,
                                                      "reasoning_content": None}, "finish_reason": fin}], "usage": usage}

def add_usage(tot, u):
    if not u: return tot
    tot["prompt_tokens"] += u.get("prompt_tokens", 0); tot["completion_tokens"] += u.get("completion_tokens", 0)
    tot["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    tot["prompt_cache_hit_tokens"] += u.get("prompt_cache_hit_tokens", 0) or 0
    return tot

def converse(a, p, it, sample):
    """One answer: a single call, or the T1 tool loop. Returns the record fields from the provider."""
    messages = [{"role": "user", "content": p["prompt"]}]
    tools = [core.CALC_TOOL] if p["tools"] else None
    tot = dict(prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, prompt_cache_hit_tokens=0)
    trace, turns, n_tc, reasoning, model_ret, fin = [], [], 0, [], None, None
    t0 = time.time()
    for rnd in range(a.max_tool_rounds + 1):
        last = tools is not None and (rnd == a.max_tool_rounds or n_tc >= a.max_tool_calls)
        tc_mode = ("none" if last else "auto") if tools else None
        if last and rnd > 0:           # tool budget used up: ask for the final answer explicitly
            messages.append({"role": "user", "content": FINAL_NUDGE})
        if a.provider == "mock": resp = mock_call(a, messages, it, p["cond"], sample, tools, tc_mode)
        elif a.provider == "codex": resp = codex_call(a, messages)
        else: resp = deepseek_call(a, messages, tools, tc_mode)
        ch = resp["choices"][0]; msg = ch["message"]; fin = ch.get("finish_reason"); model_ret = resp.get("model")
        if resp.get("codex_tool_items"): trace.append(dict(codex_tool_items=resp["codex_tool_items"]))
        add_usage(tot, resp.get("usage")); turns.append(dict(usage=resp.get("usage"), finish_reason=fin))
        if msg.get("reasoning_content"): reasoning.append(msg["reasoning_content"])
        tcs = msg.get("tool_calls") or []
        if not tools or not tcs or last:
            return dict(content=msg.get("content") or "", reasoning_content="\n\n---\n\n".join(reasoning) or None,
                        finish_reason=fin, model_returned=model_ret, usage=tot, turns=turns, tool_trace=trace,
                        n_tool_calls=n_tc, latency=round(time.time() - t0, 2))
        am = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs}
        if msg.get("reasoning_content"): am["reasoning_content"] = msg["reasoning_content"]
        messages.append(am)
        for tc in tcs:
            n_tc += 1
            try: expr = json.loads(tc["function"]["arguments"]).get("expression", "")
            except Exception: expr = ""
            res = core.safe_eval(expr) if tc["function"]["name"] == "calculate" else "error: unknown tool"
            trace.append(dict(expression=expr, result=res))
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": res})

def parse_into(rec, it, cond):
    """Every answer cut by the output cap is 'truncated' and never counts as parsed, even if JSON-like text
    appears in it."""
    if cond == "H1":
        r = core.parse_h1(it, rec.get("content")); rec["h1_struct"] = r.get("struct")
        rec["h1_fixes"] = r.get("h1_fixes"); rec["h1_action_planner"] = r.get("action_planner")
    else:
        r = core.parse_answer(it, rec.get("content")); rec["raw_action"] = r.get("raw_action")
    rec.update(parse_status=r["status"], p_benign=r["p_benign"], action=r["action"])
    if rec.get("finish_reason") == "length": rec.update(parse_status="truncated", p_benign=None, action=None)
    elif "DSML" in (rec.get("content") or "") and r["status"] != "ok": rec["parse_status"] = "tool_markup_in_content"
    return rec

def sent_temperature(model, temperature):
    return None if model in THINKING else temperature

def extra_hash(extra_body):
    if not extra_body: return ""
    return hashlib.sha256(json.dumps(json.loads(extra_body), sort_keys=True).encode()).hexdigest()[:10]

def run_key(r):
    return (r["pid"], r["model"], r["sample"], r["max_tokens"], r.get("tag") or "", r.get("temperature"),
            r.get("extra_hash") or "")

def job(a, p, it, sample, out, attempt_idx):
    rec = dict(pid=p["pid"], item=p["item"], cond=p["cond"], model=a.model, provider=a.provider, sample=sample,
               max_tokens=a.max_tokens, tag=a.tag, temperature=sent_temperature(a.model, a.temperature),
               extra_body=a.extra_body, extra_hash=extra_hash(a.extra_body), attempt=attempt_idx, call_errors=[])
    for attempt in range(a.retries):
        if attempt: rec["call_errors"].append(rec.get("error"))     # failed calls before the one recorded
        try:
            rec.update(converse(a, p, it, sample), ok=True); parse_into(rec, it, p["cond"]); break
        except BudgetExhausted:
            print("API-call limit reached; stopping this job", flush=True); return
        except urllib.error.HTTPError as e:
            body = e.read()[:300]
            rec.update(ok=False, error=f"HTTP {e.code}: {body!r}")
            if e.code in (401, 402, 403):
                print("HTTP", e.code, "- not retrying (auth or balance)", flush=True); break
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            rec.update(ok=False, error=repr(e)[:300]); time.sleep(5 * (attempt + 1))
    with lock:
        with open(out, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(("ok  " if rec.get("ok") else "ERR ") + f"{p['pid']:60s} s{sample} "
              + (f"{rec.get('parse_status')} a={rec.get('action')} p={rec.get('p_benign')} "
                 f"tok={rec['usage']['completion_tokens']} {rec['latency']}s" if rec.get("ok") else rec["error"][:100]),
              flush=True)

def done_keys(out, retry_unparsed):
    """-> (run keys that are done, number of records already written per key)."""
    s, n = set(), {}
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            try: r = json.loads(l)
            except Exception: continue
            k = run_key(r); n[k] = n.get(k, 0) + 1
            if r.get("ok") and (not retry_unparsed or r.get("parse_status") == "ok"): s.add(k)
    return s, n

# ------------------------------------------------------------------ cost estimate --------------------
def estimate(prompts, models, samples, cap=None, t1_rounds=6, verbose=True):
    rows = []
    for m in models:
        pin, pout = price_of(m); o = OUT_TOKENS.get(m, 3000)
        o = min(o, cap) if cap else o
        tin = tout = 0
        for p in prompts:
            n = len(p["prompt"]) / 3.8 + (300 if p["tools"] else 0)     # smoke test: 3.6-4.0 chars/token; tool schema ~300
            if p["tools"]: tin += t1_rounds * n + o * (t1_rounds - 1) / 2; tout += 1.2 * o
            else: tin += n; tout += o
        tin *= samples; tout *= samples
        usd = (tin * pin + tout * pout) / 1e6
        rows.append(dict(model=m, calls=len(prompts) * samples, input_Mtok=tin / 1e6, output_Mtok=tout / 1e6,
                         usd_offpeak=usd, usd_peak=2 * usd))
        if verbose:
            print(f"{m:20s} answers={len(prompts) * samples:5d} in={tin / 1e6:7.2f}M out={tout / 1e6:7.2f}M "
                  f"USD off-peak {usd:8.2f}  peak {2 * usd:8.2f}")
    return rows

# ------------------------------------------------------------------ export / ingest ------------------
def export(prompts, samples, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in prompts:
            for s in range(samples):
                f.write(json.dumps(dict(pid=p["pid"], sample=s, prompt=p["prompt"],
                                        tools=[core.CALC_TOOL] if p["tools"] else None), ensure_ascii=False) + "\n")
    print(f"exported {len(prompts) * samples} requests to {path}. For T1 rows the harness must implement "
          "calculate() exactly as core.safe_eval and loop until a final text answer (max 20 tool calls). "
          "Return one JSON line per request: {pid, sample, content, model?, usage?, latency?, finish_reason?, "
          "tool_trace?, reasoning_content?, max_tokens?}")

def ingest(path, a, items):
    P = {p["pid"]: p for p in load_prompts(None, a.prompts_file)}; n = 0
    with open(a.out, "a", encoding="utf-8") as f:
        for l in open(path, encoding="utf-8"):
            x = json.loads(l); p = P[x["pid"]]
            rec = dict(pid=p["pid"], item=p["item"], cond=p["cond"], model=x.get("model") or a.model, provider="ingest",
                       sample=x.get("sample", 0), max_tokens=x.get("max_tokens", a.max_tokens), tag=a.tag,
                       temperature=x.get("temperature"), attempt=x.get("attempt", 0), ok=x.get("ok", True),
                       content=x.get("content", ""), reasoning_content=x.get("reasoning_content"),
                       finish_reason=x.get("finish_reason"), model_returned=x.get("model_returned", x.get("model")),
                       usage=x.get("usage") or {}, latency=x.get("latency"), tool_trace=x.get("tool_trace", []),
                       n_tool_calls=len(x.get("tool_trace", [])))
            parse_into(rec, items[p["item"]], p["cond"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
    print(f"ingested {n} answers into {a.out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek", choices=["deepseek", "codex", "mock"])
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--conds", default="all", help="comma list or 'all' or 'core' (P0-P3,L1-L3,C1-C3)")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (after filtering)")
    ap.add_argument("--items", default=None, help="comma list of item ids or substrings to keep")
    ap.add_argument("--max-api-calls", type=int, default=None, help="hard cap on HTTP calls in this process")
    ap.add_argument("--max-tool-rounds", type=int, default=20)
    ap.add_argument("--max-tool-calls", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--extra-body", default=None, help="JSON merged into the request body")
    ap.add_argument("--tag", default="")
    ap.add_argument("--items-file", default=None, help="alternate items jsonl (default bench/items.jsonl)")
    ap.add_argument("--prompts-file", default=None, help="alternate prompts jsonl (default bench/prompts.jsonl)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--retry-unparsed", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="use the mock provider")
    ap.add_argument("--estimate-cost", action="store_true")
    ap.add_argument("--t1-rounds", type=int, default=6, help="assumed API rounds per T1 answer (estimate only)")
    ap.add_argument("--export", default=None)
    ap.add_argument("--ingest", default=None)
    a = ap.parse_args()
    if a.dry_run: a.provider, a.model = "mock", ("mock" if a.model == "deepseek-chat" else a.model)
    if a.conds == "all": conds = core.CONDITIONS
    elif a.conds == "core": conds = core.CORE_CONDS
    else: conds = a.conds.split(",")
    items = load_items(a.items_file)
    prompts = load_prompts(set(conds), a.prompts_file)
    if a.items: prompts = [p for p in prompts if any(s in p["item"] for s in a.items.split(","))]
    prompts.sort(key=lambda p: (conds.index(p["cond"]), p["item"]))
    if a.limit: prompts = prompts[: a.limit]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    if a.out is None: a.out = os.path.join(HERE, "results", f"{a.provider}_{a.model}{'_' + a.tag if a.tag else ''}.jsonl")
    if a.estimate_cost:
        estimate(prompts, [a.model], a.samples, cap=a.max_tokens if a.max_tokens != 8192 else None, t1_rounds=a.t1_rounds); return
    if a.export: export(prompts, a.samples, a.export); return
    if a.ingest: ingest(a.ingest, a, items); return
    have, nrec = done_keys(a.out, a.retry_unparsed)
    key = lambda pid, s: (pid, a.model, s, a.max_tokens, a.tag or "", sent_temperature(a.model, a.temperature),
                          extra_hash(a.extra_body))
    jobs = [(p, s) for p in prompts for s in range(a.samples) if key(p["pid"], s) not in have]
    print(f"{len(jobs)} answers to get ({len(prompts) * a.samples - len(jobs)} already done) -> {a.out}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for p, s in jobs: ex.submit(job, a, p, items[p["item"]], s, a.out, nrec.get(key(p["pid"], s), 0))
    print(f"finished; HTTP calls made: {api_calls[0]}", flush=True)

if __name__ == "__main__":
    main()
