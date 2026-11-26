"""
Closed-loop LLM runner. The model itself chooses calls, retries, source switches and when to stop.

  python run.py --models deepseek-chat --families SHARED --conditions K1 --episodes 1 --protocol fc
  python run.py --mock random --families PERSIST TRANSIENT SHARED MIXED --conditions K1 K2 K3 K4 --episodes 3

Protocols: fc (OpenAI-compatible function calling) and text (one JSON action per assistant turn).
Logs: one JSON line per (episode, condition, model, protocol, sample), resumable; full transcripts, usage
including reasoning tokens, latency, finish reasons and the model id returned by the API.
The API key is read at runtime from $DEEPSEEK_API_KEY or ../../api.txt and is never logged or printed.
"""
import argparse, hashlib, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from env import (HERE, Env, CORE, CONDITIONS, ALL_CONDITIONS, ELICIT, SCENARIOS, ELICIT_REQUEST, ELICIT_ACK, ELICIT_NUDGE,
                 load_episodes, system_prompt, first_user, first_user_cond, tool_schemas)
import k2decomp as K2D

KEYFILE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "api.txt")
BASE_URL = "https://api.deepseek.com"
KIMI_URL = "https://api.moonshot.cn/v1"                 # models named kimi-*: key on line 2 of api.txt
NUDGE_FC = "Please continue: call one of the tools, or call submit with your verdict."
NUDGE_TEXT = ("Please end your reply with exactly one JSON object: {\"action\": \"call\", \"tool\": \"<name>\"} or "
              "{\"action\": \"submit\", \"verdict\": \"<verdict>\"}.")
THINKING = {"deepseek-reasoner", "deepseek-v4-pro"}     # thinking-mode DeepSeek models: sampling parameters are not sent
CLI_PREFIXES = ("codex:", "claude:")                    # text-protocol CLI providers (one stateless call per turn)
lock = threading.Lock()

class Budget:
    """Hard cap on real API calls (smoke tests)."""
    def __init__(self, n): self.n, self.used = n, 0
    def take(self):
        with lock:
            if self.n is not None and self.used >= self.n: raise RuntimeError("api_call_cap_reached")
            self.used += 1

class Fatal(Exception):
    pass

# ------------------------------------------------------------------ LLM back-ends -------------
def deepseek_backend(cap, passback="auto", extra_body=None):
    """passback: how reasoning_content of earlier assistant turns is sent back (thinking models, fc protocol).
    'auto'/'always': as returned (DeepSeek requires it within a tool-call chain and ignores it for earlier user turns);
    'never': dropped from every request (only valid when no tools are sent: the thinking-mode API returns 400 for
    requests with tools that omit earlier reasoning_content). See RUNS_round5.md."""
    from openai import OpenAI
    keys = open(KEYFILE, encoding="utf-8").read().split()      # line 1: DeepSeek, line 2: Moonshot (Kimi)
    clients = {False: OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY") or keys[0], base_url=BASE_URL, timeout=900, max_retries=0)}
    kl = int(os.environ.get("KIMI_KEY_LINE", "2"))             # which line of api.txt holds the Moonshot key to use
    if len(keys) >= kl: clients[True] = OpenAI(api_key=keys[kl - 1], base_url=KIMI_URL, timeout=900, max_retries=0)
    del keys
    pace = [0.0]

    def call(model, messages, tools, max_tokens, temperature):
        client = clients[model.startswith("kimi")]
        for attempt in range(4):
            if model.startswith("kimi"):          # account limit: 3 requests per minute -> at most one request every 21 s
                with lock:
                    w = pace[0] + 21 - time.time(); pace[0] = max(time.time(), pace[0] + 21)
                if w > 0: time.sleep(w)
            cap.take(); t0 = time.time()
            try:
                kw = dict(model=model, messages=messages, max_tokens=max_tokens)
                if tools: kw["tools"] = tools
                if temperature is not None and model not in THINKING: kw["temperature"] = temperature
                if extra_body: kw["extra_body"] = extra_body
                if passback == "never" and not tools: kw["messages"] = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
                d = client.chat.completions.create(**kw).model_dump()
                break
            except Exception as e:  # noqa: BLE001
                code = getattr(e, "status_code", None)
                if code in (401, 402, 403): raise Fatal(f"{type(e).__name__} {code}")
                if attempt == 3 or code == 400: raise
                time.sleep(5 * 2 ** attempt)
        ch = d["choices"][0]; m = ch["message"]
        return dict(model=d.get("model"), content=m.get("content") or "", reasoning=m.get("reasoning_content"),
                    tool_calls=[dict(id=tc["id"], name=tc["function"]["name"], arguments=tc["function"]["arguments"])
                                for tc in (m.get("tool_calls") or [])],
                    finish_reason=ch.get("finish_reason"), usage=d.get("usage"), latency=time.time() - t0)
    return call

def codex_or(other, cap, max_waits=12):
    """Route models named 'codex:<model>@<effort>' to the Codex CLI and 'claude:<model>@<effort>' to the Claude Code
    CLI (text protocol only; same transcript rendering for both); others to `other`."""
    sys.path.insert(0, os.path.dirname(HERE))
    import codex_backend, claude_backend

    def call(model, messages, tools, max_tokens, temperature):
        if not model.startswith(CLI_PREFIXES): return other(model, messages, tools, max_tokens, temperature)
        be = codex_backend if model.startswith("codex:") else claude_backend
        if tools: raise Fatal(f"{model.split(':')[0]} models support only --protocol text")
        for attempt in range(3):
            cap.take()
            try:
                r = be.call(model, messages, max_waits=max_waits) if be is claude_backend else be.call(model, messages); break
            except Exception as e:  # noqa: BLE001
                if attempt == 2 or str(e).startswith("claude usage/rate limit persisted"): raise
                time.sleep(10 * (attempt + 1))
        out = dict(model=r["model"], content=r["content"], reasoning=None, tool_calls=[],
                   finish_reason=r["finish_reason"], usage=r["usage"], latency=r["latency"], codex_tool_items=r["tool_items"])
        if r.get("meta"): out["claude_meta"] = r["meta"]
        return out
    return call

def mock_backend(kind, seed=0):
    """Offline stand-in. 'random': random tools / verdicts with occasional invalid actions (exercises every
    branch of the loop). 'naive': calls the first tool until 3 reports, then submits its majority report."""
    def call(model, messages, tools, max_tokens, temperature):
        h = int(hashlib.md5(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng([seed, h])
        sys_ = messages[0]["content"]
        if kind in ("docreader", "nolink") and messages[-1]["role"] == "user" and \
                (ELICIT_REQUEST in messages[-1]["content"] or ELICIT_NUDGE == messages[-1]["content"]):
            # K2 decomposition mocks: 'docreader' writes the observation model a perfect reader of the docs would write
            # (tools grouped by identical backend name); 'nolink' reads every tool's doc separately (no sharing).
            sc = next(k for k, v in SCENARIOS.items() if v["tools"][0]["name"] in sys_)
            text = "Mock reasoning.\n" + json.dumps(K2D.doc_reader_model(dict(scenario=sc), sys_, link=(kind == "docreader")))
            n_in = sum(len(str(m.get("content"))) for m in messages) // 4
            return dict(model=f"mock-{kind}", content=text, reasoning=None, tool_calls=[], finish_reason="stop",
                        usage=dict(prompt_tokens=n_in, completion_tokens=200), latency=0.0)
        names = [t["function"]["name"] for t in tools] if tools else re.findall(r"^- (\w+)\(", sys_, re.M) + ["submit"]
        verdicts = re.findall(r"verdict in \{([^}]*)\}", sys_) or re.findall(r"one of: ([^\"]*)\"", sys_)
        verdicts = [v.strip() for v in verdicts[0].split(",")]
        obs = [m["content"] for m in messages if m["role"] in ("tool", "user") and '"tool":' in str(m.get("content"))]
        if kind in ("naive", "docreader", "nolink"):
            if len(obs) < 3: act = ("call", names[0])
            else:
                vals = [json.loads(o.split("\n")[0].replace("Observation: ", ""))["result"] for o in obs]
                first = [list(v.values())[1] for v in vals]
                idx = max(set(first), key=first.count); vals0 = re.findall(r"in \{([^}]*)\}", sys_)
                act = ("submit", verdicts[0])
                for line in vals0:
                    opts = [x.strip(" '") for x in line.split(",")]
                    if idx in opts: act = ("submit", verdicts[opts.index(idx)]); break
        else:
            u = rng.random()
            if u < 0.05: act = ("bad", None)
            elif u < 0.08: act = ("call", "made_up_tool")
            elif len(obs) >= rng.integers(1, 6) or u > 0.85: act = ("submit", str(rng.choice(verdicts)))
            else: act = ("call", str(rng.choice(names[:-1])))
        text = "Mock reasoning."; tcs = []
        if tools:
            if act[0] == "call": tcs = [dict(id=f"call_{h}", name=act[1], arguments=json.dumps({"service": "x"}))]
            elif act[0] == "submit": tcs = [dict(id=f"call_{h}", name="submit", arguments=json.dumps({"verdict": act[1]}))]
        else:
            if act[0] == "call": text += '\n{"action": "call", "tool": "%s"}' % act[1]
            elif act[0] == "submit": text += '\n{"action": "submit", "verdict": "%s"}' % act[1]
        n_in = sum(len(str(m.get("content"))) for m in messages) // 4 + (len(json.dumps(tools)) // 4 if tools else 0)
        return dict(model=f"mock-{kind}", content=text, reasoning=None, tool_calls=tcs, finish_reason="stop",
                    usage=dict(prompt_tokens=n_in, completion_tokens=20), latency=0.0)
    return call

# ------------------------------------------------------------------ episode loop ---------------
def parse_text_action(text):
    m = None
    for m in re.finditer(r"\{[^{}]*\"action\"[^{}]*\}", text or ""): pass
    if not m: return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if d.get("action") == "submit": return ("submit", d.get("verdict"))
    if d.get("action") == "call": return ("call", d.get("tool"))
    if isinstance(d.get("action"), str) and d["action"] not in ("call", "submit"): return ("call", d["action"])
    return None

def run_episode(ep, cond, model, protocol, llm, max_turns, max_tokens, temperature):
    env = Env(ep, cond, protocol)
    messages = [{"role": "system", "content": system_prompt(ep, cond, protocol)}, {"role": "user", "content": first_user_cond(ep, cond)}]
    tools = tool_schemas(ep) if protocol == "fc" else None
    api, status = [], "turn_cap"
    elicit, n_turns = None, max_turns
    if cond in ELICIT:
        # Observation-model elicitation (K1e, K2e, K2p): the first reply must be the observation_model JSON (no tool schemas
        # are offered for these turns; at most 2 attempts). K2p then hands every action to the exact planner of that model.
        docs = K2D.parse_docs(ep, messages[0]["content"]); spec, obj, raw, attempts = None, None, "", 0
        for attempt in range(2):
            try:
                r = llm(model, messages, None, max_tokens, temperature)
            except Fatal:
                raise
            except Exception as e:  # noqa: BLE001
                status = f"api_error: {type(e).__name__}: {str(e)[:200]}"; break
            api.append({k: r.get(k) for k in ("model", "finish_reason", "usage", "latency", "codex_tool_items", "claude_meta") if k != "claude_meta" or r.get(k)})
            attempts += 1; raw = r["content"] or ""
            em = {"role": "assistant", "content": raw}
            if r.get("reasoning") is not None: em["reasoning_content"] = r["reasoning"]   # DeepSeek: required later once tools are sent
            messages.append(em)
            obj = K2D.extract_json(raw)
            if obj is not None and isinstance(obj.get("observation_model"), dict):
                spec = K2D.normalize_model(ep, obj, docs)
                if not spec["fallback"]: break
            if attempt == 0: messages.append({"role": "user", "content": ELICIT_NUDGE})
        if status.startswith("api_error"):
            n_turns = 0
        else:
            parse_ok = spec is not None and not spec["fallback"]
            if not parse_ok: spec = K2D.default_spec(ep, docs, "elicitation failed after %d attempt(s)" % attempts)
            elicit = dict(parse_ok=parse_ok, attempts=attempts, raw=raw[-6000:], obj=obj, spec=spec)
            if cond == "K2p":
                K2D.plan_and_play(ep, spec, env); n_turns = 0
                status = "submitted" if env.done else "turn_cap"
                elicit["planner"] = "exact DP on the recovered model (oracle machinery); LLM called only for the model JSON"
            else:
                messages.append({"role": "user", "content": ELICIT_ACK if parse_ok else "No observation model recorded. Now take your first action."})
    for _ in range(n_turns):
        try:
            r = llm(model, messages, tools, max_tokens, temperature)
        except Fatal:
            raise
        except Exception as e:  # noqa: BLE001
            status = f"api_error: {type(e).__name__}: {str(e)[:200]}"; break
        api.append({k: r.get(k) for k in ("model", "finish_reason", "usage", "latency", "codex_tool_items", "claude_meta") if k != "claude_meta" or r.get(k)})
        if protocol == "fc":
            am = {"role": "assistant", "content": r["content"]}
            if r["reasoning"] is not None: am["reasoning_content"] = r["reasoning"]
            if r["tool_calls"]:
                am["tool_calls"] = [{"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": t["arguments"]}}
                                    for t in r["tool_calls"]]
            messages.append(am)
            if not r["tool_calls"]:
                env.invalid("no_tool_call" if r["finish_reason"] != "length" else "truncated", (r["content"] or "")[:200])
                messages.append({"role": "user", "content": NUDGE_FC}); continue
            for t in r["tool_calls"]:
                if env.done:
                    out = json.dumps({"error": "episode_finished"})
                elif t["name"] == "submit":
                    try: v = json.loads(t["arguments"] or "{}").get("verdict")
                    except json.JSONDecodeError: v = None
                    out = env.submit(v)
                else:
                    out = env.call(t["name"])
                messages.append({"role": "tool", "tool_call_id": t["id"], "content": out})
        else:
            messages.append({"role": "assistant", "content": r["content"]})
            a = parse_text_action(r["content"])
            if a is None:
                env.invalid("unparseable" if r["finish_reason"] != "length" else "truncated", (r["content"] or "")[-200:])
                messages.append({"role": "user", "content": NUDGE_TEXT}); continue
            out = env.submit(a[1]) if a[0] == "submit" else env.call(a[1])
            if not env.done: messages.append({"role": "user", "content": env.observation_message(out)})
        if env.done:
            status = "submitted"; break
    u = lambda k: sum((c["usage"] or {}).get(k) or 0 for c in api)
    reason = sum(((c["usage"] or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for c in api)
    extra = dict(elicit=elicit) if elicit is not None else {}
    metas = [c["claude_meta"] for c in api if c.get("claude_meta")]
    if metas:                                  # Claude CLI provider: per-episode totals (per call in api[].claude_meta)
        extra["claude"] = dict(cost_usd=sum(m.get("cost_usd") or 0 for m in metas),
                               thinking_tokens=sum(m.get("thinking_tokens") or 0 for m in metas),
                               duration_api_ms=sum(m.get("duration_api_ms") or 0 for m in metas), effort=metas[0]["effort"])
    return dict(episode=ep["id"], family=ep["family"], scenario=ep["scenario"], condition=cond, model=model,
                protocol=protocol, status=status, verdict=env.verdict, log=env.log, n_api=len(api), **extra,
                models_returned=sorted(set(str(c["model"]) for c in api)),
                finish_reasons=[c["finish_reason"] for c in api],
                tokens=dict(prompt=u("prompt_tokens"), completion=u("completion_tokens"), reasoning=reason,
                            cache_hit=u("prompt_cache_hit_tokens"), cache_miss=u("prompt_cache_miss_tokens")),
                latency=sum(c["latency"] for c in api), max_tokens=max_tokens, max_turns=max_turns,
                api=api, transcript=messages, time=time.strftime("%Y-%m-%d %H:%M:%S"))

def done_keys(path):
    s = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r["status"] in ("submitted", "turn_cap"):
                    s.add((r["episode"], r["condition"], r["model"], r["protocol"], r.get("sample", 0)))
            except (json.JSONDecodeError, KeyError):
                pass
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["deepseek-chat"])
    ap.add_argument("--families", nargs="+", default=CORE)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=ALL_CONDITIONS,
                    help="K1 K2 K3 K4 K4c (default) plus the K2 decomposition conditions K2a K1e K2e K2p")
    ap.add_argument("--episodes", type=int, default=50, help="first N episodes of each family")
    ap.add_argument("--ids", nargs="*", help="explicit episode ids (overrides --families/--episodes)")
    ap.add_argument("--protocol", choices=["fc", "text"], default="fc")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--sample-start", type=int, default=0, help="first sample index (e.g. 1 with --samples 2 = only the second sample)")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=None, help="default: 1500 chat, 32768 thinking models")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reasoning-passback", choices=["auto", "never"], default="auto",
                    help="thinking DeepSeek models, fc protocol: send reasoning_content back in later requests "
                         "(auto, required by the API inside a tool-call chain) or drop it (never; ignored when tools are sent, since the API then returns 400)")
    ap.add_argument("--extra-body", default=None, help="JSON merged into DeepSeek requests (e.g. thinking switch)")
    ap.add_argument("--max-waits", type=int, default=12, help="claude models: max usage/rate-limit waits per call")
    ap.add_argument("--max-api-calls", type=int, default=None, help="hard cap on real API calls")
    ap.add_argument("--mock", choices=["random", "naive", "docreader", "nolink"], default=None,
                    help="docreader / nolink: naive actor whose observation-model JSON (K1e/K2e/K2p) is read from the docs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    eps = load_episodes()
    ids = a.ids or [i for f in a.families for i in sorted(e for e in eps if eps[e]["family"] == f)[:a.episodes]]
    models = [f"mock-{a.mock}"] if a.mock else a.models
    out = a.out or os.path.join(HERE, "logs", f"{'mock' if a.mock else 'api'}_{a.protocol}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cap = Budget(a.max_api_calls)
    if a.mock: llm = mock_backend(a.mock)
    elif all(m.startswith(CLI_PREFIXES) for m in models): llm = codex_or(None, cap, a.max_waits)
    else: llm = codex_or(deepseek_backend(cap, a.reasoning_passback, json.loads(a.extra_body) if a.extra_body else None),
                         cap, a.max_waits)
    done = done_keys(out)
    jobs = [(i, c, m, s) for m in models for c in a.conditions for i in ids for s in range(a.sample_start, a.samples)
            if (i, c, m, a.protocol, s) not in done]
    print(f"{len(jobs)} episode runs to do ({len(done)} already in {out})")

    def work(job):
        i, c, m, s = job
        mt = a.max_tokens or (1500 if m in ("deepseek-chat",) or m.startswith("mock") else 32768)
        r = run_episode(eps[i], c, m, a.protocol, llm, a.max_turns, mt, a.temperature)
        r["sample"] = s
        with lock:
            with open(out, "a", encoding="utf-8") as f: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r

    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(work, jobs):
            print(f"{r['episode']} {r['condition']} {r['model']}: {r['status']} verdict={r['verdict']} "
                  f"calls={sum(e['type'] == 'call' for e in r['log'])} api={r['n_api']} tok={r['tokens']} "
                  f"lat={r['latency']:.1f}s")

if __name__ == "__main__":
    main()
