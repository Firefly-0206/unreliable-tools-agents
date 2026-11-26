"""
Claude Code CLI back-end (`claude -p`) for bench/run.py and agentenv/run.py.

Each call is stateless: the chat history is rendered into one transcript prompt (the SAME renderer and preamble as
codex_backend, so the text the model sees is identical to what Codex/terra sees) and sent on stdin to a fresh
Claude Code process in an empty temp directory with every tool, MCP server, settings source and session
persistence switched off. A short neutral system prompt (SYSTEM below, identical in every condition) replaces
Claude Code's default agent system prompt. NOTE: this is not a bare API call: Claude Code still prepends its SDK identity
line ("You are a Claude agent, built on Anthropic's Claude Agent SDK."), an environment block (temp working directory,
platform, shell, OS, model name, date) and a system-reminder with account metadata (~1k extra input tokens per call). Tools,
MCP, settings, hooks, skills and plugins are off. These additions are identical in every condition.

Model spec: "claude:<model>@<effort>", e.g. "claude:claude-sonnet-5-5@high".

Per call we record: model, effort, input/cache/output tokens, thinking tokens, total_cost_usd (notional when the
call is billed to a subscription plan), duration_api_ms, stop_reason. Usage-limit / rate-limit / overload errors
are waited out (sleep to the stated reset time + 5 min, else exponential backoff capped at 30 min) and never
crash the run; at most `max_waits` waits per call.
"""
import datetime, json, os, re, shutil, subprocess, tempfile, time

import codex_backend

CLAUDE = os.environ.get("CLAUDE_EXE") or shutil.which("claude") or r"C:\Users\anon\.local\bin\claude.exe"
SYSTEM = "Answer as instructed in the user message."
render = codex_backend.render            # identical prompt rendering (PREAMBLE + transcript) as the Codex provider
LIMIT_RE = re.compile(r"hit your (?:\w+ )?limit|usage limit|rate.?limit|overloaded|\b429\b|\b529\b|too many requests|\bresets?\b",
                      re.I)

class LimitError(RuntimeError):
    pass

def parse_spec(spec):
    name = spec.split(":", 1)[1]
    model, _, effort = name.partition("@")
    return model, effort or "medium"

def _reset_seconds(msg):
    """Seconds to sleep until a reset time stated in the message (+5 min), or None. Understands '|<epoch>' and
    'resets (time)' / 'resets (time)' / 'try again (time)' (local clock)."""
    m = re.search(r"\|\s*(\d{10})\b", msg)
    if m: return max(0.0, int(m.group(1)) - time.time()) + 300
    m = re.search(r"(?:resets?|try again)(?: at)?\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)\b", msg, re.I)
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
        now = datetime.datetime.now()
        t = now.replace(hour=h, minute=int(m.group(2) or 0), second=0, microsecond=0)
        if t <= now: t += datetime.timedelta(days=1)
        return (t - now).total_seconds() + 300
    return None

def wait_for_reset(msg, k):
    secs = _reset_seconds(msg)
    if secs is None: secs = min(60 * 2 ** k, 1800)
    secs = min(secs, 24 * 3600)
    print(f"claude limit/rate error ({msg[:80]!r}): sleeping {secs / 60:.0f} min", flush=True)
    time.sleep(secs)

def call(spec, messages, timeout=900, max_waits=12):
    last = None
    for k in range(max_waits + 1):
        try:
            return _call(spec, messages, timeout)
        except LimitError as e:
            last = e
            if k == max_waits: break
            wait_for_reset(str(e), k)
    raise RuntimeError(f"claude usage/rate limit persisted after {max_waits} waits: {last}")

def parse_output(stdout):
    """-> the 'result' event dict from `--output-format json` (a single object, or a list of events)."""
    s = stdout.strip()
    if not s: return None
    try: d = json.loads(s)
    except json.JSONDecodeError:
        d = None
        for line in s.splitlines():                      # tolerate JSONL
            try: x = json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(x, dict) and x.get("type") == "result": d = x
        return d
    if isinstance(d, list):
        res = [x for x in d if isinstance(x, dict) and x.get("type") == "result"]
        return res[-1] if res else None
    return d

def to_usage(d):
    u = d.get("usage") or {}
    inp, cr, cc = u.get("input_tokens", 0) or 0, u.get("cache_read_input_tokens", 0) or 0, u.get("cache_creation_input_tokens", 0) or 0
    think = (u.get("output_tokens_details") or {}).get("thinking_tokens", 0) or 0
    return dict(prompt_tokens=inp + cr + cc, completion_tokens=u.get("output_tokens", 0) or 0,
                completion_tokens_details=dict(reasoning_tokens=think), prompt_cache_hit_tokens=cr,
                input_tokens_uncached=inp, cache_creation_input_tokens=cc)

def _call(spec, messages, timeout=900):
    model, effort = parse_spec(spec)
    work = tempfile.mkdtemp(prefix="claude_")
    cmd = [CLAUDE, "-p", "--model", model, "--effort", effort, "--system-prompt", SYSTEM, "--tools", "",
           "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=render(messages).encode("utf-8"), capture_output=True, timeout=timeout, cwd=work)
    finally:
        try: os.rmdir(work)
        except OSError: pass
    lat = time.time() - t0
    out, err = p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
    d = parse_output(out)
    if d is None:
        msg = (err or out)[-400:]
        if LIMIT_RE.search(msg): raise LimitError(msg)
        raise RuntimeError(f"claude exit {p.returncode}: {msg}")
    text = d.get("result") if isinstance(d.get("result"), str) else ""
    if d.get("is_error") or d.get("subtype", "success") != "success" and not text:
        msg = (text or str(d.get("subtype")) or err)[:400]
        status = d.get("api_error_status")
        if LIMIT_RE.search(msg) or status in (429, 529, 503): raise LimitError(msg)
        raise RuntimeError(f"claude error ({d.get('subtype')}, exit {p.returncode}): {msg}")
    stop = d.get("stop_reason")
    return dict(model=f"{model}@{effort}", content=text, finish_reason="length" if stop == "max_tokens" else "stop",
                usage=to_usage(d), latency=lat, tool_items=[], errors=[],
                meta=dict(model=model, effort=effort, cost_usd=d.get("total_cost_usd"),
                          duration_api_ms=d.get("duration_api_ms"), stop_reason=stop, is_error=bool(d.get("is_error")),
                          thinking_tokens=to_usage(d)["completion_tokens_details"]["reasoning_tokens"],
                          num_turns=d.get("num_turns"), model_usage=sorted((d.get("modelUsage") or {}).keys())))
