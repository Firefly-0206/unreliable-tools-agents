"""
Codex CLI back-end (`codex exec`) for bench/run.py and agentenv/run.py.

Each call is stateless: the chat history is rendered into one transcript prompt and sent to a fresh,
ephemeral Codex session that runs in an empty directory with a read-only sandbox and is told not to use tools.
The returned events are checked for tool use; any command execution is recorded (`tool_items`) so it can be
excluded at scoring time. Codex adds its own agent system prompt, which the paper discloses.

Model spec: "codex:<model>@<effort>", e.g. "codex:gpt-6.1-terra@low".
"""
import datetime, glob, json, os, re, subprocess, tempfile, time

def _find_codex():
    """CODEX_EXE if set, else the most recently installed codex.exe (Codex updates itself into a new bin/<hash> folder)."""
    if os.environ.get("CODEX_EXE"): return os.environ["CODEX_EXE"]
    found = glob.glob(os.path.join(os.environ.get("LOCALAPPDATA", ""), "OpenAI", "Codex", "bin", "*", "codex.exe"))
    return max(found, key=os.path.getmtime) if found else "codex"

CODEX = _find_codex()
PREAMBLE = ("Answer directly from your own reasoning. Do not run any commands, do not read or write files and do "
            "not use any tools. Reply only as the assistant to the last message of the conversation below.\n\n")

def parse_spec(spec):
    name = spec.split(":", 1)[1]
    model, _, effort = name.partition("@")
    return model, effort or "medium"

def render(messages):
    if len(messages) == 1 and messages[0]["role"] == "user":
        return PREAMBLE + messages[0]["content"]
    parts = []
    for m in messages:
        parts.append(f"[{m['role'].upper()}]\n{m.get('content') or ''}")
    return PREAMBLE + "\n\n".join(parts) + "\n\n[ASSISTANT]\n"

def wait_for_reset(msg):
    """On the plan's usage-limit error, sleep until the reset time it names (+5 min), else 30 min."""
    m = re.search(r"try again at (\d{1,2}):(\d{2})\s*([AP]M)", msg)
    now = datetime.datetime.now()
    if m:
        h, mi = int(m.group(1)) % 12 + (12 if m.group(3) == "PM" else 0), int(m.group(2))
        t = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if t <= now: t += datetime.timedelta(days=1)
        secs = (t - now).total_seconds() + 300
    else:
        secs = 1800
    print(f"codex usage limit: sleeping {secs / 60:.0f} min", flush=True)
    time.sleep(secs)

def call(spec, messages, timeout=900, max_waits=12):
    for _ in range(max_waits):
        try:
            return _call(spec, messages, timeout)
        except RuntimeError as e:
            if "usage limit" not in str(e): raise
            wait_for_reset(str(e))
    raise RuntimeError("codex usage limit persisted")

def _call(spec, messages, timeout=900):
    model, effort = parse_spec(spec)
    work = tempfile.mkdtemp(prefix="codex_")
    out = os.path.join(work, "last.txt")
    cmd = [CODEX, "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"', "-s", "read-only",
           "--skip-git-repo-check", "--ephemeral", "--ignore-rules", "-C", work, "--json", "-o", out, "-"]
    t0 = time.time()
    p = subprocess.run(cmd, input=render(messages).encode("utf-8"), capture_output=True, timeout=timeout)
    lat = time.time() - t0
    usage, tool_items, errors, text = None, [], [], ""
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        try: ev = json.loads(line)
        except json.JSONDecodeError: continue
        if ev.get("type") == "turn.completed": usage = ev.get("usage")
        elif ev.get("type") == "item.completed":
            it = ev.get("item") or {}
            if it.get("type") == "agent_message": text = it.get("text") or text
            elif it.get("type") not in ("reasoning",): tool_items.append(it.get("type"))
        elif ev.get("type") in ("error", "turn.failed"): errors.append(str(ev)[:300])
    if os.path.exists(out):
        text = open(out, encoding="utf-8", errors="replace").read() or text
    try:
        for f in os.listdir(work): os.remove(os.path.join(work, f))
        os.rmdir(work)
    except OSError:
        pass
    if p.returncode != 0 and not text:
        raise RuntimeError(f"codex exit {p.returncode}: {(errors or [p.stderr.decode('utf-8', 'replace')[-300:]])[0]}")
    u = usage or {}
    usage_oai = dict(prompt_tokens=u.get("input_tokens", 0), completion_tokens=u.get("output_tokens", 0),
                     completion_tokens_details=dict(reasoning_tokens=u.get("reasoning_output_tokens", 0)),
                     prompt_cache_hit_tokens=u.get("cached_input_tokens", 0))
    return dict(model=f"{model}@{effort}", content=text, finish_reason="stop", usage=usage_oai, latency=lat,
                tool_items=tool_items, errors=errors)
