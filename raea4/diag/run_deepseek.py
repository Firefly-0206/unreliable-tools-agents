"""
Run the paired diagnostic against the DeepSeek API from your own computer.

Requirements: Python 3.8+, no extra packages.
Usage (Windows PowerShell):
    $env:DEEPSEEK_API_KEY = "sk-..."
    python run_deepseek.py
Usage (macOS / Linux):
    export DEEPSEEK_API_KEY=sk-...
    python3 run_deepseek.py

It reads diag_items.json from the same folder and appends one JSON line per call to results.jsonl.
It is resumable: rerun the same command after an interruption and finished calls are skipped.
The API key is read only from the environment and is never written to any file.
"""
import json, os, re, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = os.path.join(HERE, "diag_items.json")
OUT = os.path.join(HERE, "results.jsonl")
URL = "https://api.deepseek.com/chat/completions"
# model -> (samples per item, temperature, max output tokens incl. hidden reasoning)
MODELS = {"deepseek-chat": (3, 0.7, 8192), "deepseek-reasoner": (2, None, 131072)}
WORKERS = 8

key = os.environ.get("DEEPSEEK_API_KEY")
if not key:
    sys.exit("Set the DEEPSEEK_API_KEY environment variable first.")

lock = threading.Lock()

def done_keys():
    s = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("ok") and r.get("p_healthy") is not None and r.get("action"):
                    s.add((r["id"], r["model"], r["sample"]))   # truncated / unparsed answers are retried
            except Exception:
                pass
    return s

def call(model, prompt, temp, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    if temp is not None: body["temperature"] = temp
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())

ACTIONS = ["declare_healthy", "declare_degraded", "declare_down", "escalate", "query_A", "query_B", "query_C", "query_V"]

def parse(text):
    m = None
    for m in re.finditer(r"\{[^{}]*\"p_healthy\"[^{}]*\}", text): pass
    if m:
        try:
            d = json.loads(m.group(0)); return float(d.get("p_healthy")), str(d.get("action"))
        except Exception:
            pass
    p = re.findall(r"p_healthy\"?\s*[:=]\s*([0-9.]+)", text)
    a = re.findall(r"\"action\"\s*:\s*\"([a-z_A-Z]+)\"", text)
    if p and a and a[-1] in ACTIONS:
        try: return float(p[-1]), a[-1]
        except ValueError: pass
    return None, None

def job(item, model, sample, temp, max_tokens):
    for attempt in range(5):
        try:
            t = time.time(); resp = call(model, item["prompt"], temp, max_tokens)
            msg = resp["choices"][0]["message"]; content = msg.get("content") or ""
            p, a = parse(content)
            fin = resp["choices"][0].get("finish_reason")
            rec = dict(id=item["id"], model=model, sample=sample, ok=True, p_healthy=p, action=a, finish_reason=fin,
                       content=content, reasoning_chars=len(msg.get("reasoning_content") or ""),
                       usage=resp.get("usage"), seconds=round(time.time() - t, 1))
            break
        except urllib.error.HTTPError as e:
            body = e.read()[:300]
            rec = dict(id=item["id"], model=model, sample=sample, ok=False, error=f"HTTP {e.code}: {body!r}")
            if e.code == 402:
                print("Insufficient balance: stopping this call (rerun later; finished calls are kept).", flush=True); break
            if e.code in (401, 403): break
            if e.code == 400 and b"max_tokens" in body and max_tokens > 65536:
                max_tokens = 65536; continue          # API rejects the larger cap: fall back to 65,536
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            rec = dict(id=item["id"], model=model, sample=sample, ok=False, error=repr(e)[:300])
            time.sleep(5 * (attempt + 1))
    with lock:
        with open(OUT, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(("ok  " if rec["ok"] else "ERR ") + f"{model:18s} {item['id']:32s} s{sample} "
              + (f"p={rec.get('p_healthy')} a={rec.get('action')}" if rec["ok"] else rec["error"][:120]), flush=True)

def main():
    items = json.load(open(ITEMS, encoding="utf-8"))
    have = done_keys()
    only = sys.argv[1:]          # optional: restrict to model names given on the command line
    jobs = [(it, m, s, temp, mt) for m, (n, temp, mt) in MODELS.items() if not only or m in only
            for it in items for s in range(n) if (it["id"], m, s) not in have]
    print(f"{len(jobs)} calls to make ({len(have)} already done)", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        for j in jobs: ex.submit(job, *j)
    print("finished; upload results.jsonl", flush=True)

if __name__ == "__main__":
    main()
