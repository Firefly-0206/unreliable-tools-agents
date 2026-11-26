"""Held-out confirmation (analysis plan fixed before the runs) -> tables/confirm.tex, tables/nums_confirm.tex

Reads  raea4/bench/results/confirm_analysis.json   (analyze_confirm.py; checked against confirm_analysis.md)
       score_confirm_flash.json, score_confirm_terra.json          (standard scorer; cost, tokens)
       confirm_flash.jsonl, confirm_terra.jsonl                    (run records: usage, latency, tool calls; first record per prompt)
    python paper/scripts/make_confirm_tables.py
Every number is asserted against the markdown report before anything is written.
"""
import hashlib
import json
import math
import os
import re
import statistics

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "raea4", "bench", "results")
BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "raea4", "bench")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables")
A = json.load(open(os.path.join(RES, "confirm_analysis.json"), encoding="utf-8"))
MD = open(os.path.join(RES, "confirm_analysis.md"), encoding="utf-8").read()
MODELS = [("Flash", "flash", "flash (non-thinking)"), ("Terra", "terra", "gpt-6.1-terra")]
CONDS = [("PZero", "P0"), ("LOne", "L1"), ("LTwo", "L2"), ("HOne", "H1")]
HYP = [("Ha", "primary"), ("Hb", "primary"), ("SOne", "secondary", "S1"), ("STwo", "secondary", "S2"), ("SThree", "secondary", "S3")]
N_EXPECT = 134


def minus(s):
    return s.replace("-", "$-$")


def f(x, d=3):
    return minus(f"{x:.{d}f}")


def ci(v, d=3):
    return "[" + f(v[0], d) + ", " + f(v[1], d) + "]"


def plain(x, d=3):          # for md comparison (no TeX)
    return f"{x:.{d}f}"


def plain_ci(v, d=3):
    return f"[{v[0]:.{d}f}, {v[1]:.{d}f}]"


# ---------------------------------------------------------------- assert json == md
def md_section(name):
    m = re.search(r"^## " + name + r"\n(.*?)(?=^## |\Z)", MD, re.S | re.M)
    assert m, name
    return m.group(1)


for _, key, _ in MODELS:
    sec, d = md_section(key), A[key]
    for c in [c for _, c in CONDS]:
        r = d["conditions"][c]
        assert r["n_answers"] == N_EXPECT and r["parsed"] == 1.0 and r["statuses"] == {"ok": N_EXPECT}, (key, c)
        row = re.search(r"^\| " + c + r" \| (\d+) \| ([\d.]+) \| ([\d.]+) \[([\d.]+), ([\d.]+)\] \| \[([\d.]+), ([\d.]+)\] \| (\d+) \|", sec, re.M)
        assert row, (key, c)
        assert int(row[1]) == N_EXPECT and float(row[2]) == 1.0
        assert row[3] == plain(r["regret"]["mean"]) and row[4] == plain(r["regret"]["task_ci"][0]) and row[5] == plain(r["regret"]["task_ci"][1]), (key, c, "task")
        assert row[6] == plain(r["regret"]["cluster_ci"][0]) and row[7] == plain(r["regret"]["cluster_ci"][1]), (key, c, "cluster")
        assert int(row[8]) == round(100 * r["optimal"]["mean"]), (key, c, "optimal")
    for h, kind in (("Ha", "primary"), ("Hb", "primary"), ("S1", "secondary"), ("S2", "secondary"), ("S3", "secondary")):
        r = d[kind][h]
        assert r["n_tasks"] == N_EXPECT
        row = re.search(r"^\| " + h + r" \| (\S+) \| (-?[\d.]+) \| (\[.*?\]) \| (\[.*?\]) \| (\[.*?\]) \| (.*?) \|$", sec, re.M)
        assert row, (key, h)
        assert row[1] == r["contrast"] and row[2] == plain(r["mean"]), (key, h)
        assert row[3] == plain_ci(r["task_ci"]) and row[4] == plain_ci(r["cluster_ci"]) and row[5] == plain_ci(r["conservative"]), (key, h)
        lo = min(r["task_ci"][0], r["cluster_ci"][0])
        hi = max(r["task_ci"][1], r["cluster_ci"][1])
        assert (lo, hi) == tuple(r["conservative"]), (key, h, "conservative = union")
        if kind == "primary":
            assert (row[6] == "met") == r["criterion_met"] and (hi < 0) == r["criterion_met"], (key, h, "criterion")
        else:
            assert row[6] == "secondary"
    hc = d["Hc"]
    m = re.search(r"task-mean share ([\d.]+)% \(task CI \[([\d.]+), ([\d.]+)\], cluster CI \[([\d.]+), ([\d.]+)\]\); (\d+) of (\d+) parsed answers; criterion (met|not met)", sec)
    assert m, (key, "Hc")
    pc = lambda x: f"{100 * x:.1f}"
    assert m[1] == pc(hc["share_task_mean"]) and (m[2], m[3]) == (pc(hc["task_ci"][0]), pc(hc["task_ci"][1])), (key, "Hc task")
    assert (m[4], m[5]) == (pc(hc["cluster_ci"][0]), pc(hc["cluster_ci"][1])), (key, "Hc cluster")
    assert int(m[6]) == hc["n_answers_with_marginal_accurate_and_action_nonoptimal"] and int(m[7]) == hc["n_parsed"] == N_EXPECT
    assert (m[8] == "met") == hc["criterion_met"] and hc["criterion_met"] == (hc["conservative"][0] > 0)
    au = d["H1_audit"]
    assert au["n"] == au["ok"] == N_EXPECT and au["incomplete_model"] == 0 and au["other_failures"] == {} and au["answers_with_any_default_or_clip"] == 0
    assert '"incomplete_model": 0' in sec and '"answers_with_any_default_or_clip": 0' in sec

# ---------------------------------------------------------------- run records and cost
REC = {}
for _, key, _ in MODELS:
    first = {}
    for line in open(os.path.join(RES, f"confirm_{key}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        first.setdefault(r["pid"], r)
    assert len(first) == 4 * N_EXPECT
    u = [r["usage"] or {} for r in first.values()]
    pt = sum(x.get("prompt_tokens", 0) for x in u)
    hit = sum(x.get("prompt_cache_hit_tokens", 0) for x in u)
    ct = sum(x.get("completion_tokens", 0) for x in u)
    S = json.load(open(os.path.join(RES, f"score_confirm_{key}.json"), encoding="utf-8"))
    cost = list(S["cost"].values())[0]
    assert cost["answers"] == len(first) and cost["input"] == pt and cost["output"] == ct
    REC[key] = dict(
        n=len(first), pt=pt, hit=hit, ct=ct, usd=cost["usd"], reasoning=cost["reasoning"],
        cny=((pt - hit) * 1.0 + hit * 0.02 + ct * 4.0) / 1e6,        # DeepSeek off-peak CNY per M tokens: 1 miss, 0.02 hit, 4 out
        lat=statistics.median(r["latency"] for r in first.values()),
        tools=sum(1 for r in first.values() if r.get("n_tool_calls")),
        ok=sum(1 for r in first.values() if r.get("ok")),
        sha=hashlib.sha256(open(os.path.join(RES, f"confirm_{key}.jsonl"), "rb").read()).hexdigest(),
    )
    assert REC[key]["n"] == 536
assert REC["terra"]["tools"] == 0 and REC["flash"]["tools"] == 0
# frozen-file hashes of the analysis plan (section 6)
PRE = {"items_confirm.jsonl": "557C9DCEBB8EBD0B", "prompts_confirm.jsonl": "FE91268B0374336B", "core.py": "9359186B00C28E25",
       "run.py": "3686FB768A703CF1", "score.py": "D07915706F69DA79", "analyze_confirm.py": "638FE8C9AFFF8959"}
prereg = open(os.path.join(BENCH, "PLAN_confirm.md"), encoding="utf-8").read()
for fn, h in PRE.items():
    assert h in prereg, fn
    p = os.path.join(BENCH, fn)
    if fn == "run.py":
        # run.py later gained the `--provider claude` route (Sonnet 5.5 runs, 2026-11-04), so its hash no longer matches.
        # The planned version was recovered byte-identical (bench/frozen_confirm/run_planned.py, see README there):
        # assert its hash, and that the stored diff turns it into the current run.py exactly.
        import difflib
        fz = os.path.join(BENCH, "frozen_confirm", "run_planned.py")
        assert hashlib.sha256(open(fz, "rb").read()).hexdigest().upper().startswith(h), "recovered planned run.py hash"
        old = open(fz, encoding="utf-8").read().splitlines(True); cur = open(p, encoding="utf-8").read().splitlines(True)
        d = "".join(difflib.unified_diff(old, cur, "run_planned.py", "run.py"))
        stored = open(os.path.join(BENCH, "frozen_confirm", "run_planned_to_current.diff"), encoding="utf-8").read()
        strip = lambda t: [l for l in t.splitlines() if l[:1] in "+-" and not l.startswith(("---", "+++"))]
        assert strip(d) == strip(stored), "run.py changed beyond the recorded claude-provider diff"
        continue
    assert hashlib.sha256(open(p, "rb").read()).hexdigest().upper().startswith(h), ("hash changed since the analysis plan was written", fn)

# ---------------------------------------------------------------- confirm.tex
def verdict(r):
    return "met" if r["criterion_met"] else "not met"


rows = []
for mi, (cap, key, label) in enumerate(MODELS):
    d = A[key]
    if mi:
        rows.append("\\midrule")
    rows.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{label}}}}} \\\\")
    for _, c in CONDS:
        r = d["conditions"][c]["regret"]
        rows.append(f"\\quad regret {c} & {f(r['mean'])} & {ci(r['task_ci'])} & {ci(r['cluster_ci'])} & & \\\\")
    for h in A[key]["primary"]:
        r = d["primary"][h]
        lab = {"Ha": "$H_a$: L2$-$P0", "Hb": "$H_b$: H1$-$P0"}[h]
        rows.append(f"\\quad {lab} & {f(r['mean'])} & {ci(r['task_ci'])} & {ci(r['cluster_ci'])} & {ci(r['conservative'])} & {verdict(r)} \\\\")
    hc = d["Hc"]
    p1 = lambda x: f"{100 * x:.1f}"
    rows.append(f"\\quad $H_c$: share MCWA in P0 (\\%) & {p1(hc['share_task_mean'])} & [{p1(hc['task_ci'][0])}, {p1(hc['task_ci'][1])}] & "
                f"[{p1(hc['cluster_ci'][0])}, {p1(hc['cluster_ci'][1])}] & [{p1(hc['conservative'][0])}, {p1(hc['conservative'][1])}] & {verdict(hc)} \\\\")
    for h, lab in (("S1", "L1$-$P0"), ("S2", "L2$-$L1"), ("S3", "H1$-$L2")):
        r = d["secondary"][h]
        rows.append(f"\\quad {lab} & {f(r['mean'])} & {ci(r['task_ci'])} & {ci(r['cluster_ci'])} & {ci(r['conservative'])} & secondary \\\\")
# Sonnet 5.5: run after the plan was fixed (not named in it); same criteria, same analysis code; P0, L2, H1 only (L1 not run).
# Only read here; no assertions about this file beyond using its fields.
S55A = json.load(open(os.path.join(RES, "confirm_analysis_s55.json"), encoding="utf-8"))["sonnet"]
srows = [r"\midrule", r"\multicolumn{6}{@{}l}{\emph{Sonnet 5.5 (added after the plan; same criteria)}} \\"]
for _, c in CONDS:
    if c not in S55A["conditions"]:
        continue
    r = S55A["conditions"][c]["regret"]
    srows.append(f"\\quad regret {c} & {f(r['mean'])} & {ci(r['task_ci'])} & {ci(r['cluster_ci'])} & & \\\\")
for h in ("Ha", "Hb"):
    r = S55A["primary"][h]
    lab = {"Ha": "$H_a$: L2$-$P0", "Hb": "$H_b$: H1$-$P0"}[h]
    srows.append(f"\\quad {lab} & {f(r['mean'])} & {ci(r['task_ci'])} & {ci(r['cluster_ci'])} & {ci(r['conservative'])} & {verdict(r)} \\\\")
_hc = S55A["Hc"]
_p1 = lambda x: f"{100 * x:.1f}"
srows.append(f"\\quad $H_c$: share MCWA in P0 (\\%) & {_p1(_hc['share_task_mean'])} & [{_p1(_hc['task_ci'][0])}, {_p1(_hc['task_ci'][1])}] & "
             f"[{_p1(_hc['cluster_ci'][0])}, {_p1(_hc['cluster_ci'][1])}] & [{_p1(_hc['conservative'][0])}, {_p1(_hc['conservative'][1])}] & {verdict(_hc)} \\\\")
rows += srows
tab = ("% generated by scripts/make_confirm_tables.py from raea4/bench/results/confirm_analysis.json (asserted against confirm_analysis.md) and confirm_analysis_s55.json (Sonnet block, read only); do not edit\n"
       "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
       "Quantity (134 tasks) & Estimate & Task CI & Pattern-cluster CI & Conservative & Criterion \\\\\n\\midrule\n"
       + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
open(os.path.join(OUT, "confirm.tex"), "w", encoding="utf-8").write(tab)

# ---------------------------------------------------------------- nums_confirm.tex
M = []


def mac(name, val, src):
    M.append(f"\\providecommand{{\\{name}}}{{{val}}}% src: {src}")


mac("cfN", str(N_EXPECT), "confirm_analysis.json: n_answers per condition")
mac("cfAnswers", str(4 * N_EXPECT), "confirm_analysis.json: 4 conditions x 134")
for cap, key, _ in MODELS:
    d = A[key]
    for w, c in CONDS:
        r = d["conditions"][c]["regret"]
        mac(f"cf{cap}{w}Regret", f(r["mean"]), f"confirm_analysis.json:{key}.conditions.{c}.regret.mean")
        mac(f"cf{cap}{w}RegretCI", f"{f(r['mean'])} {ci(r['task_ci'])}", f"confirm_analysis.json:{key}.conditions.{c}.regret")
        mac(f"cf{cap}{w}Opt", str(round(100 * d["conditions"][c]["optimal"]["mean"])), f"confirm_analysis.json:{key}.conditions.{c}.optimal.mean*100")
    for h, kind in (("Ha", "primary"), ("Hb", "primary"), ("S1", "secondary"), ("S2", "secondary"), ("S3", "secondary")):
        r = d[kind][h]
        h = {"S1": "SOne", "S2": "STwo", "S3": "SThree"}.get(h, h)   # macro names are letters only
        mac(f"cf{cap}{h}", f(r["mean"]), f"confirm_analysis.json:{key}.{kind}.{h}.mean")
        mac(f"cf{cap}{h}Task", ci(r["task_ci"]), f"confirm_analysis.json:{key}.{kind}.{h}.task_ci")
        mac(f"cf{cap}{h}Clus", ci(r["cluster_ci"]), f"confirm_analysis.json:{key}.{kind}.{h}.cluster_ci")
        mac(f"cf{cap}{h}Cons", ci(r["conservative"]), f"confirm_analysis.json:{key}.{kind}.{h}.conservative")
        mac(f"cf{cap}{h}Est", f"{f(r['mean'])} {ci(r['conservative'])}", f"confirm_analysis.json:{key}.{kind}.{h}: mean, conservative interval")
        if kind == "primary":
            mac(f"cf{cap}{h}Verdict", verdict(r), f"confirm_analysis.json:{key}.{kind}.{h}.criterion_met")
    hc = d["Hc"]
    p1 = lambda x: f"{100 * x:.1f}"
    mac(f"cf{cap}HcShare", p1(hc["share_task_mean"]), f"confirm_analysis.json:{key}.Hc.share_task_mean*100")
    mac(f"cf{cap}HcCons", f"[{p1(hc['conservative'][0])}, {p1(hc['conservative'][1])}]", f"confirm_analysis.json:{key}.Hc.conservative*100")
    mac(f"cf{cap}HcCount", str(hc["n_answers_with_marginal_accurate_and_action_nonoptimal"]), f"confirm_analysis.json:{key}.Hc.n_answers_with_marginal_accurate_and_action_nonoptimal")
    mac(f"cf{cap}HcVerdict", verdict(hc), f"confirm_analysis.json:{key}.Hc.criterion_met")
    k = key
    R = REC[k]
    mac(f"cf{cap}InTokM", f"{R['pt'] / 1e6:.2f}", f"confirm_{k}.jsonl usage.prompt_tokens (first record per prompt; = score_confirm_{k}.json cost.input)")
    mac(f"cf{cap}CacheTokM", f"{R['hit'] / 1e6:.2f}", f"confirm_{k}.jsonl usage.prompt_cache_hit_tokens")
    mac(f"cf{cap}OutTokM", f"{R['ct'] / 1e6:.2f}", f"confirm_{k}.jsonl usage.completion_tokens (= score_confirm_{k}.json cost.output)")
    mac(f"cf{cap}CostUsd", f"{R['usd']:.2f}", f"score_confirm_{k}.json cost.usd (list price, no cache discount, off-peak)")
    mac(f"cf{cap}LatMedian", f"{R['lat']:.0f}", f"confirm_{k}.jsonl median latency per answer, s")
    mac(f"cf{cap}ToolCalls", str(R["tools"]), f"confirm_{k}.jsonl n_tool_calls > 0 count")
    mac(f"cf{cap}Sha", R["sha"], f"sha256 of confirm_{k}.jsonl")
mac("cfFlashCostCny", f"{REC['flash']['cny']:.1f}", "confirm_flash.jsonl usage at DeepSeek off-peak CNY 1 / 0.02 (cache hit) / 4 per M tokens (input miss / hit / output)")
mac("cfTerraReasonTokM", f"{REC['terra']['reasoning'] / 1e6:.2f}", "score_confirm_terra.json cost.reasoning")
mac("cfFlashHOneIncomplete", str(A["flash"]["H1_audit"]["incomplete_model"]), "confirm_analysis.json:flash.H1_audit.incomplete_model")
mac("cfTerraHOneIncomplete", str(A["terra"]["H1_audit"]["incomplete_model"]), "confirm_analysis.json:terra.H1_audit.incomplete_model")
mac("cfFlashHOneDefaults", str(A["flash"]["H1_audit"]["answers_with_any_default_or_clip"]), "confirm_analysis.json:flash.H1_audit.answers_with_any_default_or_clip")
mac("cfTerraHOneDefaults", str(A["terra"]["H1_audit"]["answers_with_any_default_or_clip"]), "confirm_analysis.json:terra.H1_audit.answers_with_any_default_or_clip")
# Sonnet 5.5 confirmation (added after the plan)
for w, c in CONDS:
    if c not in S55A["conditions"]:
        continue
    r = S55A["conditions"][c]["regret"]
    mac(f"cfSonnet{w}Regret", f(r["mean"]), f"confirm_analysis_s55.json:sonnet.conditions.{c}.regret.mean")
    mac(f"cfSonnet{w}RegretCI", f"{f(r['mean'])} {ci(r['task_ci'])}", f"confirm_analysis_s55.json:sonnet.conditions.{c}.regret")
for h in ("Ha", "Hb"):
    r = S55A["primary"][h]
    mac(f"cfSonnet{h}", f(r["mean"]), f"confirm_analysis_s55.json:sonnet.primary.{h}.mean")
    mac(f"cfSonnet{h}Task", ci(r["task_ci"]), f"confirm_analysis_s55.json:sonnet.primary.{h}.task_ci")
    mac(f"cfSonnet{h}Clus", ci(r["cluster_ci"]), f"confirm_analysis_s55.json:sonnet.primary.{h}.cluster_ci")
    mac(f"cfSonnet{h}Cons", ci(r["conservative"]), f"confirm_analysis_s55.json:sonnet.primary.{h}.conservative")
    mac(f"cfSonnet{h}Est", f"{f(r['mean'])} {ci(r['conservative'])}", f"confirm_analysis_s55.json:sonnet.primary.{h}: mean, conservative interval")
    mac(f"cfSonnet{h}Verdict", verdict(r), f"confirm_analysis_s55.json:sonnet.primary.{h}.criterion_met")
mac("cfSonnetHcShare", _p1(_hc["share_task_mean"]), "confirm_analysis_s55.json:sonnet.Hc.share_task_mean*100")
mac("cfSonnetHcTask", f"[{_p1(_hc['task_ci'][0])}, {_p1(_hc['task_ci'][1])}]", "confirm_analysis_s55.json:sonnet.Hc.task_ci*100")
mac("cfSonnetHcClus", f"[{_p1(_hc['cluster_ci'][0])}, {_p1(_hc['cluster_ci'][1])}]", "confirm_analysis_s55.json:sonnet.Hc.cluster_ci*100")
mac("cfSonnetHcCons", f"[{_p1(_hc['conservative'][0])}, {_p1(_hc['conservative'][1])}]", "confirm_analysis_s55.json:sonnet.Hc.conservative*100")
mac("cfSonnetHcCount", str(_hc["n_answers_with_marginal_accurate_and_action_nonoptimal"]), "confirm_analysis_s55.json:sonnet.Hc.n_answers_with_marginal_accurate_and_action_nonoptimal")
mac("cfSonnetHcVerdict", verdict(_hc), "confirm_analysis_s55.json:sonnet.Hc.criterion_met")
hdr = ("% generated by scripts/make_confirm_tables.py from raea4/bench/results/confirm_analysis.json, score_confirm_{flash,terra}.json, confirm_{flash,terra}.jsonl; do not edit\n"
       "% Macros for the held-out confirmation (sections/llm.tex, app_llm.tex, checklist.tex). Each line ends with  % src: <file>:<key path>.\n")
open(os.path.join(OUT, "nums_confirm.tex"), "w", encoding="utf-8").write(hdr + "\n".join(M) + "\n")
print("wrote confirm.tex and nums_confirm.tex:", len(M), "macros; flash CNY %.2f, USD %.2f; terra USD %.2f" % (REC["flash"]["cny"], REC["flash"]["usd"], REC["terra"]["usd"]))

