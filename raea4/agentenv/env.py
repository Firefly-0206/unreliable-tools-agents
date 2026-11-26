"""
Closed-loop diagnostic environment with injected persistent (shared) cache faults and transient noise.

Hidden state h = (s, z): s in {0: benign, 1, 2} is the status to diagnose, z is the cache-replica flag
(benign-stuck variant: z in {0 fresh, 1 stale}; value-stuck variant: z in {0 fresh, 1+v stale with frozen
value v}). Tools (fixed order):
  0 A  cache-backed status tool (e.g. health_check)            -- always on the cache replica
  1 B  second status interface (e.g. status_dashboard)          -- on the SAME replica, or on a primary DB
  2 L  independent log store (e.g. log_search)                  -- transient i.i.d. noise only
  3 P  expensive live probe (e.g. synthetic_probe)              -- transient i.i.d. noise only
The environment samples h from the episode prior and every tool output from exactly the likelihood
matrices of the episode's sim.Model, so the Oracle-DP of that Model is the exact optimum.

Controls are structure-matched twins: TRANSIENT-k has exactly the tool parameters, prior, scenario and
uniform draws of SHARED-k with only phi set to 0 (TRANSIENT_P-k likewise twins PERSIST-k), so it sees the
same status and the same outputs whenever SHARED-k's replica is fresh.
"""
import hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sim import Model, loss_matrix, healthy, source_of  # noqa: E402

K = 3
F_STUCK = 0.97                      # stale replica reports its stuck value with this probability
VPRIOR = [0.8, 0.1, 0.1]            # value-stuck variant: distribution of the frozen value
BUDGET = 6
ERR, SEV, ESC = 20.0, 60.0, 8.0
LOSS = loss_matrix(K, ERR, SEV, ESC)  # rows: verdict 0..K-1, escalate; columns: true state

# (source, accuracy range, cost) per tool; phi range; stale emission. Probe: cost 4 of the budget 6 and
# accuracy .85-.92, so one probe no longer settles the task (v1 had .95-.99 and Probe-once was near-optimal).
PROBE = ("live", (.85, .92), 4)
FAMILY_SPEC = {
    "PERSIST": dict(tools=[("cache", (.85, .95), 1), ("primary", (.70, .80), 2), ("logs", (.65, .75), 1), PROBE],
                    phi=(.25, .40), stuck="benign"),      # retries of A share a fault; B on an independent primary DB
    "SHARED": dict(tools=[("cache", (.85, .95), 1), ("cache", (.80, .90), 1), ("logs", (.65, .75), 2), PROBE],
                   phi=(.15, .35), stuck="benign"),       # two interfaces, one replica
    "MIXED": dict(tools=[("cache", (.85, .95), 1), ("cache", (.80, .90), 1), ("logs", (.70, .78), 1), PROBE],
                  phi=(.20, .35), stuck="benign"),        # shared fault + cheap noisy log store whose repeats help
    "VALUE": dict(tools=[("cache", (.85, .95), 1), ("cache", (.80, .90), 1), ("logs", (.65, .75), 2), PROBE],
                  phi=(.15, .35), stuck="value"),         # the replica shares the specific frozen value
}
TWINS = {"TRANSIENT": "SHARED", "TRANSIENT_P": "PERSIST"}   # phi = 0 twins
FAMILIES = ["PERSIST", "TRANSIENT", "SHARED", "MIXED", "VALUE", "TRANSIENT_P"]
CORE = FAMILIES[:4]
CONDITIONS = ["K1", "K2", "K3", "K4", "K4c"]
# K2 decomposition (see K2_DECOMP.md). CONDITIONS is deliberately unchanged (make_episodes / export_prompts / run.py
# defaults), so every K1-K4c prompt, episode and log stays valid; the new conditions are opt-in via --conditions.
EXTRA_CONDITIONS = ["K2a", "K1e", "K2e", "K2p", "KC", "KS1", "KS2", "KS2v", "KS3"]
ALL_CONDITIONS = CONDITIONS + EXTRA_CONDITIONS
ELICIT = {"K1e", "K2e", "K2p"}                       # conditions that first elicit the observation-model JSON
BASE_COND = {"K2a": "K2a", "K1e": "K1", "K2e": "K2", "K2p": "K2",   # information content of the system prompt
             "KC": "K1", "KS1": "K1", "KS2": "K1", "KS2v": "K1", "KS3": "K1"}
# S experiment (PLAN_s_supply.md): K1 plus a "Decision aid" block before every action. KC: a checklist without numbers;
# KS1: exact joint posterior; KS2: + next-report probabilities; KS2v: + one-step-ahead scores; KS3: KS2 + exact scores.
SUPPLY = ("KC", "KS1", "KS2", "KS2v", "KS3")
SUPPLY_NOTE = {
    "KC": "Before each of your actions you will see a short 'Decision aid' checklist.",
    **{c: "Before each of your actions you will see a 'Decision aid' block with exact quantities computed from the "
          "observation model above and all results so far. They are correct; you still choose every action yourself."
       for c in ("KS1", "KS2", "KS2v", "KS3")}}
WARNING = ("Warning: repeated or agreeing reports from tools that read the same data source are not "
           "independent confirmations.")

def _sc(role, arg, subject, noun, question, states, sev_text, backends, tools):
    return dict(role=role, arg=arg, subject=subject, noun=noun, question=question, states=states, sev_text=sev_text,
                backends=dict(zip(["cache", "primary", "logs", "live"], backends)),
                tools=[dict(name=n, field=f, values=v, what=w) for n, f, v, w in tools])

SCENARIOS = {
    "service": _sc(
        "an on-call SRE agent", "service", "checkout-api", "service",
        "whether the production service `checkout-api` is healthy, degraded or down right now",
        ["healthy", "degraded", "down"],
        "declaring \"healthy\" when the service is actually degraded or down (an outage goes unhandled)",
        ["metrics-cache-eu1", "statusdb-primary", "logstore-eu1", "the live service endpoint"],
        [("health_check", "status", ["ok", "degraded", "down"], "Returns the latest recorded health status of a service."),
         ("status_dashboard", "panel_state", ["green", "amber", "red"],
          "Reads the availability panel of the ops status dashboard for a service (green = healthy, amber = degraded, red = down)."),
         ("log_search", "summary", ["no_anomalies", "elevated_error_rate", "connection_refused_storm"],
          "Samples the last 5 minutes of application logs and summarises them "
          "(no_anomalies = healthy, elevated_error_rate = degraded, connection_refused_storm = down)."),
         ("synthetic_probe", "result", ["pass", "slow", "fail"],
          "Runs an end-to-end synthetic transaction against the live service (pass = healthy, slow = degraded, fail = down).")]),
    "pipeline": _sc(
        "a data-platform on-call agent", "job", "nightly-orders-etl", "job",
        "whether last night's run of the ETL job `nightly-orders-etl` succeeded, partially failed or failed",
        ["succeeded", "partial", "failed"],
        "declaring \"succeeded\" when the run actually partially failed or failed (bad data ships downstream)",
        ["scheduler-cache-us2", "scheduler-db-primary", "runlog-archive", "the warehouse tables"],
        [("job_status", "state", ["succeeded", "partial", "failed"], "Returns the recorded final state of a job run."),
         ("pipeline_dashboard", "badge", ["green", "amber", "red"],
          "Reads the run badge of the pipeline dashboard for a job (green = succeeded, amber = partial, red = failed)."),
         ("run_log_grep", "summary", ["clean", "row_rejects", "fatal_errors"],
          "Samples the job's run logs and summarises them (clean = succeeded, row_rejects = partial, fatal_errors = failed)."),
         ("validate_output", "result", ["rows_complete", "rows_missing", "table_empty"],
          "Runs a row-count validation query against the warehouse output tables "
          "(rows_complete = succeeded, rows_missing = partial, table_empty = failed).")]),
    "backup": _sc(
        "a database-reliability on-call agent", "database", "crm-db", "database",
        "whether last night's backup of the database `crm-db` is verified, incomplete or corrupt",
        ["verified", "incomplete", "corrupt"],
        "declaring \"verified\" when the backup is actually incomplete or corrupt (a restore would fail)",
        ["backup-catalog-cache", "backup-catalog-primary", "backup-agent-logs", "a scratch restore host"],
        [("backup_status", "state", ["verified", "incomplete", "corrupt"], "Returns the recorded state of the latest backup."),
         ("backup_console", "indicator", ["green", "yellow", "red"],
          "Reads the backup console indicator for a database (green = verified, yellow = incomplete, red = corrupt)."),
         ("agent_log_scan", "summary", ["no_errors", "skipped_files", "checksum_errors"],
          "Samples the backup agent's logs and summarises them "
          "(no_errors = verified, skipped_files = incomplete, checksum_errors = corrupt)."),
         ("restore_test", "result", ["restore_ok", "restore_partial", "restore_failed"],
          "Restores a sample of the backup on a scratch host (restore_ok = verified, restore_partial = incomplete, "
          "restore_failed = corrupt).")]),
    "dns": _sc(
        "a network on-call agent", "domain", "api.shop.example", "domain",
        "whether the domain `api.shop.example` is resolving, intermittently failing or failing to resolve right now",
        ["resolving", "intermittent", "failing"],
        "declaring \"resolving\" when resolution is actually intermittent or failing (customers cannot connect)",
        ["resolver-cache-ap1", "dns-zone-primary", "query-log-store", "the authoritative name servers"],
        [("dns_status", "status", ["resolving", "intermittent", "failing"], "Returns the latest recorded resolution status of a domain."),
         ("network_dashboard", "tile", ["green", "amber", "red"],
          "Reads the DNS tile of the network dashboard for a domain (green = resolving, amber = intermittent, red = failing)."),
         ("query_log_sample", "summary", ["all_noerror", "some_servfail", "mostly_servfail"],
          "Samples the last 5 minutes of resolver query logs and summarises them "
          "(all_noerror = resolving, some_servfail = intermittent, mostly_servfail = failing)."),
         ("dig_probe", "result", ["answer", "timeout_some", "no_answer"],
          "Queries the authoritative name servers directly from several vantage points "
          "(answer = resolving, timeout_some = intermittent, no_answer = failing).")]),
    "replication": _sc(
        "a database on-call agent", "replica", "orders-replica-2", "replica",
        "whether the database replica `orders-replica-2` is in sync, lagging or broken right now",
        ["in_sync", "lagging", "broken"],
        "declaring \"in_sync\" when the replica is actually lagging or broken (reads return wrong data)",
        ["monitoring-cache-eu3", "cluster-registry-primary", "binlog-archive", "the replica and its primary"],
        [("replication_status", "state", ["in_sync", "lagging", "broken"], "Returns the latest recorded replication state of a replica."),
         ("db_dashboard", "light", ["green", "amber", "red"],
          "Reads the replication light of the database dashboard (green = in_sync, amber = lagging, red = broken)."),
         ("binlog_scan", "summary", ["applied_promptly", "apply_backlog", "apply_halted"],
          "Samples the replica's binlog apply records and summarises them "
          "(applied_promptly = in_sync, apply_backlog = lagging, apply_halted = broken)."),
         ("checksum_compare", "result", ["match", "behind", "diverged"],
          "Compares row checksums between the replica and its primary (match = in_sync, behind = lagging, diverged = broken).")]),
    "queue": _sc(
        "a messaging on-call agent", "queue", "order-events", "queue",
        "whether the message queue `order-events` is draining normally, backlogged or stalled right now",
        ["draining", "backlogged", "stalled"],
        "declaring \"draining\" when the queue is actually backlogged or stalled (orders stop being processed)",
        ["broker-metrics-cache", "broker-admin-primary", "consumer-log-store", "the live broker"],
        [("queue_status", "state", ["draining", "backlogged", "stalled"], "Returns the latest recorded state of a queue."),
         ("broker_dashboard", "gauge", ["green", "amber", "red"],
          "Reads the queue gauge of the broker dashboard (green = draining, amber = backlogged, red = stalled)."),
         ("consumer_log_scan", "summary", ["acks_normal", "acks_slow", "no_acks"],
          "Samples the consumers' logs and summarises them (acks_normal = draining, acks_slow = backlogged, no_acks = stalled)."),
         ("canary_message", "result", ["consumed", "delayed", "not_consumed"],
          "Publishes a canary message to the live broker and waits for it "
          "(consumed = draining, delayed = backlogged, not_consumed = stalled).")]),
}
N_TOOLS = 4
DOMAIN = os.environ.get("RAEA_DOMAIN", "1")
if DOMAIN == "2":     # second task domain: five statuses, five tools, two independent fault sources (env2_data.py)
    import env2_data as _D2
    K, BUDGET, N_TOOLS = _D2.K2, _D2.BUDGET2, _D2.N_TOOLS2
    LOSS = loss_matrix(K, ERR, SEV, ESC); SCENARIOS = dict(_D2.SCENARIOS2); FAMILY_SPEC = dict(_D2.FAMILY_SPEC2)
elif DOMAIN == "3":   # third task domain: fact-checking, compound state, sub-question tools, shared wire fault (env3_data.py)
    import env3_data as _D3
    K, BUDGET, N_TOOLS = _D3.K3, _D3.BUDGET3, _D3.N_TOOLS3
    LOSS = loss_matrix(K, ERR, SEV, ESC); SCENARIOS = dict(_D3.SCENARIOS3); FAMILY_SPEC = dict(_D3.FAMILY_SPEC3)
SCENARIO_LIST = list(SCENARIOS)
OTHERS = {3: "two", 4: "three", 5: "four"}[K]; OTHERS_PLUS = {3: "three", 4: "four", 5: "five"}[K]
NTOOLS_WORD = {4: "four", 5: "five", 6: "six"}[N_TOOLS]

# ------------------------------------------------------------------ episode parameters ----------
def _u(rng, lo, hi):
    return float(np.round(rng.uniform(lo, hi), 2))

def sample_params(family, rng):
    """Every number is rounded to 2 decimals, so the numbers shown to agents ARE the model."""
    fs = FAMILY_SPEC[family]
    while True:
        if DOMAIN == "2":
            pr = np.round(rng.dirichlet(_D2.PRIOR_ALPHA2), 2); pr[0] = round(1 - pr[1:].sum(), 2)
            if pr.min() >= _D2.PRIOR_MIN2: break
            continue
        if DOMAIN == "3":
            pr = np.round(rng.dirichlet(_D3.PRIOR_ALPHA3), 2); pr[0] = round(1 - pr[1:].sum(), 2)
            if pr.min() >= _D3.PRIOR_MIN3: break
            continue
        pr = np.round(rng.dirichlet([4, 2, 2]), 2); pr[0] = round(1 - pr[1] - pr[2], 2)
        if pr.min() >= 0.05: break
    tools = [dict(q=_u(rng, *qr), cost=c, src=src) for src, qr, c in fs["tools"]]
    if DOMAIN == "3":
        for t in tools:
            if t["src"] in _D3.KIND3: t["kind"] = _D3.KIND3[t["src"]]
    out = dict(family=family, prior=[float(x) for x in pr], tools=tools, phi=_u(rng, *fs["phi"]), stuck=fs["stuck"])
    if "phi2" in fs: out["phi2"] = _u(rng, *fs["phi2"])
    return out

def twin_params(par, family):
    """Structure-matched control: identical parameters with phi = 0."""
    return dict(par, family=family, phi=0.0, stuck="benign", tools=[dict(t) for t in par["tools"]])

def stuck_emission(v):
    e = np.full(K, (1 - F_STUCK) / (K - 1)); e[v] = F_STUCK; return e

def marginal_stuck(par):
    if par["stuck"] in ("benign", "percall"): return stuck_emission(0)
    return sum(w * stuck_emission(v) for v, w in enumerate(VPRIOR))

def sim_tools(par):
    """sim.py tool dicts. For the value-stuck variant femit is the MARGINAL stale emission (used by MMI)."""
    if par["stuck"] == "percall":      # X experiment: every replica call is stale independently with probability phi
        return [dict(kind="sym", q=t["q"], cost=t["cost"], group=None, femit=stuck_emission(0) if t["src"] == "cache" else None,
                     **({"marg_phi": par["phi"]} if t["src"] == "cache" and par["phi"] > 0 else {})) for t in par["tools"]]
    return [dict(kind=t.get("kind", "sym"), q=t["q"], cost=t["cost"], group={"cache": "c", "feed": "f"}.get(t["src"]),
                 femit=marginal_stuck(par) if t["src"] == "cache" else (np.full(K, 1 / K) if t["src"] == "feed" else None)) for t in par["tools"]]

def sim_spec(par):
    groups = {"c": par["phi"]} if par["phi"] > 0 and par["stuck"] != "percall" else {}
    if par.get("phi2", 0) > 0: groups["f"] = par["phi2"]       # second domain: feed outage, independent of the replica
    return dict(family=par["family"], K=K, prior=np.array(par["prior"]), tools=sim_tools(par), groups=groups, loss=LOSS, budget=BUDGET)

class ValueStuckModel(Model):
    """Hidden state (s, z), z = 0 fresh, z = 1 + v stale with frozen value v. Same interface as sim.Model."""
    def __init__(self, prior_s, tools, phi, loss, budget):
        self.K, self.tools, self.loss, self.budget, self.temper = K, tools, np.asarray(loss), budget, 1.0
        self.G = ["c"]
        self.H = [(s, z) for s in range(K) for z in range(K + 1)]
        self.nH = len(self.H)
        self.s_of = np.array([h[0] for h in self.H])
        pz = np.r_[1 - phi, phi * np.array(VPRIOR)]
        self.prior = np.array([prior_s[s] * pz[z] for s, z in self.H]); self.prior /= self.prior.sum()
        self.lik = []
        for t in tools:
            L = np.zeros((self.nH, K))
            for i, (s, z) in enumerate(self.H):
                L[i] = stuck_emission(z - 1) if (t.get("group") == "c" and z > 0) else healthy("sym", t["q"], K, s)
            self.lik.append(L)
        self.cache = {}

def build_model(par):
    sp = sim_spec(par)
    if par["stuck"] == "value":
        return ValueStuckModel(sp["prior"], sp["tools"], par["phi"], LOSS, BUDGET)
    return Model(K, sp["prior"], sp["tools"], sp["groups"], LOSS, BUDGET)

def srcmap(par):
    return {j: source_of(t, j) for j, t in enumerate(sim_tools(par))}

def _inv(p, u):
    return int(min(np.searchsorted(np.cumsum(p), u, side="right"), len(p) - 1))

def sample_world(par, rng, M=None):
    """Common random numbers: hidden state and, for every tool, the outputs of its 1st..max-th call, all by
    inverse CDF from one fixed set of uniforms, so a phi = 0 twin drawn with the same rng sees the same s and
    the same outputs whenever the original replica is fresh."""
    M = M or build_model(par)
    u_s, u_z, u_v = rng.random(3); U = rng.random((len(par["tools"]), BUDGET))
    s = _inv(par["prior"], u_s); stale = bool(u_z < par["phi"])
    if par["stuck"] == "value":
        v = _inv(VPRIOR, u_v) if stale else None; h = (s, 1 + v if stale else 0)
    else:
        v = None; fl = {"c": stale, "f": bool(u_v < par.get("phi2", 0.0))}; h = (s, tuple(int(fl[g]) for g in M.G))
    i = M.H.index(h)
    table = [[_inv(M.lik[j][i], U[j, c]) for c in range(int(BUDGET // t["cost"]))] for j, t in enumerate(par["tools"])]
    w = dict(s=s, stale=stale, frozen=v, hidx=i)
    if "phi2" in par: w["feed_down"] = bool(u_v < par["phi2"])
    return w, table

def ticket_id(eid, salt=0):
    """Opaque 5-digit ticket number shown to agents (reveals nothing about the family)."""
    return 10000 + int(hashlib.sha256(f"raea4-ticket:{salt}:{eid}".encode()).hexdigest()[:12], 16) % 90000

# ------------------------------------------------------------------ text --------------------------
def num(x):
    return int(round(x)) if abs(x - round(x)) < 1e-9 else round(x, 6)

def pct(x):
    return f"{x:.2f}"

def backend_of(ep, j):
    sc = SCENARIOS[ep["scenario"]]; return sc["backends"][ep["params"]["tools"][j]["src"]]

def tool_doc(ep, j):
    """Per-tool documentation: identical in every condition. Sharing must be read off the backend names."""
    sc, par = SCENARIOS[ep["scenario"]], ep["params"]; t, d = par["tools"][j], sc["tools"][j]
    vals = ", ".join(f"'{v}'" for v in d["values"]); be = sc["backends"][t["src"]]
    if ep.get("doc_style") == "runbook":
        return runbook_doc(ep, j)
    if ep.get("doc_style") in ("tdecomp", "tmixed"):
        # T experiment (independent per-call fault only): every tool's report law as probabilities. "tdecomp" states the
        # replica tool's law as fresh / stale snapshot; "tmixed" states the SAME per-call law as one table by true
        # status, with no mention of staleness. The other tools' entries are identical in the two styles.
        assert par["stuck"] == "percall" and par["phi"] > 0 and t["src"] in ("cache", "primary", "logs", "live")
        s = f"{d['what']} Output field `{d['field']}` in {{{vals}}}. Cost: {t['cost']}. "
        head = "Report probabilities per call (calls are independent given the true status): "
        q, oth = t["q"], (1 - t["q"]) / (K - 1)
        if t["src"] != "cache":
            s += f"Served from {be} (primary database). " if t["src"] == "primary" else f"Queries {be} directly on every call. "
            return s + head + f"the value matching the true status {num(q)}, each other value {num(oth)}."
        s += f"Served from read replica {be}. "
        if ep["doc_style"] == "tdecomp":
            return s + (f"Each call independently hits a stale snapshot with probability {num(par['phi'])} (per call, not per session). "
                        + head + f"on a fresh snapshot, the value matching the true status {num(q)}, each other value {num(oth)}; "
                        f"on a stale snapshot, '{d['values'][0]}' {num(F_STUCK)}, each other value {num((1 - F_STUCK) / (K - 1))}, "
                        f"whatever the true status.")
        st, phi, em = sc["states"], par["phi"], stuck_emission(0); rows = []
        for k in range(K):
            p = [(1 - phi) * (q if o == k else oth) + phi * em[o] for o in range(K)]
            rows.append(f"if the true status is {st[k]}: " + ", ".join(f"'{d['values'][o]}' {num(p[o])}" for o in range(K)))
        return s + head + "; ".join(rows) + "."
    if ep.get("doc_style") in ("hq", "hi"):
        return hd_doc(ep, j)
    if ep.get("doc_style") == "spec":      # V experiment: the same facts as a terse field list (persistent benign-stuck fault only)
        assert par["stuck"] == "benign" and par["phi"] > 0 and t["src"] in ("cache", "primary", "logs", "live")
        s = f"{d['what']} | output: `{d['field']}` in {{{vals}}} | cost: {t['cost']} | "
        if t["src"] == "cache":
            return s + (f"backend: read replica {be} | fault: replica stale for the whole session, p = {par['phi']:.2f}; while stale, "
                        f"each call returns '{d['values'][0]}' with p = 0.97 | accuracy while fresh: {pct(t['q'])} per call, "
                        f"errors uniform over the other {OTHERS} values")
        kind = "primary database" if t["src"] == "primary" else "queried directly on every call"
        return s + (f"backend: {be} ({kind}) | fault: none, never stale | accuracy: {pct(t['q'])} per call, errors independent "
                    f"across calls, uniform over the other {OTHERS} values")
    s = f"{d['what']} Output field `{d['field']}` in {{{vals}}}. Cost: {t['cost']}. "
    if DOMAIN == "3" and t["src"] == "cache":
        assert par["stuck"] == "benign"
        return s + (f"Republishes the item on wire service {be}. Wire error incidents: about {par['phi']:.0%} of sessions; "
                    f"when the wire item behind this claim is wrong, the outlet keeps returning '{d['values'][0]}' "
                    f"(about 97% of calls), whatever the truth. Accuracy when the wire item is correct: {pct(t['q'])} per call, "
                    f"errors spread evenly over the other {OTHERS} values.")
    if DOMAIN == "3" and t.get("kind", "sym") != "sym":
        return s + (f"Queries {be} directly on every call (never stale). Accuracy: {pct(t['q'])} per call "
                    f"(probability that it answers its own question correctly); errors are independent across calls.")
    if t["src"] == "cache":
        s += f"Served from read replica {be}. "
        if par["phi"] == 0:
            s += "Replica staleness incidents: none (replica verified fresh for this session). "
        elif par["stuck"] == "percall":
            s += (f"Replica staleness: each call independently hits a stale snapshot with probability about "
                  f"{par['phi']:.0%} (per call, not per session); a stale answer is '{d['values'][0]}' about 97% of the time. ")
        elif par["stuck"] == "benign":
            s += (f"Replica staleness incidents: about {par['phi']:.0%} of sessions; a stale replica keeps "
                  f"returning '{d['values'][0]}' (about 97% of calls). ")
        else:
            s += (f"Replica staleness incidents: about {par['phi']:.0%} of sessions; a stale replica keeps returning "
                  f"the value frozen when it stopped refreshing (about 97% of calls; the frozen value is "
                  f"'{d['values'][0]}' in about 80% of incidents). ")
        s += f"Accuracy when the replica is fresh: {pct(t['q'])} per call, errors spread evenly over the other {OTHERS} values."
    elif t["src"] == "feed":
        s += (f"Served from aggregation feed {be}. Feed outage incidents: about {par['phi2']:.0%} of sessions; during an outage "
              f"every call returns an arbitrary value (each of the {OTHERS_PLUS} values equally likely), whatever the true status. "
              f"Accuracy when the feed is up: {pct(t['q'])} per call, errors spread evenly over the other {OTHERS} values.")
    elif t["src"] == "primary":
        s += (f"Served from {be} (primary database, never stale). Accuracy: {pct(t['q'])} per call; "
              f"errors are independent across calls and spread evenly over the other {OTHERS} values.")
    else:
        s += (f"Queries {be} directly on every call (never stale). Accuracy: {pct(t['q'])} per call; "
              f"errors are independent across calls and spread evenly over the other {OTHERS} values.")
    return s

# HD experiment (PLAN_hd_hard_description.md): rates in words. The generator draws every rate uniformly inside the range
# of its word; the ranges are NOT shown to the agent. Persistent benign-stuck fault on the replica only (PERSIST tools).
HD_PHI_BINS = {"rare": (0.05, 0.15), "occasional": (0.20, 0.40), "frequent": (0.60, 0.80)}
HD_ACC_BINS = {"highly reliable": (0.90, 0.96), "usually right": (0.78, 0.88), "noisy": (0.62, 0.74)}
HD_STALE_WORDS = "on almost every call"          # stale emission F_STUCK = 0.97

def hd_doc(ep, j):
    """'hq': fault rate and accuracies as words, otherwise the sentences of the default style (numbers -> words only).
    'hi': the same words, persistence left to inference from how the refresh job behaves. Same backends, costs,
    outputs and sentence order as the default style otherwise."""
    sc, par = SCENARIOS[ep["scenario"]], ep["params"]; t, d = par["tools"][j], sc["tools"][j]; w = ep["words"]
    assert par["stuck"] == "benign" and par["phi"] > 0
    vals = ", ".join(f"'{v}'" for v in d["values"]); be = sc["backends"][t["src"]]
    s = f"{d['what']} Output field `{d['field']}` in {{{vals}}}. Cost: {t['cost']}. "
    acc = f"errors spread evenly over the other {OTHERS} values."
    if t["src"] == "cache":
        s += f"Served from read replica {be}. "
        if ep["doc_style"] == "hq":
            s += (f"Replica staleness incidents: {w['phi']} (counted per on-call session); a stale replica keeps "
                  f"returning '{d['values'][0]}' ({HD_STALE_WORDS}). ")
        else:
            s += (f"The replica is refreshed by a background job. On-call sessions during which this job is halted: {w['phi']}. "
                  f"While the job is halted the replica serves its last snapshot, which returns '{d['values'][0]}' {HD_STALE_WORDS}. "
                  f"A halted job is restarted only between on-call sessions. ")
        return s + f"Accuracy when the replica is fresh: {w['acc'][j]}; {acc}"
    if t["src"] == "primary":
        return s + f"Served from {be} (primary database, never stale). Accuracy: {w['acc'][j]}; errors are independent across calls and {acc[7:]}"
    return s + f"Queries {be} directly on every call (never stale). Accuracy: {w['acc'][j]}; errors are independent across calls and {acc[7:]}"

RB_SESSIONS, RB_CHECKS = 40, 200          # R experiment: rates are stated as counts (incident review, spot checks)
RB_TEAMS = ["platform-observability", "core-data", "sre-tooling", "edge-reliability"]

def _rb_int(x, n, what):
    k = int(round(x * n)); assert abs(k - x * n) < 1e-9, f"{what}={x} is not a whole count out of {n}"; return k

def runbook_doc(ep, j):
    """R experiment (PLAN_r_runbook.md): the facts of tool_doc written as a runbook entry. Persistent benign-stuck
    fault only. Rates appear as counts (fault rate: sessions out of 40; accuracy and stale emission: answers out of
    200), among operational details that do not bear on the decision. A careful reader recovers the true model exactly."""
    sc, par = SCENARIOS[ep["scenario"]], ep["params"]; t, d = par["tools"][j], sc["tools"][j]
    assert par["stuck"] == "benign" and par["phi"] > 0 and t["src"] in ("cache", "primary", "logs", "live")
    vals = ", ".join(f"'{v}'" for v in d["values"]); be = sc["backends"][t["src"]]
    h = int(hashlib.sha256(f"raea4-runbook:{ep['scenario']}:{j}".encode()).hexdigest()[:8], 16)
    acc = _rb_int(t["q"], RB_CHECKS, "accuracy")
    s = (f"{d['what']} Output field `{d['field']}` in {{{vals}}}. Cost: {t['cost']}. Runbook entry, rev. {3 + h % 17} "
         f"(owner: {RB_TEAMS[(h >> 5) % 4]}; escalation via the team's pager rota). Auth: read-only service token, rotated "
         f"weekly. Rate limit: {30 * (1 + (h >> 9) % 4)} requests per minute per caller; typical latency {80 + (h >> 12) % 240} ms. ")
    if t["src"] == "cache":
        nf = _rb_int(par["phi"], RB_SESSIONS, "fault rate"); ns = _rb_int(F_STUCK, RB_CHECKS, "stale emission")
        s += (f"Data path: answers are served from read replica {be}, which an asynchronous job refreshes from the primary every "
              f"30 seconds. Known issue (OPS-{2000 + h % 7000}): the refresh job sometimes halts without raising an alert. The "
              f"replica then stops refreshing until an operator restarts the job, which has never happened within one on-call "
              f"session, so during a session the replica is either refreshing throughout or not refreshing at all. Incident "
              f"review: in {nf} of the last {RB_SESSIONS} on-call sessions the replica was not refreshing. While it is not "
              f"refreshing the tool still answers, from old data: of {RB_CHECKS} such answers sampled, {ns} were "
              f"'{d['values'][0]}' and the other {RB_CHECKS - ns} were split evenly between the other values, whatever the real "
              f"status was. Spot checks while the replica was refreshing: {acc} of {RB_CHECKS} answers matched the real "
              f"status; the mismatches were split evenly between the other {OTHERS} values, and repeated calls erred "
              f"independently of one another. ")
    else:
        path = (f"reads {be} (the primary database) synchronously on every call" if t["src"] == "primary"
                else f"queries {be} directly on every call")
        s += (f"Data path: {path}; there is no replica, cache or refresh job in front of it. Spot checks: {acc} of "
              f"{RB_CHECKS} answers matched the real status; the mismatches were split evenly between the other {OTHERS} "
              f"values, and repeated calls erred independently of one another. ")
    s += (f"Past sessions and checks are representative of today's. Changelog: v{2 + h % 3}.{(h >> 3) % 9} added the "
          f"`request_id` field to responses; v{1 + h % 2}.{(h >> 7) % 9} moved timestamps to UTC.")
    return s

def raw_output(ep, j, o, n_call, cost, budget):
    """R experiment: a tool result as a service would return it, with fields that do not bear on the decision.
    `responded_at` is the time of the response, not of the data; nothing here depends on the hidden state or the fault."""
    sc = SCENARIOS[ep["scenario"]]; d = sc["tools"][j]
    h = int(hashlib.sha256(f"raea4-raw:{ep['ticket']}:{j}:{n_call}".encode()).hexdigest()[:12], 16)
    out = {"tool": d["name"], "request_id": f"req-{h % 16 ** 8:08x}", "responded_at": f"2026-03-14T02:{17 + n_call:02d}:{h % 60:02d}Z",
           "http_status": 200, "latency_ms": 80 + (h >> 20) % 240,
           "result": {sc["arg"]: sc["subject"], d["field"]: d["values"][o], "schema_version": f"{2 + j % 2}.{3 + j}", "region": "eu1"},
           "warnings": [], "cost": cost, "budget_remaining": num(budget)}
    return json.dumps(out)

def dependence_text(ep):
    """Explicit statement of the full observation model (conditions K1, K3, K4, K4c)."""
    sc, par = SCENARIOS[ep["scenario"]], ep["params"]; T = sc["tools"]; P = par["tools"]
    cache = [j for j in range(4) if P[j]["src"] == "cache"]; names = " and ".join(T[j]["name"] for j in cache)
    be = sc["backends"]["cache"]; st = sc["states"]
    lines = []
    if par["phi"] == 0:
        lines.append(f"- {names} read the replica {be}, which is verified fresh for this session: it is never stale, "
                     f"and every call's error is independent of every other call.")
    else:
        both = "both tools" if len(cache) > 1 else "the tool"
        every = "EVERY call to either tool" if len(cache) > 1 else f"EVERY call to {names}"
        lines.append(f"- {names} {'both read' if len(cache) > 1 else 'reads'} the replica {be}. With probability "
                     f"{pct(par['phi'])} this replica is stale for the WHOLE session. "
                     + (f"Staleness is shared by {both} and persists across retries." if len(cache) > 1
                        else "Staleness persists across retries."))
        if par["stuck"] == "benign":
            lines.append(f"  While it is stale, {every} reports {st[0]} with probability 0.97 "
                         f"(each other status 0.015), whatever the true status.")
        else:
            lines.append(f"  While it is stale, {both} keep reporting one frozen status: the frozen status is {st[0]} "
                         f"with probability {VPRIOR[0]}, {st[1]} {VPRIOR[1]}, {st[2]} {VPRIOR[2]}; each call reports the "
                         f"frozen status with probability 0.97 (each other status 0.015), whatever the true status.")
        lines.append("  While it is fresh, each call is correct with the accuracy below and otherwise reports one of the "
                     "two other statuses at random, independently across calls.")
    for j in range(4):
        if P[j]["src"] == "cache":
            lines.append(f"- {T[j]['name']}: accuracy when the replica is fresh {pct(P[j]['q'])}.")
        else:
            lines.append(f"- {T[j]['name']} reads {sc['backends'][P[j]['src']]}, which is never stale: every call is "
                         f"correct with probability {pct(P[j]['q'])}, errors independent across calls.")
    lines.append("- The true status does not change during the session. Distinct data sources fail independently.")
    return "\n".join(lines)

def dependence_text_k2a(ep):
    """K2a: every NUMBER of the K1 observation-model block (phi, persistence, stale emission, fresh accuracy,
    independence of fresh calls and of distinct sources) but stated per REPLICA, never per tool: which tools are
    served from the replica is not listed and must be read off the 'Served from read replica ...' line of the tool
    docs, exactly as in K2. The rule 'staleness belongs to the replica and hits every tool served from it' is given."""
    sc, par = SCENARIOS[ep["scenario"]], ep["params"]; T = sc["tools"]; P = par["tools"]
    be = sc["backends"]["cache"]; st = sc["states"]
    lines = []
    if par["phi"] == 0:
        lines.append(f"- The read replica {be} is verified fresh for this session: it is never stale, "
                     f"and every call's error is independent of every other call.")
    else:
        lines.append(f"- The read replica {be}: with probability {pct(par['phi'])} it is stale for the WHOLE session. "
                     f"Staleness is a property of the replica: it persists across retries and affects every call to "
                     f"every tool served from that replica.")
        if par["stuck"] == "benign":
            lines.append(f"  While it is stale, EVERY call served from it reports {st[0]} with probability 0.97 "
                         f"(each other status 0.015), whatever the true status.")
        else:
            lines.append(f"  While it is stale, calls served from it keep reporting one frozen status: the frozen status is "
                         f"{st[0]} with probability {VPRIOR[0]}, {st[1]} {VPRIOR[1]}, {st[2]} {VPRIOR[2]}; each call reports "
                         f"the frozen status with probability 0.97 (each other status 0.015), whatever the true status.")
        lines.append("  While it is fresh, each call is correct with the accuracy below and otherwise reports one of the "
                     "two other statuses at random, independently across calls.")
    for j in range(4):
        if P[j]["src"] == "cache":
            lines.append(f"- {T[j]['name']}: accuracy when the replica is fresh {pct(P[j]['q'])}.")
        else:
            lines.append(f"- {T[j]['name']} reads {sc['backends'][P[j]['src']]}, which is never stale: every call is "
                         f"correct with probability {pct(P[j]['q'])}, errors independent across calls.")
    lines.append("- The true status does not change during the session. Distinct data sources fail independently.")
    return "\n".join(lines)

ELICIT_REQUEST = "Write the observation-model JSON now (no action yet)."
ELICIT_ACK = "Observation model recorded. Now take your first action."
ELICIT_NUDGE = ("No valid observation_model JSON was found. Reply with ONLY the JSON object described in the system "
                "prompt, on a single line.")

def elicit_addendum(ep):
    st = ", ".join(SCENARIOS[ep["scenario"]]["states"])
    return (
        "\n\nBefore your first action, record the observation model you have recovered from the information above. "
        "For your first reply only, write ONLY one JSON object on a single line (no tool call, no action) of this form:\n"
        "{\"observation_model\": {\"tool_groups\": [{\"tools\": [\"<tool name>\", ...], \"shared_fault\": true or false, "
        "\"phi\": <number between 0 and 1>, \"persistent\": true or false, \"stale_emission\": {\"<status>\": "
        "<probability>, ...}}, ...], \"accuracy\": {\"<tool name>\": <probability>, ...}}}\n"
        f"Fields. tool_groups must contain each of the {NTOOLS_WORD} tools exactly once; put two tools in the same group exactly when "
        "you believe they read the same backend. shared_fault: true if the group's backend can become faulty (stale) in a "
        "session, so that every call to every tool of the group is affected, including repeated calls to one tool; false if "
        "the group's errors are independent across calls and tools and it is never faulty. phi: probability that the "
        "group's backend is faulty in a session (0 when shared_fault is false). persistent: true if a fault lasts the whole "
        "session, so retries give no fresh evidence; false if every call is faulty independently. stale_emission: for a "
        f"faulty group, the probability that a call reports each of the statuses {st} (sums to 1; empty when shared_fault "
        "is false). accuracy: for each tool, the probability that a call reports the true status when its backend is not "
        "faulty" + ("; for a tool that answers one yes/no question only, the probability that it answers that question "
                    "correctly" if DOMAIN == "3" else "") + ". After this JSON the session continues and you act as usual.")

RECAP_NOTE = {"K4": "After every tool result you will also see all evidence so far, grouped by data source.",
              "K4c": "After every tool result you will also see all evidence so far, in the order it was received."}

def system_prompt(ep, cond, protocol):
    """The tool-doc block is in the system prompt in every condition and protocol; K2 differs from K1 only by the
    missing observation-model block, K3 adds one sentence, K4 / K4c add one sentence and a per-result recap."""
    sc, par = SCENARIOS[ep["scenario"]], ep["params"]; st = sc["states"]; T = sc["tools"]; P = par["tools"]
    pr = ", ".join(f"{st[k]} {pct(par['prior'][k])}" for k in range(K))
    costs = ", ".join(f"{T[j]['name']} {P[j]['cost']}" for j in tool_order(ep))
    s = (f"You are {sc['role']}. Decide {sc['question']}, using the diagnostic tools available to you, then submit exactly one "
         f"verdict.\n\nPrior (before any tool call), from historical tickets: {pr}. The true status does not change "
         f"during this session.\n\nScore = total tool cost + loss of the final verdict (lower is better):\n"
         f"- correct verdict: 0\n- {sc['sev_text']}: {SEV:g}\n- any other wrong verdict: {ERR:g}\n"
         f"- escalate (hand over to a human): {ESC:g}\n"
         f"Tool costs: {costs}. Budget: {BUDGET} cost units in total; a call you cannot afford is rejected.\n\n")
    s += "Tools:\n" + "\n".join(f"- {T[j]['name']}({sc['arg']}): {tool_doc(ep, j)}" for j in tool_order(ep)) + "\n\n"
    base = BASE_COND.get(cond, cond)     # K1e -> K1, K2e / K2p -> K2, K2a -> K2a; K1-K4c are their own base
    if base == "K2a":
        s += ("Observation model (exact; the tools served from each replica are not listed here, see the tool "
              "descriptions):\n" + dependence_text_k2a(ep) + "\n\n")
    elif base != "K2":
        s += "Observation model (exact):\n" + dependence_text(ep) + "\n\n"
    if cond == "K3":
        s += WARNING + "\n\n"
    if cond in RECAP_NOTE:
        s += RECAP_NOTE[cond] + "\n\n"
    if cond in SUPPLY:
        s += SUPPLY_NOTE[cond] + "\n\n"
    verdicts = ", ".join(st + ["escalate"])
    if protocol == "fc":
        s += f"Call one tool at a time. Finish by calling submit with verdict in {{{verdicts}}}."
    else:
        s += (f"Each turn, you may reason briefly, then end your reply with exactly one JSON object on its own line:\n"
              f"{{\"action\": \"call\", \"tool\": \"<tool name>\"}}  or  "
              f"{{\"action\": \"submit\", \"verdict\": \"<one of: {verdicts}>\"}}")
    if cond in ELICIT:
        s += elicit_addendum(ep)
    return s

def first_user(ep):
    sc = SCENARIOS[ep["scenario"]]
    return f"Ticket {ep['ticket']}: determine the current status of {sc['noun']} `{sc['subject']}`. Budget: {BUDGET} cost units."

def first_user_cond(ep, cond):
    """First user message; the elicitation conditions append the request for the observation-model JSON."""
    return (first_user(ep) + ("\n\n" + ELICIT_REQUEST if cond in ELICIT else "")
            + ("\n\n" + supply_text(ep, cond, []) if cond in SUPPLY else ""))

def supply_quantities(ep, hist):
    """Exact quantities under the true model after the calls in hist [(tool, output), ...]: status posterior, joint with
    the replica state, next-report law of every affordable tool, one-step-ahead and exact expected final scores."""
    import oracle                                   # lazy: oracle imports this module
    M = oracle.true_model(ep); b = M.prior.copy(); B = float(M.budget)
    for j, o in hist: b = M.update(b, j, o); B -= M.tools[j]["cost"]
    joint = np.zeros((K, 2))
    for w, h in zip(b, M.H): joint[h[0], int(h[1] if isinstance(h[1], int) else any(h[1]))] += w
    aff = [j for j, t in enumerate(M.tools) if t["cost"] <= B + 1e-9]
    q = oracle.qvalues(M, b, B)
    myo = {k: v for k, v in q.items() if k[0] == "stop"}
    for j in aff:
        myo[("probe", j)] = float(M.tools[j]["cost"] + sum(p * M.term(M.update(b, j, o))[0] for o, p in enumerate(M.pred(b, j)) if p > 1e-12))
    return dict(post=M.marg_s(b), joint=joint, pred={j: M.pred(b, j) for j in aff}, exact=q, myopic=myo, budget=B)

def supply_text(ep, cond, hist):
    """The 'Decision aid' block of the S experiment (shown before every action)."""
    sc = SCENARIOS[ep["scenario"]]; st = sc["states"]; T = sc["tools"]; be = sc["backends"]["cache"]
    if cond == "KC":
        return ("Decision aid (checklist): before acting, consider (1) the probability of each status, and of the replica "
                "being stale, given all results so far; (2) for each tool you can afford, the probabilities of its next "
                "report; (3) the expected final score of each available action.")
    Q = supply_quantities(ep, hist); f = lambda x: f"{x:.3f}"
    L = ["Decision aid (exact, given all results so far):",
         "- Status probabilities: " + ", ".join(f"{st[k]} {f(Q['post'][k])}" for k in range(K))
         + f". Probability that replica {be} is stale: {f(Q['joint'][:, 1].sum())}.",
         "- Joint probabilities (status, replica state): " + "; ".join(
             f"{st[k]} & fresh {f(Q['joint'][k, 0])}, {st[k]} & stale {f(Q['joint'][k, 1])}" for k in range(K)) + "."]
    if cond in ("KS2", "KS2v", "KS3"):
        if Q["pred"]:
            L.append("- Probabilities of the next report of each tool you can afford: " + "; ".join(
                f"{T[j]['name']}: " + ", ".join(f"'{T[j]['values'][o]}' {f(p)}" for o, p in enumerate(Q["pred"][j]))
                for j in tool_order(ep) if j in Q["pred"]) + ".")
        else:
            L.append("- No tool is affordable any more.")
    if cond in ("KS2v", "KS3"):
        vals = Q["myopic"] if cond == "KS2v" else Q["exact"]
        head = ("- Expected final score (tool costs from now on + verdict loss; lower is better) of each action, looking ONE "
                "step ahead (a tool call is followed by the best immediate verdict): " if cond == "KS2v" else
                "- Expected final score (tool costs from now on + verdict loss; lower is better) of each action when every "
                "later action is chosen optimally (exact): ")
        opts = st + ["escalate"]
        L.append(head + "; ".join([f"submit {opts[a]} {f(vals[('stop', a)])}" for a in range(len(opts))]
                                  + [f"call {T[j]['name']} {f(vals[('probe', j)])}" for j in tool_order(ep) if ("probe", j) in vals]) + ".")
    return "\n".join(L)

def tool_schemas(ep):
    sc = SCENARIOS[ep["scenario"]]; out = []
    for j in tool_order(ep):
        d = sc["tools"][j]
        out.append({"type": "function", "function": {
            "name": d["name"], "description": tool_doc(ep, j),
            "parameters": {"type": "object", "properties": {sc["arg"]: {"type": "string", "description": f"{sc['noun']} name"}},
                           "required": [sc["arg"]]}}})
    out.append({"type": "function", "function": {
        "name": "submit", "description": "Submit the final verdict. Ends the session.",
        "parameters": {"type": "object", "properties": {
            "verdict": {"type": "string", "enum": sc["states"] + ["escalate"]},
            "note": {"type": "string", "description": "optional one-line justification"}},
            "required": ["verdict"]}}})
    return out

# ------------------------------------------------------------------ environment -------------------
class Env:
    """One episode under one information condition. All randomness is in ep['table'] (common random numbers)."""
    def __init__(self, ep, cond="K1", protocol="fc"):
        self.ep, self.cond, self.protocol = ep, cond, protocol
        self.sc = SCENARIOS[ep["scenario"]]; self.par = ep["params"]
        self.names = [d["name"] for d in self.sc["tools"]]
        self.budget = float(BUDGET); self.n = [0] * N_TOOLS; self.hist = []; self.log = []
        self.done = False; self.verdict = None

    def _err(self, reason, raw, msg):
        self.log.append(dict(type="invalid", reason=reason, raw=raw))
        return json.dumps({"error": reason, "message": msg, "budget_remaining": num(self.budget)})

    def call(self, name):
        if self.done: return self._err("episode_finished", name, "The session has ended.")
        if name not in self.names:
            return self._err("unknown_tool", name, f"Unknown tool. Available: {', '.join(self.names[j] for j in tool_order(self.ep))}, submit.")
        j = self.names.index(name); c = self.par["tools"][j]["cost"]
        if c > self.budget + 1e-9:
            return self._err("insufficient_budget", name,
                             f"{name} costs {c} but only {num(self.budget)} budget remains. Choose an affordable tool or submit.")
        o = self.ep["table"][j][self.n[j]]; self.n[j] += 1; self.budget -= c; self.hist.append((j, o))
        self.log.append(dict(type="call", tool=name, j=j, o=o, cost=c, budget_after=self.budget))
        return self.render(j, o)

    def submit(self, verdict):
        if self.done: return self._err("episode_finished", verdict, "The session has ended.")
        opts = self.sc["states"] + ["escalate"]
        v = str(verdict).strip().lower() if verdict is not None else ""
        if v not in opts:
            return self._err("bad_verdict", verdict, f"verdict must be one of {opts}")
        self.done = True; self.verdict = opts.index(v)
        self.log.append(dict(type="submit", verdict=v, a=self.verdict))
        return json.dumps({"ok": True, "message": "Verdict recorded. Session closed."})

    def invalid(self, reason, raw=""):
        return self._err(reason, raw, "No valid action was found in your reply.")

    def render(self, j, o):
        d = self.sc["tools"][j]
        out = {"tool": d["name"], "result": {self.sc["arg"]: self.sc["subject"], d["field"]: d["values"][o]},
               "cost": self.par["tools"][j]["cost"], "budget_remaining": num(self.budget)}
        text = json.dumps(out)
        if self.ep.get("out_style") == "raw":
            text = raw_output(self.ep, j, o, len(self.hist), self.par["tools"][j]["cost"], self.budget)
        if self.budget < min(t["cost"] for t in self.par["tools"]) - 1e-9:
            text += "\nBudget exhausted: only submit is possible now."
        if self.cond == "K4": text += "\n" + self.grouped()
        if self.cond == "K4c": text += "\n" + self.chronological()
        if self.cond in SUPPLY: text += "\n" + supply_text(self.ep, self.cond, self.hist)
        return text

    def grouped(self):
        rows, seen = [], []
        for j in range(4):
            be = backend_of(self.ep, j)
            if be in seen: continue
            seen.append(be); mem = [k for k in range(4) if backend_of(self.ep, k) == be]
            ev = [f"{self.names[k]}={self.sc['tools'][k]['values'][o]}" for k, o in self.hist if k in mem]
            rows.append(f"- {be} [{', '.join(self.names[k] for k in mem)}]: {', '.join(ev) if ev else '(none)'}")
        return "Evidence so far, grouped by data source:\n" + "\n".join(rows)

    def chronological(self):
        """K4c control: the same recap without grouping (same evidence, call order)."""
        rows = [f"- call {n + 1}: {self.names[k]}={self.sc['tools'][k]['values'][o]}" for n, (k, o) in enumerate(self.hist)]
        return "Evidence so far, in the order received:\n" + "\n".join(rows)

    def observation_message(self, text):
        return text if self.protocol == "fc" else "Observation: " + text

def tool_order(ep):
    """Order in which the four tools are listed to the agent (prompt, cost line, function schemas). Tool indices in
    code, logs and tables are always the canonical ones; only episodes with an 'order' field (Y experiment) differ."""
    return ep.get("order") or range(N_TOOLS)

def register_surface(ep):
    """Y experiment: an episode with 'tool_names' gets its own scenario entry (key ep['scenario'], a copy of
    ep['base_scenario'] with the tool names replaced), so every SCENARIOS[ep['scenario']] lookup sees the new names."""
    if ep.get("tool_names") and ep["scenario"] not in SCENARIOS:
        sc = dict(SCENARIOS[ep["base_scenario"]])
        sc["tools"] = [dict(t, name=n) for t, n in zip(sc["tools"], ep["tool_names"])]
        SCENARIOS[ep["scenario"]] = sc
    return ep

def load_episodes(path=os.path.join(HERE, os.environ.get("RAEA_EPISODES", "episodes.jsonl"))):
    return {e["id"]: register_surface(e) for e in map(json.loads, open(path, encoding="utf-8"))}

def load_truth(path=os.path.join(HERE, os.environ.get("RAEA_TRUTH", "ground_truth.jsonl"))):
    return {e["id"]: e for e in map(json.loads, open(path, encoding="utf-8"))}
