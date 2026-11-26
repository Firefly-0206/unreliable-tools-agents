"""Second task domain (RAEA_DOMAIN=2): five statuses, five tools, two independent fault sources of different kinds.

Tool order (indices used in code, logs and tables):
  0 A  status tool on a read replica      cheap, accurate; replica may be STALE for the session -> reports the benign status
  1 B  status tool on an aggregation feed cheap, medium;   feed may be DOWN for the session    -> reports an arbitrary status
  2 L  log sampler                        cheap, noisy;    never faulty
  3 P  status tool on the primary store   cost 2, medium;  never faulty
  4 X  live probe                         cost 3, accurate; never faulty
The replica and the feed are different backends that fail independently. Budget 7.
"""
K2, BUDGET2, N_TOOLS2 = 5, 7, 5
SRC2 = ["cache", "feed", "logs", "primary", "live"]
FAMILY_SPEC2 = {   # (source, accuracy range, cost) per tool; phi: replica staleness; phi2: feed outage
    "D2": dict(tools=[("cache", (.82, .94), 1), ("feed", (.72, .86), 1), ("logs", (.50, .62), 1), ("primary", (.70, .80), 2), ("live", (.88, .95), 3)],
               phi=(.15, .40), phi2=(.15, .35), stuck="benign"),
}
PRIOR_ALPHA2 = [5, 2, 2, 1.5, 1.5]; PRIOR_MIN2 = 0.04


def _sc(role, arg, subject, noun, question, states, sev_text, backends, tools):
    return dict(role=role, arg=arg, subject=subject, noun=noun, question=question, states=states, sev_text=sev_text,
                backends=dict(zip(SRC2, backends)),
                tools=[dict(name=n, field=f, values=(v or states), what=w) for n, f, v, w in tools])


def _map(vals, states):
    return ", ".join(f"{v} = {s}" for v, s in zip(vals, states))


def _mk(role, arg, subject, noun, question, states, sev_text, backends, names, what, logv, probev):
    return _sc(role, arg, subject, noun, question, states, sev_text, backends,
               [(names[0], "status", None, what[0]), (names[1], "status", None, what[1]),
                (names[2], "summary", logv, f"{what[2]} ({_map(logv, states)})."),
                (names[3], "status", None, what[3]),
                (names[4], "result", probev, f"{what[4]} ({_map(probev, states)}).")])


SCENARIOS2 = {
    "service": _mk(
        "an on-call SRE agent", "service", "checkout-api", "service",
        "whether the production service `checkout-api` is healthy, slow, erroring, overloaded or down right now",
        ["healthy", "slow", "erroring", "overloaded", "down"],
        "declaring \"healthy\" when the service is actually slow, erroring, overloaded or down (an incident goes unhandled)",
        ["metrics-cache-eu1", "statusfeed-agg-eu1", "logstore-eu1", "statusdb-primary", "the live service endpoint"],
        ["health_check", "status_feed", "log_search", "status_registry", "synthetic_probe"],
        ["Returns the latest recorded health status of a service.", "Returns the status of a service as published on the aggregated status feed.",
         "Samples the last 5 minutes of application logs and summarises them", "Reads the status of a service from the status registry.",
         "Runs an end-to-end synthetic transaction against the live service"],
        ["no_anomalies", "latency_warnings", "error_bursts", "queue_saturation", "connection_refused"],
        ["pass", "pass_slow", "http_errors", "throttled", "no_response"]),
    "pipeline": _mk(
        "a data-platform on-call agent", "job", "nightly-orders-etl", "job",
        "whether last night's run of the ETL job `nightly-orders-etl` succeeded, finished late, was partial, used stale inputs or failed",
        ["succeeded", "late", "partial", "stale_inputs", "failed"],
        "declaring \"succeeded\" when the run was actually late, partial, on stale inputs or failed (bad data ships downstream)",
        ["scheduler-cache-us2", "lineage-feed-us2", "runlog-archive", "scheduler-db-primary", "the warehouse tables"],
        ["job_status", "lineage_feed", "run_log_grep", "scheduler_record", "validate_output"],
        ["Returns the recorded final state of a job run.", "Returns the state of a job run as published on the lineage feed.",
         "Samples the job's run logs and summarises them", "Reads the final state of a job run from the scheduler database.",
         "Runs validation queries against the warehouse output tables"],
        ["clean", "sla_warnings", "row_rejects", "old_partition_reads", "fatal_errors"],
        ["rows_complete", "rows_complete_late", "rows_missing", "rows_outdated", "table_empty"]),
    "backup": _mk(
        "a database-reliability on-call agent", "database", "crm-db", "database",
        "whether last night's backup of the database `crm-db` is verified, incomplete, outdated, unencrypted or corrupt",
        ["verified", "incomplete", "outdated", "unencrypted", "corrupt"],
        "declaring \"verified\" when the backup is actually incomplete, outdated, unencrypted or corrupt (a restore would fail or leak)",
        ["backup-catalog-cache", "compliance-feed-agg", "backup-agent-logs", "backup-catalog-primary", "a scratch restore host"],
        ["backup_status", "compliance_feed", "agent_log_scan", "catalog_record", "restore_test"],
        ["Returns the recorded state of the latest backup.", "Returns the state of the latest backup as published on the compliance feed.",
         "Samples the backup agent's logs and summarises them", "Reads the state of the latest backup from the backup catalog.",
         "Restores a sample of the backup on a scratch host"],
        ["no_errors", "skipped_files", "old_snapshot_used", "plaintext_warnings", "checksum_errors"],
        ["restore_ok", "restore_partial", "restore_old_data", "restore_unencrypted", "restore_failed"]),
    "dns": _mk(
        "a network on-call agent", "domain", "api.shop.example", "domain",
        "whether the domain `api.shop.example` is resolving, resolving slowly, intermittent, misrouted or failing right now",
        ["resolving", "slow", "intermittent", "misrouted", "failing"],
        "declaring \"resolving\" when resolution is actually slow, intermittent, misrouted or failing (customers cannot connect)",
        ["resolver-cache-ap1", "netstatus-feed-ap1", "query-log-store", "dns-zone-primary", "the authoritative name servers"],
        ["dns_status", "netstatus_feed", "query_log_sample", "zone_record", "dig_probe"],
        ["Returns the latest recorded resolution status of a domain.", "Returns the resolution status of a domain as published on the network status feed.",
         "Samples the last 5 minutes of resolver query logs and summarises them", "Reads the resolution status of a domain from the zone database.",
         "Queries the authoritative name servers directly from several vantage points"],
        ["all_noerror", "high_latency", "some_servfail", "unexpected_answers", "mostly_servfail"],
        ["answer", "answer_slow", "timeout_some", "wrong_address", "no_answer"]),
    "replication": _mk(
        "a database on-call agent", "replica", "orders-replica-2", "replica",
        "whether the database replica `orders-replica-2` is in sync, lagging, stalled, diverged or broken right now",
        ["in_sync", "lagging", "stalled", "diverged", "broken"],
        "declaring \"in_sync\" when the replica is actually lagging, stalled, diverged or broken (reads return wrong data)",
        ["monitoring-cache-eu3", "topology-feed-eu3", "binlog-archive", "cluster-registry-primary", "the replica and its primary"],
        ["replication_status", "topology_feed", "binlog_scan", "registry_record", "checksum_compare"],
        ["Returns the latest recorded replication state of a replica.", "Returns the replication state of a replica as published on the topology feed.",
         "Samples the replica's binlog apply records and summarises them", "Reads the replication state of a replica from the cluster registry.",
         "Compares row checksums and positions between the replica and its primary"],
        ["applied_promptly", "apply_backlog", "apply_paused", "apply_conflicts", "apply_errors"],
        ["match", "behind", "not_advancing", "mismatch", "unreachable"]),
    "queue": _mk(
        "a messaging on-call agent", "queue", "order-events", "queue",
        "whether the message queue `order-events` is draining, slow, backlogged, poisoned or stalled right now",
        ["draining", "slow", "backlogged", "poisoned", "stalled"],
        "declaring \"draining\" when the queue is actually slow, backlogged, poisoned or stalled (orders stop being processed)",
        ["broker-metrics-cache", "ops-feed-agg", "consumer-log-store", "broker-admin-primary", "the live broker"],
        ["queue_status", "ops_feed", "consumer_log_scan", "broker_record", "canary_message"],
        ["Returns the latest recorded state of a queue.", "Returns the state of a queue as published on the operations feed.",
         "Samples the consumers' logs and summarises them", "Reads the state of a queue from the broker admin database.",
         "Publishes a canary message to the live broker and waits for it"],
        ["acks_normal", "acks_slow", "lag_growing", "repeated_redeliveries", "no_acks"],
        ["consumed", "consumed_late", "queued_long", "dead_lettered", "not_consumed"]),
}
