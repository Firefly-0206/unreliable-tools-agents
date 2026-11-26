Given the first Tool A “healthy” report, integrating over the 0.3 stale-backend probability gives:

- P(healthy) ≈ 0.7386
- P(degraded) ≈ 0.1307
- P(down) ≈ 0.1307

Immediate escalation costs 4, while declaring healthy has expected cost ≈ 7.84. Querying Tool A again has the lowest expected total cost under the remaining budget and persistence of staleness.

{"p_healthy": 0.7386, "action": "query_A"}