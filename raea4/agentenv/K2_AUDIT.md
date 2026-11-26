# K2 identifiability audit

Question (reviewer, round 2): K1 -> K2 changes several things at once. Can the observation model be uniquely recovered from the K2
prompt, and what exactly does K1 add? Source: `env.py` (`system_prompt`, `tool_doc`, `dependence_text`), checked by
`selftest_k2decomp.py` (the tool-doc block is byte-identical in K1, K2, K2a, K1e, K2e, K2p and its numbers equal the episode
parameters in all 300 episodes) and `k2_audit_readings.py` (numbers below).

## 1. What the K2 prompt contains

K2 = common header + tool-doc block, nothing else. (K1 = the same plus a ~250-token "Observation model (exact)" block; measured base
prompts K1 1,656 vs K2 1,404 tokens.)

Header (all conditions): prior over the three statuses, "the true status does not change during this session", losses, tool costs,
budget. Tool docs, per tool: output field and values, cost, and

| tool kind | K2 docs state |
|---|---|
| replica tool (A; also B in SHARED / MIXED / VALUE) | backend name ("Served from read replica `X`"); "Replica staleness incidents: about p% of sessions; a stale replica keeps returning `<first value>` (about 97% of calls)"; "Accuracy when the replica is fresh: q per call, errors spread evenly over the other two values". phi = 0 twins: "none (replica verified fresh for this session)". VALUE: "...keeps returning the value frozen when it stopped refreshing (about 97% of calls; the frozen value is `<v>` in about 80% of incidents)" |
| primary DB (B in PERSIST) | backend name, "never stale", accuracy, "errors are independent across calls and spread evenly" |
| log store, live probe | backend name, "never stale", accuracy, same independence sentence |

## 2. Parameter-by-parameter audit

(phi is rounded to 2 decimals in the generator, so "about p%" is the exact value; the word "about" is still in the text.)

| item of the true observation model | K2 | K1 block | K2a (new) |
|---|---|---|---|
| which tools are on which backend | yes, by backend-name strings | yes, explicit sentence | by names only (as K2) |
| **tools on one replica share ONE stale flag** (perfectly correlated staleness across tools) | **not stated.** Only suggested by the identical replica name and the phrase "a stale replica"; a reader can equally take "about p% of sessions" as a per-tool incidence | "Staleness is shared by both tools" | the rule "staleness is a property of the replica and affects every call to every tool served from that replica" is stated; WHICH tools are served from it is not (names only) |
| incidence phi | yes ("about p% of sessions") | exact | exact |
| staleness lasts the whole session / persists across retries | implied ("of sessions", "keeps returning"); never stated | stated | stated |
| stale emission, main mass (0.97 on status 0) | yes ("about 97% of calls", first value) | yes | yes |
| **stale emission, leftover mass** (0.015 on each other status, independent of the true status: "whatever the true status") | **not stated** | stated | stated |
| VALUE only: frozen-value distribution (0.8 / 0.1 / 0.1) | only "about 80%" for the first value; the split of the other 20% is not stated | full | full |
| fresh accuracy q, errors uniform over the other two statuses | yes | repeated | repeated |
| **errors of fresh replica calls independent across calls** | **not stated for replica tools** (stated only for primary / log / probe tools) | stated | stated |
| **distinct data sources fail independently; staleness independent of the true status** | **not stated** | stated | stated |
| prior, costs, losses, budget, accuracy of primary / log / probe | yes | repeated | repeated |

Parameters absent from K2: (i) cross-tool sharing of the stale flag, (ii) the leftover 3% of the stale emission and its
independence of the true status, (iii) independence of fresh replica calls, (iv) independence across sources and between staleness and
the status, (v) explicit persistence, (vi) VALUE: the split of the remaining 20% of frozen values. Everything numeric about phi, the
main stale mass and every accuracy is present.

**Verdict (a).** The K2 text does not determine the observation model uniquely. It determines it under four conventions
(name equality = shared fault; leftover stale mass uniform and state independent; fresh calls independent; sources independent). So not
every K2 error can be booked as "failed identification": part of it is a defensible alternative reading of an under-specified text.
Conversely, the numbers are all there, so K2 is not "information removed" either, except for the items above.

## 3. How much can the unresolved readings cost? (exact, `results/k2_audit_readings.txt`)

For each alternative reading consistent with the K2 text, the EXACT planner of that reading acts and is evaluated exactly under the true
model (regret vs the oracle; first 50 episodes per family):

| reading | PERSIST | SHARED | MIXED | TRANSIENT | TRANSIENT_P | CORE (4 fam.) |
|---|---|---|---|---|---|---|
| R0 true (name = sharing) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R1 each tool has its own stale flag | 0.00 | 1.32 | 0.76 | 0.00 | 0.00 | 0.52 |
| R2 incidence per call, no session persistence (= MMI-DP) | 5.40 | 6.12 | 3.88 | 0.00 | 0.00 | 3.85 |
| R3 leftover stale mass on ONE other status | 0.05 | 0.04 | 0.05 | 0.00 | 0.00 | 0.04 |
| R4 incidence ignored (= FaultUnaware-DP) | 7.39 | 5.10 | 6.39 | 0.00 | 0.00 | 4.72 |
| R5 true model, never repeat a cache call (= Forbid-cache) | 0.28 | 0.34 | 0.19 | 1.32 | 1.03 | 0.53 |

(Rows R2, R4, R5 reproduce the README baselines MMI-DP, FaultUnaware-DP and Forbid-cache exactly, a consistency check.)

Reading the table against the terra result (K2 - K1 regret, text protocol: CORE +0.65; SHARED +0.90, PERSIST +0.37, MIXED +0.07,
TRANSIENT +1.24, TRANSIENT_P +0.94):

1. **Sharing ambiguity alone is bounded**: a perfect planner that resolves "names = sharing" the wrong way loses 1.32 (SHARED), 0.76
   (MIXED), 0 (PERSIST), 0.52 on the four-family CORE average. The observed CORE K2-K1 gap (+0.65) is of that order, but the
   per-family pattern does not match: MIXED (R1 = 0.76) shows +0.07, while the twins (R1 = 0) show the largest gaps.
2. **In the phi = 0 twins no identification question exists.** K2 states "none (replica verified fresh)" for every replica tool, so
   phi, persistence, emission and sharing are all irrelevant and fully given; readings R1-R4 give regret 0.00. The only thing K1 adds
   there is the sentence that every call's error is independent of every other call, plus the organisation (a 250-token summary).
   The K2-K1 gap in the twins (+1.24, +0.94) therefore is not a failure to identify sharing, phi or persistence (all given or irrelevant); it may still reflect the unstated independence of fresh replica calls (K2 states independence only for primary, log and probe tools), see 3.
3. What the twin gap does match numerically is R5, a "repeats of a cache tool are redundant" prior: Forbid-cache regret 1.32 / 1.03 vs
   observed K2 regret 1.27 / 0.97, with terra's repeat cache calls dropping from 0.88 to 0.12 (TRANSIENT) and from 1.32 to 0.00
   (TRANSIENT_P). This is consistent with, not proof of, the agent applying a default "do not re-query a cache" rule when the explicit
   independence statement is absent. It is a prior about replica tools, not an inability to recover phi or the sharing.

## 4. Consequence for the claim in the paper

"A failure of model identification" is not supported by K1 -> K2 alone (the contrast changes explicitness of the sharing, the
presence of persistence / leftover-emission / independence statements, and the organisation of the information). The supportable
statement is the reviewer's: "consistent with difficulty recovering the observation model from documentation". The new conditions
split it (see `K2_DECOMP.md` for the design and the decision rules):

- K2a - K2: sharing mapping inferred from names, all other statements given. In the twins K2a carries the same content as K1, so
  K2a - K2 there is the pure "explicit model block / independence statement" effect.
- K2a - K1: cost of inferring the mapping when everything else is explicit (0 by construction in the twins, up to wording).
- K2e: does the agent write down the right structure (recovery accuracy)? K2p: given the agent's own recovered model, does exact
  planning remove the regret (identification vs planning)?
