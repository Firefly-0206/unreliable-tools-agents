# Hard description (rates in words, not numbers): analysis plan fixed before the runs

Status: FROZEN 2026-11-21 before the first model call (hashes in `frozen_hd_hard_description.sha256`). Code, episodes and the analysis
exist and were run on mock logs. An independent code review (3 lenses, 2 refuters per finding) confirmed 5
defects in `hd_analysis.py`, all fixed:
- the HQ/HI tolerance filter was looser than HN's;
- a crash without the Bayes file;
- a false reading when HN is absent;
- the Bayes column was shown for HN;
- H3 included JSONs filled by defaults.

HQ also had a persistence sentence that HN does not have. The reviewers did not count this as a defect, but HQ now
uses HN's sentences, so HQ − HN differs in words only. These are local timestamps and hashes, not a third-party preregistration. Wording in the paper: "analysis plan fixed
before the runs", never "preregistered".

## Why

In every earlier condition the documentation states each rate as a number. Describing the tools is therefore
transcription, and the description loss D is close to 0 (on the strict same-JSON subset the planner's regret is
exactly 0). A reviewer can then call the replay trivial, and ask whether the executor gap would shrink or change sign
once describing takes judgement. This experiment makes describing take judgement, and asks:

1. Does describing now cost something? That is, is D > 0?
2. Does the executor gap (K2r − K2e < 0) remain when the model's own description is imprecise?
3. Does the under-use remain when it is measured against the planner on the model's OWN description? That planner
   acts on exactly the numbers the model wrote, so how the model reads the words cannot explain an under-use measured
   against it.

## Episodes (`make_episodes_hd.py` -> `episodes_hd.jsonl`, `ground_truth_hd.jsonl`; seed 2035)

These draws have never been shown to a model.

- **Draws:** 50 parameter draws in the first domain, PERSIST tools (the X sweep's persistent arm):
  - replica tool 0, cost 1;
  - primary-database tool 1, cost 2;
  - log tool 2, cost 1;
  - live probe 3, cost 4.

  Persistent benign-stuck fault on the replica, stale emission 0.97. Prior as in PERSIST, stated as numbers. Losses
  and budget are unchanged.
- **Fault-rate ranges:** the fault rate is uniform inside the range of its word (`env.HD_PHI_BINS`), with
  17 / 17 / 16 draws per word:
  - **rare** [0.05, 0.15];
  - **occasional** [0.20, 0.40];
  - **frequent** [0.60, 0.80].
- **Accuracy ranges:** each tool's accuracy is uniform inside the range of its word (`env.HD_ACC_BINS`):
  - **highly reliable** [0.90, 0.96];
  - **usually right** [0.78, 0.88];
  - **noisy** [0.62, 0.74].

  Replica and probe take highly reliable or usually right. Primary and logs take usually right or noisy. All values
  are rounded to 2 decimals.
- **Selection (as in X, whose oracle opens with the replica in 100% of draws at 0.10 and 0.30 and in 0% at 0.70):**
  - accuracies and the fault rate are redrawn, with the prior kept, until the exact oracle opens with the replica
    (rare and occasional draws) or does not (frequent draws);
  - tries per accepted draw: rare mean 1.5 (max 3), occasional 2.6 (max 5), frequent 1.0;
  - without this selection the oracle would open with the replica in only 65% and 35% of rare and occasional draws,
    which leaves too few decisions where querying the replica is right.
- **Shared worlds:** hidden status, fault flag and every tool output are pre-drawn per draw and shared by all arms.

## Arms (same draws, same pre-drawn outputs)

- **HN (numbers, control):** today's first-domain documentation ("about 30% of sessions", "0.84 per call").
- **HQ (words):** HN's sentences with every rate replaced by its word, with no range given:

  > Replica staleness incidents: occasional (counted per on-call session); a stale replica keeps returning 'ok' (on
  > almost every call). Accuracy when the replica is fresh: usually right.

  Other tools read "Accuracy: noisy; errors are independent across calls ...". Backend names, costs, outputs,
  prior, losses and sentence order are as in HN, so HQ − HN is a contrast of words against numbers alone. System
  prompt: 616 words against 620.
- **HI (words, persistence left to inference):** as HQ, except the replica sentence:

  > The replica is refreshed by a background job. On-call sessions during which this job is halted: occasional. While
  > the job is halted the replica serves its last snapshot, which returns 'ok' on almost every call. A halted job is
  > restarted only between on-call sessions.

  Persistence then follows from the last sentence instead of being stated.

Checked by `make_episodes_hd.py` (assertions):

- the arms share parameters and outputs;
- HQ/HI tool entries contain no percent, decimal or "probability", and their other digits are those of the HN entry;
- the words read back by `k2decomp.parse_docs` give the true word of every rate;
- the HN entries read back give the true numbers;
- HN's staleness sentence (in words) appears in HQ only, and the restart sentence in HI only.

Every system prompt and tool schema of every earlier first-domain episode file hashes identically before and after
the changes to `env.py` and `k2decomp.py` (1,225 episodes).

When a JSON leaves a field out in HQ/HI, the parser fills it with the middle of the word's range and records the fill
as a default. Such an episode is then not "as written", so it is excluded from the within-range subset.

## Runs (`run_hd.sh`)

Condition K2e: the model writes its observation model as JSON, then acts. One sample per episode. The models and
settings are those of the R and M runs:

- `deepseek-chat` (fc, `offpeak_runner.py`);
- `codex:gpt-6.1-terra@medium`;
- `claude:claude-sonnet-5-5@high`.

Default arms are HN + HQ, 100 episodes per model; `bash run_hd.sh <model> HN HQ HI` adds HI. Rows with `api_error`
are rerun by reissuing the command; nothing else is removed. Every cell and run is reported.

## Executors and references (all in the same world, all scored under the true model)

- **model:** the LLM's K2e trajectory.
- **planner:** exact DP on the model's written JSON (K2r).
- **oracle:** exact DP on the true model.
- **Bayes reference (`hd_bayes.py`, offline):** exact DP that reads every word as its range, uniform, on a grid of 5
  fault-rate points and 3 points per accuracy.
  - Its self-test: with a one-point grid at the true rates it reproduces the oracle's value, first action and
    zero regret in all 50 draws.
  - Result, computed already (no model involved): regret 0.13, opens with the replica in 64% of draws (oracle 68%).
  - A deterministic reader that writes each word's middle value gets planner regret 0.11 (mock check). So reading
    words well costs about 0.1 here.

## Analysis (`hd_analysis.py`; run on two mock logs before freezing)

Nested bootstrap over scenarios, then draws, 95%. Arm differences are paired by draw. H4's difference between word
ranges resamples scenarios, then draws within each range. "Low" means rare and occasional draws.

- **Within tolerance:** for HN, the strict flags (JSON as written, every parameter within 0.02). For HQ/HI, every
  one of these must hold:
  - the JSON is used as written;
  - the structure is right, persistence is right, and no non-replica tool is given a fault (as in the HN filter);
  - the fault rate and every accuracy lie inside the range of their word;
  - the stale emission on the stuck value is ≥ 0.90.
- **H0 (reading the words):** per word, the mean value written and the share inside the range. Reported only, with
  no reading attached.
- **H1 (description loss):** planner regret (K2r) per arm, and HQ − HN and HI − HN.
- **H2 (executor gap):** K2r − K2e per arm, over all episodes and within tolerance.
- **H3 (under-use against the model's own description), low draws:**
  - model minus planner in the share of first calls to the replica;
  - the own-plan pass-over: the share of the model's decisions at which the planner on its JSON, at the belief that
    JSON implies after the same reports, would query the replica and the model does not.

  Both are also computed on the low draws whose JSON is used as written (no fallback, filled default, clip or parser
  note): only for those is the planner exactly the model's own description, so this subset carries the reading. Also
  reported against the oracle and, for HQ/HI only, the Bayes reference.
- **H4 (tracks the rate):** first-call replica share at rare minus at frequent draws, for the model and the planner.
- **H5 (words vs numbers), exploratory with no direction predicted:** HQ − HN (and HI − HN) in K2e regret, in K2r −
  K2e, in the H3 first-call gap and in the model's first-call share.

Readings fixed now, per model. An interval that includes 0 is "not detected", never equivalence or absence. The
script prints each reading:

- *Describing now costs something*: HQ planner regret above 0.05, and the HQ − HN interval above 0. If not, we say
  that word-level documentation still let this model describe the task almost exactly. This reading needs the HN arm;
  without it, no reading is made. For scale: a reader that writes each word's middle value already gets 0.11, so
  this reading is about the setup, not about the model's skill.
- *Executor gap under imprecise descriptions*: the H2 interval over all HQ episodes lies below 0. Within tolerance it
  is read only if at least 20 episodes qualify, and otherwise reported without a reading.
- *Under-use is not a misreading of the rates*: on the as-written low draws, the H3 first-call interval (model −
  planner on the own JSON) lies below 0. This is read only if at least 20 episodes qualify, and otherwise reported
  without a reading. The value over all low draws is reported beside it.
- *Tracks the rate*: the H4 interval for the model lies above 0.
- HI gets the same readings. Findings that do not hold are reported in the main text.

## Placement in the paper

The analysis goes in a new appendix section. The 9-page main text has no slack, so it changes only by replacement:

- the Limitations sentence "Describing here is transcribing documentation that states every rate ..." becomes one
  sentence on the HQ result;
- tab:chain may gain a clause.

## Cost and quota (measured on the R and M logs)

- **flash:** about 8–12k prompt tokens (about half of them cache hits) and 0.5k completion tokens per K2e episode.
  That is well under 1 CNY for 100–150 episodes off-peak.
- **terra:** the coding plan. M's 100-episode K2e run took about one evening.
- **Sonnet:** the user's Claude quota, shared with this session. The 75-episode R run together with my session used
  up a 5-hour window, so 100 episodes take about 1.5 windows and 150 about 2. Run it while this session is idle.

## Decisions (user, 2026-11-21)

- **Arms:** HN + HQ for all three models; HI for flash and terra only (150 / 150 / 100 episodes). The paper says that
  Sonnet was not run on HI. HI readings are therefore reported for flash and terra only.
- The plan is frozen (`frozen_hd_hard_description.sha256`) before the first model call.

## Not covered

The second and third domains, independent faults, documentation that is wrong rather than vague, words whose ranges
differ from the generator's, live services and repeated generation.
