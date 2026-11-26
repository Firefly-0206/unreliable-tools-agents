# D3 run notes (deviations and events after the freeze; frozen_d3_domain.sha256 is not changed)

- 2026-11-16, terra pilot (DN-000, DX10-100): the pilot was launched twice by mistake (a background launch that appeared
  to have died was still running), so both episodes were run twice concurrently. The two rows written first stay in
  `logs/d3_codex_terra_k2e.jsonl` and count as the pilot; the two later rows were moved unchanged to
  `logs/d3_codex_terra_k2e_duplicates.jsonl` and are not analysed. Both copies of each episode have the same JSON
  structure (outlets in one faulty group, aggregator alone and faulty) and neither calls an outlet.
- 2026-11-16/17: terra and Sonnet runs complete (200/200 submitted each, no api errors, no duplicates). Analysis
  `python d3_analysis.py --runs terra|sonnet --workers 4` run per model (memory); per-run outputs in
  `results/d3_analysis_{terra,sonnet}.{json,txt}`, merged into `results/d3_analysis.{json,txt}`.
  Readings: R1 executor gap replicates for both; R2 under-use replicates for both; R3 tracking not detected for either
  (terra +4 [0, +17] pp; Sonnet never opens with an outlet at any fault rate). Post hoc (not a planned reading): outlets
  are almost never called at any step (0.00-0.16 calls per episode); both models open with the sub-question tools.
- 2026-11-17: flash run complete (off-peak, about CNY 0.95; 199 submitted, 1 turn cap, scored as the worst verdict per
  the frozen rules). R1 replicates (-2.92 [-3.71, -2.30]); R2 replicates (-98 [-100, -92] pp); R3 not detected (+0).
  Unlike terra and Sonnet, 39% of its DN regret is at passed-over outlets, 16% verdict, 11% passed-over sub-question
  tools; it calls an outlet 0.04-0.56 times per episode, but almost never first. Merged: results/d3_analysis.{json,txt}.
- 2026-11-17: the frozen hashes were re-verified only AFTER the three analyses (all files unchanged, `sha256sum -c
  frozen_d3_domain.sha256` all OK), not before them as in the earlier experiments; stated in the paper's D3 appendix.
