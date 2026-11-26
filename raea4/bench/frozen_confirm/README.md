# Planned version of bench/run.py (held-out confirmation)

`run_planned.py` is a byte-identical copy of `bench/run.py` as it stood when the analysis plan
(`PLAN_confirm.md`) was written: its SHA-256 starts with 3686FB768A703CF1, the prefix listed in the plan.
It was recovered on 2026-11-06 from Claude Code's per-session file-edit history
(`~/.claude/file-history/<session>/491c0c361d900d2b@v3`, last written 2026-11-02 local time; four session
folders hold identical copies).

`bench/run.py` was edited on 2026-11-04 (after the flash and terra confirmation records were last written on
2026-11-02) to add the `--provider claude` route used for Sonnet 5.5. `run_planned_to_current.diff` is the full
diff: an import, a `claude_call` function, one dispatch branch, a metadata line, the `claude` choice and a
`--max-waits` option, and a changed default output file name that applies only when `--out` is not given (both
confirmation runs passed `--out`). The deepseek and codex code paths are unchanged.
`paper/scripts/make_confirm_tables.py` asserts the hash of `run_planned.py` and that this diff still describes the
current `run.py`.

The plan file was written on 2026-11-02 under the name `PREREG_confirm.md` and renamed to `PLAN_confirm.md` on 2026-11-17, with its content unchanged (SHA-256 prefix 035B826A78516AD6). It is an analysis plan fixed locally before the runs, not a third-party preregistration; its heading and the frozen script `bench/analyze_confirm.py` (hash listed in the plan, so left untouched) still use the old wording and file name.
