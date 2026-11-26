#!/usr/bin/env bash
# Surface randomisation (part A) and repeated generation (part B); PLAN_y_surface.md. Resumable: rerun the same command.
#   bash run_y.sh flash|terra|sonnet
set -u
cd "$(dirname "$0")"
PY=<root>/.venv/Scripts/python.exe
case "$1" in
  flash)  MODEL=deepseek-chat;                 PROTO=fc;   STEM=ds_chat_fc ;;
  terra)  MODEL=codex:gpt-6.1-terra@medium;    PROTO=text; STEM=codex_terra ;;
  sonnet) MODEL=claude:claude-sonnet-5-5@high; PROTO=text; STEM=claude_s55 ;;
  *) echo "usage: run_y.sh flash|terra|sonnet"; exit 2 ;;
esac
YIDS=$($PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('episodes_y.jsonl')))")
BIDS=$($PY -c "print(' '.join(f'XP{p}-{k:03d}' for p in (10, 30) for k in range(10)))")
WRAP=""
run() {  # $1 episodes file, $2 truth file, $3 out stem, rest: run.py arguments
  local E=$1 T=$2 OUT=logs/$3.jsonl; shift 3
  if [ "$PROTO" = fc ]; then
    RAEA_EPISODES=$E RAEA_TRUTH=$T $PY ../offpeak_runner.py --out $OUT -- $PY run.py --models $MODEL --protocol $PROTO --conditions K2e --workers 4 --out $OUT "$@"
  else
    RAEA_EPISODES=$E RAEA_TRUTH=$T $PY run.py --models $MODEL --protocol $PROTO --conditions K2e --workers 4 --out $OUT "$@"
  fi
}
run episodes_y.jsonl ground_truth_y.jsonl y_${STEM}_k2e --ids $YIDS >> logs/y_${STEM}_k2e.log 2>&1; echo "part A exit $?"
run episodes_x.jsonl ground_truth_x.jsonl xrep_${STEM}_k2e --ids $BIDS --samples 5 --sample-start 1 >> logs/xrep_${STEM}_k2e.log 2>&1; echo "part B exit $?"
