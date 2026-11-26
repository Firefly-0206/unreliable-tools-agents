#!/usr/bin/env bash
# Accuracy sweep in which the optimal first query flips; PLAN_w_flip.md. Resumable: rerun the same command.
#   bash run_w.sh flash|terra|sonnet
set -u
cd "$(dirname "$0")"
PY=<root>/.venv/Scripts/python.exe
case "$1" in
  flash)  MODEL=deepseek-chat;                 PROTO=fc;   STEM=ds_chat_fc ;;
  terra)  MODEL=codex:gpt-6.1-terra@medium;    PROTO=text; STEM=codex_terra ;;
  sonnet) MODEL=claude:claude-sonnet-5-5@high; PROTO=text; STEM=claude_s55 ;;
  *) echo "usage: run_w.sh flash|terra|sonnet"; exit 2 ;;
esac
IDS=$($PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('episodes_w.jsonl')))")
OUT=logs/w_${STEM}_k2e.jsonl
export RAEA_EPISODES=episodes_w.jsonl RAEA_TRUTH=ground_truth_w.jsonl
if [ "$PROTO" = fc ]; then
  $PY ../offpeak_runner.py --out $OUT -- $PY run.py --models $MODEL --protocol $PROTO --conditions K2e --workers 4 --out $OUT --ids $IDS >> logs/w_${STEM}_k2e.log 2>&1
else
  $PY run.py --models $MODEL --protocol $PROTO --conditions K2e --workers 4 --out $OUT --ids $IDS >> logs/w_${STEM}_k2e.log 2>&1
fi
echo "exit $?"
