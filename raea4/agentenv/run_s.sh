#!/usr/bin/env bash
# Layered supply of exact quantities inside the closed loop; PLAN_s_supply.md. Resumable: rerun the same command.
#   bash run_s.sh flash|terra|sonnet
set -u
cd "$(dirname "$0")"
PY=<root>/.venv/Scripts/python.exe
case "$1" in
  flash)  MODEL=deepseek-chat;                 PROTO=fc;   STEM=ds_chat_fc ;;
  terra)  MODEL=codex:gpt-6.1-terra@medium;    PROTO=text; STEM=codex_terra ;;
  sonnet) MODEL=claude:claude-sonnet-5-5@high; PROTO=text; STEM=claude_s55 ;;
  *) echo "usage: run_s.sh flash|terra|sonnet"; exit 2 ;;
esac
IDS=$($PY -c "print(' '.join(f'XP{p}-{k:03d}' for p in (10, 30) for k in range(25)))")
OUT=logs/s_${STEM}.jsonl
CONDS="K1 KC KS1 KS2 KS2v KS3"
export RAEA_EPISODES=episodes_x.jsonl RAEA_TRUTH=ground_truth_x.jsonl
if [ "$1" = flash ]; then
  $PY ../offpeak_runner.py --out $OUT -- $PY run.py --models $MODEL --protocol $PROTO --conditions $CONDS --workers 4 --out $OUT --ids $IDS >> logs/s_${STEM}.log 2>&1
else
  $PY run.py --models $MODEL --protocol $PROTO --conditions $CONDS --workers 4 --out $OUT --ids $IDS >> logs/s_${STEM}.log 2>&1
fi
echo "exit $?"
