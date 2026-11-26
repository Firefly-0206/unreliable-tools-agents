#!/usr/bin/env bash
# Hard-description experiment; PLAN_hd_hard_description.md. Resumable: rerun the same command.
#   bash run_hd.sh flash|terra|sonnet [HN HQ HI]     (arms; default HN HQ)
set -u
cd "$(dirname "$0")"
PY=<root>/.venv/Scripts/python.exe
case "$1" in
  flash)  MODEL=deepseek-chat;                 PROTO=fc;   STEM=ds_chat_fc ;;
  terra)  MODEL=codex:gpt-6.1-terra@medium;    PROTO=text; STEM=codex_terra ;;
  sonnet) MODEL=claude:claude-sonnet-5-5@high; PROTO=text; STEM=claude_s55 ;;
  *) echo "usage: run_hd.sh flash|terra|sonnet [arms]"; exit 2 ;;
esac
shift; ARMS="${*:-HN HQ}"
IDS=$($PY -c "import json,sys;a=sys.argv[1].split();print(' '.join(e['id'] for e in map(json.loads,open('episodes_hd.jsonl')) if e['arm'] in a))" "$ARMS")
OUT=logs/hd_${STEM}_k2e.jsonl
export RAEA_EPISODES=episodes_hd.jsonl RAEA_TRUTH=ground_truth_hd.jsonl
if [ "$MODEL" = deepseek-chat ]; then
  $PY ../offpeak_runner.py --out $OUT -- $PY run.py --models $MODEL --protocol $PROTO --conditions K2e --workers 4 --out $OUT --ids $IDS >> logs/hd_${STEM}_k2e.log 2>&1
else
  $PY run.py --models $MODEL --protocol $PROTO --conditions K2e --workers 4 --out $OUT --ids $IDS >> logs/hd_${STEM}_k2e.log 2>&1
fi
echo "exit $?"
