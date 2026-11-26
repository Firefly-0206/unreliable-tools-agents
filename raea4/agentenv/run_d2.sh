#!/usr/bin/env bash
# Second task domain; PLAN_d2_domain.md. Resumable: rerun the same command.
#   bash run_d2.sh flash|terra|sonnet [extra run.py arguments, e.g. --ids DN-000 DX10-100]
set -u
cd "$(dirname "$0")"
PY=<root>/.venv/Scripts/python.exe
case "$1" in
  flash)  MODEL=deepseek-chat;                 PROTO=fc;   STEM=ds_chat_fc ;;
  terra)  MODEL=codex:gpt-6.1-terra@medium;    PROTO=text; STEM=codex_terra ;;
  sonnet) MODEL=claude:claude-sonnet-5-5@high; PROTO=text; STEM=claude_s55 ;;
  *) echo "usage: run_d2.sh flash|terra|sonnet"; exit 2 ;;
esac
shift
IDS=$($PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('episodes_d2.jsonl')))")
OUT=logs/d2_${STEM}_k2e.jsonl
export RAEA_DOMAIN=2 RAEA_EPISODES=episodes_d2.jsonl RAEA_TRUTH=ground_truth_d2.jsonl
ARGS="--models $MODEL --protocol $PROTO --conditions K2e --workers 4 --out $OUT"
if [ $# -gt 0 ]; then ARGS="$ARGS $*"; else ARGS="$ARGS --ids $IDS"; fi
if [ "$PROTO" = fc ]; then
  $PY ../offpeak_runner.py --out $OUT -- $PY run.py $ARGS >> logs/d2_${STEM}_k2e.log 2>&1
else
  $PY run.py $ARGS >> logs/d2_${STEM}_k2e.log 2>&1
fi
echo "exit $?"
