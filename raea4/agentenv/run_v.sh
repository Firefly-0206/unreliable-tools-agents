#!/usr/bin/env bash
# Cost sweep, documentation style, open-weight model; PLAN_v_cost_style.md. Resumable: rerun the same command.
#   bash run_v.sh flash|terra|sonnet|kimi
set -u
cd "$(dirname "$0")"
PY=<root>/.venv/Scripts/python.exe
EXTRA=""; W=4; EPF=episodes_v.jsonl; TRF=ground_truth_v.jsonl; TAG=v
case "$1" in
  flash)  MODEL=deepseek-chat;                 PROTO=fc;   STEM=ds_chat_fc ;;
  terra)  MODEL=codex:gpt-6.1-terra@medium;    PROTO=text; STEM=codex_terra ;;
  sonnet) MODEL=claude:claude-sonnet-5-5@high; PROTO=text; STEM=claude_s55 ;;
  kimi)   MODEL=kimi-k3.1-pro;                 PROTO=fc;   STEM=kimi_k31pro; W=1; EPF=episodes_x.jsonl; TRF=ground_truth_x.jsonl; TAG=x ;;
  kimit)  # thinking enabled (default mode): bash run_v.sh kimit <line of api.txt> <shard 0-3>; shards are consecutive
          # blocks of 19 ids of XP10, XP30, XP70 (fixed order); a shard can be resumed with another key line
          export KIMI_KEY_LINE=$2; export RAEA_EPISODES=episodes_x.jsonl RAEA_TRUTH=ground_truth_x.jsonl
          IDS=$($PY -c "ids=[f'XP{p}-{k:03d}' for p in (10, 30, 70) for k in range(25)]; print(' '.join(ids[19*$3:19*($3+1)]))")
          OUT=logs/x_kimi_k31pro_think_s$3_k2e.jsonl
          $PY run.py --models kimi-k3.1-pro --protocol fc --conditions K2e --workers 1 --out $OUT --ids $IDS >> logs/v_kimi_think_$3.log 2>&1
          echo "exit $?"; exit 0 ;;
  *) echo "usage: run_v.sh flash|terra|sonnet|kimi"; exit 2 ;;
esac
if [ "$1" = kimi ]; then
  IDS=$($PY -c "print(' '.join(f'XP{p}-{k:03d}' for p in (10, 30, 70, 50) for k in range(25)))")
else
  IDS=$($PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('episodes_v.jsonl')))")
fi
OUT=logs/${TAG}_${STEM}_k2e.jsonl
export RAEA_EPISODES=$EPF RAEA_TRUTH=$TRF
if [ "$1" = flash ]; then
  $PY ../offpeak_runner.py --out $OUT -- $PY run.py --models $MODEL --protocol $PROTO --conditions K2e --workers $W --out $OUT --ids $IDS >> logs/v_${STEM}_k2e.log 2>&1
elif [ "$1" = kimi ]; then
  $PY run.py --models $MODEL --protocol $PROTO --conditions K2e --workers $W --out $OUT --max-tokens 2000 --extra-body '{"thinking": {"type": "disabled"}}' --ids $IDS >> logs/v_${STEM}_k2e.log 2>&1
else
  $PY run.py --models $MODEL --protocol $PROTO --conditions K2e --workers $W --out $OUT --ids $IDS >> logs/v_${STEM}_k2e.log 2>&1
fi
echo "exit $?"
