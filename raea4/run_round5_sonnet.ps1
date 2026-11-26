# Round 5, Sonnet 5.5 @high, core set (user choice 2026-11-04): bench 140 P0,L1,L2,L3,C2,X1,H1; confirmation P0,L2,H1; closed loop K1,K2,K2e (+ K2r replay offline).
$ErrorActionPreference = 'Continue'
Set-Location <root>\raea4
$py  = "<root>\.venv\Scripts\python.exe"
$fam = "PERSIST","SHARED","MIXED","TRANSIENT","TRANSIENT_P"
$M   = "claude:claude-sonnet-5-5@high"
& $py bench\run.py --provider claude --model $M --conds P0,L1,L2,L3,C2,X1,H1 --samples 1 --max-tokens 0 --tag s55 --workers 4 --out bench\results\s55_main.jsonl *> bench\results\s55_main.log
& $py bench\score.py bench\results\s55_main.jsonl --name s55_main *> bench\results\s55_main_score.log
& $py bench\run.py --provider claude --model $M --items-file bench\items_confirm.jsonl --prompts-file bench\prompts_confirm.jsonl --conds P0,L2,H1 --samples 1 --max-tokens 0 --tag confirm --workers 4 --out bench\results\confirm_s55.jsonl *> bench\results\confirm_s55.log
& $py bench\score.py bench\results\confirm_s55.jsonl --items-file bench\items_confirm.jsonl --name confirm_s55 *> bench\results\confirm_s55_score.log
Set-Location agentenv
& $py run.py --models $M --protocol text --families $fam --conditions K1 K2 K2e --episodes 25 --workers 4 --out logs\claude_s55.jsonl *> logs\claude_s55.log
& $py k2e_replay.py logs\claude_s55.jsonl --out logs\claude_s55_k2e_replay.jsonl *> logs\claude_s55_k2e_replay.log
& $py score.py logs\claude_s55.jsonl logs\claude_s55_k2e_replay.jsonl --out results\score_claude_s55.json *> results\score_claude_s55.txt
"done"
