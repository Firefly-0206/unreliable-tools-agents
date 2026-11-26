# Round 5, terra via Codex: @high bench P0,L2,H1 and closed loop K1,K2e (+K2r); @medium second sample of K2e,K1 (+K2r).
$ErrorActionPreference = 'Continue'
Set-Location <root>\raea4
$py  = "<root>\.venv\Scripts\python.exe"
$fam = "PERSIST","SHARED","MIXED","TRANSIENT","TRANSIENT_P"
$H   = "codex:gpt-6.1-terra@high"
$Med = "codex:gpt-6.1-terra@medium"
& $py bench\run.py --provider codex --model $H --conds P0,L2,H1 --samples 1 --max-tokens 0 --tag high --workers 4 --out bench\results\codex_terra_high.jsonl *> bench\results\codex_terra_high.log
& $py bench\score.py bench\results\codex_terra_high.jsonl --name codex_terra_high *> bench\results\codex_terra_high_score.log
Set-Location agentenv
& $py run.py --models $H --protocol text --families $fam --conditions K1 K2e --episodes 25 --workers 4 --out logs\codex_terra_high.jsonl *> logs\codex_terra_high.log
& $py k2e_replay.py logs\codex_terra_high.jsonl --out logs\codex_terra_high_k2e_replay.jsonl *> logs\codex_terra_high_k2e_replay.log
& $py score.py logs\codex_terra_high.jsonl logs\codex_terra_high_k2e_replay.jsonl --out results\score_codex_terra_high.json *> results\score_codex_terra_high.txt
& $py run.py --models $Med --protocol text --families $fam --conditions K2e K1 --episodes 25 --samples 2 --sample-start 1 --workers 4 --out logs\codex_terra_medium_s1.jsonl *> logs\codex_terra_medium_s1.log
& $py k2e_replay.py logs\codex_terra_medium_s1.jsonl --out logs\codex_terra_medium_s1_k2e_replay.jsonl *> logs\codex_terra_medium_s1_k2e_replay.log
"done"
