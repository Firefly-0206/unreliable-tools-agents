# terra @high closed loop K1, K2e (resumes from logs\codex_terra_high.jsonl), then the offline K2r replay and scoring.
$ErrorActionPreference = 'Continue'
Set-Location <root>\raea4\agentenv
$py  = "<root>\.venv\Scripts\python.exe"
$fam = "PERSIST","SHARED","MIXED","TRANSIENT","TRANSIENT_P"
$H   = "codex:gpt-6.1-terra@high"
& $py -u run.py --models $H --protocol text --families $fam --conditions K1 K2e --episodes 25 --workers 4 --out logs\codex_terra_high.jsonl *>> logs\codex_terra_high.log
& $py k2e_replay.py logs\codex_terra_high.jsonl --out logs\codex_terra_high_k2e_replay.jsonl *> logs\codex_terra_high_k2e_replay.log
& $py score.py logs\codex_terra_high.jsonl logs\codex_terra_high_k2e_replay.jsonl --out results\score_codex_terra_high.json *> results\score_codex_terra_high.txt
"done"
