# Sonnet 5.5 @high closed loop K1, K2, K2e (resumes from logs\claude_s55.jsonl), then the offline K2r replay and scoring.
$ErrorActionPreference = 'Continue'
Set-Location <root>\raea4\agentenv
$py  = "<root>\.venv\Scripts\python.exe"
$fam = "PERSIST","SHARED","MIXED","TRANSIENT","TRANSIENT_P"
$M   = "claude:claude-sonnet-5-5@high"
& $py -u run.py --models $M --protocol text --families $fam --conditions K1 K2 K2e --episodes 25 --workers 4 --out logs\claude_s55.jsonl *>> logs\claude_s55.log
& $py k2e_replay.py logs\claude_s55.jsonl --out logs\claude_s55_k2e_replay.jsonl *> logs\claude_s55_k2e_replay.log
& $py score.py logs\claude_s55.jsonl logs\claude_s55_k2e_replay.jsonl --out results\score_claude_s55.json *> results\score_claude_s55.txt
"done"
