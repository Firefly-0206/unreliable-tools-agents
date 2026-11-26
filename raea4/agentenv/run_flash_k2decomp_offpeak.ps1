# Waits for DeepSeek off-peak (UTC 16:30-00:30), then runs the K2 decomposition on deepseek-chat (fc protocol).
$now = (Get-Date).ToUniversalTime()
$start = $now.Date.AddHours(16).AddMinutes(40)
if ($now -ge $start) { $start = $now }   # before 16:40 UTC (incl. 00:30-16:30 peak) -> wait for today's window
$secs = [int]($start - $now).TotalSeconds
if ($secs -gt 0) { Write-Output "sleeping $([math]::Round($secs/60)) min until $start UTC"; Start-Sleep -Seconds $secs }
Set-Location <root>\raea4\agentenv
$fam = "PERSIST","SHARED","MIXED","TRANSIENT","TRANSIENT_P"
& <root>\.venv\Scripts\python.exe run.py --models deepseek-chat --protocol fc --families $fam `
  --conditions K2a K2e K2p K1e --episodes 25 --workers 4 --out logs\ds_chat_fc_k2decomp.jsonl *> logs\ds_chat_fc_k2decomp.log
Write-Output "done: exit $LASTEXITCODE"
