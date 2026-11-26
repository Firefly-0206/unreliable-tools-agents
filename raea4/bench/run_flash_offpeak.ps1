# Waits for DeepSeek off-peak (UTC 16:30-00:30), then runs flash intervention conditions on all 140 tasks.
# Resumes into the same file as the P0-P3 cross-family run (done keys are skipped).
$now = (Get-Date).ToUniversalTime()
$start = $now.Date.AddHours(16).AddMinutes(35)
if ($now.Hour -ge 16 -and $now -ge $start) { $start = $now }   # already inside the window
elseif ($now.Hour -lt 1 -and $now.Minute -lt 15) { $start = $now }  # still early in the window that began yesterday
$secs = [int]($start - $now).TotalSeconds
if ($secs -gt 0) { Write-Output "sleeping $([math]::Round($secs/60)) min until $start UTC"; Start-Sleep -Seconds $secs }
Set-Location <root>\raea4
& <root>\.venv\Scripts\python.exe bench\run.py --provider deepseek --model deepseek-chat `
  --conds L1,L2,L3,C1,C2,C3,X1,X2,X3,T1,H1 --tag xfam --out bench\results\ds_chat_xfam.jsonl `
  --workers 8 --max-api-calls 12000 *> bench\results\ds_chat_xfam_interv.log
Write-Output "done: exit $LASTEXITCODE"
