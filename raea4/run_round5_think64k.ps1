# Flash thinking: re-run only the answers that hit the 16k cap in ds_reasoner_think.jsonl, now with a 64k cap (off-peak only).
# Composite protocol: answers that finished under 16k are kept (the cap was not binding); truncated ones are re-sampled at 64k.
$ErrorActionPreference = 'Continue'
Set-Location <root>\raea4
$py = "<root>\.venv\Scripts\python.exe"
$S  = "C:\Users\anon\AppData\Local\Temp\claude\d64eee42e406816a\ee019b1e-c70e-4e93-9f7c-c01a8dcf4da1\scratchpad"
$out = "bench\results\ds_reasoner_think64k.jsonl"
foreach ($c in "P0", "L2") {
  $items = Get-Content "$S\trunc_$c.txt" -Raw
  & $py -u offpeak_runner.py --out $out -- $py bench\run.py --provider deepseek --model deepseek-reasoner --conds $c --items $items --samples 1 --max-tokens 65536 --tag think --workers 8 --max-api-calls 110 --out $out *>> bench\results\ds_reasoner_think64k.log
}
"done"
