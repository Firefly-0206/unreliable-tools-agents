"""
Flash thinking (deepseek-reasoner) composite run at an effective 64k output cap.

ds_reasoner_think.jsonl ran P0, L2, H1 with a 16k cap; 191 P0/L2 answers hit it (finish_reason 'length').
Those were re-sampled with a 65,536 cap (ds_reasoner_think64k.jsonl). An answer that finished under 16k is a valid
64k-cap answer (the cap was not binding), so the composite keeps it and takes the re-sample for every truncated one.
Disclosed caveat: the truncated answers were re-sampled, not continued, so the composite is not an exact draw of a
single 64k-cap run (re-sampling can finish shorter by chance).

Each record gets max_tokens=65536 (the effective cap, so score.py treats the set as one run) and composite_source.
python bench/make_think_composite.py  ->  bench/results/ds_reasoner_think_composite.jsonl
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
first = [json.loads(l) for l in open(os.path.join(R, "ds_reasoner_think.jsonl"), encoding="utf-8")]
rerun = {r["pid"]: r for r in map(json.loads, open(os.path.join(R, "ds_reasoner_think64k.jsonl"), encoding="utf-8"))}
out, n = [], {"kept_16k": 0, "rerun_64k": 0}
for r in first:
    if r["cond"] in ("P0", "L2") and r.get("finish_reason") == "length":
        x = dict(rerun[r["pid"]]); x["composite_source"] = "rerun_64k"; n["rerun_64k"] += 1
    else:
        x = dict(r); x["composite_source"] = "kept_16k"; n["kept_16k"] += 1
    x["max_tokens"] = 65536
    out.append(x)
assert len(out) == len(first) == 420 and n["rerun_64k"] == 191, n
with open(os.path.join(R, "ds_reasoner_think_composite.jsonl"), "w", encoding="utf-8") as f:
    for x in out: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(n, "still truncated at 64k:", sum(x.get("finish_reason") == "length" for x in out))
