"""Projected token use and cost of the full closed-loop study (4 core families x 5 conditions x 50 episodes, fc).
Prompt sizes are measured from the actual prompts; chars/token (3.84) is calibrated on the pilot's DeepSeek
usage records; output tokens per turn are scenario assumptions anchored on the pilot (single-shot medians:
deepseek-chat 2,807, deepseek-reasoner 31,244 completion tokens). In fc thinking mode the reasoning of earlier
turns is sent back as input, so input grows quadratically in turns. No context-cache discount is applied
(upper bound for input). Prices per 1M tokens; peak hours double them.
Usage: python project_cost.py [--logs logs/api_fc.jsonl] -> results/cost_projection.json"""
import argparse, json, os
import numpy as np
from env import HERE, CORE, CONDITIONS, load_episodes, system_prompt, first_user, tool_schemas

CPT = 3.84
PRICE = {"deepseek-chat": (0.15, 0.60), "deepseek-reasoner": (0.15, 0.60), "deepseek-v4-pro": (0.66, 1.98)}
# (turns, output tokens per turn) for a low and a high scenario
SCEN = {"deepseek-chat": [(4, 300), (6, 2800)], "deepseek-reasoner": [(4, 5000), (6, 31000)],
        "deepseek-v4-pro": [(4, 5000), (6, 31000)]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--logs", default=None); ap.add_argument("--per-cell", type=int, default=50)
    a = ap.parse_args()
    eps = [e for e in load_episodes().values() if e["family"] in CORE]
    base = {c: np.mean([(len(system_prompt(e, c, "fc")) + len(json.dumps(tool_schemas(e))) + len(first_user(e))) / CPT
                        for e in eps]) for c in CONDITIONS}
    obs = {c: (150 if c in ("K4", "K4c") else 45) for c in CONDITIONS}
    n_ep = len(CORE) * len(CONDITIONS) * a.per_cell
    out = dict(base_prompt_tokens={c: round(v) for c, v in base.items()}, episodes_per_model=n_ep, models={})
    print(f"base prompt tokens (system + tool schemas + ticket): " + ", ".join(f"{c} {v:.0f}" for c, v in base.items()))
    for m, scs in SCEN.items():
        pin, pout = PRICE[m]; rows = []
        for T, o in scs:
            tin = np.mean([sum(base[c] + (t - 1) * (o + obs[c]) for t in range(1, T + 1)) for c in CONDITIONS])
            tout = T * o
            cost = n_ep * (tin * pin + tout * pout) / 1e6
            rows.append(dict(turns=T, out_per_turn=o, in_per_episode=round(tin), out_per_episode=tout,
                             total_in_M=round(n_ep * tin / 1e6, 2), total_out_M=round(n_ep * tout / 1e6, 2),
                             cost=round(cost, 2), cost_peak=round(2 * cost, 2)))
            print(f"{m:18s} T={T} out/turn={o:6d}: in/ep {tin:8.0f} out/ep {tout:7d}  total {n_ep*tin/1e6:6.1f}M in "
                  f"{n_ep*tout/1e6:6.1f}M out  cost {cost:7.2f} (peak {2*cost:7.2f})")
        out["models"][m] = rows
    if a.logs and os.path.exists(a.logs):
        recs = [json.loads(l) for l in open(a.logs, encoding="utf-8")]
        out["measured"] = [dict(model=r["model"], episode=r["episode"], tokens=r["tokens"], n_api=r["n_api"]) for r in recs]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "cost_projection.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
