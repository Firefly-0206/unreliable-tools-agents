"""Pre-sample episodes (common random numbers) and write
  episodes.jsonl         parameters + pre-sampled output tables (HIDDEN: read by code only)
  ground_truth.jsonl     hidden state (HIDDEN)
  export_prompts.json    agent-visible material only (prompts, schemas, tool table, observation format)
  export_hidden_outputs.json  pre-sampled outputs per tool per call index (HIDDEN: code only, never model context)
Twins: TRANSIENT-k / TRANSIENT_P-k reuse the parameters, scenario and uniforms of SHARED-k / PERSIST-k with phi = 0.
Usage: python make_episodes.py [--n 50] [--seed 2026]"""
import argparse, json, os
import numpy as np
from env import (HERE, FAMILIES, FAMILY_SPEC, TWINS, CONDITIONS, SCENARIOS, SCENARIO_LIST, BUDGET, LOSS, WARNING,
                 sample_params, twin_params, sample_world, build_model, system_prompt, first_user, tool_schemas,
                 backend_of, ticket_id)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    eps, truth = {}, {}
    for fi, fam in enumerate(FAMILY_SPEC):
        twins = [t for t, b in TWINS.items() if b == fam]
        for k, child in enumerate(np.random.SeedSequence([a.seed, fi]).spawn(a.n)):
            s_par, s_world = child.spawn(2)
            par = sample_params(fam, np.random.default_rng(s_par))
            for f, p in [(fam, par)] + [(t, twin_params(par, t)) for t in twins]:
                M = build_model(p); w, table = sample_world(p, np.random.default_rng(s_world), M)
                eid = f"{f}-{k:03d}"
                eps[eid] = dict(id=eid, ticket=ticket_id(eid), family=f, scenario=SCENARIO_LIST[k % len(SCENARIO_LIST)],
                                pair=f"{TWINS.get(f, f)}-{k:03d}", params=p, table=table)
                truth[eid] = dict(id=eid, **w, V0=float(M.V(M.prior.copy(), float(BUDGET))[0]))
    order = [f"{f}-{k:03d}" for f in FAMILIES for k in range(a.n)]
    seen = set()
    for i in order:                         # deterministic re-salting on the rare 5-digit collision
        salt = 0
        while eps[i]["ticket"] in seen: salt += 1; eps[i]["ticket"] = ticket_id(i, salt)
        seen.add(eps[i]["ticket"])
    with open(os.path.join(HERE, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for i in order: f.write(json.dumps(eps[i]) + "\n")
    with open(os.path.join(HERE, "ground_truth.jsonl"), "w", encoding="utf-8") as f:
        for i in order: f.write(json.dumps(truth[i]) + "\n")
    export([eps[i] for i in order])
    print(f"wrote {len(order)} episodes")

def export(eps):
    """Agent-visible material for an external orchestrator, and (separately) the hidden output tables."""
    out = dict(meta=dict(
        note=("Agent-visible material only. The orchestrator must not show 'id', 'family' or 'pair' to the model; "
              "the model sees only the system prompt, first_user, tool_schemas and observations. Tool outputs are in "
              "export_hidden_outputs.json, which must only be read by code and never placed in model context."),
        budget=BUDGET, loss_rows="verdict index 0..2 then escalate; columns true state", loss=LOSS.tolist(),
        conditions=CONDITIONS, protocols=["fc", "text"], warning_sentence=WARNING,
        observation=("After a valid call of tool j (its c-th call, 0-based), the observation text is "
                     "json.dumps({'tool': name, 'result': {arg: subject, field: outputs[name][c]}, 'cost': cost, "
                     "'budget_remaining': B}) with Python json.dumps separators (', ', ': '); if B < min cost append "
                     "'\\nBudget exhausted: only submit is possible now.'; in K4 append '\\n' + grouped view: "
                     "'Evidence so far, grouped by data source:' then one line per data source in tool order, "
                     "'- <source> [<tool names>]: <name=value, ...>' or '(none)'; in K4c append '\\n' + "
                     "'Evidence so far, in the order received:' then '- call <n>: <name>=<value>' per call. In the text "
                     "protocol observations are sent as user messages prefixed 'Observation: '. Invalid actions return "
                     "json.dumps({'error': reason, 'message': ..., 'budget_remaining': B}). Use step.py for a reference "
                     "implementation."),
        episodes_without_submit="flagged no_submit; scored with the worst verdict at the final belief"), episodes=[])
    hidden = dict(meta=dict(note="HIDDEN. Pre-sampled tool outputs; reveals the hidden state. Code only."), outputs={})
    for e in eps:
        sc = SCENARIOS[e["scenario"]]
        out["episodes"].append(dict(
            id=e["id"], family=e["family"], pair=e["pair"], ticket=e["ticket"], scenario=e["scenario"], arg=sc["arg"],
            subject=sc["subject"], verdicts=sc["states"] + ["escalate"],
            tools=[dict(name=d["name"], field=d["field"], values=d["values"], cost=e["params"]["tools"][j]["cost"],
                        source=backend_of(e, j)) for j, d in enumerate(sc["tools"])],
            first_user=first_user(e), tool_schemas=tool_schemas(e),
            system={c: {p: system_prompt(e, c, p) for p in ("fc", "text")} for c in CONDITIONS}))
        hidden["outputs"][e["id"]] = {d["name"]: [d["values"][o] for o in e["table"][j]] for j, d in enumerate(sc["tools"])}
    with open(os.path.join(HERE, "export_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)
    with open(os.path.join(HERE, "export_hidden_outputs.json"), "w", encoding="utf-8") as f:
        json.dump(hidden, f)

if __name__ == "__main__":
    main()
