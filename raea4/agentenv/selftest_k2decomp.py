"""Checks for the K2 decomposition (K2a, K1e, K2e, K2p). No LLM calls.
 1. K1-K4c system prompts, first user message and tool schemas are byte-identical to export_prompts.json (all 300 episodes).
 2. The tool-doc block is identical in every condition and its numbers (phi, accuracy, cost) equal the episode parameters.
 3. K2a contains every number of the K1 block and names no tool in the replica bullet; K1e / K2e / K2p carry the K1 / K2 information.
 4. A perfect doc reader's JSON normalises to the true model with zero recovery error; K2p with the TRUE model has regret 0
    on every non-VALUE episode (exact planner on the true model == oracle). VALUE is excluded: its spec class holds only the
    marginal stale emission.
 5. Parser / defaults edge cases (placeholders, fences, unknown or duplicate tools, missing fields, out-of-range numbers)."""
import json, re
import numpy as np
from env import (CONDITIONS, ALL_CONDITIONS, ELICIT, Env, SCENARIOS, load_episodes, load_truth, system_prompt, first_user,
                 first_user_cond, tool_schemas, dependence_text, dependence_text_k2a)
from oracle import score_trajectory
import k2decomp as K

def main():
    eps, truth = load_episodes(), load_truth(); ok = True
    ex = json.load(open("export_prompts.json"))
    bad = sum(system_prompt(eps[e["id"]], c, p) != e["system"][c][p] for e in ex["episodes"] for c in CONDITIONS for p in ("fc", "text"))
    bad += sum(first_user(eps[e["id"]]) != e["first_user"] or tool_schemas(eps[e["id"]]) != e["tool_schemas"] for e in ex["episodes"])
    print(f"1. existing prompts differing from export_prompts.json: {bad}"); ok &= bad == 0
    nb = 0
    for ep in eps.values():
        base = K.parse_docs(ep, system_prompt(ep, "K2", "text")); par = ep["params"]
        for c in ALL_CONDITIONS:
            for p in ("fc", "text"):
                nb += K.parse_docs(ep, system_prompt(ep, c, p)) != base
        for j, t in enumerate(par["tools"]):
            nb += abs(base[j]["acc"] - t["q"]) > 1e-9 or base[j]["cost"] != t["cost"]
            nb += t["src"] == "cache" and abs(base[j]["phi"] - par["phi"]) > 1e-9
    print(f"2. doc-block / parameter mismatches over 300 episodes x {len(ALL_CONDITIONS)} conditions x 2 protocols: {nb}"); ok &= nb == 0
    nb = 0
    for ep in eps.values():
        names = [t["name"] for t in SCENARIOS[ep["scenario"]]["tools"]]; k2a = dependence_text_k2a(ep); k1 = dependence_text(ep)
        first = k2a.split("\n- ")[0]                       # the replica bullet (with its sub-lines)
        nb += any(n in first for n in names)
        for tok in re.findall(r"\b0\.\d+\b", k1):          # every number of the K1 block appears in the K2a block
            nb += tok not in k2a
        s1, s2 = system_prompt(ep, "K1", "text"), system_prompt(ep, "K2", "text")
        nb += system_prompt(ep, "K1e", "text") != s1 + K.__dict__.get("_", "") + system_prompt(ep, "K1e", "text")[len(s1):]
        nb += system_prompt(ep, "K2e", "text") != system_prompt(ep, "K2p", "text")
        nb += not system_prompt(ep, "K2e", "text").startswith(s2) or not system_prompt(ep, "K1e", "text").startswith(s1)
        nb += not first_user_cond(ep, "K2p").startswith(first_user(ep)) or first_user_cond(ep, "K2") != first_user(ep)
    print(f"3. K2a / elicitation prompt violations: {nb}"); ok &= nb == 0
    worst, rec_bad, n = 0.0, 0, 0
    for eid, ep in eps.items():
        if ep["family"] == "VALUE": continue
        sysm = system_prompt(ep, "K2e", "text"); docs = K.parse_docs(ep, sysm)
        spec = K.normalize_model(ep, K.extract_json("blah\n" + json.dumps(K.doc_reader_model(ep, sysm))), docs)
        r = K.score_recovery(ep, spec)
        rec_bad += not (r["struct_exact"] == 1 and r["phi_abs_err"] < 1e-9 and r["acc_mae"] < 1e-9 and not spec["defaults"]
                        and (r["emission_tv"] is None or r["emission_tv"] < 1e-3))
        env = Env(ep, "K2p", "text"); K.plan_and_play(ep, K.truth_spec(ep), env)
        s = score_trajectory(ep, env.log, truth[eid]); worst = max(worst, abs(s["regret"])); n += 1
    print(f"4. perfect-reader recovery errors: {rec_bad}; K2p-with-true-model max |regret| over {n} non-VALUE episodes: {worst:.2e}")
    ok &= rec_bad == 0 and worst < 1e-6
    ep = eps["SHARED-002"]; docs = K.parse_docs(ep, system_prompt(ep, "K2", "text")); nm = [t["name"] for t in SCENARIOS[ep["scenario"]]["tools"]]
    chk = []
    chk.append(K.extract_json('first {"observation_model": {"tool_groups": [{"tools": ["<tool name>", ...]}]}} then no') is None)
    chk.append(K.extract_json('```json\n{"observation_model": {"tool_groups": [], "accuracy": {}}}\n```') is not None)
    s = K.normalize_model(ep, {"observation_model": {"tool_groups": [{"tools": [nm[0], nm[1], "bogus"], "shared_fault": True}]}}, docs)
    chk.append(s["groups"][0]["tools"] == [0, 1] and abs(s["groups"][0]["phi"] - ep["params"]["phi"]) < 1e-9 and len(s["groups"]) == 3
               and any("phi" in d for d in s["defaults"]) and any("bogus" in x for x in s["notes"]))
    s = K.normalize_model(ep, {"observation_model": {"tool_groups": [{"tools": [nm[0]], "phi": 1.7, "shared_fault": True, "persistent": False},
                                                                      {"tools": [nm[0], nm[1]], "phi": .2, "shared_fault": True}],
                                                      "accuracy": {nm[0]: 1.5}}}, docs)
    chk.append(s["groups"][0]["phi"] == 0.95 and not s["groups"][0]["persistent"] and s["accuracy"][0] == 0.98 and any("twice" in x for x in s["notes"]))
    chk.append(K.normalize_model(ep, {"observation_model": {}}, docs)["fallback"] and K.normalize_model(ep, None, docs)["fallback"])
    em = K.normalize_model(ep, {"observation_model": {"tool_groups": [{"tools": nm[:2], "shared_fault": True, "phi": .3,
                                                                       "stale_emission": dict(zip(SCENARIOS[ep["scenario"]]["tools"][1]["values"], [8, 1, 1]))}]}}, docs)
    chk.append(abs(em["groups"][0]["emission"][0] - 0.8) < 1e-9)       # aliases (tool output values) and normalisation
    chk.append(K.normalize_model(ep, {"observation_model": {"tool_groups": [{"tools": [n.upper() for n in nm], "shared_fault": True}]}}, docs)["fallback"])   # no valid tool names -> documented defaults
    chk.append(abs(K.normalize_model(ep, {"observation_model": {"tool_groups": [{"tools": nm[:2], "shared_fault": True, "phi": 25}]}}, docs)["groups"][0]["phi"] - 0.25) < 1e-9)   # percent without sign
    pe = K.normalize_model(ep, {"observation_model": {"tool_groups": [{"tools": nm[:2], "shared_fault": True, "phi": .3, "stale_emission": {SCENARIOS[ep["scenario"]]["tools"][1]["values"][0]: 0.9}}]}}, docs)
    chk.append(abs(sum(pe["groups"][0]["emission"]) - 1) < 1e-9 and pe["groups"][0]["emission"][0] > 0.89 and pe["groups"][0]["emission"][1] > 0 and any("partial" in d for d in pe["defaults"]))   # partial emission recorded    print(f"5. parser / default edge cases passed: {sum(chk)}/{len(chk)}"); ok &= all(chk)
    print("SELFTEST_K2DECOMP", "PASSED" if ok else "FAILED")

if __name__ == "__main__":
    main()

