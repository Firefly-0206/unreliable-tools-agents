"""
External stepping CLI: replay an action history in one episode and print the next observation as JSON.

  python step.py --episode SHARED-003 --condition K1 --protocol fc --history ""
      -> {"system": ..., "user": ..., "tools": [...], "budget_remaining": 6, "done": false}
  python step.py --episode SHARED-003 --condition K4 --history "health_check,status_dashboard,submit:healthy"
      -> {"observation": <text for the LAST action>, "budget_remaining": ..., "done": ..., "valid": ..., "log": [...]}

Actions: a tool name, "submit:<verdict>", or "invalid:<reason>". No ground truth is ever printed.
"""
import argparse, json
from env import Env, load_episodes, system_prompt, first_user, tool_schemas, num

def step(ep, cond, protocol, history):
    env = Env(ep, cond, protocol); out = None
    for a in history:
        if a.startswith("submit:"): out = env.submit(a.split(":", 1)[1])
        elif a.startswith("invalid:"): out = env.invalid(a.split(":", 1)[1])
        else: out = env.call(a)
    if not history:
        return dict(system=system_prompt(ep, cond, protocol), user=first_user(ep),
                    tools=tool_schemas(ep) if protocol == "fc" else None, budget_remaining=num(env.budget), done=False)
    return dict(observation=env.observation_message(out), budget_remaining=num(env.budget), done=env.done,
                valid=env.log[-1]["type"] != "invalid",
                log=[{k: v for k, v in e.items() if k not in ("j", "o")} for e in env.log])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True); ap.add_argument("--condition", default="K1")
    ap.add_argument("--protocol", choices=["fc", "text"], default="fc")
    ap.add_argument("--history", default="", nargs="?", const="", help="comma-separated actions")
    a = ap.parse_args()
    hist = [x.strip() for x in a.history.split(",") if x.strip()]
    print(json.dumps(step(load_episodes()[a.episode], a.condition, a.protocol, hist), ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
