"""Hash of every system prompt and tool schema of the earlier episode files (first domain), to check that a change of
env.py leaves them untouched. python t_prompt_hash.py -> prints one sha256 per episode file."""
import hashlib, json, os, subprocess, sys

FILES = ["episodes.jsonl", "episodes_x.jsonl", "episodes_y.jsonl", "episodes_w.jsonl", "episodes_v.jsonl"]
CODE = ("import os, sys, json, hashlib; from env import load_episodes, system_prompt, tool_schemas, first_user_cond\n"
        "h = hashlib.sha256()\n"
        "for i, e in sorted(load_episodes().items()):\n"
        "    for c in ('K1', 'K2', 'K2e'):\n"
        "        for p in ('text', 'fc'):\n"
        "            h.update(system_prompt(e, c, p).encode()); h.update(first_user_cond(e, c).encode())\n"
        "    h.update(json.dumps(tool_schemas(e), sort_keys=True).encode())\n"
        "print(h.hexdigest())\n")

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for f in FILES:
        env = dict(os.environ, RAEA_EPISODES=f, RAEA_TRUTH=f.replace("episodes", "ground_truth"))
        out = subprocess.run([sys.executable, "-c", CODE], cwd=here, env=env, capture_output=True, text=True)
        print(f, out.stdout.strip() or out.stderr.strip()[-300:])
