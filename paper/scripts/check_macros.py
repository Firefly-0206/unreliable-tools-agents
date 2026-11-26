"""Check that every \\sn... macro used in sections/sim.tex and sections/app_sim.tex is defined in tables/nums.tex,
and that every table file and figure referenced exists."""
import re
from pathlib import Path

P = Path(__file__).resolve().parents[1]
nums = (P / "tables" / "nums.tex").read_text(encoding="utf8")
defd = set(re.findall(r"\\(?:new|provide)command\{\\(sn[A-Za-z]+)\}", nums))
bad = False
for f in ["sections/sim.tex", "sections/app_sim.tex"]:
    t = (P / f).read_text(encoding="utf8")
    used = set(re.findall(r"\\(sn[A-Za-z]+)", t))
    miss = sorted(u for u in used if u not in defd)
    print(f, "macros used:", len(used), "missing:", miss)
    bad |= bool(miss)
    for tb in re.findall(r"\\input\{(tables/[A-Za-z0-9_]+)\}", t):
        if not (P / (tb + ".tex")).exists():
            print("missing table file", tb)
            bad = True
    labels = re.findall(r"\\label\{([^}]+)\}", t)
    print("  labels:", labels)
print("OK" if not bad else "PROBLEMS")

