"""Paired per-instance contrasts with cluster-bootstrap CIs (same instances, same calibration logs)."""
import json, numpy as np
from exact import run
from runs import M
def ci(d, rng=np.random.default_rng(0)):
    b = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
out = {}
for fam in ["RETRY", "SHARED", "IND"]:
    pols = {"Oracle-DP": (("dp",), "true"), "fit": (("dp",), "fit"), "fit_dedup": (("dp",), "fit_dedup"),
            "oracle_dedup": (("dp",), "oracle_dedup"), "mmi_dp": (("dp",), "mmi"), "mmi_voi1": (("look", 1), "mmi"),
            "unaware": (("dp",), "fault_unaware"), "unaware_dedup": (("dp",), "unaware_dedup")}
    r = run(fam, M, pols, 300, 101, n_cal_max=100)       # same seed as X1: same instances and logs
    t = {k: v["total"] for k, v in r.items()}
    out[fam] = {"fit_dedup - fit": ci(t["fit_dedup"] - t["fit"]),
                "oracle_dedup - exact": ci(t["oracle_dedup"] - t["Oracle-DP"]),
                "mmi DP - mmi VOI1": ci(t["mmi_dp"] - t["mmi_voi1"]),
                "unaware_dedup - unaware": ci(t["unaware_dedup"] - t["unaware"]),
                "share of instances where oracle_dedup is worse by > 0.01": float(np.mean(t["oracle_dedup"] - t["Oracle-DP"] > 0.01))}
    print(fam, json.dumps(out[fam]), flush=True)
json.dump(out, open("results/contrasts.json", "w"), indent=1)
