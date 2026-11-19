"""
Applies babylm-plan.md section 11 to the three seeds: seed 0 in reports/babylm/babylm_{s,p,t}, seeds 1-2 in
reports/babylm_seeds/babylm_{s,p,t}_seed{1,2}, diagnoses in reports/babylm/diag_length.json and
reports/babylm_seeds/diag_length_seed{1,2}.json (missing ones are skipped).

$ python scripts/babylm/summarize_seeds.py scripts/reports
"""
import json
import os
import re
import statistics
import sys

R = sys.argv[1]
TIE_LOSS, TIE_BLIMP = 0.02, 2.0


def run(v, k):
    d = os.path.join(R, "babylm", f"babylm_{v}") if k == 0 else os.path.join(R, "babylm_seeds", f"babylm_{v}_seed{k}")
    e = json.load(open(os.path.join(d, "eval.json")))
    vals = [float(x) for x in re.findall(r"val loss ([0-9.]+)", open(os.path.join(d, "train.log")).read())]
    return {"seed": e["seed"], "loss": e["val"]["512"]["nll"], "blimp": 100 * e["blimp"]["accuracy"],
            "best": min(vals), "last_train_val": vals[-1], "pos": e["val"].get("2048", {}).get("by_position")}


def verdict(d, tie):  # d = S minus other; lower loss is better, higher BLiMP is better
    return "tie" if abs(d) < tie else ("S better" if d < 0 else "S worse")


def main():
    seeds = [k for k in (0, 1, 2) if all(os.path.exists(os.path.join(R, *(("babylm", f"babylm_{v}") if k == 0 else ("babylm_seeds", f"babylm_{v}_seed{k}")), "eval.json")) for v in "spt")]
    runs = {k: {v: run(v, k) for v in "spt"} for k in seeds}
    rows = {}
    for k in seeds:
        s, p, t = (runs[k][v] for v in "spt")
        lp = lambda x, a, b: x["pos"][f"{a}-{b}"] if isinstance(x["pos"], dict) else None
        rows[k] = {
            "check": {v: round(runs[k][v]["loss"] - runs[k][v]["last_train_val"], 4) for v in "spt"},
            "S-T loss": s["loss"] - t["loss"], "S-T blimp": -(s["blimp"] - t["blimp"]),
            "S-P loss": s["loss"] - p["loss"], "S-P blimp": -(s["blimp"] - p["blimp"]),
            "S-P best": s["best"] - p["best"],
            "S own": lp(s, 1024, 2048) - lp(s, 256, 512), "P own": lp(p, 1024, 2048) - lp(p, 256, 512),
        }
        rows[k]["long"] = rows[k]["S own"] - rows[k]["P own"]
        rows[k]["blimp"] = {v: round(runs[k][v]["blimp"], 2) for v in "spt"}
        rows[k]["loss"] = {v: round(runs[k][v]["loss"], 4) for v in "spt"}
    out = {"seeds": {k: runs[k]["s"]["seed"] for k in seeds}, "per_seed": rows, "summary": {}}
    for key, tie in (("S-T loss", TIE_LOSS), ("S-T blimp", TIE_BLIMP), ("S-P loss", TIE_LOSS), ("S-P blimp", TIE_BLIMP),
                     ("S-P best", TIE_LOSS), ("long", TIE_LOSS)):
        xs = [rows[k][key] for k in seeds]
        sign = 1 if "loss" in key or key in ("S-P best", "long") else -1  # report BLiMP as S minus other
        vs = [verdict(x, tie) for x in xs]
        out["summary"][key] = {"S minus other": [round(sign * x, 4) for x in xs], "mean": round(sign * statistics.mean(xs), 4),
                               "verdicts": vs, "holds": len(set(vs)) == 1}
    for key in ("S own", "P own"):
        xs = [rows[k][key] for k in seeds]
        out["summary"][key] = {"values": [round(x, 4) for x in xs], "no collapse (<=0.05)": [x <= 0.05 for x in xs]}
    diags = {0: os.path.join(R, "babylm", "diag_length.json")}
    diags.update({k: os.path.join(R, "babylm_seeds", f"diag_length_seed{k}.json") for k in (1, 2)})
    for k, path in diags.items():
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        P = d["P"]
        base, normal = P["normal"]["all"]["nll"]["256-512"], P["normal"]["all"]["nll"]["1024-2048"]
        gap = normal - base
        rec = lambda m, w="all": (normal - P[m][w]["nll"]["1024-2048"]) / gap if gap >= 0.10 else None
        growth = lambda v: statistics.median(a / b for a, b in zip(d[v]["normal"]["all"]["context_share"]["1024-2048"],
                                                                   d[v]["normal"]["all"]["context_share"]["256-512"]))
        per_block = {w: round(rec("window", w), 3) for w in P["window"] if w != "all"} if gap >= 0.10 else {}
        out.setdefault("diag", {})[k] = {
            "base": round(base, 4), "normal": round(normal, 4), "gap": round(gap, 4),
            "recovered clamp": rec("clamp") and round(rec("clamp"), 3), "recovered window": rec("window") and round(rec("window"), 3),
            "S change window": round(d["S"]["window"]["all"]["nll"]["1024-2048"] - d["S"]["normal"]["all"]["nll"]["1024-2048"], 4),
            "S change clamp": round(d["S"]["clamp"]["all"]["nll"]["1024-2048"] - d["S"]["normal"]["all"]["nll"]["1024-2048"], 4),
            "best single block": max(per_block, key=per_block.get) if per_block else None, "per block window": per_block,
            "median share growth": {"S": round(growth("S"), 3), "P": round(growth("P"), 3)},
            "time constants": d["time_constant"], "check": d["check_rel_err"],
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
