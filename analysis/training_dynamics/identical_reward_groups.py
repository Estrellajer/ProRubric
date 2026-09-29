# -*- coding: utf-8 -*-
"""Share of sampled groups whose rollouts all receive the same reward (K=1 arm).

Produces the claim that about two thirds of K=1's sampled groups receive identical rewards on every seed
(Sec. 5.3; appendix "Aggregation controls at 8B": 63-66%, medical 4B K=1 seeds 42/43/44: 63.9 / 65.5 / 63.4%).

Input: each run's training log `tracking/events.jsonl`, one JSON object per line {"step": int, "data": {metric:
value}}. The trainer logs, per step, `critic/nonconstant_reward_group_fraction` = the fraction of the step's
prompt groups (8 rollouts each) whose rewards are not all equal (older logs name it
`critic/reward_groups/nonconstant_fraction`). A group with identical rewards has zero advantage under GRPO and
contributes no policy gradient. The reported share is 100 x (1 - mean over logged training steps of that fraction).
Several files for one run (a resumed run) are merged in order, later steps overriding earlier ones.

    python3 identical_reward_groups.py --run "K=1 s42=events.jsonl" --run "K=1 s43=events.jsonl" --run "K=1 s44=a.jsonl,b.jsonl"
"""
import argparse, json, statistics

KEYS = ("critic/nonconstant_reward_group_fraction", "critic/reward_groups/nonconstant_fraction")


def load(paths):
    rows = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("step") is not None:
                rows.setdefault(int(r["step"]), {}).update(r.get("data") or {})
    return rows


def identical_share(rows):
    fr = []
    for step in sorted(rows):
        v = next((rows[step][k] for k in KEYS if rows[step].get(k) is not None), None)
        if v is not None:
            fr.append(v)
    return {"steps": len(fr), "identical_reward_group_pct": round(100 * (1 - statistics.mean(fr)), 2) if fr else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=PATH[,PATH]", help="events.jsonl of one run (repeatable)")
    ap.add_argument("--out", help="optional output JSON")
    a = ap.parse_args()
    res = {}
    for spec in a.run:
        name, _, paths = spec.partition("=")
        res[name] = identical_share(load(paths.split(",")))
        print("%s: %s%% of groups have identical rewards (%d steps)" % (name, res[name]["identical_reward_group_pct"], res[name]["steps"]))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
