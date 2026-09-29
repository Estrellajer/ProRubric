#!/usr/bin/env python3
"""RuscaRL-style pre-expanded rollout groups over an atomic checklist.

Each training step t (1..T) has GROUPS groups; each group = the same prompt in
G rows that share a `uid`, where row i carries a rubric scaffold of strength
lambda_{S,i} = lambda_step(t) * (G-i)/(G-1),
lambda_step(t) = 1/(1+exp(alpha*(t/T - t0))) with t0=0.2, alpha=125
(RuscaRL, arXiv 2508.16949). The scaffold shows round(lambda * N) of the N
atomic criteria (seeded subset) appended to the last user message; row order is
step-major so a sequential (shuffle=False) dataloader with batch 512 = 64x8
reproduces the schedule. The reward rubric (extra_info.rubric) is untouched;
the judge never sees the scaffold.

  python3 build_ruscarl.py --input rubrichub-medical-v1/train.parquet \
      --output release/rubrichub-medical-ruscarl-v1 \
      --artifact-id rubrichub-medical-ruscarl-v1 --derived-from rubrichub-medical-v1
"""
import argparse
import json
import math
import random

from _common import read_rows, tree_hash, write_release

HEAD = ("[Guidance] A high-quality response should satisfy the following criteria. Address them naturally in your answer; "
        "do not list, quote or refer to these criteria explicitly:\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="source atomic checklist train.parquet")
    ap.add_argument("--output", required=True, help="output release directory")
    ap.add_argument("--artifact-id", default="rubrichub-medical-ruscarl-v1")
    ap.add_argument("--derived-from", default="rubrichub-medical-v1",
                    help="artifact id of the source atomic checklist")
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--groups", type=int, default=64)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--t0", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=125.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=14000)
    args = ap.parse_args()
    T, GROUPS, G, T0, ALPHA, SEED, MAX_CHARS = (
        args.T, args.groups, args.G, args.t0, args.alpha, args.seed, args.max_chars)

    def lam_step(t):
        return 1.0 / (1.0 + math.exp(ALPHA * (t / T - T0)))

    rows = read_rows(args.input)
    rng = random.Random(SEED)

    def ex(r):
        e = r.get("extra_info")
        return json.loads(e) if isinstance(e, str) else e

    def plen(r):
        p = r["prompt"]
        txt = " ".join(m.get("content", "") for m in p) if isinstance(p, list) else str(p)
        return len(txt) + sum(len(c["criterion"]) for c in ex(r)["rubric"])

    keep = [r for r in rows if plen(r) <= MAX_CHARS]
    dropped = len(rows) - len(keep)
    order = list(range(len(keep)))
    rng.shuffle(order)
    need = T * GROUPS
    seq = [keep[order[i % len(order)]] for i in range(need)]   # 1.53 epochs, seeded permutation, cycled
    out = []
    kstats = []
    for step in range(1, T + 1):
        ls = lam_step(step)
        for g in range(GROUPS):
            r = seq[(step - 1) * GROUPS + g]
            e = ex(r)
            rub = e["rubric"]
            N = len(rub)
            rid = str(e.get("id"))
            uid = f"ruscarl:{step}:{g}:{rid}"
            for i in range(1, G + 1):
                lam = ls * (G - i) / (G - 1)
                k = int(round(lam * N))
                kstats.append(k / max(1, N))
                prompt = json.loads(json.dumps(r["prompt"]))  # deep copy
                if k > 0:
                    sub = random.Random(f"{SEED}:{rid}:{step}:{i}").sample(range(N), k)
                    sub.sort()
                    scaffold = HEAD + "\n".join(f"- {rub[j]['criterion']}" for j in sub)
                    if isinstance(prompt, list):
                        for m in reversed(prompt):
                            if m.get("role") == "user":
                                m["content"] = m["content"].rstrip() + "\n\n" + scaffold
                                break
                    else:
                        prompt = str(prompt).rstrip() + "\n\n" + scaffold
                e2 = dict(e)
                e2["ruscarl"] = {"step": step, "i": i, "lambda_step": round(ls, 4),
                                 "lambda": round(lam, 4), "k": k, "n_criteria": N}
                out.append({"prompt": prompt, "data_source": r["data_source"],
                            "extra_info": json.dumps(e2) if isinstance(r["extra_info"], str) else e2,
                            "uid": uid})
    scaff_rows = sum(1 for k in kstats if k > 0)
    manifest = {
        "artifact_id": args.artifact_id,
        "derived_from": {"artifact": args.derived_from},
        "rows_out": len(out), "groups": T * GROUPS, "group_size": G,
        "steps": T, "prompts_used": len(keep),
        "prompts_dropped_over_%d_chars" % MAX_CHARS: dropped,
        "schedule": {"t0": T0, "alpha": ALPHA, "lambda_i": "(G-i)/(G-1)",
                     "subset": "seeded random subset of round(lambda*N) criteria"},
        "scaffolded_row_share": scaff_rows / len(out),
        "mean_criteria_fraction_shown": sum(kstats) / len(kstats),
        "usage": "data.shuffle=False, actor_rollout_ref.rollout.n=1, data.train_batch_size=512 (=64 groups x 8), uid column groups advantages (needs trainer uid passthrough, release git-909636bdd1e7)",
    }
    readme = (f"# {args.artifact_id}\n\n"
              "RuscaRL-style pre-expanded groups (G=8, sigmoid temporal decay t0=0.2 alpha=125, "
              "intra-group linear strength) over the medical atomic checklist. Reward rubric untouched. "
              "Val sets NOT included; bind val to rubrichub-medical-v1.\n")
    m = write_release(args.output, out, manifest, readme)
    hh, file_count, total = tree_hash(args.output)
    print(json.dumps({"rows": len(out), "dropped": dropped,
                      "scaffolded_share": round(scaff_rows / len(out), 3),
                      "mean_frac_shown": round(sum(kstats) / len(kstats), 3),
                      "tree": hh, "file_count": file_count, "total_bytes": total}))
    print("SAMPLE step1 i1 uid:", out[0]["uid"], "| last user msg tail:",
          (out[0]["prompt"][-1]["content"] if isinstance(out[0]["prompt"], list) else out[0]["prompt"])[-400:].replace("\n", " | "))
    print("step 70 lambda_step:", round(lam_step(70), 4), "| step 55:", round(lam_step(55), 4))


if __name__ == "__main__":
    main()
