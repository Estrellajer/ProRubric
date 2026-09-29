#!/usr/bin/env python3
"""Append one fixed appropriateness criterion ("+ appr. criterion" rows of Table 3 / Appendix C).

prorubric mode: ProRubric's dimensions + one fixed appropriateness criterion per
question. Deterministic post-process of the ProRubric parquet: every row keeps
its dimensions byte-identical and gains exactly one extra criterion, so the arm
is a clean +1-criterion delta against ProRubric. Weight = mean of that
question's existing dimension weights (= total absolute atomic weight / K).

atomic mode: Rubric-RL's atomic checklist + the SAME fixed criterion, scored
independently and summed (explicit aggregation). Weight = sum(|atomic w|) / K,
where K is the question's number of ProRubric dimensions, i.e. the same share
of the total (1/(K+1)) as in the ProRubric arm, so the two arms differ only in
the base rubric. Requires --prorubric-input to read each question's K.

In both modes the new criterion is scored on its own and enters the weighted
reward like any other; failing it does not zero the other criteria.

  python3 build_appr_criterion.py --mode prorubric --input medical-prorubric/train.parquet \
      --output release/medical-prorubric-appr
  python3 build_appr_criterion.py --mode atomic --input medical-atomic/train.parquet \
      --prorubric-input medical-prorubric/train.parquet --output release/medical-atomic-appr
"""
import argparse
import json

from _common import read_rows, sha256_file, tree_hash, write_release

APPR_CRITERION = (
    "A good answer achieves appropriateness to what was actually asked and to whom it is addressed. It responds to "
    "the user's direct question first, before background, framing or caveats; it stays inside the scope of what was "
    "asked instead of expanding into adjacent material the user did not request; and it is pitched at the person "
    "asking -- their role, situation and level of expertise -- rather than at a generic reader. It fails this "
    "dimension if the direct question is answered only late, only by implication, or not at all; if substantial "
    "content is added beyond what the question calls for; or if the register and depth do not match who is asking. "
    "This dimension is judged on its own: an answer that is complete and accurate on every other dimension still "
    "fails here if it is not responsive, scoped and addressed to this particular asker."
)
APPR_NAME = "Appropriateness and directness"


def run_prorubric(input_path, outdir, artifact_id, derived_from):
    rows = read_rows(input_path)
    n_before = []
    for r in rows:
        ei = r["extra_info"]
        rub = list(ei["rubric"])
        n_before.append(len(rub))
        w = sum(float(c["weight"]) for c in rub) / len(rub)
        rub.append({"criterion": APPR_CRITERION, "weight": float(w)})
        ei["rubric"] = rub
        gen = ei.get("rir_generation")
        if isinstance(gen, str):
            try:
                g = json.loads(gen)
                g["appr_criterion"] = {"name": APPR_NAME, "weight_rule": "mean of the question's dimension weights", "weight": float(w)}
                ei["rir_generation"] = json.dumps(g)
            except json.JSONDecodeError:
                pass
    after = [len(r["extra_info"]["rubric"]) for r in rows]
    manifest = {
        "schema_version": "prorubric-release/appr-criterion-v1",
        "artifact_id": artifact_id,
        "supersedes": None,
        "derived_from": {"artifact": derived_from, "train_sha256": sha256_file(input_path)},
        "rows_out": len(rows),
        "transform": ("append exactly one fixed appropriateness criterion to extra_info.rubric of every row; "
                      "all pre-existing criteria, weights and every other column are untouched"),
        "appr_criterion": {"name": APPR_NAME, "text": APPR_CRITERION,
                           "weight_rule": "mean of that question's existing dimension weights"},
        "criteria_per_question": {"before": {"min": min(n_before), "max": max(n_before), "mean": sum(n_before) / len(n_before)},
                                  "after": {"min": min(after), "max": max(after), "mean": sum(after) / len(after)}},
    }
    readme = (f"# {artifact_id}\n\n"
              "ProRubric's dimensions plus one fixed appropriateness criterion per question "
              "(responsive to the direct question / in scope / addressed to this asker). Everything else is identical, "
              "so the arm differs from ProRubric by exactly one criterion. Val sets NOT included; bind val to the atomic arm.\n")
    m = write_release(outdir, rows, manifest, readme)
    print(json.dumps({k: m[k] for k in ("rows_out", "outputs", "criteria_per_question")}, indent=1))


def run_atomic(input_path, prorubric_path, outdir, artifact_id, derived_from):
    def ex(r):
        e = r.get("extra_info")
        return json.loads(e) if isinstance(e, str) else (e or {})

    k_dims = {}
    for r in read_rows(prorubric_path):
        e = ex(r)
        k_dims[str(e.get("id"))] = len(e.get("rubric") or [])
    kmean = sum(k_dims.values()) / len(k_dims)
    rows = read_rows(input_path)
    shares = []
    nk = 0
    for r in rows:
        e = r["extra_info"]
        isstr = isinstance(e, str)
        ei = json.loads(e) if isstr else e
        rub = list(ei["rubric"])
        tot = sum(abs(float(c["weight"])) for c in rub)
        k = k_dims.get(str(ei.get("id")))
        if k is None:
            k = kmean
            nk += 1
        w = tot / k
        rub.append({"criterion": APPR_CRITERION, "weight": float(w)})
        shares.append(w / (tot + w))
        ei["rubric"] = rub
        ei["appr_criterion"] = {"weight_rule": "sum(|atomic w|)/K -> same share 1/(K+1) as the ProRubric arm", "k_dimensions": k, "weight": w}
        r["extra_info"] = json.dumps(ei) if isstr else ei
    manifest = {
        "artifact_id": artifact_id,
        "derived_from": {"artifact": derived_from},
        "rows_out": len(rows),
        "appr_criterion": APPR_CRITERION,
        "appr_share_mean": sum(shares) / len(shares),
        "rows_without_k": nk,
        "transform": "atomic checklist unchanged + one fixed appropriateness criterion (identical text to the ProRubric arm) appended per question, scored independently and summed; weight = sum(|atomic w|)/K",
    }
    readme = (f"# {artifact_id}\n\n"
              "Atomic checklist + the same appropriateness criterion as the ProRubric arm, explicit aggregation. "
              "Tests whether one constraint survives additive aggregation among ~30 positive atoms. "
              "Val sets NOT included; bind val to rubrichub-medical-v1.\n")
    m = write_release(outdir, rows, manifest, readme)
    hh, file_count, total = tree_hash(outdir)
    print(json.dumps({"rows": len(rows), "appr_share_mean": round(m["appr_share_mean"], 3),
                      "rows_without_k": nk, "tree": hh, "file_count": file_count, "total_bytes": total}))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("prorubric", "atomic"), required=True)
    ap.add_argument("--input", required=True, help="base train.parquet (ProRubric in prorubric mode, atomic checklist in atomic mode)")
    ap.add_argument("--prorubric-input", help="ProRubric train.parquet (atomic mode only; supplies K per question)")
    ap.add_argument("--output", required=True, help="output release directory")
    ap.add_argument("--artifact-id")
    ap.add_argument("--derived-from")
    args = ap.parse_args()

    if args.mode == "prorubric":
        run_prorubric(args.input, args.output,
                args.artifact_id or "medical-prorubric-appr",
                args.derived_from or "medical-prorubric")
    else:
        if not args.prorubric_input:
            ap.error("atomic mode requires --prorubric-input")
        run_atomic(args.input, args.prorubric_input, args.output,
                   args.artifact_id or "medical-atomic-appr",
                   args.derived_from or "medical-atomic")


if __name__ == "__main__":
    main()
