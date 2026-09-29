#!/usr/bin/env python3
"""How often the fixed appropriateness criterion is judged unmet, and how often that verdict is unwarranted.

Produces the appendix "A Fixed Appropriateness Criterion": the rate at which the training judge marks the criterion
unmet on the untrained model's answers (orig) and on each incentive-audit edit (E1 adds rubric terms, E4 deletes the
key sentence, E3 inserts a mildly inappropriate recommendation, E5 paraphrases), and the share of a sample of those
verdicts that the evaluation judge, adjudicating, calls unwarranted -- on detail-only variants (E1, and E1b where the
leak check found the padding clean) and on the untrained model's answers.

No new training-judge calls are needed: the incentive audit already scored every variant with the arm
"ProRubric + appr. criterion" (key V by default), whose LAST criterion is the appropriateness criterion.

  step 1  unmet rate of the criterion per variant, from <audit>/scores/<ARM>.jsonl (first draw's verdicts)
  step 2  for a sample of unmet verdicts (--n on detail-only variants, max(30, n // 3) on base answers, drawn with
          random.Random(0) after one shuffle), ask the evaluation judge whether the verdict is defensible
  step 3  unwarranted share = sampled verdicts the evaluation judge calls not a failure

Inputs: --audit-dir is the incentive audit's medicine directory (<workdir>/medicine) with prompts.jsonl,
responses.jsonl, edits.jsonl, e1b_leak.jsonl and scores/<ARM>.jsonl.
Outputs (--out-dir): adjudications.jsonl {"id", "variant", "fails": bool|null, "why"} (resumable) and report.json.

Judge: the evaluation judge (no role; paper: DeepSeek-V4-Pro), temperature 0, max_tokens 3000, thinking disabled,
three attempts per item.

  python3 appr_criterion_audit.py --audit-dir W/medicine --out-dir appr_fp --n 150
"""
import argparse, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reward"))
from judge_client import JudgeUnavailableError, judge_from_env  # noqa: E402

ADJUDICATE = """A grader applied ONE criterion to an answer and judged that the answer FAILS it. Decide whether that
judgement is defensible.

QUESTION:
{q}

ANSWER:
{r}

CRITERION THE ANSWER WAS JUDGED TO FAIL:
{c}

Answer as a careful clinician reviewing the grader. The answer fails the criterion only if it genuinely does what the
criterion forbids (for example: it buries the direct answer under background, it wanders outside what was asked, or it
addresses the wrong audience). Extra detail or length is NOT by itself a failure; neither is imperfect polish.

Return ONLY a JSON object {{"fails": true|false, "why": "<one short sentence>"}}."""


def log(msg):
    print(time.strftime("[%H:%M:%SZ]", time.gmtime()) + " " + msg, flush=True)


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--audit-dir", required=True, help="incentive audit medicine directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", default="V", help="arm key of 'ProRubric + appr. criterion' in the audit (default V)")
    ap.add_argument("--n", type=int, default=150, help="unmet verdicts to adjudicate on detail-only variants")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--judge-role", default=None, help="RUBRIC_JUDGE_<ROLE>_* role (default: evaluation judge)")
    ap.add_argument("--thinking", default="disabled", choices=["disabled", "enabled", "env"])
    a = ap.parse_args()
    AUDIT, WORK = a.audit_dir, a.out_dir
    os.makedirs(WORK, exist_ok=True)
    prompts = {p["id"]: p for p in jl(f"{AUDIT}/prompts.jsonl")}
    resps = {r["id"]: r["response"] for r in jl(f"{AUDIT}/responses.jsonl")}
    edits = {(r["id"], r["edit"]): r["text"] for r in jl(f"{AUDIT}/edits.jsonl") if r.get("text")}
    leak = {r["id"]: r["leak"] for r in jl(f"{AUDIT}/e1b_leak.jsonl")}
    scores = jl(f"{AUDIT}/scores/{a.arm}.jsonl")
    assert prompts and scores, "the medicine incentive audit has to have run first"

    # step 1: unmet rate per variant. The appropriateness criterion is the last criterion of the arm's rubric.
    fired, per_variant = {}, {}
    for row in scores:
        sat = row.get("satisfied")
        if not sat:
            continue
        met = bool(sat[0][-1])
        fired[(row["id"], row["variant"])] = not met
        per_variant.setdefault(row["variant"], []).append(not met)
    log("appropriateness criterion unmet rate by variant (unmet / scored):")
    for v in sorted(per_variant):
        xs = per_variant[v]
        log("  %-14s %3d/%3d = %5.1f%%" % (v, sum(xs), len(xs), 100 * sum(xs) / len(xs)))
    orig = per_variant.get("orig", [])
    detail = [x for v in ("E1", "E1b") for x in per_variant.get(v, [])]
    if orig and detail:
        log("detail-only variants are marked unmet %.1f pp more often than the base answer (%.1f%% vs %.1f%%)"
            % (100 * sum(detail) / len(detail) - 100 * sum(orig) / len(orig),
               100 * sum(detail) / len(detail), 100 * sum(orig) / len(orig)))

    # step 2: adjudicate a sample of unmet verdicts on detail-only variants (E1b only where the padding leaked nothing)
    cands = []
    for (pid, variant), f in fired.items():
        if not f or variant not in ("E1", "E1b", "orig"):
            continue
        if variant == "E1b" and leak.get(pid, True):
            continue
        text = resps.get(pid) if variant == "orig" else edits.get((pid, variant))
        if text and pid in prompts:
            cands.append((pid, variant, text))
    random.Random(0).shuffle(cands)
    detail_c = [c for c in cands if c[1] != "orig"][: a.n]
    orig_c = [c for c in cands if c[1] == "orig"][: max(30, a.n // 3)]
    todo = detail_c + orig_c
    out_path = f"{WORK}/adjudications.jsonl"
    done = {(r["id"], r["variant"]) for r in jl(out_path)}
    todo = [c for c in todo if (c[0], c[1]) not in done]
    log("adjudicating %d unmet verdicts (%d detail-only, %d base answers); %d already done"
        % (len(todo), len(detail_c), len(orig_c), len(done)))
    judge = judge_from_env(a.judge_role, max_tokens=3000, max_concurrency=a.workers,
                           thinking=None if a.thinking == "env" else a.thinking)

    def one(c):
        pid, variant, text = c
        criterion = prompts[pid]["rubrics"][a.arm][-1]["criterion"]
        prompt = ADJUDICATE.format(q=prompts[pid]["problem"][:6000], r=text[:16000], c=criterion)
        for _ in range(3):
            try:
                content = judge.complete([{"role": "user", "content": prompt}], max_tokens=3000, temperature=0).raw
            except JudgeUnavailableError:
                continue
            try:
                d = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
                if isinstance(d.get("fails"), bool):
                    return {"id": pid, "variant": variant, "fails": d["fails"], "why": str(d.get("why", ""))[:300]}
            except Exception:
                continue
        return {"id": pid, "variant": variant, "fails": None}

    if todo:
        with ThreadPoolExecutor(a.workers) as ex, open(out_path, "a") as f:
            n = 0
            for fut in as_completed([ex.submit(one, c) for c in todo]):
                r = fut.result(); f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); n += 1
                if n % 25 == 0:
                    log("  adjudicated %d/%d" % (n, len(todo)))

    # step 3: report
    rows = [r for r in jl(out_path) if r.get("fails") is not None]
    report = {"appr_unmet_rate_pct": {v: round(100 * sum(xs) / len(xs), 2) for v, xs in sorted(per_variant.items())}}
    for label, want in (("detail_only", lambda v: v in ("E1", "E1b")), ("base_answer", lambda v: v == "orig")):
        sub = [r for r in rows if want(r["variant"])]
        if not sub:
            continue
        fp = [r for r in sub if not r["fails"]]
        report[label] = {"adjudicated_unmet": len(sub), "judge_says_not_a_failure": len(fp),
                         "unwarranted_pct": round(100 * len(fp) / len(sub), 1),
                         "examples": [{"id": r["id"], "variant": r["variant"], "why": r["why"]} for r in fp[:5]]}
    json.dump(report, open(f"{WORK}/report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
