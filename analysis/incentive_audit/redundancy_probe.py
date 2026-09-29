#!/usr/bin/env python3
"""Redundancy sub-score of the appropriateness probe, applied to the incentive audit's controlled variants.

The appropriateness probe (appendix "Appropriateness probe across domains") scores four criteria; the third,
"includes only material that helps and avoids repetition", is the redundancy sub-score. This script applies that
one criterion, with the evaluation judge and the official HealthBench grader template, to the audit's base answer
and its edits (E1 names checklist topics, E6 appends correct unrequested substance, E7 / E7L integrate material
into the answer, E6L appends at the same 50% budget). The drop in pass rate from the base answer to an edit is
what the appropriateness axis charges for that edit.

Produces:
  Sec. 5.4 / 5.5 and the cross-domain audit table: redundancy cost of added content, 17 to 38 points over both
    edits and 25 to 38 for added substance (all prompts)   -> --variants orig,E1,E6 --n 300
  Appendix "Where added material sits": the unequal-budget pass (E7 sized to E6, domains 31 points apart)
                                     -> --variants orig,E1,E6,E7 --n 300
    and the common-budget pass       -> --variants orig,E6L,E7L --n 300

Inputs: an incentive-audit work directory (incentive_audit.py): <workdir>/<domain>/{prompts,responses,edits}.jsonl.
Output: --out JSONL {"domain", "id", "variant", "met": bool|null, "chars"} (resumable; one row per judged variant)
and <out>.report.json with per-variant pass rates and the paired comparison on the prompts that have every variant.
The per-domain prompt set is the first --n ids in sorted order; the report is restricted to that set.

Judge: the evaluation judge (no role; paper: DeepSeek-V4-Pro), temperature 0, max_tokens 3000, thinking disabled,
three attempts per item.

  python3 redundancy_probe.py --workdir W --domains science,medicine,dialogue,writing --n 300 \
      --variants orig,E1,E6 --out c3_e1_e6.jsonl
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "reward"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "eval"))
from judge_client import JudgeUnavailableError, judge_from_env  # noqa: E402
from healthbench_judge import conversation_text, grader_prompt  # noqa: E402  (official HealthBench template)

C3 = ("The response should include only information that materially helps satisfy the request and should avoid "
      "unnecessary repetition or digression.")


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def strip_fence(v):
    v = v.strip()
    if v.startswith("```"):
        v = v.partition("\n")[2]
        if v.endswith("```"):
            v = v[:-3]
    return v.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--workdir", required=True, help="incentive-audit work directory")
    ap.add_argument("--domains", default="science,medicine", help="comma-separated: medicine,writing,dialogue,science")
    ap.add_argument("--n", type=int, default=150, help="first N prompt ids (sorted) per domain")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--variants", default="orig,E1,E6", help="which audit variants to judge")
    ap.add_argument("--out", required=True, help="verdict JSONL (resumable)")
    ap.add_argument("--judge-role", default=None, help="RUBRIC_JUDGE_<ROLE>_* role (default: evaluation judge)")
    ap.add_argument("--thinking", default="disabled", choices=["disabled", "enabled", "env"])
    a = ap.parse_args()
    judge = judge_from_env(a.judge_role, max_tokens=3000, max_concurrency=a.workers,
                           thinking=None if a.thinking == "env" else a.thinking)
    A = a.workdir
    out_path = a.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = {(r["domain"], r["id"], r["variant"]) for r in jl(out_path)}
    tasks, in_scope = [], set()
    for domain in a.domains.split(","):
        prompts = {p["id"]: p for p in jl(f"{A}/{domain}/prompts.jsonl")}
        resps = {r["id"]: r["response"] for r in jl(f"{A}/{domain}/responses.jsonl")}
        edits = {(r["id"], r["edit"]): r["text"] for r in jl(f"{A}/{domain}/edits.jsonl") if r.get("text")}
        ids = sorted(resps)[: a.n]
        in_scope |= {(domain, pid) for pid in ids}
        for pid in ids:
            want = a.variants.split(",")
            avail = {"orig": resps[pid],
                     **{k: edits.get((pid, k)) for k in ("E1", "E1b", "E6", "E7", "E6L", "E7L")}}
            for variant, text in [(v, avail.get(v)) for v in want]:
                if text and (domain, pid, variant) not in done:
                    tasks.append((domain, pid, variant, prompts[pid]["problem"], text))
    print("tasks", len(tasks), flush=True)

    def one(t):
        domain, pid, variant, q, text = t
        rendered = grader_prompt(conversation_text(q, text), {"weight": 1.0, "criterion": C3})
        for _ in range(3):
            try:
                content = judge.complete([{"role": "user", "content": rendered}], max_tokens=3000, temperature=0).raw
            except JudgeUnavailableError:
                continue
            try:
                met = json.loads(strip_fence(content)).get("criteria_met")
            except Exception:
                continue
            if isinstance(met, bool):
                return {"domain": domain, "id": pid, "variant": variant, "met": met, "chars": len(text)}
        return {"domain": domain, "id": pid, "variant": variant, "met": None, "chars": len(text)}

    if tasks:
        with ThreadPoolExecutor(a.workers) as ex, open(out_path, "a") as f:
            n = 0
            for fut in as_completed([ex.submit(one, t) for t in tasks]):
                f.write(json.dumps(fut.result()) + "\n"); f.flush(); n += 1
                if n % 100 == 0:
                    print("  %d/%d" % (n, len(tasks)), flush=True)
    # the report covers the first --n ids only, so passes of different size can share one --out file
    rows = [r for r in jl(out_path) if r.get("met") is not None and (r["domain"], r["id"]) in in_scope]
    report = {}
    for domain in a.domains.split(","):
        per = {}
        for v in a.variants.split(","):
            xs = [r for r in rows if r["domain"] == domain and r["variant"] == v]
            if xs:
                per[v] = {"n": len(xs), "pass_pct": round(100 * sum(r["met"] for r in xs) / len(xs), 1),
                          "mean_chars": round(sum(r["chars"] for r in xs) / len(xs))}
        # paired on the questions that have every requested variant
        want = [v for v in a.variants.split(",") if v in per]
        ids = set.intersection(*[{r["id"] for r in rows if r["domain"] == domain and r["variant"] == v} for v in want]) if len(want) > 1 else set()
        if ids:
            g = {(r["id"], r["variant"]): r["met"] for r in rows if r["domain"] == domain}
            per["paired"] = {"n": len(ids), **{f"{v}_pass": round(100 * sum(g[(i, v)] for i in ids) / len(ids), 1) for v in want}}
            if "orig" in want:   # the charge: pass-rate change from the base answer, in points
                per["paired"].update({f"{v}_minus_orig": round(per["paired"][f"{v}_pass"] - per["paired"]["orig_pass"], 1)
                                      for v in want if v != "orig"})
        report[domain] = per
    json.dump(report, open(out_path + ".report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
