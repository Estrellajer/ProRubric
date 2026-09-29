# -*- coding: utf-8 -*-
"""Training dynamics: score saved checkpoints of the dynamics replicates.

Produces the numbers behind Figure `fig:training_dynamics`, the first paragraph of Sec. 5.4 ("Under additive
aggregation, appropriateness collapses early") and the appendix paragraph "Training dynamics" (the ordering of
fixing rates by the untrained answer's state).

The dynamics replicates repeat the medical 4B recipes of Rubric-RL and ProRubric (seeds 42/43/44) with
two config-only changes: more checkpoints kept and rollout dumps on. Every 50th checkpoint (a "point" =
arm x seed x step) answers a fixed 600-prompt set, decoded as in every other evaluation:

  * 300 medical training prompts of the incentive audit (analysis/incentive_audit, `<workdir>/medicine`), scored by
    the TRAINING judge (role TRAIN, hard verdicts, max_tokens 1024, thinking disabled; one whole-rubric call per
    prompt, exactly the call the incentive audit makes) under two rubrics of the same prompt:
      - the checklist (key A: the atomic checklist Rubric-RL trains on)   -> g_train
      - the ProRubric dimensions (key B)                                  -> dimension satisfaction, fixing rates
  * 300 HealthBench-consensus prompts (healthbench_consensus_300_ids.json: a fixed draw from the common set of
    Table `tab:consensus_ablations`) -> appropriateness c, graded with the official HealthBench GRADER_TEMPLATE, one
    call per criterion (eval/healthbench_judge.py's prompt, parser and score()); request settings as in the paper:
    temperature 0, thinking disabled, max_tokens 3000, up to 3 attempts per criterion. One judge deployment grades
    the whole trajectory, step 0 included (paper: a second deployment of DeepSeek-V4-Pro; role selectable).

Step 0 is the untrained model, shared by every arm and seed: its audit verdicts are the incentive audit's
`scores/<KEY>.jsonl` rows with variant "orig" (same judge, same settings), and its consensus answers are graded
under the key "base" from `--base`.

Inputs
  --point ARM:SEED:STEP=PATH   (repeatable) or --points-json FILE {"ARM": {"SEED": {"STEP": PATH}}}.
      PATH is a responses JSONL with one object per row: {"id", "response"}. Rows are joined BY ID: audit ids are the
      ids of <audit-dir>/prompts.jsonl, consensus ids those of --ids; any other rows are ignored.
  --audit-dir   incentive-audit medicine directory: prompts.jsonl {"id", "problem", "rubrics": {KEY: [{"criterion",
      "weight"}]}} and scores/<KEY>.jsonl {"id", "variant", "satisfied": [[bool]], "score"} for the untrained model.
  --questions   HealthBench-consensus parquet (eval/healthbench_judge.py layout);  --ids  JSON list of consensus ids.

Outputs (in --out-dir, append-only, resumable):
  audit_verdicts.jsonl     {"point", "rubric", "id", "satisfied": [[bool]], "score", "chars"} (or "error")
  consensus_grades.jsonl   {"arm": point, "id", "k", "met"}  -- the grades layout of eval/healthbench_judge.py
  summary.json / summary.md (subcommand summary)
A point key is "ARM/sSEED/stNNN"; the untrained model's consensus key is "base".

  python3 score_checkpoints.py audit     --audit-dir W/medicine --out-dir OUT --point prorubric:42:50=resp.jsonl ...
  python3 score_checkpoints.py consensus --questions healthbench_consensus.parquet --ids healthbench_consensus_300_ids.json \
          --out-dir OUT --base base_answers.jsonl --point prorubric:42:50=resp.jsonl ...
  python3 score_checkpoints.py summary   --audit-dir W/medicine --questions ... --ids ... --out-dir OUT \
          [--reference-grades table3_grades.jsonl --reference prorubric=B42,B43,B44 ...] \
          [--events prorubric:42=events.jsonl ... --original-events prorubric:42=events.jsonl ...]
"""
import argparse, json, os, re, statistics, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "reward"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "eval"))
from judge_client import judge_from_env, JudgeUnavailableError  # noqa: E402
import healthbench_judge as HB  # noqa: E402

CONSENSUS_MAX_TOKENS = 3000   # request body of the consensus grader: temperature 0, thinking disabled, 3000 tokens
CONSENSUS_ATTEMPTS = 3        # a criterion whose call fails or whose verdict does not parse is asked again, up to 3x
AUDIT_MAX_TOKENS = 1024       # training-judge call of the incentive audit
MIN_AUDIT_ROWS = 290          # a point enters the base-state table once >= 290 of the 300 audit prompts are scored
DRIFT_MARGIN = 0.05           # entropy-drift flag: replicate drift > max(drift of that arm's original seeds) + 0.05
SMETRICS = ("actor/entropy", "actor/grad_norm", "critic/score/mean", "response_length/mean")
STATES = (("zero", "no dimension met"), ("middle", "partly met"), ("one_short", "one dimension short"))
BASE_KEY = "base"
KEY_RE = re.compile(r"^(.*)/s([^/]+)/st(\d+)$")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p and os.path.exists(p) else []


def pkey(arm, seed, step):
    return "%s/s%s/st%03d" % (arm, seed, int(step))


def split_key(key):
    m = KEY_RE.match(key)
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def parse_points(a):
    pts = {}
    for spec in a.point or []:
        head, sep, path = spec.partition("=")
        parts = head.split(":")
        if not sep or len(parts) != 3:
            raise SystemExit("--point must be ARM:SEED:STEP=PATH, got %r" % spec)
        pts[pkey(*parts)] = path
    if a.points_json:
        for arm, seeds in json.load(open(a.points_json)).items():
            for seed, steps in seeds.items():
                for step, path in steps.items():
                    pts[pkey(arm, seed, step)] = path
    return pts


def responses(path):
    return {str(r["id"]): r["response"] for r in jl(path) if r.get("response")}


def sat_row(r):
    s = r["satisfied"]
    s = json.loads(s) if isinstance(s, str) else s
    return [bool(x) for x in s[0]]


# ---------------------------------------------------------------- (a) training judge on the audit prompts
def cmd_audit(a):
    prompts = {p["id"]: p for p in jl(os.path.join(a.audit_dir, "prompts.jsonl"))}
    out = os.path.join(a.out_dir, "audit_verdicts.jsonl")
    os.makedirs(a.out_dir, exist_ok=True)
    done = {(r["point"], r["rubric"], r["id"]) for r in jl(out) if r.get("satisfied")}
    jd = judge_from_env(a.judge_role or None, verdict_mode="hard", thinking="disabled", max_tokens=AUDIT_MAX_TOKENS,
                        max_concurrency=a.workers, qpm=a.qpm)
    for key, path in sorted(parse_points(a).items()):
        resp = responses(path)
        for rubric in a.rubrics.split(","):
            todo = [pid for pid in prompts if pid in resp and (key, rubric, pid) not in done]
            if not todo:
                continue

            def one(pid, rubric=rubric, key=key):
                try:
                    r = jd.score(prompts[pid]["problem"], resp[pid], prompts[pid]["rubrics"][rubric])
                    return {"point": key, "rubric": rubric, "id": pid, "satisfied": [list(map(bool, r.satisfied))],
                            "score": r.weighted_score, "chars": len(resp[pid])}
                except JudgeUnavailableError as exc:   # recorded; the next run retries it
                    return {"point": key, "rubric": rubric, "id": pid, "error": "%s: %s" % (type(exc).__name__, str(exc)[:160])}
            bad = 0
            with ThreadPoolExecutor(a.workers) as ex, open(out, "a", encoding="utf-8") as f:
                for row in ex.map(one, todo):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); bad += "error" in row
            print("audit %s rubric %s: %d scored, %d errors" % (key, rubric, len(todo), bad), flush=True)


# ---------------------------------------------------------------- (b) consensus c, official template
def grade_one(judge, task):
    key, rid, k, conversation, item = task
    err = None
    for _ in range(CONSENSUS_ATTEMPTS):
        try:
            c = judge.complete([{"role": "user", "content": HB.grader_prompt(conversation, item)}],
                               max_tokens=CONSENSUS_MAX_TOKENS, temperature=0)
            return {"arm": key, "id": rid, "k": k, "met": HB.parse_verdict(c.raw), "chars": len(c.raw)}
        except (JudgeUnavailableError, ValueError) as exc:   # json.JSONDecodeError is a ValueError
            err = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    return {"arm": key, "id": rid, "k": k, "met": None, "err": err}


def cmd_consensus(a):
    qs = HB.load_questions(a.questions)
    ids = json.load(open(a.ids))
    grades = a.grades or os.path.join(a.out_dir, "consensus_grades.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(grades)), exist_ok=True)
    done = HB.load_grades(grades)
    points = parse_points(a)
    if a.base:
        points[BASE_KEY] = a.base
    judge = judge_from_env(a.judge_role or None, thinking="disabled", max_tokens=CONSENSUS_MAX_TOKENS,
                           max_concurrency=a.workers, qpm=a.qpm)
    for key, path in sorted(points.items()):
        resp = responses(path)
        tasks = []
        for rid in ids:
            if rid not in resp:
                continue
            conv = HB.conversation_text(qs[rid]["prompt"], resp[rid])
            tasks += [(key, rid, k, conv, it) for k, it in enumerate(qs[rid]["rubric"]) if (key, rid, k) not in done]
        if not tasks:
            continue
        bad = 0
        with ThreadPoolExecutor(a.workers) as ex, open(grades, "a", encoding="utf-8") as f:
            for fu in as_completed([ex.submit(grade_one, judge, t) for t in tasks]):
                r = fu.result()
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); bad += r["met"] is None
        print("consensus %s: %d calls, %d errors" % (key, len(tasks), bad), flush=True)


# ---------------------------------------------------------------- summary
def audit_metrics(rows):
    rows = [r for r in rows if r.get("satisfied")]
    if not rows:
        return None
    dims = sum(len(sat_row(r)) for r in rows); sat = sum(sum(sat_row(r)) for r in rows)
    return {"n": len(rows), "dim_sat": round(100 * sat / dims, 2), "score": round(100 * statistics.mean(r["score"] for r in rows), 2)}


def cons_metric(grades, key, ids, qs):
    vals = []
    for rid in ids:
        rub = qs[rid]["rubric"]
        mets = [grades.get((key, rid, k)) for k in range(len(rub))]
        if all(m is not None for m in mets):
            vals.append(HB.score(mets, rub))
    return {"n": len(vals), "c": round(100 * statistics.mean(vals), 2)} if vals else None


def base_state(v):
    """Question state from the untrained answer: s0 = met dimensions of k."""
    k, s0 = len(v), sum(v)
    return "zero" if s0 == 0 else ("all" if s0 == k else ("one_short" if s0 == k - 1 else "middle"))


def state_table(base, after):
    """Per base state: questions, dimension satisfaction after training, and the fixing rate = of the dimensions unmet
    at step 0, the share met at step N. 'overall' adds the totals: satisfaction before/after, fixed, lost."""
    g, tot = {}, {"k": 0, "sat0": 0, "sat": 0, "uns": 0, "fix": 0, "lost": 0, "q": 0}
    for pid, b in base.items():
        a = after.get(pid)
        if a is None or len(a) != len(b):
            continue
        x = g.setdefault(base_state(b), {"q": 0, "k": 0, "sat": 0, "uns": 0, "fix": 0})
        fix = sum(1 for bv, av in zip(b, a) if not bv and av)
        x["q"] += 1; x["k"] += len(b); x["sat"] += sum(a); x["uns"] += sum(1 for v in b if not v); x["fix"] += fix
        tot["q"] += 1; tot["k"] += len(b); tot["sat0"] += sum(b); tot["sat"] += sum(a)
        tot["uns"] += sum(1 for v in b if not v); tot["fix"] += fix; tot["lost"] += sum(1 for bv, av in zip(b, a) if bv and not av)
    out = {k: {"questions": v["q"], "dim_sat": round(100 * v["sat"] / v["k"], 1),
               "fix_rate": round(100 * v["fix"] / v["uns"], 1) if v["uns"] else None, "unmet_at_0": v["uns"]} for k, v in g.items()}
    if tot["k"]:
        met0 = tot["k"] - tot["uns"]
        out["overall"] = {"questions": tot["q"], "dims": tot["k"], "dim_sat_0": round(100 * tot["sat0"] / tot["k"], 1),
                          "dim_sat": round(100 * tot["sat"] / tot["k"], 1),
                          "fix_rate": round(100 * tot["fix"] / tot["uns"], 1) if tot["uns"] else None,
                          "lost_rate": round(100 * tot["lost"] / met0, 1) if met0 else None}
    return out


def read_events(paths):
    """tracking/events.jsonl rows {"step", "data": {metric: value}}; several files (a resumed run) merge in order."""
    out = {}
    for p in paths:
        for l in open(p, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("step") is not None:
                out.setdefault(r["step"], {}).update({k: v for k, v in (r.get("data") or {}).items() if k in SMETRICS})
    return out


def ent_drift(e):
    """mean entropy over steps 251-300 minus steps 151-200 (None unless both windows have >= 40 steps)."""
    w = lambda lo, hi: [e[x]["actor/entropy"] for x in range(lo, hi + 1) if "actor/entropy" in e.get(x, {})]
    a, b = w(251, 300), w(151, 200)
    return round(statistics.mean(a) - statistics.mean(b), 4) if len(a) >= 40 and len(b) >= 40 else None


def parse_events(specs):
    out = {}
    for spec in specs or []:
        head, _, paths = spec.partition("=")
        arm, _, seed = head.partition(":")
        out[(arm, seed)] = paths.split(",")
    return out


def stability(rep, orig):
    """grad-norm spikes (> 3x the run's median) and entropy drift per replicate; a replicate is flagged when its drift
    exceeds the largest drift among that arm's original (table) seeds by more than DRIFT_MARGIN."""
    od_all = {k: ent_drift(read_events(p)) for k, p in orig.items()}
    out = {}
    for (arm, seed), paths in sorted(rep.items()):
        e = read_events(paths)
        g = [e[x]["actor/grad_norm"] for x in sorted(e) if "actor/grad_norm" in e[x]]
        med = statistics.median(g) if g else None
        spikes = [(x, round(e[x]["actor/grad_norm"], 3)) for x in sorted(e) if med and e[x].get("actor/grad_norm", 0) > 3 * med]
        dr = ent_drift(e); od = [v for (a2, _), v in od_all.items() if a2 == arm and v is not None]
        every10 = {x: {k.split("/")[-2] if k.endswith("/mean") else k.split("/")[-1]: round(e[x][k], 4) for k in SMETRICS if k in e[x]}
                   for x in range(0, max(e) + 1, 10) if x in e} if e else {}
        out["%s/s%s" % (arm, seed)] = {"steps": max(e) if e else 0, "grad_median": round(med, 4) if med else None,
                                       "grad_spikes_gt3x_median": spikes, "entropy_drift_251_300_minus_151_200": dr,
                                       "original_seeds_drift": {"s%s" % s2: v for (a2, s2), v in od_all.items() if a2 == arm},
                                       "entropy_drift_flag": dr is not None and bool(od) and dr > max(od) + DRIFT_MARGIN,
                                       "every10": every10}
    return out


def agg(xs, nseeds):
    """mean +- sd (ddof=1) over seeds, filled only when every seed has the value."""
    if len(xs) == nseeds and nseeds >= 2:
        return {"mean": round(statistics.mean(xs), 2), "sd": round(statistics.stdev(xs), 2), "n": nseeds}
    return {"n": len(xs)}


def cmd_summary(a):
    qs = HB.load_questions(a.questions)
    ids = json.load(open(a.ids))
    grades = HB.load_grades(a.grades or os.path.join(a.out_dir, "consensus_grades.jsonl"))
    chk, dims = a.checklist_key, a.dimensions_key
    audit = {}
    for r in jl(os.path.join(a.out_dir, "audit_verdicts.jsonl")):
        audit.setdefault((r["point"], r["rubric"]), []).append(r)
    base_rows = {rb: [r for r in jl(os.path.join(a.audit_dir, "scores", "%s.jsonl" % rb)) if r.get("variant") == "orig"]
                 for rb in (chk, dims)}
    base_dims = {r["id"]: sat_row(r) for r in base_rows[dims] if r.get("satisfied")}

    keys = {p for p, _ in audit} | {k for k, _, _ in grades}
    pts = sorted(filter(None, (split_key(k) for k in keys)), key=lambda t: (t[0], t[1], t[2]))
    arms = list(dict.fromkeys(t[0] for t in pts))
    seeds = a.seeds.split(",") if a.seeds else sorted({t[1] for t in pts})
    steps = sorted({t[2] for t in pts})

    out = {"base": {"g_train": audit_metrics(base_rows[chk]), "dimensions": audit_metrics(base_rows[dims]),
                    "consensus_c": cons_metric(grades, BASE_KEY, ids, qs)},
           "base_state_table": state_table(base_dims, base_dims), "seeds": {}}
    for sd in seeds:
        S = {"points": {}, "dimensions_by_base_state": {}}
        for arm in arms:
            for st in steps:
                key = pkey(arm, sd, st)
                m = {"g_train": audit_metrics(audit.get((key, chk), [])),
                     "dimensions": audit_metrics(audit.get((key, dims), [])),
                     "consensus_c": cons_metric(grades, key, ids, qs)}
                if any(m.values()):
                    S["points"]["%s/%d" % (arm, st)] = m
                after = {r["id"]: sat_row(r) for r in audit.get((key, dims), []) if r.get("satisfied")}
                if len(after) >= MIN_AUDIT_ROWS:
                    S["dimensions_by_base_state"]["%s/%d" % (arm, st)] = state_table(base_dims, after)
        out["seeds"][sd] = S

    # mean +- sd over seeds, per arm x step
    mean, bsm = {}, {}
    for arm in arms:
        for st in steps:
            p = "%s/%d" % (arm, st)
            per = [out["seeds"][sd]["points"].get(p) or {} for sd in seeds]
            mean[p] = {"g_train_score": agg([x["g_train"]["score"] for x in per if x.get("g_train")], len(seeds)),
                       "dimensions_dim_sat": agg([x["dimensions"]["dim_sat"] for x in per if x.get("dimensions")], len(seeds)),
                       "consensus_c": agg([x["consensus_c"]["c"] for x in per if x.get("consensus_c")], len(seeds))}
            tabs = [out["seeds"][sd]["dimensions_by_base_state"].get(p) for sd in seeds]
            tabs = [t for t in tabs if t]
            if tabs:   # mean over the seeds available
                bsm[p] = {k: {"dim_sat": round(statistics.mean(t[k]["dim_sat"] for t in tabs if k in t), 1),
                              "fix_rate": round(statistics.mean(t[k]["fix_rate"] for t in tabs if k in t and t[k]["fix_rate"] is not None), 1),
                              "n_seeds": sum(1 for t in tabs if k in t)} for k, _ in STATES if any(k in t for t in tabs)}
    out["mean_over_seeds"] = mean
    out["dimensions_by_base_state_mean"] = bsm

    # fixing-rate ordering zero < middle < one_short at the check step, per arm and seed
    order = {}
    for arm in arms:
        for sd in seeds:
            t = out["seeds"][sd]["dimensions_by_base_state"].get("%s/%d" % (arm, a.check_step))
            if t and all(k in t and t[k]["fix_rate"] is not None for k, _ in STATES):
                r = [t[k]["fix_rate"] for k, _ in STATES]
                order["%s/s%s" % (arm, sd)] = {"fix_rates": r, "zero<middle<one_short": r[0] < r[1] < r[2]}
    out["fixing_rate_ordering_at_check_step"] = order

    # per step: every seed of one arm below every seed of another?
    sep = {}
    for st in steps:
        vals = {arm: [(out["seeds"][sd]["points"].get("%s/%d" % (arm, st)) or {}).get("consensus_c") for sd in seeds] for arm in arms}
        vals = {arm: [v["c"] for v in vs if v] for arm, vs in vals.items()}
        vals = {arm: vs for arm, vs in vals.items() if len(vs) == len(seeds)}
        sep[str(st)] = {"%s<%s" % (x, y): max(vals[x]) < min(vals[y]) for x in vals for y in vals if x != y}
    out["consensus_c_all_seeds_below"] = sep

    # the same 300 ids graded for other checkpoints (e.g. the table's step-300 checkpoints) on a reference grades file
    if a.reference_grades and a.reference:
        ref = HB.load_grades(a.reference_grades)
        out["reference_check"] = {}
        for spec in a.reference:
            arm, _, names = spec.partition("=")
            refs = {n: cons_metric(ref, n, ids, qs) for n in names.split(",")}
            vals = [v["c"] for v in refs.values() if v]
            per = {}
            for sd in seeds:
                c = cons_metric(grades, pkey(arm, sd, a.check_step), ids, qs)
                if c:
                    per["s%s" % sd] = {"replicate": c, "inside_reference_range": (min(vals) <= c["c"] <= max(vals)) if vals else None}
            out["reference_check"][arm] = {"reference": refs, "range": [min(vals), max(vals)] if vals else None, "replicates": per}

    if a.events:
        out["stability"] = stability(parse_events(a.events), parse_events(a.original_events))

    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1, ensure_ascii=False)
    md = markdown(out, arms, seeds, steps, a.check_step)
    open(os.path.join(a.out_dir, "summary.md"), "w").write(md)
    print(md)


def markdown(out, arms, seeds, steps, check_step):
    fm = lambda d: ("%.2f ± %.2f" % (d["mean"], d["sd"])) if "mean" in d else "(%d seeds)" % d.get("n", 0)
    v = lambda m, k: str(m[k]) if m else "-"
    b = out["base"]
    L = ["# Training dynamics: checkpoint scores", "",
         "g_train = checklist score, training judge; dimensions = ProRubric dimensions, same judge (300 audit prompts); "
         "c = HealthBench consensus on the fixed 300 ids. Values x100. Step 0 (untrained): g_train %s, dimension "
         "satisfaction %s, c %s." % (v(b["g_train"], "score"), v(b["dimensions"], "dim_sat"), v(b["consensus_c"], "c")), "",
         "## Mean ± sd over seeds %s (ddof=1; filled only when every seed has the point)" % "/".join(seeds), "",
         "| arm | step | g_train score | dimension satisfaction | consensus c |", "|---|---|---|---|---|"]
    for p, d in out["mean_over_seeds"].items():
        arm, st = p.rsplit("/", 1)
        L.append("| %s | %s | %s | %s | %s |" % (arm, st, fm(d["g_train_score"]), fm(d["dimensions_dim_sat"]), fm(d["consensus_c"])))
    for sd in seeds:
        L += ["", "### seed %s" % sd, "", "| arm | step | g_train score | g_train dim-sat | dimensions score | dimension satisfaction | c (n) |",
              "|---|---|---|---|---|---|---|"]
        for p, m in out["seeds"][sd]["points"].items():
            arm, st = p.rsplit("/", 1)
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (arm, st, v(m["g_train"], "score"), v(m["g_train"], "dim_sat"),
                     v(m["dimensions"], "score"), v(m["dimensions"], "dim_sat"),
                     "%s (%d)" % (m["consensus_c"]["c"], m["consensus_c"]["n"]) if m["consensus_c"] else "-"))
    b0 = out["base_state_table"]
    cell = lambda d, k: ("%s / %s" % (d[k]["dim_sat"], d[k]["fix_rate"])) if k in d else "-"
    L += ["", "## ProRubric dimensions by the untrained answer's state (dimension satisfaction % / fixing rate %)", "",
          "State per question from the untrained answer: zero = no dimension met, one-short = all but one met, middle = "
          "otherwise; questions with every dimension met are omitted. Fixing rate = of the dimensions unmet at step 0, "
          "the share met at step N. Questions (dimensions unmet at step 0): " +
          ", ".join("%s %d (%d)" % (lab, b0[k]["questions"], b0[k]["unmet_at_0"]) for k, lab in STATES if k in b0), "",
          "| arm | step | zero | middle | one-short | seeds |", "|---|---|---|---|---|---|"]
    for p, d in out["dimensions_by_base_state_mean"].items():
        arm, st = p.rsplit("/", 1)
        L.append("| %s | %s | %s | %s |" % (arm, st, " | ".join(cell(d, k) for k, _ in STATES), max((d[k]["n_seeds"] for k in d), default=0)))
    L += ["", "Fixing-rate ordering at step %d (zero < middle < one-short):" % check_step]
    L += ["- %s: %s -> %s" % (k, x["fix_rates"], x["zero<middle<one_short"]) for k, x in out["fixing_rate_ordering_at_check_step"].items()]
    L += ["", "Consensus c, every seed of arm X below every seed of arm Y:"]
    for st, d in out["consensus_c_all_seeds_below"].items():
        L.append("- step %s: %s" % (st, ", ".join(k for k, ok in d.items() if ok) or "none"))
    for arm, r in (out.get("reference_check") or {}).items():
        L.append("- step-%d reference check %s: replicates %s vs reference %s" % (
            check_step, arm, {s: x["replicate"]["c"] for s, x in r["replicates"].items()}, {k: (o["c"] if o else None) for k, o in r["reference"].items()}))
    if out.get("stability"):
        L += ["", "## Training stability (events.jsonl)", "", "| replicate | steps | grad spikes (>3x median) | entropy drift | originals' drift | flag |",
              "|---|---|---|---|---|---|"]
        for k, s in out["stability"].items():
            L.append("| %s | %d | %s | %s | %s | %s |" % (k, s["steps"], ", ".join("%d (%.2f)" % t for t in s["grad_spikes_gt3x_median"][:5]) or "none",
                     s["entropy_drift_251_300_minus_151_200"], s["original_seeds_drift"], "DRIFT" if s["entropy_drift_flag"] else ""))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def points(p):
        p.add_argument("--point", action="append", metavar="ARM:SEED:STEP=PATH", help="responses JSONL of one checkpoint (repeatable)")
        p.add_argument("--points-json", help='{"ARM": {"SEED": {"STEP": PATH}}}')
        p.add_argument("--out-dir", required=True)

    p = sub.add_parser("audit", help="training-judge verdicts on the audit prompts (checklist and dimensions)")
    points(p)
    p.add_argument("--audit-dir", required=True, help="incentive-audit medicine directory (prompts.jsonl)")
    p.add_argument("--rubrics", default="A,B", help="rubric keys of prompts.jsonl to score (default A,B)")
    p.add_argument("--judge-role", default="TRAIN", help="judge role (default TRAIN: the training-reward judge)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--qpm", type=int, default=600)

    p = sub.add_parser("consensus", help="HealthBench-consensus c, official template, one call per criterion")
    points(p)
    p.add_argument("--questions", required=True, help="HealthBench-consensus parquet")
    p.add_argument("--ids", required=True, help="JSON list of consensus ids (healthbench_consensus_300_ids.json)")
    p.add_argument("--base", help="untrained model's responses JSONL, graded under the key 'base'")
    p.add_argument("--grades", help="grades JSONL (default OUT_DIR/consensus_grades.jsonl)")
    p.add_argument("--judge-role", default="", help="judge role (default: none = the evaluation judge)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--qpm", type=int, default=70)

    p = sub.add_parser("summary", help="write summary.json / summary.md")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--audit-dir", required=True, help="incentive-audit medicine directory (scores/<KEY>.jsonl = step 0)")
    p.add_argument("--questions", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--grades", help="grades JSONL (default OUT_DIR/consensus_grades.jsonl)")
    p.add_argument("--checklist-key", default="A", help="rubric key of the checklist (g_train); default A")
    p.add_argument("--dimensions-key", default="B", help="rubric key of the ProRubric dimensions; default B")
    p.add_argument("--seeds", help="comma-separated seed order (default: every seed found)")
    p.add_argument("--check-step", type=int, default=300, help="step of the ordering / reference checks (default 300)")
    p.add_argument("--reference-grades", help="grades JSONL holding other runs graded on the same ids")
    p.add_argument("--reference", action="append", metavar="ARM=KEY1,KEY2,...",
                   help="compare ARM's check-step c per seed with the range of these keys in --reference-grades")
    p.add_argument("--events", action="append", metavar="ARM:SEED=PATH[,PATH]", help="replicate events.jsonl (stability)")
    p.add_argument("--original-events", action="append", metavar="ARM:SEED=PATH[,PATH]",
                   help="events.jsonl of the arm's original seeds (entropy-drift threshold)")

    a = ap.parse_args()
    {"audit": cmd_audit, "consensus": cmd_consensus, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
