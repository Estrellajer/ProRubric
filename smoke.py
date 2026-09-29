#!/usr/bin/env python3
"""End-to-end smoke test: run the whole package against a stub judge.

    python3 smoke.py [--keep]

Builds a six-question fixture, serves a stub OpenAI-compatible endpoint on
localhost, and runs generate -> package -> the control transforms (raw-AND,
w/o failure clauses, + appr. criterion, weighted, atomic-rw, grouped-mean) ->
grade -> summarize -> ghost share -> pairwise, plus the training judge's hard
and Graded verdict modes, asserting each step. It also checks that a
missing judge model fails loudly. Nothing but pyarrow and the standard library;
no network, no credentials, no GPU.

Run it after copying this directory somewhere else -- that is what it is for:
it catches the generator and the packager disagreeing about where the atomic
rubric and the row id live, and a judge error that hides its own cause.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


class StubJudge(BaseHTTPRequestHandler):
    """Chat-completions stub: splits the atomic list in half for the generator,
    passes every criterion for the grader, and always picks B for pairwise (so
    the order-swap check must resolve it as a tie)."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        if "atomic_indices" in prompt:
            import re
            n = int(re.search(r"index from 1 to (\d+)", prompt).group(1))
            half = max(1, n // 2)
            content = json.dumps({"criteria": [
                {"name": "Core answer", "description": "The answer gets the core question right and is safe.",
                 "weight": 0, "atomic_indices": list(range(1, half + 1))},
                {"name": "Coverage", "description": "The answer covers the remaining necessary ground.",
                 "weight": 0, "atomic_indices": list(range(half + 1, n + 1))}]})
        elif "# Atomic items (idx: text)" in prompt:
            import re
            idx = [int(m) for m in re.findall(r"^(\d+): ", prompt.split("# Atomic items (idx: text)")[1], re.M)]
            content = json.dumps({"items": [{"idx": i, "criterion": "A good answer achieves point %d; it fails "
                                             "if it does not." % i} for i in idx]})
        elif "CHECKLIST ITEMS:" in prompt:
            import re
            n = int(re.search(r"with exactly (\d+) labels", prompt).group(1))
            content = json.dumps({"labels": (["CRITICAL", "FORMAT"] + ["NORMAL"] * n)[:n]})
        elif "grade each criterion on a 0-3 scale" in prompt:
            import re
            n = len(re.findall(r'"index": \d+', prompt)) or 1
            content = json.dumps({"grades": {str(i): 2 for i in range(1, n + 1)}})
        elif '"winner"' in prompt:
            content = json.dumps({"winner": "B", "reason": "stub"})
        else:
            content = json.dumps({"explanation": "stub", "criteria_met": True})
        out = json.dumps({"choices": [{"message": {"content": content}}],
                          "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def write_fixture(work):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = [{
        "prompt": [{"role": "user", "content": "question %d?" % i}],
        "data_source": "smoke",
        "extra_info": {
            "id": "q%d" % i,
            "rubric": [{"criterion": "mentions fact %d" % k,
                        "weight": float(k + 1) * (-1 if k == 3 else 1)} for k in range(5)],
            "rubric_correct": "the reference answer mentions all five facts",
            "problem": "question %d?" % i,
        },
    } for i in range(6)]
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(work, "atomic.parquet"))
    for name in ("atomic", "prorubric"):
        with open(os.path.join(work, "%s_scores.jsonl" % name), "w") as f:
            for i in range(6):
                f.write(json.dumps({"id": "q%d" % i, "response": "answer %d" % i,
                                    "satisfied": [True, False], "score": 0.5}) + "\n")
    return len(rows)


def check_verdict_modes(env):
    """The training judge's reward-relevant verdict modes, in process, against the stub."""
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)  # proxy variables already stripped
    try:
        sys.path.insert(0, os.path.join(HERE, "reward"))
        import judge_client
        import rubric_judge  # noqa: F401  -- the reward manager imports without verl
        import rubric_judge_repeat  # noqa: F401
        rubric = [{"criterion": "states the diagnosis", "weight": 2.0}, {"criterion": "orders a test", "weight": 1.0}]
        graded = judge_client.judge_from_env("TRAIN", verdict_mode="graded").score("q?", "a", rubric, temperature=0.7)
        assert abs(graded.weighted_score - 2 / 3) < 1e-9 and graded.satisfied == [True, True], graded
        print("%-24s %s" % ("verdict mode: graded", "ok"))
        return 0
    except Exception as exc:  # noqa: BLE001 -- report and count
        print("%-24s FAILED  %s: %s" % ("verdict mode: graded", type(exc).__name__, exc))
        return 1
    finally:
        os.environ.clear()
        os.environ.update(saved)


def check_scripts(env):
    """Every script answers --help; every library module imports. Run from an unrelated cwd."""
    import glob
    bad = 0
    files = sorted(f for f in glob.glob(os.path.join(HERE, "**", "*.py"), recursive=True)
                   if os.path.basename(f) != "smoke.py" and "__pycache__" not in f)
    cwd = tempfile.mkdtemp(prefix="prorubric-help-")
    for f in files:
        src = open(f, encoding="utf-8").read()
        if '__name__ == "__main__"' in src or "argparse" in src:
            cmd = [sys.executable, f, "--help"]
        else:
            cmd = [sys.executable, "-c", "import sys, runpy; sys.path.insert(0, %r); sys.path.insert(0, %r); "
                   "runpy.run_path(%r, run_name='smoke_import')" % (os.path.dirname(f), os.path.join(HERE, "reward"), f)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, timeout=120)
        if r.returncode != 0:
            bad += 1
            print("%-24s FAILED  %s\n%s" % ("--help/import", os.path.relpath(f, HERE), (r.stderr or r.stdout).strip()[-400:]))
    shutil.rmtree(cwd, ignore_errors=True)
    print("%-24s %s (%d files)" % ("--help / import", "ok" if not bad else "FAILED", len(files)))
    return bad


def check_engine():
    """Compose each arm's override list against the vendored engine's config tree.

    This is the gate the configs fail first when the engine is wrong: Hydra
    rejects an override for a key the schema does not have. It needs hydra and
    omegaconf, but no GPU, no ray and no rollout backend.
    """
    import glob
    try:
        import yaml
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError as exc:
        print("\nengine check skipped (%s)" % exc)
        return 0

    cfg_dir = os.environ.get("VERL_CONFIG_DIR", "")
    if not os.path.isdir(cfg_dir):
        print("\nengine check skipped: set VERL_CONFIG_DIR to a verl checkout's "
              "verl/trainer/config to run it")
        return 0
    for k in ("PRORUBRIC_MODEL_PATH", "PRORUBRIC_TRAIN_PARQUET", "PRORUBRIC_VAL_HELDOUT_PARQUET",
              "PRORUBRIC_VAL_BENCH_PARQUET", "PRORUBRIC_OUTPUT_DIR", "PRORUBRIC_PROJECT", "PRORUBRIC_EXPERIMENT",
              "PRORUBRIC_TEACHER_PATH", "PRORUBRIC_CONTEXT_PROFILE"):
        os.environ.setdefault(k, "/placeholder")

    print()
    bad = 0
    for f in sorted(glob.glob(os.path.join(HERE, "configs", "*.yaml"))):
        overrides = [x for x in (yaml.safe_load(open(f)) or [])
                     if isinstance(x, str) and "=" in x and not x.startswith("hydra.")]
        try:
            with initialize_config_dir(config_dir=cfg_dir, version_base=None):
                name = "sft_trainer_engine" if os.path.basename(f) == "sft.yaml" else "ppo_trainer"
                cfg = compose(config_name=name, overrides=overrides)
            OmegaConf.resolve(cfg)
            print("engine: %-13s composes (%d overrides)" % (os.path.basename(f), len(overrides)))
        except Exception as exc:
            bad += 1
            print("engine: %-13s FAILED  %s" % (os.path.basename(f), str(exc).replace("\n", " ")[:160]))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="keep the temporary work directory")
    ap.add_argument("--engine", action="store_true",
                    help="also compose every arm config against a verl checkout "
                         "(set VERL_CONFIG_DIR to its verl/trainer/config; needs hydra)")
    ap.add_argument("--port", type=int, default=0, help="0 picks a free port")
    args = ap.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), StubJudge)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    work = tempfile.mkdtemp(prefix="prorubric-smoke-")
    n = write_fixture(work)
    p = lambda *a: os.path.join(work, *a)

    env = {k: v for k, v in os.environ.items()
           if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
    env.update(RUBRIC_JUDGE_BASE_URL="http://127.0.0.1:%d" % port,
               RUBRIC_JUDGE_API_KEY="stub", RUBRIC_JUDGE_MODEL="stub-model")

    responses = ["--responses", "atomic=" + p("atomic_scores.jsonl"),
                 "--responses", "prorubric=" + p("prorubric_scores.jsonl")]
    steps = [
        ("generate", [os.path.join(HERE, "generate/generate_dimensions.py"), "--variant", "anchored",
                      "--input", p("atomic.parquet"), "--output", p("dimensions.jsonl"),
                      "--workers", "2", "--rpm", "600"]),
        ("package", [os.path.join(HERE, "data/build_prorubric_release.py"),
                     "--base-parquet", p("atomic.parquet"), "--gen-jsonl", p("dimensions.jsonl"),
                     "--out-dir", p("rel"), "--artifact-id", "smoke", "--mode", "protocol"]),
        ("arm: raw-AND", [os.path.join(HERE, "data/build_raw_and.py"),
                           "--input", p("rel/train.parquet"), "--output", p("and")]),
        ("arm: w/o failure", [os.path.join(HERE, "data/build_no_failure_clauses.py"),
                          "--input", p("rel/train.parquet"), "--output", p("noveto")]),
        ("arm: grouped-mean", [os.path.join(HERE, "data/build_grouped_mean.py"),
                               "--input", p("rel/train.parquet"), "--output", p("gmean")]),
        ("arm: +appr (ProRubric)", [os.path.join(HERE, "data/build_appr_criterion.py"), "--mode", "prorubric",
                                    "--input", p("rel/train.parquet"), "--output", p("appr_pr")]),
        ("arm: +appr (Rubric-RL)", [os.path.join(HERE, "data/build_appr_criterion.py"), "--mode", "atomic",
                                    "--input", p("atomic.parquet"), "--prorubric-input", p("rel/train.parquet"),
                                    "--output", p("appr_atomic")]),
        ("arm: weighted", [os.path.join(HERE, "data/build_weighted_atomic.py"), "--input", p("atomic.parquet"),
                           "--work", p("weighted_work"), "--output", p("weighted"), "--workers", "2"]),
        ("atomic-rw: rewrite", [os.path.join(HERE, "generate/rewrite_atomic.py"), "run",
                                "--input", p("atomic.parquet"), "--output", p("rw.jsonl"), "--workers", "2"]),
        ("atomic-rw: package", [os.path.join(HERE, "data/build_atomic_rw.py"), "--input", p("atomic.parquet"),
                                "--rewrites", p("rw.jsonl"), "--output", p("atomic_rw")]),
        ("grade", [os.path.join(HERE, "eval/healthbench_judge.py"), "run",
                   "--questions", p("rel/train.parquet"), *responses,
                   "--grades", p("grades.jsonl"), "--workers", "4", "--qpm", "600"]),
        ("grade (resume)", [os.path.join(HERE, "eval/healthbench_judge.py"), "run",
                            "--questions", p("rel/train.parquet"), *responses,
                            "--grades", p("grades.jsonl")]),
        ("summarize", [os.path.join(HERE, "eval/healthbench_judge.py"), "summarize",
                       "--questions", p("rel/train.parquet"), *responses,
                       "--grades", p("grades.jsonl")]),
        ("ghost share", [os.path.join(HERE, "eval/ghost_share.py"),
                         "--questions", p("rel/train.parquet"), *responses,
                         "--grades", p("grades.jsonl")]),
        ("harness: lite scores", [os.path.join(HERE, "eval/harness/score_records.py"),
                                  "--questions", p("rel/train.parquet"), "--responses", p("atomic_scores.jsonl"),
                                  "--scores", p("lite_scores.jsonl"), "--workers", "2"]),
        ("pairwise", [os.path.join(HERE, "eval/pairwise.py"), "run",
                      "--questions", p("rel/train.parquet"), *responses,
                      "--verdicts", p("pairwise.jsonl"), "--workers", "4", "--qpm", "600"]),
    ]

    failed = 0
    # a missing judge model must fail loudly, naming the variable to set
    no_model = {k: v for k, v in env.items() if k != "RUBRIC_JUDGE_MODEL"}
    r = subprocess.run([sys.executable, os.path.join(HERE, "generate/generate_dimensions.py"), "--variant",
                        "anchored", "--input", p("atomic.parquet"), "--output", p("never.jsonl")],
                       capture_output=True, text=True, env=no_model)
    ok = r.returncode != 0 and "RUBRIC_JUDGE_MODEL" in (r.stderr + r.stdout)
    print("%-24s %s" % ("no model -> clear error", "ok" if ok else "FAILED"))
    failed += not ok

    failed += check_verdict_modes(env)
    failed += check_scripts(env)

    for label, cmd in steps:
        r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, env=env)
        print("%-24s %s" % (label, "ok" if r.returncode == 0 else "FAILED"))
        if r.returncode != 0:
            failed += 1
            print((r.stderr or r.stdout).strip()[-800:])

    if not failed:
        kept = [json.loads(line) for line in open(p("dimensions.jsonl"))]
        assert len(kept) == n, "generator dropped questions: %d of %d" % (len(kept), n)
        assert all(r["id"].startswith("q") for r in kept), "generator ids do not match the fixture ids"
        import pyarrow.parquet as pq
        packaged = pq.read_table(p("rel/train.parquet")).to_pylist()
        assert len(packaged) == n, ("packaging dropped rows -- generator and packager disagree "
                                    "about the id or rubric field: %d of %d" % (len(packaged), n))
        ei = packaged[0]["extra_info"]
        assert [c["weight"] for c in ei["rubric"]] == [3.0, 12.0], \
            "criterion weight must be the sum of |atomic weight|, got %s" % [c["weight"] for c in ei["rubric"]]
        assert len(json.load(open(p("rel/manifest.json")))["outputs"]) == 1
        appr = pq.read_table(p("appr_pr/train.parquet")).to_pylist()[0]["extra_info"]["rubric"]
        assert len(appr) == 3 and appr[-1]["weight"] == 7.5, "appr. criterion weight must be the mean dimension weight"
        wt = pq.read_table(p("weighted/train.parquet")).to_pylist()[0]["extra_info"]["rubric"]
        assert [c["weight"] for c in wt][:2] == [3.0, 0.0], "weighted: CRITICAL x3, FORMAT x0"
        lite = [json.loads(line) for line in open(p("lite_scores.jsonl"))]
        assert len(lite) == n and all(r.get("score") == 1.0 and not r.get("error") for r in lite), lite[:1]
        rw = pq.read_table(p("atomic_rw/train.parquet")).to_pylist()[0]["extra_info"]["rubric"]
        assert len(rw) == 5 and rw[0]["criterion"].startswith("A good answer") and rw[3]["weight"] == -4.0
        print("\nassertions ok: %d questions survived generate -> package with matching ids and weights" % n)

    if args.engine and not failed:
        failed += check_engine()

    server.shutdown()
    if args.keep:
        print("work dir:", work)
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
