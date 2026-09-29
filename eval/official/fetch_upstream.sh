#!/usr/bin/env bash
# Download the pinned upstream files the official scorers read (prompts, criteria, questions,
# baseline answers). Nothing from these repositories is redistributed in this package.
#
#   bash fetch_upstream.sh DEST_DIR
#
# Layout produced (pass the per-benchmark directory as --upstream-dir / --questions):
#   DEST_DIR/WritingBench/{prompt.py,benchmark_query/benchmark_all.jsonl}
#   DEST_DIR/creative-writing-bench/data/{creative_writing_prompts_v3.json,creative_writing_criteria.txt,
#                                         negative_criteria.txt,creative_writing_judging_prompt.txt}
#   DEST_DIR/arena-hard-auto/{utils/judge_utils.py,data/arena-hard-v2.0/question.jsonl,
#                             data/arena-hard-v2.0/model_answer/{o3-mini-2025-01-31,gemini-2.0-flash-001}.jsonl}
#   DEST_DIR/ResearchQA/valid.json   (Hugging Face dataset realliyifei/ResearchQA)
# Each repository's own license applies to its files; see the repositories.
set -euo pipefail
DEST="${1:?usage: fetch_upstream.sh DEST_DIR}"
GH=https://raw.githubusercontent.com

WB_COMMIT=ae2d5176449b7b769815482641d35926f26793eb      # X-PLUG/WritingBench
CW_COMMIT=c7c3ceef54c40a8ae02dc1c2e1a5e40970fe5c0b      # EQ-bench/creative-writing-bench
AH_COMMIT=196f6b826783b3da7310e361a805fa36f0be83f3      # lmarena/arena-hard-auto
RQA_REV=bf8a4cfef073ecfc0275c57acf8ca960e4dc79d6        # HF dataset realliyifei/ResearchQA

get() {  # repo commit path dest_subdir
  mkdir -p "$(dirname "$DEST/$4/$3")"
  curl -fsSL "$GH/$1/$2/$3" -o "$DEST/$4/$3"
}
get X-PLUG/WritingBench "$WB_COMMIT" prompt.py WritingBench
get X-PLUG/WritingBench "$WB_COMMIT" benchmark_query/benchmark_all.jsonl WritingBench
for f in creative_writing_prompts_v3.json creative_writing_criteria.txt negative_criteria.txt creative_writing_judging_prompt.txt; do
  get EQ-bench/creative-writing-bench "$CW_COMMIT" "data/$f" creative-writing-bench
done
get lmarena/arena-hard-auto "$AH_COMMIT" utils/judge_utils.py arena-hard-auto
get lmarena/arena-hard-auto "$AH_COMMIT" data/arena-hard-v2.0/question.jsonl arena-hard-auto
for m in o3-mini-2025-01-31 gemini-2.0-flash-001; do
  get lmarena/arena-hard-auto "$AH_COMMIT" "data/arena-hard-v2.0/model_answer/$m.jsonl" arena-hard-auto
done
mkdir -p "$DEST/ResearchQA"
curl -fsSL "https://huggingface.co/datasets/realliyifei/ResearchQA/resolve/$RQA_REV/valid.json" -o "$DEST/ResearchQA/valid.json"

# MD5 of the files the reported numbers were produced with.
cat > "$DEST/MD5SUMS" <<'EOF'
f40c11b4956887be55632bb0ed9870ab  WritingBench/prompt.py
f4218d10824c19bc14daeaf25fb0d46e  WritingBench/benchmark_query/benchmark_all.jsonl
001a9cf215b4ba1b0b5632b8afd0eaae  creative-writing-bench/data/creative_writing_prompts_v3.json
b9a38853ede11c7b7beffbcb1258efaa  creative-writing-bench/data/creative_writing_criteria.txt
2d6c34cacfd9a15f645b25d6b5959f2f  creative-writing-bench/data/negative_criteria.txt
3c3dd047291f325a3f2bd61f3e28bc42  creative-writing-bench/data/creative_writing_judging_prompt.txt
ffba33626930a08d1876ee09c192494f  arena-hard-auto/utils/judge_utils.py
d65122e6e4fb2790a40b38228580ef07  arena-hard-auto/data/arena-hard-v2.0/question.jsonl
e7f1c9ec0c32e1b1b5e08d8d0e8b9aba  arena-hard-auto/data/arena-hard-v2.0/model_answer/o3-mini-2025-01-31.jsonl
4eec36dda880c6afe1b09d414e8eda19  arena-hard-auto/data/arena-hard-v2.0/model_answer/gemini-2.0-flash-001.jsonl
cda51b2bc527e76aade954efc5e7c224  ResearchQA/valid.json
EOF
( cd "$DEST" && if command -v md5sum >/dev/null; then md5sum -c MD5SUMS; else
    while read -r sum f; do [ "$(md5 -q "$f")" = "$sum" ] && echo "$f: OK" || { echo "$f: FAILED"; exit 1; }; done < MD5SUMS; fi )
echo "upstream files -> $DEST"
