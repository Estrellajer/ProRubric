#!/usr/bin/env bash
# Download the pinned openai/simple-evals files used by the HealthBench meta-evaluation
# scripts, plus the public meta-evaluation data file.
#
#   bash fetch_simple_evals.sh DEST_DIR
#
# Produces
#   DEST_DIR/simple_evals/{common.py,types.py,healthbench_eval.py,healthbench_meta_eval.py,
#                          sampler/chat_completion_sampler.py,LICENSE,__init__.py,sampler/__init__.py}
#   DEST_DIR/2025-05-07-06-14-12_oss_meta_eval.jsonl        (about 136 MB)
#
# simple-evals is MIT-licensed (Copyright (c) 2024 OpenAI); its LICENSE file is downloaded
# alongside the code. The upstream repository has no __init__.py files; empty ones are created
# so that `import simple_evals.healthbench_meta_eval` works with DEST_DIR on sys.path
# (pass it as --simple-evals-dir DEST_DIR). Importing healthbench_eval.py requires
# blobfile, numpy, pandas, jinja2, requests, tqdm and openai.
set -euo pipefail
DEST="${1:?usage: fetch_simple_evals.sh DEST_DIR}"
COMMIT=652c89d0ca9df547706735883097e9537d40dc47
RAW="https://raw.githubusercontent.com/openai/simple-evals/${COMMIT}"
DATA_URL="https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_meta_eval.jsonl"

mkdir -p "$DEST/simple_evals/sampler"
for f in LICENSE common.py types.py healthbench_eval.py healthbench_meta_eval.py sampler/chat_completion_sampler.py; do
  curl -fsSL "$RAW/$f" -o "$DEST/simple_evals/$f"
done
: > "$DEST/simple_evals/__init__.py"
: > "$DEST/simple_evals/sampler/__init__.py"
curl -fsSL "$DATA_URL" -o "$DEST/2025-05-07-06-14-12_oss_meta_eval.jsonl"

# Expected MD5 of the code files at the pinned commit.
cat > "$DEST/simple_evals/MD5SUMS" <<'EOF'
5b4b1ee0d04e064089d230f4516ab5e6  common.py
eceace2269610872ffe30131533da574  types.py
b9eddbb2f13808c486dd5ebc3a3540d9  healthbench_eval.py
df167e5bd0f2bd7dde83bfe20b590ceb  healthbench_meta_eval.py
EOF
( cd "$DEST/simple_evals" && if command -v md5sum >/dev/null; then md5sum -c MD5SUMS; else
    while read -r sum f; do [ "$(md5 -q "$f")" = "$sum" ] && echo "$f: OK" || { echo "$f: FAILED"; exit 1; }; done < MD5SUMS; fi )
echo "simple-evals @ ${COMMIT} -> $DEST/simple_evals"
