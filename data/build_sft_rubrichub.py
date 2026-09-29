#!/usr/bin/env python3
"""SFT baseline data: RubricHub's own supervised corpus, prepared for verl's SFT trainer (configs/sft.yaml).

The SFT arm tests whether training on the rubric data without an RL objective collapses appropriateness. The data is
RubricHub's own, not ours: `sft_RuFT/rurbichub_v1_best_of_6samples_26k_sft_data.parquet` from the `sojuL/RubricHub_v1`
dataset (26,194 rows, each the best of six sampled responses by rubric score), the four domains mixed, as in
RISE-RL's supervised baseline. Each row becomes `messages = [user: query, assistant: answer]`.

Rows whose chat-templated length exceeds 20,000 tokens (the SFT max sequence length) are dropped rather than truncated:
a truncated target teaches the model to stop mid-sentence. The paper's build kept 26,162 rows (30 over budget,
2 empty). The count is printed and recorded in the manifest.

    python3 build_sft_rubrichub.py --src rurbichub_v1_best_of_6samples_26k_sft_data.parquet \
        --model /path/to/Qwen3-4B --output release/rubrichub-sft
    python3 build_sft_rubrichub.py --src ... --model ... --dry-run 500

Needs `transformers` and the base model's tokenizer for an exact token count.
"""
import argparse
import json
import os
from datetime import datetime, timezone

from _common import tree_hash

MAX_LEN = 20_000
UPSTREAM = ("hf:sojuL/RubricHub_v1 sft_RuFT/rurbichub_v1_best_of_6samples_26k_sft_data.parquet "
            "(26,194 rows, best of 6 samples by rubric score)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="local copy of the RubricHub SFT parquet")
    ap.add_argument("--model", required=True, help="base model path or hub id (tokenizer used for the length filter)")
    ap.add_argument("--output", help="release directory to write")
    ap.add_argument("--dry-run", type=int, default=0)
    a = ap.parse_args()
    if not a.dry_run and not a.output:
        ap.error("--output is required unless --dry-run is given")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    rows = pq.read_table(a.src).to_pylist()
    if a.dry_run:
        rows = rows[: a.dry_run]
    print("source rows:", len(rows))

    kept, dropped, lens = [], 0, []
    for r in rows:
        q, ans = (r.get("query") or "").strip(), (r.get("answer") or "").strip()
        if not q or not ans:
            dropped += 1
            continue
        messages = [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        # return_dict=False is required: transformers >= 5 returns a BatchEncoding by default, whose len() is 2.
        n = len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False))
        if n > MAX_LEN:
            dropped += 1
            continue
        lens.append(n)
        kept.append({"messages": messages, "source": r.get("source"), "sample_id": r.get("sample_id"),
                     "rubric_score": r.get("rubric_score")})
    lens.sort()
    print("kept %d, dropped %d (over %d tokens or empty)" % (len(kept), dropped, MAX_LEN))
    print("sequence tokens: median %d, p90 %d, max %d" % (lens[len(lens) // 2], lens[int(0.9 * len(lens))], lens[-1]))
    if a.dry_run:
        return

    os.makedirs(a.output, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(kept), os.path.join(a.output, "train.parquet"))
    man = {"arm": "SFT", "rows_out": len(kept), "rows_dropped": dropped, "max_length": MAX_LEN,
           "tokenizer": os.path.basename(a.model.rstrip("/")), "upstream": UPSTREAM,
           "rule": "messages = [user: query, assistant: answer]; rows whose chat-templated length exceeds "
                   "max_length are dropped, never truncated",
           "alignment": "RISE-RL supervised recipe (3 epochs, lr 1e-5, warmup 0.05, bf16, full-parameter, "
                        "max sequence length 20000, global batch 64; four domains mixed)",
           "built_at": datetime.now(timezone.utc).isoformat()}
    json.dump(man, open(os.path.join(a.output, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    open(os.path.join(a.output, "README.md"), "w").write(
        "# RubricHub SFT corpus\n\nRubricHub's own SFT corpus (best of 6 samples per prompt), four domains mixed, "
        "rendered as\n`messages` for verl's MultiTurnSFTDataset. Kept %d of %d rows; the rest exceed the %d-token "
        "budget or are empty.\n" % (len(kept), len(rows), MAX_LEN))
    tree, nfiles, total = tree_hash(a.output)
    print(json.dumps({"rows": len(kept), "dropped": dropped, "tree": tree, "files": nfiles, "bytes": total}))


if __name__ == "__main__":
    main()
