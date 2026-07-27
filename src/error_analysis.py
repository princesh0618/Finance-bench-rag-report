"""
error_analysis.py — categorise failures of a run for the report's
"Error Analysis" section.

Failure taxonomy (assigned per question):
    RETRIEVAL_MISS   : no retrieved chunk came from the gold document
                       -> the generator never had a chance
    EVIDENCE_MISS    : right document, but the gold evidence passage was not
                       among the retrieved chunks (chunking/embedding issue)
    GENERATION_ERROR : correct evidence retrieved, but wrong answer produced
                       (extraction/reasoning failure of the LLM)
    ABSTAINED        : model said it cannot answer although evidence existed
    CORRECT          : numeric_match == 1

Usage:
    python src/error_analysis.py --run-name baseline [--show 5]
"""

import argparse
import json
from collections import Counter

from config import RESULTS_DIR


def categorise(rec: dict) -> str:
    if rec["numeric_match"] == 1:
        return "CORRECT"
    if "cannot answer" in rec["pred_answer"].lower():
        return "ABSTAINED"
    if rec["doc_hit"] == 0:
        return "RETRIEVAL_MISS"
    if rec["evidence_recall"] == 0:
        return "EVIDENCE_MISS"
    return "GENERATION_ERROR"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--show", type=int, default=3,
                        help="print N examples per failure category")
    args = parser.parse_args()

    path = RESULTS_DIR / f"{args.run_name}.jsonl"
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    for r in records:
        r["category"] = categorise(r)

    counts = Counter(r["category"] for r in records)
    n = len(records)
    print(f"[ERROR ANALYSIS] run '{args.run_name}' — {n} questions")
    print("-" * 50)
    for cat in ["CORRECT", "RETRIEVAL_MISS", "EVIDENCE_MISS",
                "GENERATION_ERROR", "ABSTAINED"]:
        c = counts.get(cat, 0)
        print(f"{cat:<18} {c:>4}  ({100 * c / n:.1f}%)")

    for cat in ["RETRIEVAL_MISS", "EVIDENCE_MISS", "GENERATION_ERROR", "ABSTAINED"]:
        examples = [r for r in records if r["category"] == cat][: args.show]
        if not examples:
            continue
        print(f"\n===== Examples: {cat} =====")
        for r in examples:
            print(f"\nQ : {r['question'][:160]}")
            print(f"G : {r['gold_answer'][:120]}")
            print(f"P : {r['pred_answer'][:200]}")
            print(f"gold_doc={r['gold_doc']}  retrieved={set(r['retrieved_docs'])}")

    out = RESULTS_DIR / f"{args.run_name}.error_categories.json"
    out.write_text(json.dumps(dict(counts), indent=2))
    print(f"\n[SAVED] category counts -> {out}")


if __name__ == "__main__":
    main()
