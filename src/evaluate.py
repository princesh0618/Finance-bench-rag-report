"""
evaluate.py — STEP 3 of the pipeline: run the evaluation loop.

For each of the 150 items in financebench_open_source.jsonl:
  1. extract the question,
  2. pass it to the RAG system, capture the generated answer AND the
     retrieved contexts,
  3. compare against the ground-truth `answer` and `evidence_text`.

METRICS
-------
Retrieval quality (needs no LLM, fully deterministic):
  * evidence_recall@k : 1 if at least one retrieved chunk overlaps the gold
                        evidence_text (ROUGE-L-style token containment >= 0.5)
  * doc_hit@k         : 1 if at least one retrieved chunk comes from the gold
                        doc_name (a cheap but very informative proxy)
  * MRR               : reciprocal rank of the first chunk from the gold doc

Answer quality:
  * numeric_match     : all gold numbers appear in the prediction
                        (FinanceBench answers are mostly figures)
  * token_f1          : SQuAD-style token-level F1
  * llm_judge (opt.)  : local-LLM-as-a-judge scores correctness 0/0.5/1
                        (enabled with --judge; uses the same local backend)

All raw per-question records are saved to results/<run_name>.jsonl and the
aggregated metrics to results/<run_name>.summary.json, so the report's error
analysis can dig into individual failures.

Usage:
    python src/evaluate.py --run-name baseline
    python src/evaluate.py --run-name k10 --k 10 --n 50
    python src/evaluate.py --run-name judged --judge
"""

import argparse
import json
import re
import string
import time
from collections import Counter

from config import Config, QA_FILE, RESULTS_DIR
from rag_pipeline import RAGPipeline, Generator


# ---------------------------------------------------------------------------
# Text-normalisation helpers
# ---------------------------------------------------------------------------
def normalize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision, recall = n_common / len(p), n_common / len(g)
    return 2 * precision * recall / (precision + recall)


NUM_RE = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?%?")


def extract_numbers(text: str) -> set[str]:
    """Extract numeric tokens, normalised (strip $ , % parens; drop trailing .00)."""
    out = set()
    for m in NUM_RE.findall(text):
        s = m.strip("$%()").replace(",", "")
        try:
            v = float(s)
            out.add(f"{v:g}")
        except ValueError:
            pass
    return out


def numeric_match(pred: str, gold: str) -> float:
    gold_nums = extract_numbers(gold)
    if not gold_nums:               # non-numeric gold answer: fall back to F1>=0.5
        return float(token_f1(pred, gold) >= 0.5)
    pred_nums = extract_numbers(pred)
    return float(gold_nums.issubset(pred_nums))


def evidence_overlap(chunk: str, evidence: str) -> float:
    """Fraction of gold-evidence tokens contained in the chunk."""
    ev, ch = normalize(evidence), Counter(normalize(chunk))
    if not ev:
        return 0.0
    hit = sum(1 for t in ev if ch[t] > 0)
    return hit / len(ev)


# ---------------------------------------------------------------------------
# Optional local-LLM-as-a-judge
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are grading a question-answering system on financial documents.

Question: {question}
Gold answer: {gold}
System answer: {pred}

Grade the system answer: reply with exactly one token:
CORRECT   - factually equivalent to the gold answer (rounding/format differences are fine)
PARTIAL   - partially correct or correct but with unsupported extra claims
INCORRECT - wrong or refuses although the gold answer exists
Reply with only the single word."""


def judge_score(generator: Generator, question: str, gold: str, pred: str) -> float:
    reply = generator.generate(
        JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)).upper()
    if "CORRECT" in reply and "INCORRECT" not in reply:
        return 1.0
    if "PARTIAL" in reply:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def load_qa(n: int | None = None) -> list[dict]:
    items = []
    with open(QA_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            items.append({
                "id": row["financebench_id"],
                "doc_name": row["doc_name"],
                "question": row["question"],
                "answer": str(row["answer"]),
                "evidence_texts": [e["evidence_text"] for e in row.get("evidence", [])],
            })
    return items[:n] if n else items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--n", type=int, default=None, help="evaluate only N questions")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--generation-model", type=str, default=None)
    parser.add_argument("--no-structure", action="store_true",
                        help="target the collection built without structure preservation")
    parser.add_argument("--judge", action="store_true", help="enable local LLM-as-a-judge")
    parser.add_argument("--resume", action="store_true",
                        help="skip questions already recorded in results/<run_name>.jsonl")
    args = parser.parse_args()

    cfg = Config()
    for attr, val in [("top_k", args.k), ("chunk_size", args.chunk_size),
                      ("chunk_overlap", args.chunk_overlap),
                      ("embedding_model", args.embedding_model),
                      ("generation_model", args.generation_model)]:
        if val is not None:
            setattr(cfg, attr, val)
    if args.no_structure:
        cfg.preserve_structure = False

    print("[CONFIG]", json.dumps(cfg.to_dict(), indent=2))
    qa = load_qa(args.n)
    print(f"[EVAL] {len(qa)} questions loaded from {QA_FILE}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.run_name}.jsonl"

    records = []
    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)
                done_ids.add(rec["id"])
        print(f"[EVAL] resuming: {len(done_ids)} questions already recorded in {out_path}")
        qa = [item for item in qa if item["id"] not in done_ids]

    rag = RAGPipeline(cfg, verbose=False)

    t0 = time.time()
    mode = "a" if (args.resume and done_ids) else "w"
    with open(out_path, mode, encoding="utf-8") as fout:
        for i, item in enumerate(qa, 1):
            result = rag.answer(item["question"])
            passages = result["passages"]

            # ---- retrieval metrics ----
            doc_ranks = [r for r, p in enumerate(passages, 1)
                         if p["doc_name"] == item["doc_name"]]
            doc_hit = float(bool(doc_ranks))
            mrr = 1.0 / doc_ranks[0] if doc_ranks else 0.0
            ev_recall = 0.0
            for ev in item["evidence_texts"]:
                for p in passages:
                    if evidence_overlap(p["text"], ev) >= 0.5:
                        ev_recall = 1.0
                        break
                if ev_recall:
                    break

            # ---- answer metrics ----
            rec = {
                "id": item["id"],
                "question": item["question"],
                "gold_answer": item["answer"],
                "pred_answer": result["answer"],
                "gold_doc": item["doc_name"],
                "retrieved_docs": [p["doc_name"] for p in passages],
                "doc_hit": doc_hit,
                "mrr": mrr,
                "evidence_recall": ev_recall,
                "numeric_match": numeric_match(result["answer"], item["answer"]),
                "token_f1": token_f1(result["answer"], item["answer"]),
            }
            if args.judge:
                rec["judge"] = judge_score(rag.generator, item["question"],
                                           item["answer"], result["answer"])
            records.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            print(f"[{i:>3}/{len(qa)}] {item['id']}  "
                  f"doc_hit={doc_hit:.0f} ev_recall={ev_recall:.0f} "
                  f"num_match={rec['numeric_match']:.0f} f1={rec['token_f1']:.2f}"
                  + (f" judge={rec['judge']:.1f}" if args.judge else ""), flush=True)

    # ---- aggregate ----
    n = len(records)
    summary = {
        "run_name": args.run_name,
        "config": cfg.to_dict(),
        "n_questions": n,
        "doc_hit@k": sum(r["doc_hit"] for r in records) / n,
        "evidence_recall@k": sum(r["evidence_recall"] for r in records) / n,
        "MRR": sum(r["mrr"] for r in records) / n,
        "numeric_match": sum(r["numeric_match"] for r in records) / n,
        "token_f1": sum(r["token_f1"] for r in records) / n,
        "runtime_s": round(time.time() - t0, 1),
    }
    if args.judge:
        summary["llm_judge"] = sum(r["judge"] for r in records) / n

    sum_path = RESULTS_DIR / f"{args.run_name}.summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print("\n[SUMMARY] ============================")
    for k, v in summary.items():
        if k != "config":
            print(f"[SUMMARY] {k:<20} {v}")
    print(f"[SUMMARY] per-question records -> {out_path}")
    print(f"[SUMMARY] aggregate metrics    -> {sum_path}")


if __name__ == "__main__":
    main()
