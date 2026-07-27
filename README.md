# FinanceBench RAG — NLP Graded Project A

An end-to-end Retrieval-Augmented Generation (RAG) system that answers
questions about a closed collection of financial documents (SEC 10-K/10-Q
filings and earnings reports) from the
[FinanceBench](https://github.com/patronus-ai/financebench) dataset, and a
full evaluation + experimentation pipeline.

**Team:** Prince Sharma, Syed Amaan, Harshitha Prasanna Karle

**Constraint compliance:** only free, locally executable models are used —
Sentence-Transformers embeddings and a local HuggingFace Transformers model
(Qwen2.5-1.5B-Instruct by default; CPU-only, no API keys) — and every
generated answer is instructed to cite the retrieved passages as `[1]`,
`[2]`, … or explicitly abstain rather than answer ungrounded.

## Project structure

```
financebench_rag_project/
├── data/
│   ├── financebench_document_information.jsonl   # catalog (doc_name -> PDF URL)
│   ├── financebench_open_source.jsonl            # 150 gold Q/A + evidence
│   └── raw_pdfs/                                 # downloaded PDFs (created by step 1)
├── src/
│   ├── config.py            # all experimental settings in one place
│   ├── download_pdfs.py     # STEP 1 — automate PDF retrieval
│   ├── ingest.py            # STEP 2 — parse, clean, chunk, embed, index
│   ├── rag_pipeline.py      # retrieval + local-LLM generation with citations
│   ├── evaluate.py          # STEP 3 — evaluation loop + metrics
│   ├── experiments.py       # runs the full comparison grid + plots
│   ├── error_analysis.py    # failure taxonomy for the report
│   ├── server.py            # FastAPI backend for the interactive web UI
│   └── static/              # web UI frontend (HTML/CSS/JS)
├── chroma_store/            # persistent ChromaDB vector store (created by ingest.py)
├── results/                 # per-run .jsonl records, summaries, plots, comparison.csv
├── report/
│   ├── Project_Report.pdf   # scientific report (deliverable 2)
│   └── screenshots/         # web UI screenshots referenced in the report's appendix
└── requirements.txt
```

## Installation

Tested with Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No API keys and no GPU are needed — the default configuration runs the
Qwen2.5-1.5B-Instruct generation model locally via HuggingFace Transformers
on CPU. (If you'd rather use [Ollama](https://ollama.com/download), set
`generation_backend="ollama"` and `generation_model="qwen2.5:3b-instruct"`
in `src/config.py`, having already run `ollama pull qwen2.5:3b-instruct`.)

## Reproducing the experiments

All commands are run from the project root. Every script prints detailed
progress logs (`[LOAD]`, `[INGEST]`, `[RETRIEVER]`, `[SUMMARY]`, …).

```bash
# STEP 1 — download the corpus (~530 MB; already-downloaded files are
# skipped on re-run, failed ones are retried). 263 of the catalog's
# documents download successfully; the rest (dead links / no-longer-hosted
# filings) are logged to data/raw_pdfs/_failed_downloads.txt each run.
python src/download_pdfs.py
# quick smoke test instead: python src/download_pdfs.py --limit 10

# STEP 2 — build the knowledge base (default: 512-token chunks, 64 overlap,
#          all-MiniLM-L6-v2 embeddings, persistent ChromaDB in ./chroma_store)
python src/ingest.py

# Try the system interactively
python src/rag_pipeline.py --question "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"

# STEP 3 — evaluate on all 150 FinanceBench questions
python src/evaluate.py --run-name baseline
# add local-LLM-as-a-judge scoring:
python src/evaluate.py --run-name baseline_judged --judge

# Error analysis for the report
python src/error_analysis.py --run-name baseline

# Full experiment grid (chunk size/overlap, embedding models, k, generation
# models, structure preservation) — re-ingests the corpus for each axis that
# needs it; use --n 50 for a faster pass
python src/experiments.py --n 50
# rebuild the comparison table + plots from existing runs, without re-running:
python src/experiments.py --plots-only
```

Outputs:

* `results/<run>.jsonl` — per-question record (question, gold, prediction,
  retrieved docs, all metrics) for error analysis
* `results/<run>.summary.json` — aggregated metrics
* `results/comparison.csv` and `results/plots/*.png` — cross-configuration
  comparison used in the report

## Web UI

A FastAPI + vanilla-JS interactive demo is included:

```bash
python src/server.py
# then open http://127.0.0.1:8000
```

It provides a query console (ask questions, inspect the cited evidence
passages live), a benchmark analytics tab (reads `results/comparison.csv`
directly), and a corpus browser. It's a convenience layer for exploring the
system, not a substitute for the evaluation above — see
`report/Project_Report.pdf`, Appendix A, for annotated screenshots.

Only the `chroma_store` collection matching the default config
(`chunk_size=512, chunk_overlap=64, all-MiniLM-L6-v2`) is kept persisted;
switching the UI's embedding/chunking parameters to one of the other
experiment configs requires re-running `ingest.py` for that config first
(`experiments.py` does this automatically when reproducing the full grid).

## Metrics

| Metric | What it measures |
|---|---|
| `doc_hit@k` | at least one retrieved chunk from the gold document |
| `evidence_recall@k` | a retrieved chunk covers ≥ 50 % of the gold `evidence_text` tokens |
| `MRR` | rank of the first gold-document chunk |
| `numeric_match` | all gold-answer numbers appear in the prediction |
| `token_f1` | SQuAD-style token overlap with the gold answer |
| `llm_judge` (optional) | local LLM grades CORRECT / PARTIAL / INCORRECT |

## Design choices (summary — full justification in the report)

* **PyMuPDF** for extraction: fastest open-source parser and keeps a usable
  reading order for financial tables.
* **Structure-aware chunking**: paragraphs are never split (unless longer
  than the chunk size), a sliding-window overlap avoids cutting evidence in
  half at chunk boundaries.
* **Cosine similarity over normalised embeddings** in a persistent ChromaDB.
* **Grounded prompt with mandatory `[i]` citations** and an explicit
  abstention instruction to limit hallucination — see report §3.5 for the
  citation-compliance fix and its measured effect.
* **Temperature 0** for reproducible evaluation.
