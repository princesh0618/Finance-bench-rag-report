"""
server.py — FastAPI backend web server for the FinanceBench RAG system.

Provides REST API endpoints for:
  - POST /api/query          : runs end-to-end RAG retrieval + generation
  - GET  /api/experiments    : returns benchmark comparison table data
  - GET  /api/sample-questions: returns sample benchmark questions for quick testing
  - GET  /api/documents      : returns indexed document catalog and chunk counts
  - GET  /                   : serves the Web UI frontend
"""

import csv
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chromadb

from config import Config, RESULTS_DIR, RAW_PDF_DIR, CHROMA_DIR
from rag_pipeline import RAGPipeline

app = FastAPI(title="FinanceBench RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Embedding models the frontend may request. Each must already have a
# persisted ChromaDB collection (built by the experiment grid) — switching to
# an arbitrary untested embedding model would otherwise silently trigger a
# multi-hour full-corpus re-ingestion on first use, which a live query
# endpoint must never do.
ALLOWED_EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
}
ALLOWED_GENERATION_MODELS = {
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Pipelines are cached per (embedding_model, generation_model) combo so
# switching models in the UI doesn't reload from scratch on every request —
# only the first request for a given combo pays the model-load cost.
_pipeline_cache: dict[tuple[str, str], RAGPipeline] = {}


def _collection_exists(cfg: Config) -> bool:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return any(c.name == cfg.collection_name() for c in client.list_collections())


def get_pipeline(embedding_model: Optional[str], generation_model: Optional[str]) -> RAGPipeline:
    emb = embedding_model or Config().embedding_model
    gen = generation_model or Config().generation_model
    if emb not in ALLOWED_EMBEDDING_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported embedding model: {emb}")
    if gen not in ALLOWED_GENERATION_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported generation model: {gen}")

    key = (emb, gen)
    if key not in _pipeline_cache:
        cfg = Config(embedding_model=emb, generation_model=gen)
        if not _collection_exists(cfg):
            raise HTTPException(
                status_code=400,
                detail=f"No ingested collection for '{emb}' at the default chunk size — "
                       f"run the experiment grid for this embedding model first.",
            )
        _pipeline_cache[key] = RAGPipeline(cfg, verbose=True)
    return _pipeline_cache[key]


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 5
    embedding_model: Optional[str] = None
    generation_model: Optional[str] = None


@app.get("/")
def read_root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"message": "Server running. Web UI loading..."}, status_code=200)
    return FileResponse(index_path)


@app.post("/api/query")
def run_query(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    start_time = time.time()
    try:
        pipeline = get_pipeline(req.embedding_model, req.generation_model)

        passages = pipeline.retriever.retrieve(req.question, k=req.k)
        from rag_pipeline import build_prompt
        prompt = build_prompt(req.question, passages, tokenizer=getattr(pipeline.generator, "tok", None))
        answer = pipeline.generator.generate(prompt)

        elapsed_s = round(time.time() - start_time, 2)
        
        # Clean passages for JSON response
        cleaned_passages = []
        for i, p in enumerate(passages, 1):
            cleaned_passages.append({
                "citation_id": i,
                "doc_name": p.get("doc_name", "Unknown Document"),
                "page": p.get("page", 1),
                "score": round(float(p.get("score", 0.0)), 4),
                "similarity_pct": round(float(p.get("score", 0.0)) * 100, 1),
                "text": p.get("text", "")
            })

        return {
            "status": "success",
            "question": req.question,
            "answer": answer,
            "passages": cleaned_passages,
            "latency_seconds": elapsed_s,
            "config": {
                "embedding_model": pipeline.cfg.embedding_model,
                "top_k": req.k or pipeline.cfg.top_k,
                "chunk_size": pipeline.cfg.chunk_size,
                "generation_model": pipeline.cfg.generation_model
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/available-models")
def get_available_models():
    """Which (embedding, generation) model options are actually usable right now —
    i.e. have a persisted ChromaDB collection — so the UI never offers a choice
    that would silently trigger a multi-hour re-ingestion on first use."""
    available_embeddings = []
    for emb in sorted(ALLOWED_EMBEDDING_MODELS):
        cfg = Config(embedding_model=emb)
        available_embeddings.append({"value": emb, "ready": _collection_exists(cfg)})
    return {
        "embedding_models": available_embeddings,
        "generation_models": sorted(ALLOWED_GENERATION_MODELS),
    }


@app.get("/api/experiments")
def get_experiments():
    comp_file = RESULTS_DIR / "comparison.csv"
    if not comp_file.exists():
        return {"experiments": []}
    
    records = []
    with open(comp_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Format numbers nicely
            for key in ["doc_hit@k", "evidence_recall@k", "MRR", "numeric_match", "token_f1"]:
                if key in row and row[key]:
                    try:
                        row[key] = round(float(row[key]), 4)
                    except ValueError:
                        pass
            records.append(row)
            
    return {"experiments": records}


@app.get("/api/sample-questions")
def get_sample_questions():
    samples = [
        {
            "id": "sample_1",
            "category": "Capital Expenditures",
            "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
        },
        {
            "id": "sample_2",
            "category": "Balance Sheet Analysis",
            "question": "What is the year end FY2018 net PPNE for 3M? Answer in USD billions."
        },
        {
            "id": "sample_3",
            "category": "Liquidity & Solvency",
            "question": "Does 3M have a reasonably healthy liquidity profile based on its quick ratio for Q2 of FY2023?"
        },
        {
            "id": "sample_4",
            "category": "Profitability & Margin",
            "question": "What drove operating margin change as of FY2022 for 3M?"
        },
        {
            "id": "sample_5",
            "category": "Segment Analysis",
            "question": "If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?"
        }
    ]
    return {"samples": samples}


@app.get("/api/documents")
def get_documents():
    pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf")) if RAW_PDF_DIR.exists() else []
    docs = []
    for pdf in pdf_files:
        docs.append({
            "name": pdf.name,
            "size_mb": round(pdf.stat().st_size / (1024 * 1024), 2),
            "doc_type": "10-K" if "10K" in pdf.name else ("10-Q" if "10Q" in pdf.name else "8-K")
        })
    return {"total_pdfs": len(docs), "documents": docs[:50]}


# Mount static assets & plots
app.mount("/plots", StaticFiles(directory=str(RESULTS_DIR / "plots")), name="plots")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

