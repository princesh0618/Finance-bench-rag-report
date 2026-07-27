"""
config.py — Central configuration for the FinanceBench RAG project.

Every experiment is described by a Config object, so that runs are fully
reproducible and comparable. Change values here (or pass CLI overrides)
rather than editing the pipeline code.
"""

from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
CHROMA_DIR = PROJECT_ROOT / "chroma_store"
RESULTS_DIR = PROJECT_ROOT / "results"

DOC_INFO_FILE = DATA_DIR / "financebench_document_information.jsonl"
QA_FILE = DATA_DIR / "financebench_open_source.jsonl"


@dataclass
class Config:
    """One experimental configuration of the RAG pipeline."""

    # ----- Chunking (Task 1: corpus processing) -----
    chunk_size: int = 512            # target chunk size, in tokens (word-approx)
    chunk_overlap: int = 64          # overlap between consecutive chunks
    preserve_structure: bool = True  # split on page/paragraph boundaries first

    # ----- Retrieval (Task 2) -----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # alternatives to compare in experiments:
    #   "BAAI/bge-small-en-v1.5"
    #   "sentence-transformers/all-mpnet-base-v2"
    top_k: int = 5                   # number of passages given to the generator

    # ----- Generation (Task 3) -----
    # "transformers" runs a local HuggingFace model (default, used throughout
    # this project's results); "ollama" talks to a local Ollama server instead,
    # if you have it installed (e.g. generation_model="qwen2.5:3b-instruct").
    generation_backend: str = "transformers"      # "ollama" or "transformers"
    generation_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 300
    temperature: float = 0.0         # deterministic answers for evaluation

    # ----- Misc -----
    collection_prefix: str = "financebench"
    n_eval_questions: int = 150      # use fewer for quick smoke tests
    seed: int = 42

    def collection_name(self) -> str:
        """Unique Chroma collection name per (embedding, chunking, structure) setting."""
        emb = self.embedding_model.split("/")[-1].replace(".", "-")
        name = f"{self.collection_prefix}_{emb}_cs{self.chunk_size}_ov{self.chunk_overlap}"
        if not self.preserve_structure:
            name += "_nostruct"
        return name

    def to_dict(self) -> dict:
        return asdict(self)
