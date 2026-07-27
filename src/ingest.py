"""
ingest.py — STEP 2 of the pipeline: build the knowledge base.

For every downloaded PDF:
  1. TEXT EXTRACTION  : PyMuPDF (fitz) extracts text page by page. It is fast
                        and preserves the reading order of financial tables
                        reasonably well.
  2. DATA CLEANING    : de-hyphenation, whitespace normalisation, removal of
                        repeated headers/footers and page numbers.
  3. CHUNKING         : structure-aware sliding window. We first split on
                        paragraph boundaries (preserving document structure),
                        then pack paragraphs into chunks of ~chunk_size tokens
                        with chunk_overlap tokens of overlap.
  4. METADATA         : every chunk carries doc_name, company, doc_type,
                        period, page number and chunk id — this is what lets
                        the generator cite its sources.
  5. EMBEDDING + STORE: Sentence-Transformers embeddings stored in a local
                        persistent ChromaDB collection (cosine similarity).

Usage:
    python src/ingest.py                          # default config
    python src/ingest.py --chunk-size 256 --chunk-overlap 32
    python src/ingest.py --embedding-model BAAI/bge-small-en-v1.5
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

from config import Config, DOC_INFO_FILE, RAW_PDF_DIR, CHROMA_DIR


# ---------------------------------------------------------------------------
# 1+2. Extraction and cleaning
# ---------------------------------------------------------------------------
def extract_pages(pdf_path: Path) -> list[str]:
    """Extract raw text of each page of a PDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


def detect_repeated_lines(pages: list[str], threshold: float = 0.5) -> set[str]:
    """Find header/footer lines that repeat on > threshold of pages."""
    counter: Counter = Counter()
    for p in pages:
        for line in {l.strip() for l in p.splitlines() if l.strip()}:
            counter[line] += 1
    n = max(len(pages), 1)
    return {line for line, c in counter.items() if c / n > threshold and len(line) < 120}


def clean_page(text: str, repeated: set[str]) -> str:
    """Clean one page of extracted text."""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s in repeated:
            continue
        if re.fullmatch(r"(page\s*)?\d{1,4}", s, flags=re.I):  # bare page numbers
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)     # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)               # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)            # collapse blank lines
    return text.strip()


# ---------------------------------------------------------------------------
# 3. Chunking
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Cheap whitespace tokenisation — good enough to control chunk size."""
    return text.split()


def chunk_pages(pages: list[str], cfg: Config) -> list[dict]:
    """Structure-aware sliding-window chunking.

    Returns a list of {"text": ..., "page": ...} dicts.
    If cfg.preserve_structure is True we never split inside a paragraph
    (unless a single paragraph is longer than chunk_size).
    """
    chunks: list[dict] = []

    # Build a flat list of (paragraph, page_number)
    paragraphs: list[tuple[str, int]] = []
    for page_no, page in enumerate(pages, 1):
        if cfg.preserve_structure:
            for para in re.split(r"\n\s*\n", page):
                para = para.strip()
                if para:
                    paragraphs.append((para, page_no))
        else:
            if page.strip():
                paragraphs.append((page.strip(), page_no))

    buf: list[str] = []
    buf_page = 1
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append({"text": " ".join(buf), "page": buf_page})
            # keep the overlap for the next chunk
            overlap_tokens = tokenize(" ".join(buf))[-cfg.chunk_overlap:]
            buf = [" ".join(overlap_tokens)] if overlap_tokens else []
            buf_len = len(overlap_tokens)

    for para, page_no in paragraphs:
        ptoks = tokenize(para)
        # very long paragraph: hard-split it
        while len(ptoks) > cfg.chunk_size:
            head, ptoks = ptoks[: cfg.chunk_size], ptoks[cfg.chunk_size - cfg.chunk_overlap:]
            if buf:
                flush()
            chunks.append({"text": " ".join(head), "page": page_no})
        if buf_len == 0:
            buf_page = page_no
        buf.append(para if len(ptoks) == len(tokenize(para)) else " ".join(ptoks))
        buf_len += len(ptoks)
        if buf_len >= cfg.chunk_size:
            flush()
    if buf and buf_len > cfg.chunk_overlap:  # don't store a pure-overlap tail
        chunks.append({"text": " ".join(buf), "page": buf_page})

    return chunks


# ---------------------------------------------------------------------------
# 4+5. Metadata, embedding, storage
# ---------------------------------------------------------------------------
def load_metadata() -> dict[str, dict]:
    meta = {}
    with open(DOC_INFO_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            meta[row["doc_name"]] = {
                "company": row.get("company", ""),
                "doc_type": row.get("doc_type", ""),
                "doc_period": str(row.get("doc_period", "")),
            }
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--no-structure", action="store_true",
                        help="disable structure preservation (page-level split only)")
    parser.add_argument("--limit-docs", type=int, default=None,
                        help="ingest only N PDFs (smoke test)")
    parser.add_argument("--resume", action="store_true",
                        help="keep the existing collection and skip already-ingested docs")
    args = parser.parse_args()

    cfg = Config()
    if args.chunk_size:
        cfg.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        cfg.chunk_overlap = args.chunk_overlap
    if args.embedding_model:
        cfg.embedding_model = args.embedding_model
    if args.no_structure:
        cfg.preserve_structure = False

    print("[CONFIG]", json.dumps(cfg.to_dict(), indent=2))

    pdfs = sorted(RAW_PDF_DIR.glob("*.pdf"))
    if args.limit_docs:
        pdfs = pdfs[: args.limit_docs]
    print(f"[INGEST] {len(pdfs)} PDFs found in {RAW_PDF_DIR}")
    if not pdfs:
        print("[ERROR] No PDFs found. Run `python src/download_pdfs.py` first.")
        return

    print(f"[EMBED] Loading embedding model: {cfg.embedding_model}")
    model = SentenceTransformer(cfg.embedding_model)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll_name = cfg.collection_name()
    already_ingested: set[str] = set()
    if args.resume:
        try:
            collection = client.get_collection(coll_name)
            n = collection.count()
            page = 5000
            for off in range(0, n, page):
                batch = collection.get(include=["metadatas"], limit=page, offset=off)
                already_ingested.update(m["doc_name"] for m in batch["metadatas"])
            print(f"[STORE] Resuming collection '{coll_name}' "
                  f"({len(already_ingested)} docs already ingested)")
        except Exception:
            collection = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})
            print(f"[STORE] Created Chroma collection '{coll_name}'")
    else:
        try:
            client.delete_collection(coll_name)
            print(f"[STORE] Deleted stale collection '{coll_name}' (rebuilding)")
        except Exception:
            pass
        collection = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})
        print(f"[STORE] Created Chroma collection '{coll_name}'")

    doc_meta = load_metadata()
    total_chunks = 0

    for i, pdf in enumerate(pdfs, 1):
        doc_name = pdf.stem
        if doc_name in already_ingested:
            print(f"[{i:>3}/{len(pdfs)}] {doc_name:<40} skipped (already ingested)")
            continue
        pages = extract_pages(pdf)
        repeated = detect_repeated_lines(pages)
        cleaned = [clean_page(p, repeated) for p in pages]
        chunks = chunk_pages(cleaned, cfg)
        print(f"[{i:>3}/{len(pdfs)}] {doc_name:<40} pages={len(pages):>4} chunks={len(chunks):>5}")

        if not chunks:
            print(f"    [WARN] no text extracted from {doc_name} — possibly a scanned PDF")
            continue

        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False,
                                  normalize_embeddings=True)
        metadatas = [{
            "doc_name": doc_name,
            "page": c["page"],
            **doc_meta.get(doc_name, {}),
        } for c in chunks]
        ids = [f"{doc_name}__chunk{j:05d}" for j in range(len(chunks))]

        # Chroma add() in batches to keep memory bounded
        B = 2000
        for s in range(0, len(chunks), B):
            collection.add(
                ids=ids[s:s + B],
                documents=texts[s:s + B],
                embeddings=embeddings[s:s + B].tolist(),
                metadatas=metadatas[s:s + B],
            )
        total_chunks += len(chunks)

    print("\n[SUMMARY] ============================")
    print(f"[SUMMARY] collection : {coll_name}")
    print(f"[SUMMARY] documents  : {len(pdfs)}")
    print(f"[SUMMARY] chunks     : {total_chunks}")
    print("[SUMMARY] The knowledge base is loaded and ready to be queried.")


if __name__ == "__main__":
    main()
