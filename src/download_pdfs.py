"""
download_pdfs.py — STEP 1 of the pipeline: automate PDF retrieval.

Reads financebench_document_information.jsonl, extracts (doc_name, doc_link)
pairs and downloads every PDF into data/raw_pdfs/.

Features required by the project instructions:
  * error handling (timeouts, HTTP errors, retries with backoff)
  * a User-Agent header so SEC EDGAR / investor-relations servers
    do not block the requests
  * idempotent: already-downloaded files are skipped, so the script
    can be re-run safely after a partial failure

Usage:
    python src/download_pdfs.py            # download everything
    python src/download_pdfs.py --limit 10 # smoke test on 10 docs
"""

import argparse
import json
import time
from pathlib import Path

import requests

from config import DOC_INFO_FILE, RAW_PDF_DIR

# SEC EDGAR explicitly requires a descriptive User-Agent with contact info.
HEADERS = {
    "User-Agent": "FinanceBench-RAG-student-project (contact: your.email@example.com)",
    "Accept": "application/pdf,*/*",
}

MAX_RETRIES = 3
TIMEOUT = 30  # seconds


def load_catalog() -> list[dict]:
    """Load the document catalog and keep unique (doc_name, doc_link) pairs."""
    print(f"[LOAD] Reading catalog from {DOC_INFO_FILE}")
    records, seen = [], set()
    with open(DOC_INFO_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["doc_name"] not in seen:
                seen.add(row["doc_name"])
                records.append({"doc_name": row["doc_name"], "doc_link": row["doc_link"]})
    print(f"[LOAD] Found {len(records)} unique documents in the catalog")
    return records


def download_one(doc_name: str, url: str, out_dir: Path) -> str:
    """Download a single PDF with retries. Returns a status string."""
    out_path = out_dir / f"{doc_name}.pdf"
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return "skipped (already downloaded)"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            content = resp.content
            # sanity check: a real PDF starts with %PDF
            if not content[:5].startswith(b"%PDF"):
                raise ValueError("response is not a PDF (probably an HTML block page)")
            out_path.write_bytes(content)
            return f"ok ({len(content) / 1e6:.1f} MB)"
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    [WARN] attempt {attempt}/{MAX_RETRIES} failed for {doc_name}: {exc} "
                  f"-> retrying in {wait}s")
            time.sleep(wait)
    return "FAILED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only download the first N documents (smoke test)")
    args = parser.parse_args()

    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    if args.limit:
        catalog = catalog[: args.limit]
        print(f"[INFO] --limit set: downloading only {len(catalog)} documents")

    ok, failed = 0, []
    for i, rec in enumerate(catalog, 1):
        status = download_one(rec["doc_name"], rec["doc_link"], RAW_PDF_DIR)
        print(f"[{i:>3}/{len(catalog)}] {rec['doc_name']:<40} {status}")
        if status == "FAILED":
            failed.append(rec["doc_name"])
        else:
            ok += 1
        time.sleep(0.5)  # be polite to the servers, avoid rate limiting

    print("\n[SUMMARY] ============================")
    print(f"[SUMMARY] downloaded/skipped : {ok}")
    print(f"[SUMMARY] failed             : {len(failed)}")
    if failed:
        fail_file = RAW_PDF_DIR / "_failed_downloads.txt"
        fail_file.write_text("\n".join(failed))
        print(f"[SUMMARY] failed doc_names written to {fail_file}")
        print("[SUMMARY] Tip: some investor-relations links rot over time; you can often")
        print("[SUMMARY] find the same filing on https://www.sec.gov/cgi-bin/browse-edgar")


if __name__ == "__main__":
    main()
