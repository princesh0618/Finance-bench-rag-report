"""
experiments.py — Task 5: experimental analysis.

Runs the comparison grid along the six axes required by the project
instructions ("investigate the impact of: chunk size, chunk overlap,
document structure preservation" plus "compare ... different embedding
models, different chunk sizes, different values of k, different generation
models"):
    1. chunk size / overlap (coupled)   — full corpus, needs re-ingestion
    2. embedding model                  — full corpus, needs re-ingestion
    3. k (number of retrieved passages) — re-uses the default index
    4. generation model                 — re-uses the default index
    5. document structure preservation  — on/off, 30-doc subset (speed)
    6. chunk overlap (independent of size) — 30-doc subset (speed)

Axes 5-6 use a reduced, but *identical-across-conditions*, corpus subset
purely for compute feasibility on CPU-only hardware — see the inline
comments at each axis for why the specific parameter values were chosen to
avoid accidentally reusing the full-corpus index for only one arm of the
comparison (which would confound corpus size with the variable under test).

For each configuration it calls ingest + evaluate as subprocesses, then
collects all results/*.summary.json files into a comparison table
(results/comparison.csv) and bar plots (results/plots/*.png).

Usage:
    python src/experiments.py --n 50        # 50 questions per config (faster)
    python src/experiments.py               # full 150 questions
    python src/experiments.py --plots-only  # just rebuild table + plots
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from config import RESULTS_DIR, CHROMA_DIR, Config

SRC = Path(__file__).parent


def run(cmd: list[str], retries: int = 2) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    for attempt in range(retries + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise
            wait = 30 * (attempt + 1)
            print(f"[RETRY] subprocess failed (attempt {attempt + 1}/{retries + 1}), "
                  f"retrying in {wait}s — likely a transient HF Hub / network hiccup")
            time.sleep(wait)


def eval_if_needed(run_name: str, cmd: list[str]) -> None:
    """Skip configs whose summary already exists; resume partially-done ones."""
    summary_path = RESULTS_DIR / f"{run_name}.summary.json"
    if summary_path.exists():
        print(f"[SKIP] '{run_name}' already has a summary")
        return
    run(cmd + ["--resume"])


def ingest_if_needed(cfg: Config, limit_docs: int | None = None) -> None:
    """Ingest (resuming any partially-built collection) for this config.

    limit_docs restricts ingestion to the first N PDFs (alphabetical) — used
    for lightweight supplementary ablations (structure preservation, overlap)
    where the axis under test only needs to be internally comparable, not
    matched against the full-corpus main axes.
    """
    from pathlib import Path as _Path
    all_pdfs = sorted((_Path(CHROMA_DIR).parent / "data" / "raw_pdfs").glob("*.pdf"))
    target_pdfs = all_pdfs[:limit_docs] if limit_docs else all_pdfs
    n_pdfs = len(target_pdfs)

    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = {c.name: c for c in client.list_collections()}
    coll = existing.get(cfg.collection_name())
    if coll is not None:
        # rough completeness check: enough chunks to plausibly cover all target PDFs
        if coll.count() > 0:
            n = coll.count()
            page = 5000
            docs = set()
            for off in range(0, n, page):
                batch = coll.get(include=["metadatas"], limit=page, offset=off)
                docs.update(m["doc_name"] for m in batch["metadatas"])
            n_docs = len(docs)
            if n_docs >= n_pdfs:
                print(f"[SKIP] collection '{cfg.collection_name()}' looks complete "
                      f"({n_docs} docs)")
                return
        print(f"[RESUME] collection '{cfg.collection_name()}' is incomplete, continuing")

    cmd = [sys.executable, str(SRC / "ingest.py"),
           "--chunk-size", str(cfg.chunk_size),
           "--chunk-overlap", str(cfg.chunk_overlap),
           "--embedding-model", cfg.embedding_model,
           "--resume"]
    if not cfg.preserve_structure:
        cmd.append("--no-structure")
    if limit_docs:
        cmd += ["--limit-docs", str(limit_docs)]
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="questions per config")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()

    if not args.plots_only:
        # ---------------- Axis 1: chunking ----------------
        for cs, ov in [(256, 32), (512, 64), (1024, 128)]:
            cfg = Config(chunk_size=cs, chunk_overlap=ov)
            ingest_if_needed(cfg)
            run_name = f"chunk{cs}"
            cmd = [sys.executable, str(SRC / "evaluate.py"),
                   "--run-name", run_name,
                   "--chunk-size", str(cs), "--chunk-overlap", str(ov)]
            if args.n:
                cmd += ["--n", str(args.n)]
            eval_if_needed(run_name, cmd)

        # ---------------- Axis 2: embedding model ----------------
        for emb, tag in [("sentence-transformers/all-MiniLM-L6-v2", "minilm"),
                         ("BAAI/bge-small-en-v1.5", "bge"),
                         ("sentence-transformers/all-mpnet-base-v2", "mpnet")]:
            cfg = Config(embedding_model=emb)
            ingest_if_needed(cfg)
            run_name = f"emb_{tag}"
            cmd = [sys.executable, str(SRC / "evaluate.py"),
                   "--run-name", run_name, "--embedding-model", emb]
            if args.n:
                cmd += ["--n", str(args.n)]
            eval_if_needed(run_name, cmd)

        # ---------------- Axis 3: k ----------------
        for k in [1, 3, 5, 10]:
            run_name = f"k{k}"
            cmd = [sys.executable, str(SRC / "evaluate.py"),
                   "--run-name", run_name, "--k", str(k)]
            if args.n:
                cmd += ["--n", str(args.n)]
            eval_if_needed(run_name, cmd)

        # ---------------- Axis 4: generation model ----------------
        # Ollama is not installed in this environment, so we compare local
        # transformers-backend models instead of the Ollama tags — still a
        # genuine "different generation models" comparison (project instructions
        # list Qwen and a Llama-family model as suitable examples). Reuses the
        # default full-corpus index (cs512/minilm), so no re-ingestion needed.
        for gen, tag in [("Qwen/Qwen2.5-1.5B-Instruct", "qwen1_5b"),
                         ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama1_1b")]:
            run_name = f"gen_{tag}"
            cmd = [sys.executable, str(SRC / "evaluate.py"),
                   "--run-name", run_name, "--generation-model", gen]
            if args.n:
                cmd += ["--n", str(args.n)]
            eval_if_needed(run_name, cmd)

        # ---------------- Axis 5: document structure preservation ----------------
        # Explicitly required by the project instructions alongside chunk size
        # and overlap. Uses a reduced corpus subset (first 30 PDFs alphabetical,
        # which covers all documents referenced by the first --n eval questions)
        # purely for compute feasibility. A non-default chunk_size/overlap pair
        # (384/48) is used deliberately so BOTH conditions are freshly ingested
        # on the identical 30-doc subset — reusing the full 263-doc cs512_ov64
        # collection for the "on" condition would make the comparison unfair.
        STRUCT_LIMIT = 30
        for preserve, tag in [(True, "struct_on"), (False, "struct_off")]:
            cfg = Config(chunk_size=384, chunk_overlap=48, preserve_structure=preserve)
            ingest_if_needed(cfg, limit_docs=STRUCT_LIMIT)
            run_name = tag
            cmd = [sys.executable, str(SRC / "evaluate.py"), "--run-name", run_name,
                   "--chunk-size", "384", "--chunk-overlap", "48"]
            if not preserve:
                cmd.append("--no-structure")
            if args.n:
                cmd += ["--n", str(args.n)]
            eval_if_needed(run_name, cmd)

        # ---------------- Axis 6: chunk overlap (independent of chunk size) ----------------
        # Axis 1 varies chunk_size and chunk_overlap together (proportionally);
        # this isolates overlap's effect at a fixed chunk_size, as separately
        # required by the instructions. Overlap values deliberately avoid 64
        # (the default, already fully ingested) so all three points here are
        # freshly built on the identical 30-doc subset — a fair, isolated
        # comparison rather than mixing full-corpus and subset results.
        OVERLAP_LIMIT = 30
        for ov in [0, 32, 96]:
            cfg = Config(chunk_size=512, chunk_overlap=ov)
            ingest_if_needed(cfg, limit_docs=OVERLAP_LIMIT)
            run_name = f"ov{ov}"
            cmd = [sys.executable, str(SRC / "evaluate.py"),
                   "--run-name", run_name, "--chunk-overlap", str(ov)]
            if args.n:
                cmd += ["--n", str(args.n)]
            eval_if_needed(run_name, cmd)

    build_table_and_plots()


def build_table_and_plots() -> None:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for f in sorted(RESULTS_DIR.glob("*.summary.json")):
        s = json.loads(f.read_text())
        rows.append({
            "run": s["run_name"],
            "chunk_size": s["config"]["chunk_size"],
            "chunk_overlap": s["config"]["chunk_overlap"],
            "preserve_structure": s["config"]["preserve_structure"],
            "embedding": s["config"]["embedding_model"].split("/")[-1],
            "k": s["config"]["top_k"],
            "gen_model": s["config"]["generation_model"],
            "doc_hit@k": round(s["doc_hit@k"], 3),
            "evidence_recall@k": round(s["evidence_recall@k"], 3),
            "MRR": round(s["MRR"], 3),
            "numeric_match": round(s["numeric_match"], 3),
            "token_f1": round(s["token_f1"], 3),
            **({"llm_judge": round(s["llm_judge"], 3)} if "llm_judge" in s else {}),
        })
    if not rows:
        print("[WARN] no summary files found in results/ — run evaluations first")
        return

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[TABLE] written to {csv_path}\n")
    print(df.to_string(index=False))

    plot_dir = RESULTS_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)
    for metric in ["evidence_recall@k", "numeric_match", "token_f1"]:
        if metric not in df:
            continue
        ax = df.plot.bar(x="run", y=metric, legend=False, figsize=(9, 4),
                         title=f"{metric} per configuration")
        ax.set_ylabel(metric)
        plt.tight_layout()
        out = plot_dir / f"{metric.replace('@', '_at_')}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[PLOT] {out}")


if __name__ == "__main__":
    main()
