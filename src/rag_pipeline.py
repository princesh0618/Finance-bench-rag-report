"""
rag_pipeline.py — Tasks 2 & 3: retrieval + answer generation.

RETRIEVAL:
    The user question is embedded with the same Sentence-Transformers model
    used at ingestion time and the top-k most similar chunks are retrieved
    from ChromaDB (cosine similarity).

GENERATION:
    The retrieved passages are injected into a grounded-QA prompt and sent
    to a FREE, LOCALLY EXECUTABLE model (project constraint). Two backends:

      * "ollama"       : talks to a local Ollama server (http://localhost:11434).
                         Recommended — run e.g. `ollama pull qwen2.5:3b-instruct`.
      * "transformers" : loads a small HF model (e.g. Qwen/Qwen2.5-1.5B-Instruct)
                         directly with transformers. Slower on CPU but has no
                         external dependency.

    The prompt forces the model to cite its evidence as [1], [2], ... which
    map to the retrieved passages (constraint: "the generated answer should
    include references to the retrieved passages used as evidence").

Usage (interactive demo):
    python src/rag_pipeline.py --question "What was 3M's FY2018 capex?"
"""

import argparse
import json

import chromadb
from sentence_transformers import SentenceTransformer

from config import Config, CHROMA_DIR

SYSTEM_PROMPT = (
    "You are a careful financial analyst assistant. Answer the question using ONLY "
    "the numbered context passages below. "
    "MANDATORY CITATION RULE: every sentence that states a fact or figure MUST end with "
    "a bracketed reference to the passage it came from, like [1] or [2]. A sentence with "
    "a number or claim and no bracketed citation is not acceptable. "
    "If the context does not contain the answer, respond with EXACTLY this sentence and "
    "nothing else: \"I cannot answer this from the provided documents.\" "
    "Be concise and give exact figures with their units when available."
)
CITATION_REMINDER = (
    "Reminder before you answer: cite every factual sentence as [1], [2], etc. "
    "Do not write an uncited sentence."
)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class Retriever:
    def __init__(self, cfg: Config, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        if verbose:
            print(f"[RETRIEVER] loading embedding model {cfg.embedding_model}")
        self.model = SentenceTransformer(cfg.embedding_model)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(cfg.collection_name())
        if verbose:
            print(f"[RETRIEVER] collection '{cfg.collection_name()}' "
                  f"({self.collection.count()} chunks)")

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        k = k or self.cfg.top_k
        q_emb = self.model.encode([question], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=k)
        passages = []
        for text, meta, dist in zip(res["documents"][0],
                                    res["metadatas"][0],
                                    res["distances"][0]):
            passages.append({
                "text": text,
                "doc_name": meta["doc_name"],
                "page": meta.get("page"),
                "company": meta.get("company", ""),
                "score": 1 - dist,  # cosine similarity
            })
        if self.verbose:
            print(f"[RETRIEVER] top-{k} passages for: {question[:80]!r}")
            for i, p in enumerate(passages, 1):
                print(f"    [{i}] {p['doc_name']} p.{p['page']} (sim={p['score']:.3f})")
        return passages


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _assemble_prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(
        f"[{i}] (source: {p['doc_name']}, page {p['page']})\n{p['text']}"
        for i, p in enumerate(passages, 1)
    )
    return (f"{SYSTEM_PROMPT}\n\n### Context passages\n{ctx}\n\n### Question\n{question}\n\n"
            f"{CITATION_REMINDER}\n\n### Answer")


def build_prompt(question: str, passages: list[dict], tokenizer=None,
                  max_total_tokens: int = 1200) -> str:
    """Assemble the grounded prompt. If a tokenizer is given, the context is fit to
    max_total_tokens by dropping the *weakest* (lowest-similarity, i.e. last) passages
    first — never by truncating the raw token sequence, which would silently cut off
    the system prompt (containing the citation/abstention rules) whenever it sits
    before the truncation point. Passages are already ordered by descending similarity
    from retrieval, so dropping from the end drops the least relevant evidence first."""
    if tokenizer is None:
        return _assemble_prompt(question, passages)

    kept = list(passages)
    while True:
        prompt = _assemble_prompt(question, kept)
        n_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tokens <= max_total_tokens or len(kept) <= 1:
            return prompt
        kept = kept[:-1]


class Generator:
    def __init__(self, cfg: Config, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        if cfg.generation_backend == "transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            name = cfg.generation_model
            if verbose:
                print(f"[GENERATOR] loading HF model {name} (this can take a while)")
            self.tok = AutoTokenizer.from_pretrained(name)
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                name, dtype=torch.float32).to("cpu")
        elif verbose:
            print(f"[GENERATOR] using local Ollama model '{cfg.generation_model}'")

    def generate(self, prompt: str) -> str:
        if self.cfg.generation_backend == "ollama":
            return self._generate_ollama(prompt)
        return self._generate_transformers(prompt)

    def _generate_ollama(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.cfg.generation_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.max_new_tokens,
                },
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def _generate_transformers(self, prompt: str) -> str:
        import time
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _TimeLimit(StoppingCriteria):
            """Hard wall-clock cap so a degenerate (non-EOS-terminating) generation
            can never hang the pipeline, regardless of root cause (repetition loop,
            memory-pressure slowdown, ...) — cheap CPU hardware has no other
            backstop against a single pathological question stalling a whole run."""
            def __init__(self, max_seconds: float):
                self.deadline = time.time() + max_seconds

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return time.time() > self.deadline

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.hf_model.device)
        attention_mask = inputs["attention_mask"].to(self.hf_model.device)
        
        # Last-resort safety net only: build_prompt() should already have fit the
        # context to budget (dropping weak passages, keeping the system prompt
        # intact) whenever it was given a tokenizer. This blind tail-keeping slice
        # is a fallback for callers that build a prompt without one — it prevents a
        # CPU memory-allocator crash on pathologically long input, at the cost of
        # potentially cutting off the system prompt, so it should rarely fire.
        MAX_CPU_INPUT_TOKENS = 1600
        if input_ids.shape[1] > MAX_CPU_INPUT_TOKENS:
            input_ids = input_ids[:, -MAX_CPU_INPUT_TOKENS:]
            attention_mask = attention_mask[:, -MAX_CPU_INPUT_TOKENS:]

        with torch.no_grad():
            out = self.hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=max(self.cfg.temperature, 1e-5),
                pad_token_id=self.tok.eos_token_id,
                stopping_criteria=StoppingCriteriaList([_TimeLimit(120)]),
            )
        return self.tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
class RAGPipeline:
    def __init__(self, cfg: Config | None = None, verbose: bool = True):
        self.cfg = cfg or Config()
        self.retriever = Retriever(self.cfg, verbose)
        self.generator = Generator(self.cfg, verbose)
        self.verbose = verbose

    def answer(self, question: str) -> dict:
        passages = self.retriever.retrieve(question)
        tok = getattr(self.generator, "tok", None)
        prompt = build_prompt(question, passages, tokenizer=tok)
        if self.verbose:
            print("[GENERATOR] generating answer ...")
        answer = self.generator.generate(prompt)
        if self.verbose:
            print("\n================= ANSWER =================")
            print(answer)
            print("\n----------------- SOURCES ----------------")
            for i, p in enumerate(passages, 1):
                print(f"[{i}] {p['doc_name']}, page {p['page']} (sim={p['score']:.3f})")
        return {"question": question, "answer": answer, "passages": passages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.k:
        cfg.top_k = args.k
    rag = RAGPipeline(cfg)
    result = rag.answer(args.question)
    print("\n[JSON]", json.dumps(
        {k: v for k, v in result.items() if k != "passages"}, indent=2))
