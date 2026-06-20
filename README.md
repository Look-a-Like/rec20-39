# Plain-Vanilla RAG Baseline

A small, runnable retrieval-augmented generation system for the supplied PDF corpus. It has explicit ingest, query, and evaluation commands and uses only retrieved text to answer.

## 40-minute implementation plan

- **0-5 minutes — inspect:** confirm the corpus, deliverables, question formats, and available environment.
- **5-15 minutes — ingest:** page-aware PDF loading, 1,000/200 chunking, Gemini embeddings, and a fresh local Chroma index.
- **15-25 minutes — answer:** top-four retrieval, a strict context-only prompt, fixed abstention behavior, and verified source/page/chunk citations.
- **25-32 minutes — evaluate:** JSONL runner plus LLM-as-a-judge groundedness; add retrieval-hit and abstention metrics when labels exist.
- **32-40 minutes — harden and hand off:** CLI errors, interactive mode, dependency/env files, README, brief, and an end-to-end smoke test.

If time compresses further, keep ingest, query, citation validation, and one grounding metric; defer OCR, reranking, hybrid search, and UI work.

## Quick start

Requires Python 3.10+ and a Gemini API key from Google AI Studio.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your key to `.env`, then run:

```powershell
# 1. Extract, chunk, embed, and persist the local Chroma index
python rag.py ingest RegsNavyIV.pdf

# 2. Ask one question
python rag.py query "Your question here"

# Or enter interactive mode
python rag.py query

# 3. Run the implemented evaluation metric
python rag.py evaluate eval_questions.example.jsonl --output eval_results.json
```

To index a larger downloaded corpus, pass its directory:

```powershell
python rag.py ingest path\to\corpus
```

Each ingest creates a fresh index, so reruns cannot silently duplicate or retain stale chunks.

## How it works

1. `PyPDFLoader` extracts text and page metadata from every PDF.
2. `RecursiveCharacterTextSplitter` makes 1,000-character chunks with 200-character overlap.
3. `gemini-embedding-001` embeds chunks into a persistent local Chroma cosine index.
4. A query retrieves the top four chunks and sends only those chunks to `gemini-2.5-flash`.
5. The strict prompt permits only context-supported claims and requires inline `[S1]`-style citations.
6. The program validates citation labels and maps them deterministically to file, page, and chunk ID. An uncited or invalidly cited answer is converted to the fixed cannot-answer response.

The defaults can be changed with `--chunk-size`, `--overlap`, `-k`, or the model settings in `.env`.

## Key choices

- **Chunking: 1,000 / 200 characters.** Large enough to preserve a policy clause and nearby qualification; overlap reduces boundary loss.
- **Retriever: cosine similarity, k=4.** A compact amount of relevant context keeps the prompt focused and inexpensive.
- **Vector store: Chroma.** It is local, persistent, and needs no service or cloud account.
- **Generation: temperature 0.** This favors stable, reproducible answers.
- **Unanswerable handling.** The model must return exactly `I cannot answer this based on the provided documents.` when evidence is insufficient. Citation validation provides a second guardrail.

## Evaluation

`evaluate` implements **LLM-as-a-judge groundedness**. For each JSONL question it checks whether every answer claim and citation is supported by the retrieved context, then reports `groundedness_pass_rate`.

Optional labels add two transparent diagnostics:

- `expected_answerable` measures correct abstention behavior.
- `expected_sources` measures whether at least one expected document appeared in the top-k retrieval.

Example row:

```json
{"question":"...","expected_answerable":true,"expected_sources":["RegsNavyIV.pdf"]}
```

The included example is deliberately unanswerable. Replace or extend it with the provided competition questions for meaningful results. The judge is fast to add and directly tests grounding, but it is model-dependent; a labeled evaluation set with retrieval recall and answer correctness would be stronger.

## Scope and next improvements

This is intentionally a canonical baseline, not a production system. With more time: evaluate chunk size and k on labeled questions, add OCR for scanned PDFs, use hybrid lexical/vector retrieval, rerank candidates, cache embeddings, and add automated tests. See [brief.md](brief.md) for the complete rationale.
