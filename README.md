# Plain-Vanilla RAG Baseline

A small, runnable RAG system for the supplied procurement and policy PDFs. It deliberately optimizes for a clean, explainable baseline that can be completed and tested in a 75-minute coding round.

## Scope for the time limit

The implementation includes only the canonical RAG path:

1. Load PDFs with page metadata.
2. Split text recursively into 1,000-character chunks with 200-character overlap.
3. Embed chunks with Gemini and persist them in local Chroma.
4. Retrieve the top four relevant chunks for a question.
5. Ask Gemini to answer using only that context, with source citations.
6. Abstain rather than guess when the context is insufficient.
7. Run one grounding evaluation metric.

The following are intentionally out of scope for this round: structure-aware PDF parsing, BM25/hybrid retrieval, reciprocal-rank fusion, incremental indexing, automated version resolution, OCR, reranking, and UI work. They are sensible later improvements, but each adds setup and failure modes that are not justified for a static corpus and short time budget.

## 75-minute delivery plan

- **0-10 minutes:** inspect corpus, question format, page metadata, and API setup.
- **10-25 minutes:** load PDFs, recursively chunk at 1,000/200, embed, and persist Chroma.
- **25-40 minutes:** top-four retrieval and a strict context-only Gemini prompt.
- **40-52 minutes:** source/page/chunk citations and fixed cannot-answer behavior.
- **52-62 minutes:** JSONL evaluator with LLM-as-a-judge grounding.
- **62-75 minutes:** run a small smoke test, document choices, and fix obvious CLI failures.

If time shrinks further, preserve ingest, query, citations, and abstention. Do not trade a working baseline for untested retrieval enhancements.

## Quick start

Requires Python 3.10+ and a Gemini API key from Google AI Studio.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`, then run:

```powershell
# Build a fresh local index
python rag.py ingest RegsNavyIV.pdf

# Ask one question
python rag.py query "Your question here"

# Or start the interactive prompt
python rag.py query

# Evaluate grounding
python rag.py evaluate eval_questions.example.jsonl --output eval_results.json
```

To index a downloaded corpus directory:

```powershell
python rag.py ingest path\to\corpus
```

## How it works

`PyPDFLoader` extracts text and zero-based page metadata; the application converts pages to human-readable one-based values. `RecursiveCharacterTextSplitter` makes 1,000-character chunks with 200-character overlap. `gemini-embedding-001` stores those chunks in a local persistent Chroma cosine index.

For each query, the system retrieves `k=4` chunks and supplies only them to `gemini-2.5-flash`. The answer prompt requires inline `[S1]` citations. Application code maps each label to a source filename, page, and content-derived chunk ID, then rejects answers with missing or invented labels.

The system returns exactly `I cannot answer this based on the provided documents.` when evidence is insufficient. Temperature is zero for more repeatable answers.

## Evaluation

`evaluate` implements LLM-as-a-judge grounding: a second deterministic Gemini call receives the question, generated answer, and retrieved context, then returns whether the answer is fully supported. It reports `groundedness_pass_rate`.

Optional labels in a JSONL question file add:

- `expected_answerable` for abstention accuracy;
- `expected_sources` for a basic retrieval-hit rate.

The included example is deliberately unanswerable. Replace or extend it with the competition questions before reporting results. The judge is a pragmatic baseline metric, not proof of correctness; future work should use labeled retrieval and answer-correctness evaluation.

## Tradeoffs

Character chunks can split tables or clauses, dense retrieval can miss exact identifiers, and a fixed top-four may miss cross-document evidence. These are known constraints accepted to keep this submission small, runnable, and explainable. If more time becomes available, validate these failure modes against real questions before adding structure-aware chunking, hybrid retrieval, or other complexity.
