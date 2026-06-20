# Plain-Vanilla RAG

A small RAG CLI for answering questions from PDF documents using Gemini, LangChain, and a local Chroma vector index.

## What it does

- Loads PDFs while preserving page metadata.
- Splits text into 1,000-character chunks with 200-character overlap.
- Uses `gemini-embedding-001` and retrieves the top four chunks.
- Generates answers with `gemini-2.5-flash` using only retrieved context.
- Returns source, page, and chunk citations.
- Uses a fixed response for unsupported questions.
- Includes an LLM grounding evaluator.

## Setup

Requires Python 3.10+ and a Gemini API key.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add `GOOGLE_API_KEY` to `.env`.

## Run

```powershell
# Build the local index
python rag.py ingest RegsNavyIV.pdf

# Ask one question
python rag.py query "What new part did Amendment No. 82 insert?"

# Run the example evaluation
python rag.py evaluate eval_questions.example.jsonl --output eval_results.json
```

Run `python rag.py query` without a question for interactive mode.

## Verified results

The supplied 99-page PDF produced 262 chunks. The two-row smoke test passed:

- groundedness: `1.0`;
- answerable/unanswerable accuracy: `1.0`;
- retrieval hit rate: `1.0`.

This confirms the pipeline runs end to end; it is not a full quality benchmark.

## Design scope

The solution intentionally uses simple recursive chunking, dense top-four retrieval, strict citation validation, and a local vector store. Hybrid retrieval, OCR, reranking, and document-version handling were excluded to keep the timed baseline reliable and explainable.

For detailed setup, expected output, and troubleshooting, see [SMOKE_TEST.md](SMOKE_TEST.md).
