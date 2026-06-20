# Plain-Vanilla RAG Baseline

A runnable retrieval-augmented generation baseline for procurement and policy PDFs. The current code provides explicit ingest, query, citation, abstention, and evaluation commands. The revised plan below addresses the main retrieval and evaluation weaknesses discovered during review.

## Current status

The repository currently implements:

- page-aware PDF extraction;
- recursive 1,000/200-character chunks;
- Gemini embeddings in a persistent Chroma index;
- top-four dense retrieval;
- context-only generation with validated source/page/chunk citations;
- a fixed cannot-answer response;
- Gemini-based groundedness judging plus optional retrieval-hit and abstention metrics.

This is the time-boxed baseline, not the final target design. In particular, citation validation proves that a cited label was retrieved; it does not prove that retrieval found the correct clause, year, or document version.

## Revised 40-minute improvement plan

- **0-5 minutes - establish tests:** create a small gold slice containing an exact clause ID, a date, a cross-document question, a superseded-policy conflict, and an unanswerable question.
- **5-15 minutes - preserve structure:** detect headings and numbered clauses, attach `section_id` and section-title metadata, and use recursive character splitting only as a fallback for oversized sections.
- **15-25 minutes - add hybrid retrieval:** combine BM25 lexical candidates with dense-vector candidates using reciprocal-rank fusion. Preserve exact identifier hits.
- **25-31 minutes - diversify context:** retrieve a wider candidate pool, deduplicate it, cap repeated chunks from one document, and select six to eight passages when a question is cross-document.
- **31-36 minutes - harden answers:** expose version/effective-date metadata, require both sides of a detected conflict, and abstain when the evidence does not establish the requested identifier or time scope.
- **36-40 minutes - evaluate and report:** run the gold slice, record retrieval recall, citation correctness, abstention accuracy, and answer correctness, then document failures honestly.

Under a hard deadline, the priority order is structure-aware chunks, hybrid retrieval, a small human-audited test set, and retrieval diversity. OCR, reranking, incremental indexing, and a second model provider follow afterward.

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
# Build a fresh local index
python rag.py ingest RegsNavyIV.pdf

# Ask one question
python rag.py query "Your question here"

# Or enter interactive mode
python rag.py query

# Run the implemented evaluation
python rag.py evaluate eval_questions.example.jsonl --output eval_results.json
```

To index a directory of PDFs:

```powershell
python rag.py ingest path\to\corpus
```

## How the current baseline works

1. `PyPDFLoader` extracts text and page metadata.
2. `RecursiveCharacterTextSplitter` creates 1,000-character chunks with 200-character overlap.
3. `gemini-embedding-001` embeds chunks into a local Chroma cosine index.
4. A query retrieves four dense-vector matches and sends only those passages to `gemini-2.5-flash`.
5. The prompt requires context-supported claims and inline `[S1]` citations.
6. Application code maps labels to file, page, and chunk ID and rejects missing or invented labels.

The defaults can be changed with `--chunk-size`, `--overlap`, `-k`, or model settings in `.env`.

## Known limitations

- Character chunks can split numbered clauses, tables, and qualifiers.
- Pure vector search can miss exact contract numbers, clause IDs, and dates.
- Fixed `k=4` can overrepresent one document and miss cross-document evidence.
- A valid citation can still point to a plausible but wrong year or policy version.
- The generator and judge share a model family and therefore correlated blind spots.
- Rebuilding the entire index is simple but inefficient for a large changing corpus.
- The current index does not identify duplicates, superseded policies, or contradictions.

## Revised target design

### Structure-aware ingestion

Detect section headings and identifiers such as `5.2.1(b)` and retain `document_id`, `section_id`, section title, page range, effective date, revision, and status where available. Keep a complete clause together when it fits; split only oversized sections with overlap. Citations should resolve to both section and page.

### Hybrid and diverse retrieval

Run BM25 and vector retrieval in parallel, fuse their rankings, then deduplicate and diversify the candidate set. Lexical retrieval protects exact identifiers and dates; vector retrieval handles paraphrases. Cross-document questions should allow more context and apply a per-document cap rather than using a universal four-passage limit.

### Version and contradiction handling

Extract revision/effective-date metadata during ingestion. When retrieved passages disagree, present the conflict with both citations and their versions instead of silently selecting one. If version precedence cannot be established from metadata, say so.

### Evaluation

Use a roughly 20-row human-audited gold set covering single facts, exact IDs, dates, cross-document synthesis, conflicting versions, and unanswerable questions. Report retrieval Recall@k or MRR separately from answer correctness, citation entailment, and abstention accuracy.

The existing LLM judge remains a cheap diagnostic, not ground truth. Calibrate it against human labels and use a different model family when another provider is available.

## Later reliability work

- Maintain a content-hash manifest and update only changed documents.
- Add OCR only for pages that fail extraction checks.
- Rerank fused candidates if hybrid recall is good but ordering is weak.
- Detect near-duplicate clauses and superseded documents.
- Add automated regression tests for every audited failure.

See [brief.md](brief.md) for the detailed reasoning and acceptance criteria.
