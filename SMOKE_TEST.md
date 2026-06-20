# RAG Smoke-Test Guide

Run every command from the repository directory containing `rag.py`:

```powershell
cd C:\Users\Admin\rec20-39\rec20-39
```

## 1. Select Python

This workspace already has an isolated Python 3.12 installation:

```powershell
$python = "..\.python\python.exe"
& $python --version
```

Expected: `Python 3.12.10`.

On another machine, use a normal Python 3.10+ installation instead:

```powershell
$python = "python"
```

## 2. Install and verify dependencies

```powershell
& $python -m pip install -r requirements.txt
& $python -m pip check
& $python -m py_compile rag.py
```

Expected:

```text
No broken requirements found.
```

`py_compile` succeeds silently.

## 3. Configure Gemini

If `.env` does not exist:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set:

```dotenv
GOOGLE_API_KEY=your-gemini-api-key
```

Confirm that the variable loads without printing the secret:

```powershell
& $python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API key loaded:', bool(os.getenv('GOOGLE_API_KEY')))"
```

Expected: `API key loaded: True`.

## 4. Build the index

```powershell
& $python rag.py ingest RegsNavyIV.pdf
```

Expected final line:

```text
Indexed 262 chunks from 1 PDF(s) (99 pages) in .rag_index.
```

Gemini's free embedding tier is rate limited. The script embeds 90 chunks at a time and pauses for 61 seconds, so this step takes roughly two to three minutes. Re-running ingest safely creates a fresh index.

## 5. Test an answerable question

```powershell
& $python rag.py query "What new part did Amendment No. 82 insert into the Regulations for the Navy, 1965?" --json
```

The result should state that Amendment No. 82 inserted Part IV, the Indian Naval Auxiliary Service Regulations, 1973. It should include a source similar to:

```json
{
  "label": "S1",
  "source": "RegsNavyIV.pdf",
  "page": 1
}
```

The chunk ID and relevance score may vary if ingestion settings change.

## 6. Test an unanswerable question

```powershell
& $python rag.py query "What was the closing price of Apple stock yesterday?" --json
```

Expected answer and sources:

```json
{
  "answer": "I cannot answer this based on the provided documents.",
  "sources": []
}
```

The wording must match exactly and no source should be cited.

## 7. Run the evaluator

```powershell
& $python rag.py evaluate eval_questions.example.jsonl --output eval_results.json
```

Expected smoke-test summary:

```json
{
  "questions": 2,
  "groundedness_pass_rate": 1.0,
  "unanswerable_correct_rate": 1.0,
  "retrieval_hit_rate": 1.0
}
```

`eval_results.json` contains the detailed answers and is intentionally ignored by Git. These two rows prove that the pipeline runs; they are not a statistically meaningful quality benchmark.

## 8. Try interactive mode

```powershell
& $python rag.py query
```

Type questions at the prompt. Enter `exit` or `quit` to stop.

## Troubleshooting

### `OPENAI_API_KEY` or `GOOGLE_API_KEY` is missing

This project uses Gemini. Ensure `.env` contains `GOOGLE_API_KEY`, not an OpenAI key variable, and run commands from the repository directory.

### `Index not found`

Run the ingest command before querying or evaluating.

### `RESOURCE_EXHAUSTED` or HTTP 429

Wait one minute and retry. Keep these free-tier-safe defaults in `.env`:

```dotenv
GEMINI_EMBEDDING_BATCH_SIZE=90
GEMINI_EMBEDDING_BATCH_PAUSE_SECONDS=61
```

### `DLL load failed while importing chromadb_rust_bindings`

Install the Microsoft Visual C++ 2015-2022 Redistributable for x64, then open a new terminal and retry.

### Final repository check

```powershell
git status --short
git log -5 --oneline
```

Generated secrets and artifacts such as `.env`, `.rag_index`, the PDF, images, and `eval_results.json` should not appear as tracked changes.
