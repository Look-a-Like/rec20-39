"""Plain-vanilla retrieval-augmented generation CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


UNANSWERABLE = "I cannot answer this based on the provided documents."
COLLECTION_NAME = "procurement_policy"
DEFAULT_INDEX = Path(".rag_index")
LEXICAL_INDEX_FILE = "lexical_documents.json"
RRF_K = 60

NUMBERED_SECTION_RE = re.compile(
    r"^(?P<section>(?:§\s*)?\d+(?:\.\d+)*(?:\([A-Za-z0-9]+\))*)[.)]?\s+"
    r"(?P<title>[A-Z][\w ,;:/&'’()\-]{2,160})$"
)
NAMED_SECTION_RE = re.compile(
    r"^(?P<section>(?:CHAPTER|SECTION|PART)\s+[A-Z0-9IVXLC]+)"
    r"(?:\s*[-:.]\s*(?P<title>.+))?$",
    re.IGNORECASE,
)
EFFECTIVE_DATE_RE = re.compile(
    r"(?:effective|issued|dated)\s*(?:as\s+of|on)?\s*[:\-]?\s*"
    r"(?P<date>[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    re.IGNORECASE,
)
REVISION_RE = re.compile(
    r"\b(?:revision|rev\.?|version|amendment)\s*[:#-]?\s*"
    r"(?P<revision>[A-Za-z0-9][A-Za-z0-9.\- ]{0,30})",
    re.IGNORECASE,
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a careful procurement and policy research assistant.

Answer the question using ONLY the supplied context. Every material claim must be
explicitly supported by that context. Do not use prior knowledge or guess.

Citation rules:
- Cite supporting passages inline with their exact labels, for example [S1].
- Use only labels present in the context.
- If multiple passages support a claim, cite each relevant label.

If the context is insufficient, respond with exactly this sentence and nothing else:
I cannot answer this based on the provided documents.

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You evaluate whether a RAG answer is grounded in retrieved context.
Return only YES or NO.

Return YES only when every factual claim in the answer is supported by the context
and every citation points to a passage that supports its claim. If the answer is the
specified cannot-answer sentence, return YES only when the context truly lacks enough
information to answer the question. Otherwise return NO.""",
        ),
        (
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}",
        ),
    ]
)


def require_api_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )


def model_name() -> str:
    return os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")


def embedding_model_name() -> str:
    return os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")


def embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=embedding_model_name())


def chat_model(model: str | None = None) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model or model_name(), temperature=0)


def discover_pdfs(inputs: list[str]) -> list[Path]:
    files: set[Path] = set()
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_file() and path.suffix.lower() == ".pdf":
            files.add(path.resolve())
        elif path.is_dir():
            files.update(p.resolve() for p in path.rglob("*.pdf"))
        else:
            raise SystemExit(f"Not a PDF file or directory: {path}")
    if not files:
        raise SystemExit("No PDF files found.")
    return sorted(files)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def document_attributes(text: str) -> dict[str, str]:
    """Extract conservative document-level metadata used for citations and conflicts."""
    effective = EFFECTIVE_DATE_RE.search(text)
    revision = REVISION_RE.search(text)
    lowered = text.lower()
    if "rescinded" in lowered:
        status = "rescinded"
    elif "superseded" in lowered or "obsolete" in lowered:
        status = "superseded"
    else:
        status = "unknown"
    return {
        "effective_date": effective.group("date") if effective else "unknown",
        "revision": revision.group("revision").strip() if revision else "unknown",
        "document_status": status,
    }


def heading_from_line(line: str) -> tuple[str, str] | None:
    """Recognize the conservative heading forms common in policy PDFs."""
    normalized = " ".join(line.split())
    numbered = NUMBERED_SECTION_RE.match(normalized)
    if numbered:
        return numbered.group("section").replace("§ ", "§"), numbered.group("title")
    named = NAMED_SECTION_RE.match(normalized)
    if named:
        return named.group("section").upper(), (named.group("title") or "").strip()
    return None


def section_documents(pdf: Path) -> tuple[list[Document], int]:
    """Load a PDF into clause-aware documents, retaining page-range metadata."""
    pages = PyPDFLoader(str(pdf)).load()
    source = display_path(pdf)
    attributes = document_attributes("\n".join(page.page_content for page in pages))
    sections: list[Document] = []
    current_lines: list[str] = []
    current_pages: list[int] = []
    section_id: str | None = None
    section_title = ""

    def flush() -> None:
        nonlocal current_lines, current_pages
        content = "\n".join(current_lines).strip()
        if not content:
            current_lines, current_pages = [], []
            return
        start_page = min(current_pages)
        end_page = max(current_pages)
        sections.append(
            Document(
                page_content=content,
                metadata={
                    "source": source,
                    "page_number": start_page,
                    "page_start": start_page,
                    "page_end": end_page,
                    "section_id": section_id or f"page-{start_page}",
                    "section_title": section_title or f"Unheaded text on page {start_page}",
                    **attributes,
                },
            )
        )
        current_lines, current_pages = [], []

    for page in pages:
        page_number = int(page.metadata.get("page", 0)) + 1
        for raw_line in page.page_content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            heading = heading_from_line(line)
            if heading:
                flush()
                section_id, section_title = heading
            current_lines.append(line)
            current_pages.append(page_number)
        if current_lines and current_pages[-1] != page_number:
            current_pages.append(page_number)
    flush()
    return sections, len(pages)


def split_sections(
    sections: list[Document], chunk_size: int, overlap: int
) -> list[Document]:
    """Keep complete sections when possible and split only oversized sections."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for section in sections:
        if len(section.page_content) <= chunk_size:
            section.metadata["start_index"] = 0
            chunks.append(section)
        else:
            chunks.extend(splitter.split_documents([section]))
    return chunks


def write_lexical_index(index_dir: Path, chunks: list[Document]) -> None:
    """Persist the text and metadata needed for local BM25 retrieval."""
    records = [
        {"text": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
    (index_dir / LEXICAL_INDEX_FILE).write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )


def open_lexical_index(index_dir: Path) -> list[dict[str, Any]]:
    lexical_path = index_dir / LEXICAL_INDEX_FILE
    if not lexical_path.exists():
        raise SystemExit(
            f"Lexical index not found at {lexical_path}. Re-run the ingest command."
        )
    try:
        records = json.loads(lexical_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Lexical index is invalid: {error}") from error
    if not isinstance(records, list):
        raise SystemExit("Lexical index is invalid: expected a list of documents.")
    return records


def tokenize(text: str) -> list[str]:
    """Keep clause-like tokens intact while normalizing ordinary words."""
    return re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())


def identifier_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if any(character.isdigit() for character in token)
        and ("." in token or "-" in token or len(token) >= 4)
    }


def passage_from_metadata(
    text: str, metadata: dict[str, Any], score: float, method: str
) -> dict[str, Any]:
    return {
        "text": text,
        "score": float(score),
        "source": str(metadata.get("source", "unknown")),
        "page": int(metadata.get("page_number", 0)),
        "page_start": int(metadata.get("page_start", 0)),
        "page_end": int(metadata.get("page_end", 0)),
        "section_id": str(metadata.get("section_id", "unknown")),
        "section_title": str(metadata.get("section_title", "")),
        "effective_date": str(metadata.get("effective_date", "unknown")),
        "revision": str(metadata.get("revision", "unknown")),
        "document_status": str(metadata.get("document_status", "unknown")),
        "chunk_id": str(metadata.get("chunk_id", "unknown")),
        "retrieval_methods": [method],
    }


def dense_candidates(store: Chroma, question: str, candidate_k: int) -> list[dict[str, Any]]:
    results = store.similarity_search_with_relevance_scores(question, k=candidate_k)
    return [
        passage_from_metadata(document.page_content, document.metadata, score, "dense")
        for document, score in results
    ]


def bm25_candidates(
    lexical_documents: list[dict[str, Any]], question: str, candidate_k: int
) -> list[dict[str, Any]]:
    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    tokenized_documents = [tokenize(str(document.get("text", ""))) for document in lexical_documents]
    document_count = len(tokenized_documents)
    if document_count == 0:
        return []

    document_frequency: dict[str, int] = {}
    for tokens in tokenized_documents:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    average_length = sum(len(tokens) for tokens in tokenized_documents) / document_count
    query_identifiers = identifier_tokens(question)
    scored: list[tuple[float, dict[str, Any]]] = []

    for document, tokens in zip(lexical_documents, tokenized_documents):
        term_frequency: dict[str, int] = {}
        for token in tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            frequency = term_frequency.get(token, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (document_count - document_frequency.get(token, 0) + 0.5)
                / (document_frequency.get(token, 0) + 0.5)
            )
            score += inverse_frequency * (frequency * 2.0) / (
                frequency + 1.0 * (1 - 0.75 + 0.75 * len(tokens) / average_length)
            )
        text = str(document.get("text", ""))
        normalized_text = text.lower()
        score += 10.0 * sum(identifier in normalized_text for identifier in query_identifiers)
        if score > 0:
            scored.append(
                (
                    score,
                    passage_from_metadata(
                        text, dict(document.get("metadata", {})), score, "lexical"
                    ),
                )
            )
    return [passage for _, passage in sorted(scored, reverse=True, key=lambda item: item[0])[:candidate_k]]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]], rrf_k: int = RRF_K
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for ranked_list in ranked_lists:
        for rank, passage in enumerate(ranked_list, start=1):
            chunk_id = passage["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = {**passage, "rrf_score": 0.0}
            fused[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)
            method = passage["retrieval_methods"][0]
            if method not in fused[chunk_id]["retrieval_methods"]:
                fused[chunk_id]["retrieval_methods"].append(method)
    return sorted(fused.values(), key=lambda passage: passage["rrf_score"], reverse=True)


def is_cross_document_question(question: str) -> bool:
    lowered = question.lower()
    return any(
        phrase in lowered
        for phrase in ("compare", "contrast", "conflict", "difference", "between", "both")
    )


def diversify_passages(
    passages: list[dict[str, Any]], question: str, k: int
) -> list[dict[str, Any]]:
    """Avoid spending a cross-document context window on one repeated source."""
    target_k = max(k, 6) if is_cross_document_question(question) else k
    per_source_limit = 2 if is_cross_document_question(question) else max(3, k)
    selected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for passage in passages:
        source = passage["source"]
        if source_counts.get(source, 0) >= per_source_limit:
            continue
        selected.append(passage)
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(selected) == target_k:
            return selected
    for passage in passages:
        if passage not in selected:
            selected.append(passage)
        if len(selected) == target_k:
            break
    return selected


def reset_index(index_dir: Path) -> None:
    if not index_dir.exists():
        return
    resolved = index_dir.resolve()
    protected = {Path.cwd().resolve(), Path.home().resolve(), Path(resolved.anchor)}
    if resolved in protected:
        raise SystemExit(f"Refusing to delete unsafe index path: {resolved}")
    shutil.rmtree(resolved)


def ingest(inputs: list[str], index_dir: Path, chunk_size: int, overlap: int) -> None:
    require_api_key()
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise SystemExit("Require chunk_size > 0 and 0 <= overlap < chunk_size.")

    pdfs = discover_pdfs(inputs)
    sections: list[Document] = []
    page_count = 0
    for pdf in pdfs:
        loaded_sections, pages = section_documents(pdf)
        sections.extend(loaded_sections)
        page_count += pages

    chunks = split_sections(sections, chunk_size, overlap)
    if not chunks:
        raise SystemExit("The PDFs contained no extractable text.")

    ids: list[str] = []
    for chunk in chunks:
        identity = "|".join(
            [
                str(chunk.metadata.get("source", "unknown")),
                str(chunk.metadata.get("page_number", "?")),
                str(chunk.metadata.get("start_index", 0)),
                chunk.page_content,
            ]
        )
        chunk_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
        chunk.metadata["chunk_id"] = chunk_id
        ids.append(chunk_id)

    reset_index(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=chunks,
        ids=ids,
        embedding=embeddings(),
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=str(index_dir),
    )
    write_lexical_index(index_dir, chunks)
    print(
        f"Indexed {len(chunks)} chunks from {len(pdfs)} PDF(s) "
        f"({page_count} pages, {len(sections)} sections) in {index_dir}."
    )


def open_store(index_dir: Path) -> Chroma:
    if not index_dir.exists():
        raise SystemExit(f"Index not found at {index_dir}. Run the ingest command first.")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings(),
        persist_directory=str(index_dir),
    )


def retrieve(
    store: Chroma,
    lexical_documents: list[dict[str, Any]],
    question: str,
    k: int,
    candidate_k: int,
) -> list[dict[str, Any]]:
    if k <= 0 or candidate_k <= 0:
        raise SystemExit("k and candidate_k must be greater than zero.")
    dense = dense_candidates(store, question, candidate_k)
    lexical = bm25_candidates(lexical_documents, question, candidate_k)
    fused = reciprocal_rank_fusion([dense, lexical])
    diversified = diversify_passages(fused, question, k)
    for number, passage in enumerate(diversified, start=1):
        passage["label"] = f"S{number}"
    return diversified


def format_context(passages: list[dict[str, Any]]) -> str:
    if not passages:
        return "(No passages retrieved.)"
    sections = []
    for passage in passages:
        header = (
            f"[{passage['label']}] source={passage['source']}; "
            f"section={passage['section_id']} ({passage['section_title']}); "
            f"pages={passage['page_start']}-{passage['page_end']}; "
            f"effective_date={passage['effective_date']}; "
            f"revision={passage['revision']}; status={passage['document_status']}; "
            f"retrieval={'+'.join(passage['retrieval_methods'])}; "
            f"chunk={passage['chunk_id']}"
        )
        sections.append(f"{header}\n{passage['text']}")
    return "\n\n".join(sections)


def answer_question(
    store: Chroma,
    lexical_documents: list[dict[str, Any]],
    question: str,
    k: int,
    candidate_k: int,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    passages = retrieve(store, lexical_documents, question, k, candidate_k)
    context = format_context(passages)
    response = (ANSWER_PROMPT | chat_model()).invoke(
        {"question": question, "context": context}
    )
    answer = str(response.content).strip()

    valid_labels = {f"[{passage['label']}]" for passage in passages}
    cited_labels = set(re.findall(r"\[S\d+\]", answer))
    if answer != UNANSWERABLE and (
        not cited_labels or not cited_labels.issubset(valid_labels)
    ):
        answer = UNANSWERABLE
        cited_labels = set()

    sources = [
        {
            "label": passage["label"],
            "source": passage["source"],
            "page": passage["page"],
            "page_start": passage["page_start"],
            "page_end": passage["page_end"],
            "section_id": passage["section_id"],
            "section_title": passage["section_title"],
            "effective_date": passage["effective_date"],
            "revision": passage["revision"],
            "document_status": passage["document_status"],
            "retrieval_methods": passage["retrieval_methods"],
            "rrf_score": round(passage["rrf_score"], 4),
            "chunk_id": passage["chunk_id"],
            "relevance": round(passage["score"], 4),
        }
        for passage in passages
        if f"[{passage['label']}]" in cited_labels
    ]
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "passages": passages,
        "context": context,
    }


def print_result(result: dict[str, Any]) -> None:
    print(f"\n{result['answer']}")
    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(
                f"[{source['label']}] {source['source']}, section {source['section_id']}, "
                f"pages {source['page_start']}-{source['page_end']}, "
                f"revision {source['revision']}, status {source['document_status']}, "
                f"via {'+'.join(source['retrieval_methods'])}, "
                f"chunk {source['chunk_id']} (RRF {source['rrf_score']:.4f})"
            )


def run_query(
    index_dir: Path, question: str | None, k: int, candidate_k: int, as_json: bool
) -> None:
    require_api_key()
    store = open_store(index_dir)
    lexical_documents = open_lexical_index(index_dir)
    if question:
        result = answer_question(store, lexical_documents, question, k, candidate_k)
        if as_json:
            printable = {key: result[key] for key in ("question", "answer", "sources")}
            print(json.dumps(printable, indent=2))
        else:
            print_result(result)
        return

    print("Interactive mode. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user_input.lower() in {"exit", "quit"}:
            return
        if not user_input:
            continue
        print_result(answer_question(store, lexical_documents, user_input, k, candidate_k))


def load_eval_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON on {path}:{line_number}: {error}") from error
            if not isinstance(row.get("question"), str):
                raise SystemExit(f"Missing string 'question' on {path}:{line_number}.")
            rows.append(row)
    if not rows:
        raise SystemExit(f"No evaluation questions found in {path}.")
    return rows


def judge_grounding(question: str, answer: str, context: str) -> bool:
    judge = chat_model(os.getenv("GEMINI_JUDGE_MODEL", model_name()))
    response = (JUDGE_PROMPT | judge).invoke(
        {"question": question, "answer": answer, "context": context}
    )
    verdict = str(response.content).strip().upper()
    return verdict == "YES"


def evaluate(
    index_dir: Path,
    questions_path: Path,
    output: Path | None,
    k: int,
    candidate_k: int,
) -> None:
    require_api_key()
    store = open_store(index_dir)
    lexical_documents = open_lexical_index(index_dir)
    rows = load_eval_rows(questions_path)
    results = []

    for position, row in enumerate(rows, start=1):
        result = answer_question(
            store, lexical_documents, row["question"], k, candidate_k
        )
        grounded = judge_grounding(
            result["question"], result["answer"], result["context"]
        )
        item: dict[str, Any] = {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "grounded": grounded,
        }

        if "expected_answerable" in row:
            predicted_answerable = result["answer"] != UNANSWERABLE
            item["unanswerable_correct"] = (
                predicted_answerable == bool(row["expected_answerable"])
            )

        expected_sources = row.get("expected_sources")
        if expected_sources:
            retrieved_sources = {p["source"] for p in result["passages"]}
            item["retrieval_hit"] = any(
                expected in retrieved_sources for expected in expected_sources
            )

        results.append(item)
        print(
            f"[{position}/{len(rows)}] grounded={'YES' if grounded else 'NO'} - "
            f"{row['question']}"
        )

    summary: dict[str, Any] = {
        "questions": len(results),
        "groundedness_pass_rate": round(
            sum(item["grounded"] for item in results) / len(results), 4
        ),
    }
    for metric in ("unanswerable_correct", "retrieval_hit"):
        scored = [item[metric] for item in results if metric in item]
        if scored:
            summary[f"{metric}_rate"] = round(sum(scored) / len(scored), 4)

    report = {"summary": summary, "results": results}
    print("\n" + json.dumps(summary, indent=2))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote detailed results to {output}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build a fresh vector index")
    ingest_parser.add_argument(
        "inputs", nargs="+", help="One or more PDF files/directories"
    )
    ingest_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ingest_parser.add_argument("--chunk-size", type=int, default=1000)
    ingest_parser.add_argument("--overlap", type=int, default=200)

    query_parser = subparsers.add_parser("query", help="Ask one question or start a REPL")
    query_parser.add_argument("question", nargs="?", help="Omit for interactive mode")
    query_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    query_parser.add_argument("-k", type=int, default=6, help="Final context passages")
    query_parser.add_argument(
        "--candidate-k",
        type=int,
        default=12,
        help="Candidates fetched from each retriever before fusion",
    )
    query_parser.add_argument("--json", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Run JSONL evaluation questions")
    eval_parser.add_argument("questions", type=Path)
    eval_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    eval_parser.add_argument("--output", type=Path)
    eval_parser.add_argument("-k", type=int, default=6, help="Final context passages")
    eval_parser.add_argument(
        "--candidate-k",
        type=int,
        default=12,
        help="Candidates fetched from each retriever before fusion",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    try:
        if args.command == "ingest":
            ingest(args.inputs, args.index, args.chunk_size, args.overlap)
        elif args.command == "query":
            run_query(args.index, args.question, args.k, args.candidate_k, args.json)
        elif args.command == "evaluate":
            evaluate(args.index, args.questions, args.output, args.k, args.candidate_k)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
