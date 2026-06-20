"""Plain-vanilla retrieval-augmented generation CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def retrieve(store: Chroma, question: str, k: int) -> list[dict[str, Any]]:
    if k <= 0:
        raise SystemExit("k must be greater than zero.")
    results = store.similarity_search_with_relevance_scores(question, k=k)
    passages = []
    for number, (document, score) in enumerate(results, start=1):
        passages.append(
            {
                "label": f"S{number}",
                "text": document.page_content,
                "score": float(score),
                "source": str(document.metadata.get("source", "unknown")),
                "page": int(document.metadata.get("page_number", 0)),
                "page_start": int(document.metadata.get("page_start", 0)),
                "page_end": int(document.metadata.get("page_end", 0)),
                "section_id": str(document.metadata.get("section_id", "unknown")),
                "section_title": str(document.metadata.get("section_title", "")),
                "effective_date": str(
                    document.metadata.get("effective_date", "unknown")
                ),
                "revision": str(document.metadata.get("revision", "unknown")),
                "document_status": str(
                    document.metadata.get("document_status", "unknown")
                ),
                "chunk_id": str(document.metadata.get("chunk_id", "unknown")),
            }
        )
    return passages


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
            f"chunk={passage['chunk_id']}"
        )
        sections.append(f"{header}\n{passage['text']}")
    return "\n\n".join(sections)


def answer_question(store: Chroma, question: str, k: int) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    passages = retrieve(store, question, k)
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
                f"chunk {source['chunk_id']} (relevance {source['relevance']:.4f})"
            )


def run_query(index_dir: Path, question: str | None, k: int, as_json: bool) -> None:
    require_api_key()
    store = open_store(index_dir)
    if question:
        result = answer_question(store, question, k)
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
        print_result(answer_question(store, user_input, k))


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


def evaluate(index_dir: Path, questions_path: Path, output: Path | None, k: int) -> None:
    require_api_key()
    store = open_store(index_dir)
    rows = load_eval_rows(questions_path)
    results = []

    for position, row in enumerate(rows, start=1):
        result = answer_question(store, row["question"], k)
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
    query_parser.add_argument("-k", type=int, default=4)
    query_parser.add_argument("--json", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Run JSONL evaluation questions")
    eval_parser.add_argument("questions", type=Path)
    eval_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    eval_parser.add_argument("--output", type=Path)
    eval_parser.add_argument("-k", type=int, default=4)
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    try:
        if args.command == "ingest":
            ingest(args.inputs, args.index, args.chunk_size, args.overlap)
        elif args.command == "query":
            run_query(args.index, args.question, args.k, args.json)
        elif args.command == "evaluate":
            evaluate(args.index, args.questions, args.output, args.k)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
