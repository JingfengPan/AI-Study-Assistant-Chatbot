"""Run the stuffing baseline and production RAG evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings, get_settings
from ingest import DocumentChunk, ingest_document
from llm import (
    ANSWER_PROMPT_VERSION,
    REFUSAL_TEXT,
    STUFFING_PROMPT_VERSION,
    answer_with_sources,
    answer_with_stuffed_context,
    create_client,
    generate_summary,
    validate_citations,
)
from retrieve import NumpyVectorIndex, embed_texts

from .judge import JUDGE_PROMPT_VERSION, JudgeError, judge_answer


DEFAULT_FIXTURES = Path(__file__).parent / "fixtures"
DEFAULT_GOLDEN_SET = Path(__file__).parent / "golden_set.json"
DEFAULT_OUTPUT = Path(__file__).parent / "results.md"
SUPPORTED_FIXTURES = {".pdf", ".pptx", ".docx", ".txt"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stuffing", "rag", "both"), default="both")
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-judge", action="store_true")
    return parser.parse_args()


def _load_golden_set(path: Path, limit: int | None) -> list[dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Golden set must be a JSON list.")
    ids = [item.get("id") for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Golden-set IDs must be unique.")
    required = {
        "id",
        "question",
        "expected_answer",
        "expected_sources",
        "answerable",
        "notes",
    }
    for item in items:
        if set(item) != required:
            raise ValueError(f"Golden-set item {item.get('id')} has invalid fields.")
        if item["answerable"] and not item["expected_sources"]:
            raise ValueError(f"Answerable item {item['id']} has no expected source.")
        if not item["answerable"] and (
            item["expected_answer"] is not None or item["expected_sources"]
        ):
            raise ValueError(
                f"Unanswerable item {item['id']} must have no answer or sources."
            )
    return items[:limit] if limit else items


def _load_corpus(
    documents_dir: Path,
    settings: Settings,
    client,
) -> tuple[list[dict[str, Any]], NumpyVectorIndex, str]:
    documents = []
    all_chunks: list[DocumentChunk] = []
    for path in sorted(documents_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_FIXTURES:
            continue
        _, units, chunks = ingest_document(
            path.read_bytes(),
            path.name,
            "Reading Materials",
            settings.chunk_size_tokens,
            settings.chunk_overlap_tokens,
        )
        documents.append({"name": path.name, "units": units, "chunks": chunks})
        all_chunks.extend(chunks)
    if not documents:
        raise ValueError(f"No evaluation documents found in {documents_dir}.")

    matrix = embed_texts(
        [chunk.text for chunk in all_chunks],
        client,
        settings.embedding_model,
        settings.embedding_batch_size,
    )
    for chunk, vector in zip(all_chunks, matrix, strict=True):
        chunk.embedding = vector.tolist()
    index = NumpyVectorIndex(all_chunks, matrix)

    summary_sections = []
    for document in documents:
        text = "\n\n".join(unit.text for unit in document["units"])
        summary = generate_summary(
            text,
            "Reading Materials",
            "Synthetic Computer Science Study Corpus",
            client,
            settings,
        )
        document["summary"] = summary
        summary_sections.append(f"{document['name']}:\n{summary.text}")
    return documents, index, "\n\n".join(summary_sections)


def _source_tuple(item: dict[str, str]) -> tuple[str, str]:
    return item["file"], item["location"]


def _retrieval_source_pairs(retrieved) -> list[tuple[str, str]]:
    return [(item.chunk.source, item.chunk.location) for item in retrieved]


def _reference_match(generated: str, expected: str | None, answerable: bool) -> bool:
    if not answerable:
        return generated.strip() == REFUSAL_TEXT
    if not expected:
        return False
    normalize = lambda value: " ".join(  # noqa: E731
        "".join(character.lower() if character.isalnum() else " " for character in value)
        .split()
    )
    generated_words = set(normalize(generated).split())
    expected_words = set(normalize(expected).split())
    if not expected_words:
        return False
    return len(generated_words & expected_words) / len(expected_words) >= 0.65


def _run_item(
    *,
    mode: str,
    item: dict[str, Any],
    index: NumpyVectorIndex,
    stuffing_context: str,
    top_k: int,
    client,
    settings: Settings,
    skip_judge: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    retrieved = []
    retrieval_ms = 0.0
    query_embedding_ms = 0.0
    if mode == "rag":
        embedding_started = time.perf_counter()
        query_vector = embed_texts(
            [item["question"]],
            client,
            settings.embedding_model,
            settings.embedding_batch_size,
        )[0]
        query_embedding_ms = (time.perf_counter() - embedding_started) * 1000
        retrieval_started = time.perf_counter()
        retrieved = index.search_by_vector(query_vector, top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        result = answer_with_sources(
            item["question"],
            retrieved,
            [],
            "Synthetic Computer Science Study Corpus",
            client,
            settings,
        )
        validation = validate_citations(result.text, retrieved)
    else:
        result = answer_with_stuffed_context(
            item["question"],
            stuffing_context,
            [],
            "Synthetic Computer Science Study Corpus",
            client,
            settings,
        )
        validation = None
    end_to_end_ms = (time.perf_counter() - started) * 1000

    expected_sources = {_source_tuple(source) for source in item["expected_sources"]}
    retrieval_pairs = _retrieval_source_pairs(retrieved)
    cited_pairs = {
        retrieval_pairs[number - 1]
        for number in (validation.cited_numbers if validation else set())
        if 1 <= number <= len(retrieval_pairs)
    }
    evidence = [
        f"{result.chunk.source} — {result.chunk.location}\n{result.chunk.text}"
        for result in retrieved
    ]
    if mode == "stuffing":
        evidence = [stuffing_context]

    judge_payload = None
    judge_error = None
    # Judge the production RAG output by default. The stuffing baseline remains a
    # deterministic cost/token comparison and does not need a second model pass.
    if not skip_judge and mode == "rag":
        try:
            judge_payload = asdict(
                judge_answer(
                    question=item["question"],
                    reference_answer=item["expected_answer"],
                    evidence=evidence,
                    generated_answer=result.text,
                    answerable=item["answerable"],
                    client=client,
                    settings=settings,
                )
            )
        except JudgeError as exc:
            judge_error = str(exc)

    return {
        "id": item["id"],
        "mode": mode,
        "question": item["question"],
        "answerable": item["answerable"],
        "expected_answer": item["expected_answer"],
        "expected_sources": item["expected_sources"],
        "answer": result.text,
        "refused": result.text.strip() == REFUSAL_TEXT,
        "reference_match": _reference_match(
            result.text, item["expected_answer"], item["answerable"]
        ),
        "retrievals": [
            {
                "rank": value.rank,
                "file": value.chunk.source,
                "location": value.chunk.location,
                "chunk_id": value.chunk.chunk_id,
                "score": value.score,
            }
            for value in retrieved
        ],
        "recall_at_1": bool(expected_sources & set(retrieval_pairs[:1])),
        "recall_at_3": bool(expected_sources & set(retrieval_pairs[:3])),
        "recall_at_4": bool(expected_sources & set(retrieval_pairs[:4])),
        "citation_valid": validation.is_valid if validation else None,
        "citation_source_accurate": bool(expected_sources & cited_pairs)
        if validation and item["answerable"]
        else None,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "generation_latency_ms": result.latency_ms,
        "query_embedding_ms": query_embedding_ms,
        "retrieval_ms": retrieval_ms,
        "end_to_end_ms": end_to_end_ms,
        "judge": judge_payload,
        "judge_error": judge_error,
    }


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    judged = [row for row in rows if row["judge"]]
    tokens = [row["total_tokens"] for row in rows if row["total_tokens"] is not None]
    prompt_tokens = [
        row["prompt_tokens"] for row in rows if row["prompt_tokens"] is not None
    ]
    latencies = [row["end_to_end_ms"] for row in rows]
    return {
        "mode": mode,
        "items": len(rows),
        "recall_at_1": _rate([row["recall_at_1"] for row in answerable])
        if mode == "rag"
        else None,
        "recall_at_3": _rate([row["recall_at_3"] for row in answerable])
        if mode == "rag"
        else None,
        "recall_at_4": _rate([row["recall_at_4"] for row in answerable])
        if mode == "rag"
        else None,
        "answer_accuracy": _rate([row["reference_match"] for row in rows]),
        "citation_validity": _rate(
            [row["citation_valid"] for row in answerable if not row["refused"]]
        )
        if mode == "rag"
        else None,
        "citation_source_accuracy": _rate(
            [row["citation_source_accurate"] for row in answerable]
        )
        if mode == "rag"
        else None,
        "refusal_accuracy": _rate([row["refused"] for row in unanswerable]),
        "false_refusal_rate": _rate([row["refused"] for row in answerable]),
        "mean_prompt_tokens": _mean(prompt_tokens),
        "median_prompt_tokens": _median(prompt_tokens),
        "mean_total_tokens": _mean(tokens),
        "median_total_tokens": _median(tokens),
        "mean_latency_ms": _mean(latencies),
        "median_latency_ms": _median(latencies),
        "judge_correctness": _mean(
            [row["judge"]["correctness"] for row in judged]
        ),
        "judge_groundedness": _mean(
            [row["judge"]["groundedness"] for row in judged]
        ),
        "judge_relevance": _mean([row["judge"]["relevance"] for row in judged]),
        "judge_failures": sum(bool(row["judge_error"]) for row in rows),
    }


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _number(value: float | None, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:,.1f}{suffix}"


def _write_markdown(
    path: Path,
    settings: Settings,
    items: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    top_k: int,
    skip_judge: bool,
) -> None:
    lines = [
        "# Evaluation Results",
        "",
        f"- Date: {datetime.now(timezone.utc).isoformat()}",
        f"- Chat/judge model: `{settings.chat_model}`",
        f"- Embedding model: `{settings.embedding_model}`",
        f"- Answer prompt: `{ANSWER_PROMPT_VERSION}`",
        f"- Stuffing prompt: `{STUFFING_PROMPT_VERSION}`",
        f"- Judge prompt: `{JUDGE_PROMPT_VERSION}`"
        + (" (skipped)" if skip_judge else ""),
        f"- Chunk size / overlap: {settings.chunk_size_tokens} / "
        f"{settings.chunk_overlap_tokens} tokens",
        f"- Top-k: {top_k}",
        f"- Dataset: {len(items)} questions",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@4 | Answer Accuracy | "
        "Citation Source Accuracy | Refusal Accuracy | False Refusal Rate | "
        "Avg. Input Tokens | Median Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for aggregate in aggregates:
        lines.append(
            f"| {aggregate['mode'].title()} | {_pct(aggregate['recall_at_1'])} | "
            f"{_pct(aggregate['recall_at_3'])} | {_pct(aggregate['recall_at_4'])} | "
            f"{_pct(aggregate['answer_accuracy'])} | "
            f"{_pct(aggregate['citation_source_accuracy'])} | "
            f"{_pct(aggregate['refusal_accuracy'])} | "
            f"{_pct(aggregate['false_refusal_rate'])} | "
            f"{_number(aggregate['mean_prompt_tokens'])} | "
            f"{_number(aggregate['median_latency_ms'], ' ms')} |"
        )

    lines.extend(["", "## Supplemental judge metrics", ""])
    for aggregate in aggregates:
        lines.append(
            f"- **{aggregate['mode'].title()}**: correctness "
            f"{_pct(aggregate['judge_correctness'])}, groundedness "
            f"{_pct(aggregate['judge_groundedness'])}, relevance "
            f"{_pct(aggregate['judge_relevance'])}; "
            f"{aggregate['judge_failures']} judge failures."
        )

    failures = [
        row
        for row in rows
        if (
            not row["reference_match"]
            or (row["mode"] == "rag" and row["answerable"] and not row["recall_at_4"])
            or row["judge_error"]
        )
    ]
    lines.extend(["", "## Failure examples", ""])
    if not failures:
        lines.append("No failures under the reported automatic criteria.")
    for row in failures[:10]:
        lines.extend(
            [
                f"### {row['mode'].title()} · {row['id']}",
                "",
                f"**Question:** {row['question']}",
                "",
                f"**Answer:** {row['answer']}",
                "",
                f"Reference match: {row['reference_match']}; "
                f"Recall@4: {row['recall_at_4'] if row['mode'] == 'rag' else 'N/A'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Limitations",
            "",
            "- The corpus and golden set are intentionally small and synthetic.",
            "- Reference-match accuracy is a deterministic lexical metric; supplemental "
            "judge scores are model-based and not ground truth.",
            "- Citation syntax and source matching do not prove that every claim is "
            "entailed by its cited passage.",
            "- Results depend on the selected models, prompts, chunking, and API behavior "
            "at the recorded date.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    top_k = args.top_k or settings.retrieval_top_k
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")
    items = _load_golden_set(args.golden_set, args.limit)
    client = create_client(settings)
    _, index, stuffing_context = _load_corpus(
        args.documents_dir,
        settings,
        client,
    )
    modes = ("stuffing", "rag") if args.mode == "both" else (args.mode,)
    rows = [
        _run_item(
            mode=mode,
            item=item,
            index=index,
            stuffing_context=stuffing_context,
            top_k=top_k,
            client=client,
            settings=settings,
            skip_judge=args.skip_judge,
        )
        for mode in modes
        for item in items
    ]
    aggregates = [
        _aggregate(mode, [row for row in rows if row["mode"] == mode])
        for mode in modes
    ]
    _write_markdown(
        args.output,
        settings,
        items,
        aggregates,
        rows,
        top_k,
        args.skip_judge,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "settings": {
                        "chat_model": settings.chat_model,
                        "embedding_model": settings.embedding_model,
                        "chunk_size_tokens": settings.chunk_size_tokens,
                        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
                        "top_k": top_k,
                    },
                    "aggregates": aggregates,
                    "items": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
