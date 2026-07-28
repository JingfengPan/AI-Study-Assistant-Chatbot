"""Streamlit interface for the citation-grounded study assistant."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import streamlit as st
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from config import ConfigurationError, Settings, get_settings
from ingest import (
    DocumentExtractionError,
    EmptyDocumentError,
    IngestionError,
    UnsupportedFileTypeError,
    ingest_document,
)
from llm import (
    LLMError,
    answer_with_sources,
    authoritative_source_mapping,
    create_client,
    generate_summary,
    validate_citations,
)
from retrieve import (
    EmbeddingError,
    InvalidIndexError,
    NumpyVectorIndex,
    embed_texts,
)


st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
)
st.markdown(
    """
    <style>
    h1 {
        font-size: 2.2rem !important;
        line-height: 1.08 !important;
        white-space: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _initialize_state() -> None:
    defaults: dict[str, Any] = {
        "course_name": "",
        "documents": [],
        "vector_index": None,
        "messages": [],
        "selected_document_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _all_chunks() -> list:
    return [
        chunk
        for document in st.session_state["documents"]
        for chunk in document["chunks"]
    ]


def _rebuild_index() -> None:
    chunks = _all_chunks()
    if not chunks:
        st.session_state["vector_index"] = None
        return
    embeddings = np.asarray([chunk.embedding for chunk in chunks], dtype=np.float32)
    st.session_state["vector_index"] = NumpyVectorIndex(chunks, embeddings)


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY and try again."
    if isinstance(exc, RateLimitError):
        return "The OpenAI rate limit was reached. Wait briefly and try again."
    if isinstance(exc, APITimeoutError):
        return "The OpenAI request timed out after retries. Try again."
    if isinstance(exc, APIConnectionError):
        return "Could not connect to OpenAI. Check the network and try again."
    if isinstance(exc, UnsupportedFileTypeError):
        return str(exc)
    if isinstance(exc, (EmptyDocumentError, DocumentExtractionError)):
        return str(exc)
    if isinstance(exc, (EmbeddingError, InvalidIndexError, LLMError)):
        return str(exc)
    if isinstance(exc, IngestionError):
        return "The document could not be processed."
    return "Something went wrong while processing the request."


def _document_id(file_hash: str, category: str) -> str:
    return hashlib.sha256(f"{file_hash}:{category}".encode()).hexdigest()


def _process_upload(uploaded_file, category: str, settings: Settings, client) -> None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    duplicate = next(
        (
            document
            for document in st.session_state["documents"]
            if document["file_hash"] == file_hash
        ),
        None,
    )
    if duplicate:
        st.session_state["selected_document_id"] = duplicate["document_id"]
        st.info(f"{uploaded_file.name} is already indexed; reused the existing copy.")
        return

    with st.spinner(f"Extracting and indexing {uploaded_file.name}…"):
        file_hash, units, chunks = ingest_document(
            file_bytes,
            uploaded_file.name,
            category,
            settings.chunk_size_tokens,
            settings.chunk_overlap_tokens,
        )
        vectors = embed_texts(
            [chunk.text for chunk in chunks],
            client,
            settings.embedding_model,
            settings.embedding_batch_size,
        )
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector.tolist()
        full_text = "\n\n".join(unit.text for unit in units)
        summary = generate_summary(
            full_text,
            category,
            st.session_state["course_name"],
            client,
            settings,
        )
        document_id = _document_id(file_hash, category)
        st.session_state["documents"].append(
            {
                "document_id": document_id,
                "file_hash": file_hash,
                "name": uploaded_file.name,
                "category": category,
                "source_units": units,
                "chunks": chunks,
                "summary": summary.text,
                "summary_usage": summary,
            }
        )
        st.session_state["selected_document_id"] = document_id
        _rebuild_index()
    st.success(
        f"Indexed {uploaded_file.name}: {len(units)} source locations, "
        f"{len(chunks)} chunks."
    )


def _render_sidebar(settings: Settings, client) -> None:
    with st.sidebar:
        st.header("Course workspace")
        st.session_state["course_name"] = st.text_input(
            "Course name",
            value=st.session_state["course_name"],
            placeholder="e.g. Algorithms",
        ).strip()
        category = st.radio(
            "Material type",
            ("Reading Materials", "Homework"),
            horizontal=True,
        )
        uploads = st.file_uploader(
            "Upload course materials",
            type=["pdf", "pptx", "docx", "txt"],
            accept_multiple_files=True,
            help="Text extraction only; image-only files are not supported.",
        )
        if st.button("Process uploads", type="primary", use_container_width=True):
            if not st.session_state["course_name"]:
                st.warning("Enter a course name first.")
            elif not uploads:
                st.warning("Choose at least one file.")
            else:
                for uploaded in uploads:
                    try:
                        _process_upload(uploaded, category, settings, client)
                    except Exception as exc:
                        st.error(f"{uploaded.name}: {_safe_error_message(exc)}")

        st.divider()
        st.caption(
            f"Chat: `{settings.chat_model}`  \n"
            f"Embeddings: `{settings.embedding_model}`"
        )


def _render_documents() -> None:
    st.subheader("Materials")
    documents = st.session_state["documents"]
    if not documents:
        st.info("Upload a PDF, PPTX, DOCX, or TXT file to begin.")
        return
    labels = {
        document["document_id"]: (
            f"{document['name']} · {document['category']} · "
            f"{len(document['chunks'])} chunks"
        )
        for document in documents
    }
    selected = st.selectbox(
        "Document summary",
        options=list(labels),
        format_func=labels.get,
        index=max(
            0,
            next(
                (
                    index
                    for index, document in enumerate(documents)
                    if document["document_id"]
                    == st.session_state["selected_document_id"]
                ),
                0,
            ),
        ),
    )
    st.session_state["selected_document_id"] = selected
    document = next(item for item in documents if item["document_id"] == selected)
    st.markdown(document["summary"])


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.caption("Authoritative source mapping")
            for number, source, location in message["sources"]:
                st.markdown(f"`[{number}]` **{source}** — {location}")
            with st.expander("Retrieved excerpts"):
                for item in message.get("retrievals", []):
                    st.markdown(
                        f"**[{item['number']}] {item['source']} — "
                        f"{item['location']}**"
                    )
                    st.write(item["excerpt"])
                    st.caption(f"Similarity: {item['score']:.3f}")
        usage = message.get("usage")
        if usage:
            st.caption(
                f"{usage['total_tokens'] or '—'} tokens · "
                f"{usage['latency_ms']:.0f} ms"
            )


def _render_chat(settings: Settings, client) -> None:
    st.subheader("Ask across all uploaded materials")
    for message in st.session_state["messages"]:
        _render_message(message)

    question = st.chat_input(
        "Ask a question grounded in the uploaded materials",
        disabled=not st.session_state["documents"],
    )
    if not question:
        return
    question = question.strip()
    if not question:
        st.warning("Enter a non-empty question.")
        return
    user_message = {"role": "user", "content": question}
    st.session_state["messages"].append(user_message)
    with st.chat_message("user"):
        st.markdown(question)

    try:
        index = st.session_state["vector_index"]
        if not isinstance(index, NumpyVectorIndex):
            raise InvalidIndexError("No valid vector index is available.")

        def embed_query(text: str) -> np.ndarray:
            return embed_texts(
                [text],
                client,
                settings.embedding_model,
                settings.embedding_batch_size,
            )[0]

        retrieved = index.search(question, embed_query, settings.retrieval_top_k)
        result = answer_with_sources(
            question,
            retrieved,
            st.session_state["messages"][:-1],
            st.session_state["course_name"],
            client,
            settings,
        )
        validation = validate_citations(result.text, retrieved)
        sources = authoritative_source_mapping(
            retrieved,
            validation.cited_numbers if validation.cited_numbers else None,
        )
        retrievals = [
            {
                "number": number,
                "chunk_id": item.chunk.chunk_id,
                "source": item.chunk.source,
                "location": item.chunk.location,
                "score": item.score,
                "excerpt": item.chunk.text[:500],
            }
            for number, item in enumerate(retrieved, start=1)
        ]
        assistant_message = {
            "role": "assistant",
            "content": result.text,
            "sources": sources,
            "citation_valid": validation.is_valid,
            "retrievals": retrievals,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
            },
        }
        st.session_state["messages"].append(assistant_message)
        _render_message(assistant_message)
        if not validation.is_valid:
            st.warning(
                "The answer's citation structure was invalid. Verify the displayed "
                "source excerpts before relying on it."
            )
    except Exception as exc:
        st.error(_safe_error_message(exc))


def main() -> None:
    _initialize_state()
    st.title("AI Study Assistant")
    st.caption(
        "Summarize course materials and ask citation-grounded questions across "
        "documents."
    )
    try:
        settings = get_settings()
        client = create_client(settings)
    except ConfigurationError as exc:
        st.error(str(exc))
        st.code(
            "OPENAI_API_KEY=your-key\n"
            "OPENAI_CHAT_MODEL=gpt-5.6-terra\n"
            "OPENAI_EMBEDDING_MODEL=text-embedding-3-small",
            language="dotenv",
        )
        st.stop()

    _render_sidebar(settings, client)
    materials, chat = st.tabs(("Materials & summaries", "Grounded chat"))
    with materials:
        _render_documents()
    with chat:
        _render_chat(settings, client)


if __name__ == "__main__":
    main()
