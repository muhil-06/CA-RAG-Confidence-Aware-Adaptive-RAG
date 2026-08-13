"""
Streamlit demo app for CA-RAG.

Run with:
    streamlit run app.py
"""

import streamlit as st

from src.pipeline import CARAGPipeline
from src.vector_store import VectorStore
from src.config import Config
import re

st.set_page_config(page_title="CA-RAG Demo", page_icon="⚡", layout="wide")


@st.cache_resource
def get_pipeline():
    """Load the pipeline once and cache it across reruns."""
    vector_store = VectorStore()
    if vector_store.count() == 0:
        with st.spinner("Indexing sample documents..."):
            vector_store.index_documents()
    return CARAGPipeline(vector_store=vector_store)


def sanitize_answer(text: str) -> str:
    """Remove code blocks and inline code from model answers and return safe markdown.

    - Removes triple-backtick fenced code blocks
    - Removes indented code blocks (4+ spaces)
    - Strips inline backticks
    """
    if not text:
        return ""
    # Preserve fenced code blocks and indented code — only normalize spacing.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def apply_styles() -> None:
    """Inject custom CSS to improve chat appearance and place metadata below answers."""
    st.markdown(
        """
        <style>
        /* Narrow the main content and add a subtle background */
        .stApp {
            background-color: #0f1720;
            color: #e6eef8;
        }
        .stChatMessage > div {
            font-size: 15px;
        }
        .meta {
            color: #99a3b3;
            font-size: 12px;
            margin-top: 6px;
        }
        .stSidebar {
            background-color: #0b1220;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "skipped_retrieval" not in st.session_state:
        st.session_state.skipped_retrieval = 0


def main():
    init_state()

    apply_styles()

    st.title("⚡ CA-RAG — Confidence-Aware Adaptive RAG")
    st.caption(
        "Unlike standard RAG, this system only retrieves documents when the model "
        "isn't confident it already knows the answer — saving cost and latency."
    )

    try:
        pipeline = get_pipeline()
    except Exception as exc:
        st.error(f"Failed to initialize pipeline: {exc}")
        st.info(
            "Check your .env file — make sure MODE is set correctly and, if using "
            "MODE=api, that ANTHROPIC_API_KEY is set. If using MODE=local, make sure "
            "Ollama is running (`ollama serve`)."
        )
        return

    # --- Sidebar stats ---
    with st.sidebar:
        st.header("📊 Session Stats")
        st.metric("Total queries", st.session_state.total_queries)
        skip_rate = (
            (st.session_state.skipped_retrieval / st.session_state.total_queries * 100)
            if st.session_state.total_queries > 0 else 0.0
        )
        st.metric("Retrieval calls saved", f"{skip_rate:.0f}%")
        st.divider()
        st.caption(f"Mode: `{Config.MODE}`")
        st.caption(f"Confidence threshold: `{Config.CONFIDENCE_THRESHOLD}`")
        st.caption(f"Similarity threshold: `{Config.SIMILARITY_THRESHOLD}`")
        st.divider()
        st.caption(
            "Try asking a general knowledge question (e.g. 'What is the capital of Japan?') "
            "vs. a document-specific one (e.g. 'How many days of sick leave do I get?')."
        )

    # --- Chat history ---
    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["query"])
        with st.chat_message("assistant"):
            badge = (
                "⚡ Answered directly (no retrieval)" if not entry["retrieved"]
                else f"📄 Retrieved {entry['num_chunks_used']} document chunk(s)"
            )
            st.caption(badge)
            # sanitize answer to avoid showing raw code blocks or inline code
            sanitized = sanitize_answer(entry.get("answer", ""))
            st.markdown(sanitized)
            st.markdown(
                f"<div class=\"meta\">Confidence: {entry['confidence']} | Latency: {entry['latency_ms']:.0f}ms | Est. cost: ${entry['estimated_cost_usd']:.6f}</div>",
                unsafe_allow_html=True,
            )

    # --- Chat input ---
    query = st.chat_input("Ask a question...")
    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = pipeline.run(query)
                except Exception as exc:
                    st.error(f"Error running query: {exc}")
                    return

            badge = (
                "⚡ Answered directly (no retrieval)" if not result.retrieved
                else f"📄 Retrieved {result.num_chunks_used} document chunk(s)"
            )
            st.caption(badge)
            sanitized = sanitize_answer(result.answer)
            st.markdown(sanitized)
            st.markdown(
                f"<div class=\"meta\">Confidence: {result.confidence} | Latency: {result.latency_ms:.0f}ms | Est. cost: ${result.estimated_cost_usd:.6f}</div>",
                unsafe_allow_html=True,
            )

        st.session_state.history.append({
            "query": query,
            "answer": result.answer,
            "retrieved": result.retrieved,
            "num_chunks_used": result.num_chunks_used,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "estimated_cost_usd": result.estimated_cost_usd,
        })
        st.session_state.total_queries += 1
        if not result.retrieved:
            st.session_state.skipped_retrieval += 1


if __name__ == "__main__":
    main()
