"""
app.py – Streamlit UI for Document Navigator
=============================================
Run with:
    streamlit run app.py

Features
--------
* Sidebar: top-k slider, trace toggle, chunk-text expander, sample questions
* Answer panel: colour-coded by confidence (green/yellow/red)
* Confidence metric + mode badge
* Citations listed as inline code badges
* Retrieval trace table (rank, chunk_id, source, score, text preview)
* Full chunk text expanders (optional)
* Build-index button (runs ingest.py inline if index not found)
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Ensure the project root is importable regardless of how streamlit is launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TOP_K, LOW_CONFIDENCE_THRESHOLD, CLARIFY_THRESHOLD


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Navigator – PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading retrieval index …")
def _load_retriever():
    from retriever import Retriever
    return Retriever()


@st.cache_resource(show_spinner="Loading answer generator …")
def _load_generator(use_llm: bool):
    from generator import AnswerGenerator
    return AnswerGenerator(use_llm=use_llm)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Document Navigator")
    st.caption("Transparent PDF Q&A with Retrieval Traces")
    st.divider()

    st.subheader("Retrieval settings")
    top_k       = st.slider("Top-K chunks", 1, 10, TOP_K)
    show_trace  = st.checkbox("Show retrieval trace",  value=True)
    show_chunks = st.checkbox("Show full chunk texts", value=False)

    st.divider()
    st.subheader("Sample questions")
    SAMPLES = [
        "What is the standard delivery timeline?",
        "What is the return window for most products?",
        "How long do refunds take after quality check?",
        "What does Precision@k measure?",
        "What is hybrid retrieval?",
        "What should you do if evidence is weak?",
        "When is Cash on Delivery available?",
        "Give one privacy best practice for payment data.",
        "What chunk size is recommended for narrative PDFs?",
        "When should a support issue be escalated to a human?",
    ]
    selected_sample = st.selectbox("Pick a sample ↓", ["— type your own —"] + SAMPLES)

    st.divider()
    use_llm = st.checkbox(
        "Use LLM (OpenAI)",
        value=bool(os.getenv("OPENAI_API_KEY")),
        help="Requires OPENAI_API_KEY environment variable.",
    )

    if st.button("Rebuild index from PDFs"):
        with st.spinner("Running ingestion pipeline …"):
            try:
                from ingest import build_index
                st.cache_resource.clear()
                build_index()
                st.success("Index rebuilt successfully.")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")


# ── Main area ─────────────────────────────────────────────────────────────────

st.header("Ask a question about your documents")

default_query = (
    selected_sample
    if selected_sample != "— type your own —"
    else ""
)
query = st.text_input(
    "Question:",
    value=default_query,
    placeholder="e.g.  What is the standard delivery timeline?",
    label_visibility="collapsed",
)

search_clicked = st.button("🔍  Search", type="primary", use_container_width=False)

if search_clicked and query.strip():
    # ── Load components (will use cache after first call) ──────────────────
    index_missing = False
    try:
        retriever = _load_retriever()
    except FileNotFoundError as exc:
        st.error(
            f"**Index not found.**  Click **Rebuild index from PDFs** in the "
            f"sidebar to build it.\n\n`{exc}`"
        )
        index_missing = True

    if not index_missing:
        generator = _load_generator(use_llm=use_llm)

        with st.spinner("Retrieving …"):
            chunks = retriever.retrieve(query, top_k=top_k)

        with st.spinner("Generating answer …"):
            result = generator.generate(query, chunks)

        # ── Layout: answer (left) + metrics (right) ────────────────────────
        st.divider()
        col_answer, col_meta = st.columns([3, 1])

        with col_answer:
            st.subheader("Answer")
            mode = result["mode"]
            conf = result["confidence"]

            if mode == "refused":
                st.error(result["answer"])
            elif result["low_confidence_flag"]:
                st.warning(result["answer"])
            else:
                st.success(result["answer"])

        with col_meta:
            # Confidence gauge
            if conf < LOW_CONFIDENCE_THRESHOLD:
                conf_label, conf_color = "LOW", "🔴"
            elif conf < CLARIFY_THRESHOLD:
                conf_label, conf_color = "MEDIUM", "🟡"
            else:
                conf_label, conf_color = "HIGH", "🟢"

            st.metric("Confidence", f"{conf:.4f}")
            st.markdown(f"**Level:** {conf_color} {conf_label}")
            st.metric("Mode", mode.capitalize())

            if result["citations"]:
                st.markdown("**Citations**")
                for c in result["citations"]:
                    st.code(f"[{c}]", language=None)

        # ── Retrieval trace table ──────────────────────────────────────────
        if show_trace and chunks:
            st.divider()
            st.subheader("🔍 Retrieval Trace")

            trace_rows = [
                {
                    "Rank":     c["rank"],
                    "Chunk ID": c["chunk_id"],
                    "Citation": c["citation"],
                    "Score":    round(c["score"], 4),
                    "Preview":  (
                        c["text"][:110] + "…"
                        if len(c["text"]) > 110
                        else c["text"]
                    ),
                }
                for c in chunks
            ]
            st.dataframe(trace_rows, use_container_width=True, hide_index=True)

        # ── Full chunk texts ───────────────────────────────────────────────
        if show_chunks and chunks:
            st.subheader("📄 Full Chunk Texts")
            for c in chunks:
                label = (
                    f"[Rank {c['rank']}]  {c['citation']}  "
                    f"·  chunk_id={c['chunk_id']}  ·  score={c['score']:.4f}"
                )
                with st.expander(label):
                    st.text(c["text"])

elif not query.strip():
    st.info(
        "Type a question above (or pick a sample from the sidebar) "
        "and click **Search**."
    )
