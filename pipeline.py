"""
pipeline.py – End-to-end CLI for the PDF Q&A system
=====================================================
Orchestrates: Retriever → AnswerGenerator → pretty-printed trace + answer.

Usage:
    python pipeline.py "What is the standard delivery timeline?"
    python pipeline.py "What does Precision@k measure?" --top-k 3
    python pipeline.py "Some out-of-scope question" --top-k 5
    python pipeline.py "Your question" --llm          # use OpenAI if key set
"""

from __future__ import annotations

import uvicorn
import argparse
import sys
from typing import Any

from generator import AnswerGenerator
from retriever import Retriever
from config import TOP_K


# ── Core function (importable) ────────────────────────────────────────────────

def run_query(
    query: str,
    top_k: int  = TOP_K,
    use_llm: bool = False,
) -> dict[str, Any]:
    """
    Full RAG pipeline for a single question.

    Returns a dict with:
        query, answer, citations, confidence, mode,
        low_confidence_flag, trace (str), retrieved_chunks (list)
    """
    retriever = Retriever()
    generator = AnswerGenerator(use_llm=use_llm)

    chunks = retriever.retrieve(query, top_k=top_k)
    trace  = retriever.format_trace(chunks)
    result = generator.generate(query, chunks)

    return {
        "query":               query,
        "answer":              result["answer"],
        "citations":           result["citations"],
        "confidence":          result["confidence"],
        "mode":                result["mode"],
        "low_confidence_flag": result["low_confidence_flag"],
        "trace":               trace,
        "retrieved_chunks":    chunks,
    }


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_result(result: dict[str, Any]) -> None:
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  QUERY: {result['query']}")
    print(f"{bar}")

    print(f"\n  Answer:\n  {result['answer']}")

    if result["citations"]:
        print(f"\n  Citations : {', '.join(result['citations'])}")

    confidence_label = (
        "LOW"    if result["confidence"] < 0.30 else
        "MEDIUM" if result["confidence"] < 0.45 else
        "HIGH"
    )
    print(
        f"  Confidence: {result['confidence']:.4f} [{confidence_label}]"
        f"  |  Mode: {result['mode']}"
    )

    if result["low_confidence_flag"]:
        print("  ⚠  Low/medium confidence – please verify cited sources.")

    print(f"\n{result['trace']}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDF Q&A – ask questions grounded in your document corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python pipeline.py "What is the standard delivery timeline?"\n'
            '  python pipeline.py "What does Precision@k measure?" --top-k 3\n'
            '  python pipeline.py "Some topic not in PDFs"  # triggers low-conf\n'
        ),
    )
    parser.add_argument("query", help="The question to answer")
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K})"
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Use OpenAI LLM for synthesis (requires OPENAI_API_KEY)"
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    try:
        result = run_query(args.query, top_k=args.top_k, use_llm=args.llm)
        print_result(result)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
