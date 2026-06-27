"""
evaluate.py – Retrieval quality evaluation  (precision@k + answer checks)
==========================================================================
Runs every question in eval_set.csv through the RAG pipeline and measures:

  precision@k  – fraction of top-k chunks whose *source file* matches the
                  gold citation.  (All PDFs in this dataset are 1-page, so
                  the filename alone is the discriminating key.)

  top-1 accuracy – whether the single highest-scored chunk comes from the
                   correct document.

  key-phrase match – lightweight check that core words from gold_key_phrase
                     appear somewhere in the generated answer.

Outputs:
  stdout          – formatted report
  retrieval_report.md – filled-in version of retrieval_report_template.md
  eval_results.json   – full per-question data for further analysis

Usage:
    python evaluate.py
    python evaluate.py --llm      # use OpenAI if OPENAI_API_KEY is set
    python evaluate.py --top-k 3  # override default k
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any

from generator import AnswerGenerator
from retriever import Retriever
from config import TOP_K


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_eval_set(path: str = "eval_set.csv") -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def precision_at_k(
    retrieved: list[dict[str, Any]],
    gold_source: str,
    k: int,
) -> float:
    """Fraction of top-k retrieved chunks whose source == gold_source."""
    top_k    = retrieved[:k]
    relevant = sum(1 for c in top_k if c["source"] == gold_source)
    return relevant / k if k > 0 else 0.0


def key_phrase_match(answer: str, key_phrase: str) -> bool:
    """
    True if all words of the first 4 tokens of `key_phrase` appear in `answer`.
    This is intentionally lenient – it checks presence, not order.
    """
    answer_lower = answer.lower()
    probe_words  = key_phrase.lower().split()[:4]
    return all(w in answer_lower for w in probe_words)


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(
    eval_path: str     = "eval_set.csv",
    k_values:  list[int] = None,
    use_llm:   bool    = False,
) -> dict[str, Any]:
    if k_values is None:
        k_values = [3, 5]

    retriever = Retriever()
    generator = AnswerGenerator(use_llm=use_llm)
    eval_set  = _load_eval_set(eval_path)

    per_question: list[dict[str, Any]] = []

    for row in eval_set:
        qid          = row["id"]
        question     = row["question"]
        gold_citation  = row["gold_citation"]
        gold_key_phrase = row["gold_key_phrase"]
        gold_source  = gold_citation.split(":")[0]   # filename only

        # Retrieve
        chunks = retriever.retrieve(question, top_k=max(k_values))

        # Generate
        gen = generator.generate(question, chunks)
        answer = gen["answer"]

        # Metrics
        p_at_k = {
            f"p@{k}": precision_at_k(chunks, gold_source, k)
            for k in k_values
        }
        top1_correct = bool(chunks and chunks[0]["source"] == gold_source)
        kp_match     = key_phrase_match(answer, gold_key_phrase)

        per_question.append(
            {
                "id":             qid,
                "question":       question,
                "gold_citation":  gold_citation,
                "gold_key_phrase":gold_key_phrase,
                "answer":         answer,
                "top_score":      chunks[0]["score"] if chunks else 0.0,
                "top_source":     chunks[0]["source"] if chunks else "N/A",
                "top1_correct":   top1_correct,
                "kp_match":       kp_match,
                **p_at_k,
            }
        )

    n = len(per_question)
    metrics: dict[str, float] = {
        f"precision@{k}": sum(r[f"p@{k}"] for r in per_question) / n
        for k in k_values
    }
    metrics["top1_accuracy"]        = sum(r["top1_correct"] for r in per_question) / n
    metrics["key_phrase_match_rate"] = sum(r["kp_match"] for r in per_question) / n

    return {"metrics": metrics, "per_question": per_question}


# ── Report printers ───────────────────────────────────────────────────────────

def print_report(results: dict[str, Any]) -> None:
    metrics = results["metrics"]
    per_q   = results["per_question"]

    bar = "=" * 64
    print(f"\n{bar}")
    print("  RETRIEVAL EVALUATION REPORT")
    print(f"{bar}")

    print("\nOverall Metrics:")
    for key, val in metrics.items():
        print(f"  {key:<28}: {val:.4f}  ({val * 100:.1f} %)")

    # Per-question table
    k_keys = sorted(k for k in per_q[0] if k.startswith("p@"))
    header_cols = ["ID", *[k.upper() for k in k_keys], "TOP1", "KP", "SCORE", "QUESTION"]
    print(
        f"\n{'ID':<5} "
        + "".join(f"{k.upper():<7}" for k in k_keys)
        + f"{'TOP1':<6}{'KP':<6}{'SCORE':<8}  QUESTION"
    )
    print("-" * 80)
    for r in per_q:
        p_cols = "".join(f"{r[k]:<7.2f}" for k in k_keys)
        print(
            f"{r['id']:<5} {p_cols}"
            f"{'✓' if r['top1_correct'] else '✗':<6}"
            f"{'✓' if r['kp_match'] else '✗':<6}"
            f"{r['top_score']:<8.4f}  {r['question'][:42]}"
        )

    # Failures
    failures = [r for r in per_q if not r["top1_correct"]]
    if failures:
        print(f"\nRetrieval failures ({len(failures)} / {len(per_q)}):")
        for r in failures:
            print(
                f"  {r['id']:>3}: expected '{r['gold_citation']}', "
                f"got '{r['top_source']}'"
            )
    else:
        print(f"\nAll {len(per_q)} questions retrieved top-1 correctly!")

    # Good examples
    goods = [r for r in per_q if r["top1_correct"] and r["kp_match"]]
    if goods:
        print(f"\nGood examples ({len(goods)}):")
        for r in goods[:3]:
            print(f"  {r['id']}: {r['question']}")
            print(f"    Answer  : {r['answer'][:120]}")
            print(f"    Gold    : {r['gold_citation']}")


def save_report(
    results: dict[str, Any],
    output_path: str = "retrieval_report.md",
) -> None:
    metrics = results["metrics"]
    per_q   = results["per_question"]
    goods   = [r for r in per_q if r["top1_correct"] and r["kp_match"]]
    failures = [r for r in per_q if not r["top1_correct"]]

    k_values = sorted(
        int(k.split("@")[1])
        for k in metrics
        if k.startswith("precision@")
    )

    lines = [
        "# retrieval_report.md – Document Navigator",
        "",
        "## Setup",
        "- PDFs: 10",
        f"- Chunk size : {400} characters",
        f"- Overlap    : {80} characters",
        "- Embeddings model : all-MiniLM-L6-v2  (sentence-transformers)",
        "- Vector index     : FAISS IndexFlatIP  (cosine similarity)",
        "",
        "## Metrics",
    ]
    for k in k_values:
        v = metrics[f"precision@{k}"]
        lines.append(f"- precision@{k}           : {v:.4f}  ({v * 100:.1f} %)")
    v = metrics["top1_accuracy"]
    lines.append(f"- top-1 accuracy          : {v:.4f}  ({v * 100:.1f} %)")
    v = metrics["key_phrase_match_rate"]
    lines.append(f"- key-phrase match rate   : {v:.4f}  ({v * 100:.1f} %)")

    lines += ["", "## Examples (Good)"]
    for r in goods[:3]:
        lines += [
            f"- Q: {r['question']}",
            f"  - Retrieved : {r['top_source']} (score={r['top_score']:.4f})",
            f"  - Answer    : {r['answer'][:160]}",
            "",
        ]

    lines += ["## Examples (Failures)"]
    if failures:
        for r in failures[:3]:
            lines += [
                f"- Q: {r['question']}",
                f"  - Retrieved : {r['top_source']}  "
                f"(expected {r['gold_citation'].split(':')[0]})",
                f"  - Why it failed : Top-1 source did not match gold.",
                f"  - Fix attempted : Tune chunk size / try hybrid retrieval.",
                "",
            ]
    else:
        lines.append("- No retrieval failures recorded.")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nReport saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    p.add_argument("--top-k", type=int, default=TOP_K,
                   help=f"Maximum k for precision@k (default: {TOP_K})")
    p.add_argument("--llm", action="store_true",
                   help="Use OpenAI LLM for answer generation")
    p.add_argument("--eval-set", default="eval_set.csv",
                   help="Path to evaluation CSV (default: eval_set.csv)")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    k_values = sorted({3, min(5, args.top_k), args.top_k})

    try:
        results = run_evaluation(
            eval_path=args.eval_set,
            k_values=k_values,
            use_llm=args.llm,
        )
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print_report(results)
    save_report(results)

    with open("eval_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("Raw results saved → eval_results.json")
