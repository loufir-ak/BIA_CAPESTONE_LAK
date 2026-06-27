"""
generator.py – Grounded answer generation with citations and confidence gating
===============================================================================
Generates answers from retrieved chunks.  Two modes:

  extractive (default, zero-cost)
      Picks the most query-relevant sentence from the top chunk and appends
      a [filename:page] citation.  No API key required.

  llm (optional)
      Calls OpenAI chat completions with a strict grounding prompt.
      Activated automatically when OPENAI_API_KEY is set in the environment,
      or when AnswerGenerator(use_llm=True) is instantiated.

Low-confidence handling
-----------------------
  score < LOW_CONFIDENCE_THRESHOLD  → refuse ("not enough evidence")
  score < CLARIFY_THRESHOLD         → answer with a visible caveat banner
"""

from __future__ import annotations

import os
from typing import Any

from config import CLARIFY_THRESHOLD, LOW_CONFIDENCE_THRESHOLD


class AnswerGenerator:
    """Generate a grounded, cited answer from a set of retrieved chunks."""

    def __init__(
        self,
        use_llm: bool = False,
        openai_api_key: str | None = None,
    ) -> None:
        self._llm_client = None
        self._llm_model  = "gpt-3.5-turbo"

        effective_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if use_llm and effective_key:
            try:
                from openai import OpenAI  # type: ignore

                self._llm_client = OpenAI(api_key=effective_key)
                self._mode = "llm"
            except ImportError:
                print(
                    "[generator] openai package not installed – "
                    "falling back to extractive mode."
                )
                self._mode = "extractive"
        else:
            self._mode = "extractive"

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Build a grounded answer.

        Returns a dict with keys:
            answer            – the answer string (may include caveats)
            citations         – deduplicated list of "[filename:page]" strings
            confidence        – top-1 cosine similarity score
            mode              – "refused" | "extractive" | "llm"
            low_confidence_flag – True when confidence is below CLARIFY_THRESHOLD
        """
        # ── No results ──────────────────────────────────────────────────────
        if not retrieved_chunks:
            return {
                "answer": (
                    "I could not find any relevant information in the "
                    "available documents.  Please try rephrasing your question."
                ),
                "citations":           [],
                "confidence":          0.0,
                "mode":                "refused",
                "low_confidence_flag": True,
            }

        top_score = retrieved_chunks[0]["score"]

        # ── Hard refuse ─────────────────────────────────────────────────────
        if top_score < LOW_CONFIDENCE_THRESHOLD:
            return {
                "answer": (
                    "I don't have enough evidence in the documents to answer "
                    "this question confidently.  Could you rephrase, or ask "
                    "about a topic covered in the available PDFs?"
                ),
                "citations":           [],
                "confidence":          top_score,
                "mode":                "refused",
                "low_confidence_flag": True,
            }

        # ── Generate answer ─────────────────────────────────────────────────
        needs_caveat = top_score < CLARIFY_THRESHOLD

        if self._mode == "llm" and self._llm_client is not None:
            answer, citations = self._llm_answer(query, retrieved_chunks)
            mode = "llm"
        else:
            answer, citations = self._extractive_answer(query, retrieved_chunks)
            mode = "extractive"

        if needs_caveat:
            caveat = (
                f"[Confidence: {top_score:.2f} – moderate.  "
                "The answer below is based on the best available match; "
                "please verify with the cited source.]\n\n"
            )
            answer = caveat + answer

        return {
            "answer":            answer,
            "citations":         citations,
            "confidence":        top_score,
            "mode":              mode,
            "low_confidence_flag": needs_caveat,
        }

    # ── Extractive answer ─────────────────────────────────────────────────

    def _extractive_answer(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> tuple[str, list[str]]:
        """
        Select the sentence from the top chunk that shares the most content
        words with the query, then format it with a citation.
        """
        top_chunk = chunks[0]
        citation  = top_chunk["citation"]
        text      = top_chunk["text"]

        # Tokenise query into meaningful words (≥4 chars, ignoring stopwords)
        _STOPWORDS = {
            "what", "which", "when", "where", "does", "have", "should",
            "would", "could", "there", "their", "this", "that", "with",
            "from", "about", "into", "after", "give", "name",
        }
        query_words = {
            w.lower().strip("?.,")
            for w in query.split()
            if len(w) >= 4 and w.lower() not in _STOPWORDS
        }

        # Split chunk into sentences
        sentences = [
            s.strip()
            for s in text.replace("\n", " ").split(".")
            if len(s.strip()) > 15
        ]

        best_sentence = sentences[0] if sentences else text[:300]
        best_overlap  = -1

        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap    = len(query_words & sent_words)
            if overlap > best_overlap:
                best_overlap  = overlap
                best_sentence = sent

        answer = f"{best_sentence.strip()}. [{citation}]"

        # Deduplicated citations from top-3 chunks
        citations = list(
            dict.fromkeys(c["citation"] for c in chunks[:3])
        )
        return answer, citations

    # ── LLM answer ────────────────────────────────────────────────────────

    def _llm_answer(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> tuple[str, list[str]]:
        """
        Send a strictly grounded prompt to the OpenAI chat API.
        Uses at most 5 chunks to keep token usage low.
        """
        context = "\n\n---\n\n".join(
            f"[Source: {c['citation']}]\n{c['text']}"
            for c in chunks[:5]
        )

        system_prompt = (
            "You are a precise document assistant.  "
            "Answer ONLY using the provided context – do not add outside knowledge.  "
            "End every factual claim with its citation in the format [filename:page].  "
            "If the context lacks sufficient information, say so explicitly.  "
            "Be concise (2–4 sentences max)."
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer (with inline citations):"
        )

        response = self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        answer    = response.choices[0].message.content.strip()
        citations = list(dict.fromkeys(c["citation"] for c in chunks[:5]))
        return answer, citations
