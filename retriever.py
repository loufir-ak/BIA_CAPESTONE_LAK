"""
retriever.py – Vector retrieval with full trace output
=======================================================
Loads the FAISS index built by ingest.py and exposes a Retriever class
that returns top-k chunks together with a structured retrieval trace
(rank, chunk_id, source, page, citation, similarity score, text preview).

Usage:
    r = Retriever()
    chunks = r.retrieve("What is the return window?", top_k=5)
    print(r.format_trace(chunks))
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from config import EMBEDDING_MODEL, INDEX_DIR, TOP_K


class Retriever:
    """Thin wrapper around a FAISS index + chunk metadata store."""

    def __init__(self, index_dir: str = INDEX_DIR) -> None:
        index_path = os.path.join(index_dir, "chunks.index")
        meta_path  = os.path.join(index_dir, "chunks_meta.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index not found at '{index_path}'.\n"
                "Run  python ingest.py  to build the index first."
            )

        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._index = faiss.read_index(index_path)

        with open(meta_path, encoding="utf-8") as fh:
            self._chunks: list[dict[str, Any]] = json.load(fh)

    # ── Public API ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Embed `query`, search the FAISS index, and return a retrieval trace.

        Each result dict contains:
            rank       – 1-based position
            chunk_id   – unique integer ID from ingestion
            source     – PDF filename
            page       – page number
            citation   – "filename:page"  (matches eval_set.csv gold format)
            score      – cosine similarity (float, 0–1 for normalised vecs)
            text       – full chunk text
        """
        query_vec = (
            self._model.encode([query], normalize_embeddings=True)
            .astype(np.float32)
        )
        scores, indices = self._index.search(query_vec, top_k)

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0]), start=1
        ):
            if idx == -1:         # FAISS returns -1 when fewer results exist
                continue
            entry = dict(self._chunks[idx])   # shallow copy
            entry["rank"]  = rank
            entry["score"] = float(score)
            results.append(entry)

        return results

    # ── Formatting helpers ─────────────────────────────────────────────────

    def format_trace(self, results: list[dict[str, Any]]) -> str:
        """
        Render a human-readable retrieval trace table.

        Example output:
          ─── Retrieval Trace ──────────────────────────────────────────────
          [1] chunk_id=7  |  guide_rag_basics.pdf:1  |  score=0.7832
              "Citations allow users to verify the source of each claim…"
          [2] …
        """
        if not results:
            return "--- Retrieval Trace: no results ---"

        lines = ["─── Retrieval Trace " + "─" * 44]
        for r in results:
            preview = r["text"][:140].replace("\n", " ")
            if len(r["text"]) > 140:
                preview += "…"
            lines.append(
                f"  [{r['rank']}] chunk_id={r['chunk_id']:<4} | "
                f"{r['citation']:<40} | score={r['score']:.4f}"
            )
            lines.append(f'      "{preview}"')
        lines.append("─" * 64)
        return "\n".join(lines)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
