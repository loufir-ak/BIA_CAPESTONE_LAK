"""
ingest.py – PDF ingestion pipeline
====================================
Reads every PDF in PDF_DIR, extracts text, splits into overlapping character
chunks, embeds with sentence-transformers, and stores a FAISS IndexFlatIP
(inner-product = cosine for L2-normalised vectors).

Outputs (written to INDEX_DIR):
  chunks.index      – FAISS binary index
  chunks_meta.json  – list of chunk dicts {chunk_id, source, page, text, citation}

Usage:
  python ingest.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDEX_DIR,
    PDF_DIR,
)


# ── Text extraction ──────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Collapse excessive whitespace while keeping paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)          # multi-space → single space
    text = re.sub(r"\n{3,}", "\n\n", text)        # triple+ newlines → double
    return text.strip()


def extract_pages(pdf_path: str) -> list[dict[str, Any]]:
    """
    Return a list of {page: int, text: str} dicts, one per page.
    Tries PyPDF2 first; falls back to pdfminer.six on failure.
    """
    pages: list[dict[str, Any]] = []

    # ── attempt 1: PyPDF2 ──────────────────────────────────────────────────
    try:
        import PyPDF2  # type: ignore

        with open(pdf_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page_num, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                text = _clean_text(raw)
                if text:
                    pages.append({"page": page_num, "text": text})
        if pages:
            return pages
    except Exception as exc:
        print(f"  [PyPDF2 warn] {Path(pdf_path).name}: {exc}")

    # ── attempt 2: pdfminer.six ────────────────────────────────────────────
    try:
        from pdfminer.high_level import extract_pages as pm_extract  # type: ignore
        from pdfminer.layout import LTTextContainer  # type: ignore

        for page_num, layout in enumerate(pm_extract(pdf_path), start=1):
            raw = "".join(
                el.get_text()
                for el in layout
                if isinstance(el, LTTextContainer)
            )
            text = _clean_text(raw)
            if text:
                pages.append({"page": page_num, "text": text})
        return pages
    except Exception as exc:
        print(f"  [pdfminer error] {Path(pdf_path).name}: {exc}")

    return pages


# ── Chunking ─────────────────────────────────────────────────────────────────

def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Slide a window of `chunk_size` characters over `text` with `overlap`.
    Tries to break at a sentence boundary ('. ') within the last quarter of
    the window rather than cutting mid-word.
    """
    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        segment = text[start:end]

        # Prefer breaking at the last '. ' in the final quarter of the window
        if end < length:
            boundary_search_start = max(0, len(segment) - chunk_size // 4)
            last_period = segment.rfind(". ", boundary_search_start)
            if last_period != -1:
                end = start + last_period + 2   # include the '. '
                segment = text[start:end]

        segment = segment.strip()
        if segment:
            chunks.append(segment)

        if end >= length:
            break
        start = end - overlap

    return chunks


# ── Indexing ─────────────────────────────────────────────────────────────────

def build_index(
    pdf_dir: str = PDF_DIR,
    index_dir: str = INDEX_DIR,
) -> tuple[list[dict], Any]:
    """
    Full ingestion pipeline.

    Returns:
        all_chunks – list of chunk metadata dicts
        faiss_index – the built FAISS index object
    """
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    os.makedirs(index_dir, exist_ok=True)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_chunks: list[dict] = []
    chunk_id = 0

    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{pdf_dir}'")

    print(f"\nFound {len(pdf_files)} PDF(s) in '{pdf_dir}'\n")

    for pdf_path in pdf_files:
        filename = pdf_path.name
        print(f"  Processing: {filename}")
        pages = extract_pages(str(pdf_path))

        if not pages:
            print(f"    [skip] No extractable text.")
            continue

        for page_data in pages:
            page_num  = page_data["page"]
            page_text = page_data["text"]
            chunks    = split_into_chunks(page_text)

            for chunk_text in chunks:
                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source":   filename,
                        "page":     page_num,
                        "text":     chunk_text,
                        "citation": f"{filename}:{page_num}",
                    }
                )
                chunk_id += 1

        print(f"    → {len(pages)} page(s), {chunk_id} total chunks so far")

    if not all_chunks:
        raise RuntimeError("No chunks generated – check PDF content.")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Generating embeddings …")

    texts      = [c["text"] for c in all_chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise → IP == cosine
        batch_size=64,
    ).astype(np.float32)

    # FAISS index: inner product (= cosine for normalised vectors)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist
    index_path = os.path.join(index_dir, "chunks.index")
    meta_path  = os.path.join(index_dir, "chunks_meta.json")

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, indent=2, ensure_ascii=False)

    print(f"\nIndex saved to '{index_dir}/'")
    print(f"  chunks.index      – FAISS IndexFlatIP ({len(all_chunks)} vectors, dim={dim})")
    print(f"  chunks_meta.json  – chunk metadata")

    return all_chunks, index


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_index()
