# ─── RAG Pipeline Configuration ──────────────────────────────────────────────

# Directories
PDF_DIR   = "pdfs"
INDEX_DIR = "index"

# Chunking  (character-based; ~400 chars ≈ 80–100 tokens for typical prose)
CHUNK_SIZE    = 400   # max characters per chunk
CHUNK_OVERLAP = 80    # characters of overlap between consecutive chunks

# Embedding model (sentence-transformers, runs fully offline)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval
TOP_K = 5             # default number of chunks to retrieve per query

# Confidence thresholds  (cosine similarity, 0 → 1 after L2-normalisation)
LOW_CONFIDENCE_THRESHOLD = 0.30   # below → refuse to answer
CLARIFY_THRESHOLD        = 0.45   # below → answer with a confidence caveat
