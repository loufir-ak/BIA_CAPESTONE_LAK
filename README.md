# Document Navigator – Transparent PDF Q&A System

A local RAG (Retrieval-Augmented Generation) pipeline over a PDF corpus.
Every answer comes with **retrieval traces**, **[filename:page] citations**,
and transparent **confidence gating**.

---

## Architecture

```
pdfs/
 └─ *.pdf
     │  extract_pages()
     ▼
  chunks (400 char, 80 overlap)
     │  SentenceTransformer.encode()
     ▼
  FAISS IndexFlatIP          ← index/chunks.index
  chunk metadata JSON        ← index/chunks_meta.json
     │
     │  query ──► embed ──► search(top-k)
     ▼
  Retriever  →  retrieval trace (rank, chunk_id, citation, score, text)
     │
     ▼
  AnswerGenerator
     ├─ score < 0.30 → REFUSE  (low confidence)
     ├─ score < 0.45 → ANSWER + caveat  (medium confidence)
     └─ score ≥ 0.45 → ANSWER  (high confidence)
          │
          ├─ extractive mode (default, no API key needed)
          └─ LLM mode (OpenAI, set OPENAI_API_KEY)
```

---

## Quick Start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

> `sentence-transformers` pulls in PyTorch (~600 MB first install).
> All subsequent runs use the cached model.

### 2 – Ingest PDFs and build the index

```bash
python ingest.py
```

Creates `index/chunks.index` and `index/chunks_meta.json`.

### 3 – Ask a question (CLI)

```bash
python pipeline.py "What is the standard delivery timeline?"
python pipeline.py "What does Precision@k measure?" --top-k 3
python pipeline.py "Something completely off-topic"          # triggers low-conf
python pipeline.py "What is hybrid retrieval?" --llm        # OpenAI synthesis
```

### 4 – Run evaluation

```bash
python evaluate.py
```

Outputs:

- `retrieval_report.md` – precision@3, precision@5, top-1 accuracy, key-phrase match
- `eval_results.json`   – raw per-question data

### 5 – Launch the Streamlit UI

```bash
streamlit run app.py
```

---

## File Reference

| File | Purpose |
|------|---------|
| `config.py` | All tunable constants (chunk size, thresholds, model name) |
| `ingest.py` | PDF → chunks → embeddings → FAISS index |
| `retriever.py` | FAISS search + retrieval trace formatting |
| `generator.py` | Extractive / LLM answer generation with citations |
| `pipeline.py` | CLI orchestration (`ingest → retrieve → generate`) |
| `evaluate.py` | Precision@k evaluation over `eval_set.csv` |
| `app.py` | Streamlit demo UI |
| `requirements.txt` | Python dependencies |

---

## Key Design Decisions

### Chunking

- **400 characters with 80-char overlap** – sized for 1-page synthetic PDFs;
  the 80-char overlap preserves context across chunk boundaries.
- Sentence-boundary snapping: the splitter tries to break at `.` rather
  than mid-word.

### Embeddings

- `all-MiniLM-L6-v2` (22 M parameters) – fast, offline, good semantic
  quality for short passages.  L2-normalised before indexing so inner
  product equals cosine similarity.

### Confidence gating

| Score | Behaviour |
|-------|-----------|
| < 0.30 | Refuse – "not enough evidence" |
| 0.30 – 0.45 | Answer with visible caveat banner |
| ≥ 0.45 | Full answer |

### Citations

Format: `[filename.pdf:page]`  matches `eval_set.csv` `gold_citation` column.

### LLM usage

Optional and minimal – only one `chat.completions.create` call per query,
with a strict grounding prompt that forbids outside knowledge.

---

## Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| precision@k | `relevant_in_top_k / k`; relevant = source file matches gold |
| top-1 accuracy | `top1_source == gold_source` across all questions |
| key-phrase match | all first-4 tokens of `gold_key_phrase` appear in answer |

---

## Extending the System

- **More PDFs**: drop files into `pdfs/` and re-run `python ingest.py`.
- **Larger chunk size**: edit `CHUNK_SIZE` in `config.py` (try 800 for longer docs).
- **Hybrid retrieval**: add BM25 scoring with `rank_bm25` and combine scores.
- **Better LLM**: swap `gpt-3.5-turbo` for `gpt-4o` in `generator.py`.
- **Persistent UI state**: Streamlit `session_state` already handles query
  history; extend `app.py` to store past Q&A pairs.
